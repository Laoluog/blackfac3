"""Export a trained backbone to ONNX and benchmark it for edge deployment.

The "edge" half of the story: take the distilled student checkpoint, export it
to ONNX (the portable format mobile/embedded runtimes consume), and report the
numbers a deployment section needs — parameter count, on-disk size, and CPU
latency. Embeddings from the exported model are checked against the PyTorch
model so the export is verified, not assumed.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from train import add_repo_to_path, build_backbone

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


class EmbeddingOnly(torch.nn.Module):
    """Wrap a backbone so forward returns only the embedding (clean ONNX graph)."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def load_backbone(arch: str, ckpt: str, adaface_repo: str) -> torch.nn.Module:
    """Build the backbone and load weights from a checkpoint (strips head/optimizer)."""
    model = build_backbone(arch, adaface_repo)
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)
    return model.eval()


def count_params(model: torch.nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def benchmark_cpu(model: torch.nn.Module, runs: int = 50, warmup: int = 10) -> float:
    """Return mean single-image CPU latency in milliseconds."""
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        dt = time.perf_counter() - t0
    return 1000.0 * dt / runs


def export_onnx(model: torch.nn.Module, out_path: Path, opset: int = 12) -> None:
    """Export to ONNX with a dynamic batch dimension."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # legacy TorchScript exporter: no onnxscript dependency
    )


def verify_onnx(model: torch.nn.Module, onnx_path: Path) -> float | None:
    """Return max abs difference between torch and ONNX embeddings, or None if ORT missing."""
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        logger.warning("onnxruntime not installed; skipping numerical verification")
        return None
    x = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        torch_out = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.numpy()})[0]
    return float(np.abs(torch_out - onnx_out).max())


def parse_args() -> argparse.Namespace:
    """Build CLI parser."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--adaface-repo", required=True)
    p.add_argument("--arch", default="ir_18")
    p.add_argument("--out", required=True, help="Output .onnx path")
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--bench-runs", type=int, default=50)
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = parse_args()
    add_repo_to_path(args.adaface_repo)

    backbone = load_backbone(args.arch, args.checkpoint, args.adaface_repo)
    model = EmbeddingOnly(backbone).eval()

    params = count_params(model)
    latency_ms = benchmark_cpu(model, runs=args.bench_runs)

    out_path = Path(args.out)
    export_onnx(model, out_path, opset=args.opset)
    size_mb = out_path.stat().st_size / 1e6
    max_diff = verify_onnx(model, out_path)

    logger.info("arch=%s", args.arch)
    logger.info("params=%.2fM", params / 1e6)
    logger.info("onnx_size=%.2f MB (%s)", size_mb, out_path)
    logger.info("cpu_latency=%.2f ms/image", latency_ms)
    if max_diff is not None:
        logger.info("onnx vs torch max abs diff=%.2e (%s)", max_diff, "OK" if max_diff < 1e-3 else "CHECK")

    print("\n=== edge export summary ===")
    print(f"arch          : {args.arch}")
    print(f"params        : {params/1e6:.2f} M")
    print(f"onnx size     : {size_mb:.2f} MB")
    print(f"cpu latency   : {latency_ms:.2f} ms/image")
    if max_diff is not None:
        print(f"onnx vs torch : {max_diff:.2e} max abs diff")


if __name__ == "__main__":
    main()
