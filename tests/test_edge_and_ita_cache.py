"""Unit tests for the edge-export and ITA-cache utilities.

These avoid the AdaFace repo dependency by using a tiny stand-in backbone, so
they run anywhere the eval tests run (CPU, no GPU, no mxnet).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def _write_solid(path: Path, bgr: tuple[int, int, int], size: int = 112) -> None:
    """Write a solid-color BGR JPEG."""
    arr = np.full((size, size, 3), bgr, dtype=np.uint8)
    cv2.imwrite(str(path), arr)


# ---------- precompute_ita ----------

def test_ita_one_matches_compute_ita(tmp_path: Path) -> None:
    """The worker wrapper returns (path, ita) consistent with compute_ita."""
    from evaluate_ita import compute_ita
    from precompute_ita import _ita_one

    img = tmp_path / "x.jpg"
    _write_solid(img, (180, 200, 230))
    path, v = _ita_one(str(img))
    assert path == str(img)
    assert v == compute_ita(img)


def test_compute_cache_keys_and_none(tmp_path: Path) -> None:
    """compute_cache returns one entry per path; unreadable paths map to None."""
    from precompute_ita import compute_cache

    good = tmp_path / "good.jpg"
    _write_solid(good, (210, 220, 235))
    bad = tmp_path / "missing.jpg"
    cache = compute_cache([str(good), str(bad)], workers=1)
    assert set(cache) == {str(good), str(bad)}
    assert cache[str(good)] is not None
    assert cache[str(bad)] is None


# ---------- export_edge ----------

class _TinyBackbone(torch.nn.Module):
    """Returns (embedding, norm) like the AdaFace backbone, but minimal."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, dim, 3, stride=2, padding=1)
        self.fc = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        feat = self.conv(x).mean(dim=(2, 3))
        emb = self.fc(feat)
        norm = emb.norm(dim=1, keepdim=True)
        return emb, norm


def test_embedding_only_returns_first_output() -> None:
    """EmbeddingOnly drops the norm and returns just the embedding tensor."""
    from export_edge import EmbeddingOnly

    m = EmbeddingOnly(_TinyBackbone())
    out = m(torch.randn(2, 3, 112, 112))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8)


def test_count_params_positive() -> None:
    """count_params counts all backbone parameters."""
    from export_edge import EmbeddingOnly, count_params

    n = count_params(EmbeddingOnly(_TinyBackbone()))
    assert n > 0


def test_benchmark_cpu_returns_positive_latency() -> None:
    """benchmark_cpu returns a positive millisecond latency."""
    from export_edge import EmbeddingOnly, benchmark_cpu

    ms = benchmark_cpu(EmbeddingOnly(_TinyBackbone()), runs=3, warmup=1)
    assert ms > 0.0


def test_export_onnx_roundtrip(tmp_path: Path) -> None:
    """export_onnx writes a file; if onnxruntime is present, outputs match torch."""
    import pytest

    pytest.importorskip("onnx", reason="onnx package required to serialize ONNX")
    from export_edge import EmbeddingOnly, export_onnx, verify_onnx

    model = EmbeddingOnly(_TinyBackbone()).eval()
    out = tmp_path / "tiny.onnx"
    export_onnx(model, out)
    assert out.exists() and out.stat().st_size > 0
    diff = verify_onnx(model, out)
    if diff is not None:  # onnxruntime installed
        assert diff < 1e-3
