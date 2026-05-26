"""Evaluate a face recognition model on the RFW dataset.

Loads an AdaFace-compatible IResNet checkpoint, runs verification on all
RFW pairs for each of the four race subsets (African, Asian, Caucasian,
Indian), and reports TAR@FAR, accuracy at best threshold, and AUC.

Per-pair cosine-similarity scores and labels are also persisted so that
downstream tools (e.g. evaluate_ita.py) can re-bin them without re-running
inference.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve
from tqdm import tqdm

logger = logging.getLogger(__name__)

RACES = ("African", "Asian", "Caucasian", "Indian")
DEFAULT_FAR_TARGETS = (0.001, 0.01, 0.1)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with timestamped format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def load_model(
    checkpoint_path: str,
    adaface_repo: str,
    arch: str = "ir_101",
) -> tuple[torch.nn.Module, torch.device]:
    """Build an AdaFace IResNet model and load weights from a checkpoint.

    Args:
        checkpoint_path: Path to a `.ckpt` (Lightning) or `.pt` state dict.
        adaface_repo: Path to AdaFace repository (must contain ``net.py``).
        arch: Architecture identifier accepted by ``net.build_model``.

    Returns:
        Tuple of (eval-mode model on best device, the device).
    """
    repo = str(Path(adaface_repo).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import net  # type: ignore

    model = net.build_model(arch)
    state = torch.load(checkpoint_path, map_location="cpu")
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    return model, device


def preprocess_image(image_path: Path | str) -> torch.Tensor | None:
    """Read an image and convert to model input tensor.

    Resizes to 112x112, converts BGR->RGB, normalizes to [-1, 1], and
    returns a tensor of shape (1, 3, 112, 112). Returns None if the image
    cannot be read.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return torch.from_numpy(img)


@torch.no_grad()
def get_embedding(
    model: torch.nn.Module, tensor: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Forward pass and return L2-normalized embedding as a 1-D numpy array."""
    out = model(tensor.to(device))
    emb = out[0] if isinstance(out, (tuple, list)) else out
    emb = emb.detach().cpu().numpy()[0].astype(np.float32)
    n = np.linalg.norm(emb)
    return emb / (n + 1e-10)


def parse_pairs_file(
    pairs_file: Path | str, race: str, rfw_root: Path | str
) -> list[tuple[Path, Path, int]]:
    """Parse an RFW pairs file into (img1, img2, label) tuples.

    Genuine (label=1): "name<ws>idx1<ws>idx2"
    Imposter (label=0): "name1<ws>idx1<ws>name2<ws>idx2"

    Image filenames follow ``{name}_{idx:04d}.jpg`` under
    ``{rfw_root}/data/{race}/{name}/``.
    """
    race_dir = Path(rfw_root) / "data" / race
    pairs: list[tuple[Path, Path, int]] = []
    with open(pairs_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 3:
                name, i1, i2 = parts
                p1 = race_dir / name / f"{name}_{int(i1):04d}.jpg"
                p2 = race_dir / name / f"{name}_{int(i2):04d}.jpg"
                pairs.append((p1, p2, 1))
            elif len(parts) == 4:
                n1, i1, n2, i2 = parts
                p1 = race_dir / n1 / f"{n1}_{int(i1):04d}.jpg"
                p2 = race_dir / n2 / f"{n2}_{int(i2):04d}.jpg"
                pairs.append((p1, p2, 0))
            else:
                logger.debug("Skipping malformed pair line: %r", line)
    return pairs


def compute_pair_scores(
    model: torch.nn.Module,
    device: torch.device,
    pairs: list[tuple[Path, Path, int]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]], int]:
    """Compute cosine similarity for each pair, caching embeddings.

    Returns:
        scores: float32 array of cosine similarities for kept pairs.
        labels: int array of 0/1 ground-truth labels for kept pairs.
        kept_paths: list of (p1, p2) strings for kept pairs.
        missing: number of pair-images that could not be read.
    """
    cache: dict[Path, np.ndarray] = {}
    scores: list[float] = []
    labels: list[int] = []
    kept_paths: list[tuple[str, str]] = []
    missing = 0
    for p1, p2, label in tqdm(pairs, desc="pairs"):
        embs: list[np.ndarray] = []
        ok = True
        for p in (p1, p2):
            e = cache.get(p)
            if e is None:
                t = preprocess_image(p)
                if t is None:
                    missing += 1
                    ok = False
                    break
                e = get_embedding(model, t, device)
                cache[p] = e
            embs.append(e)
        if not ok:
            continue
        scores.append(float(np.dot(embs[0], embs[1])))
        labels.append(label)
        kept_paths.append((str(p1), str(p2)))
    return (
        np.asarray(scores, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        kept_paths,
        missing,
    )


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    far_targets: Iterable[float] = DEFAULT_FAR_TARGETS,
) -> dict:
    """Compute AUC, best-threshold accuracy, and TAR@FAR for given targets."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    metrics: dict = {"AUC": float(sk_auc(fpr, tpr))}
    for far in far_targets:
        idx = int(np.searchsorted(fpr, far, side="right") - 1)
        idx = max(idx, 0)
        metrics[f"TAR@FAR={far}"] = float(tpr[idx])
    accs = (tpr + (1.0 - fpr)) / 2.0
    best = int(np.argmax(accs))
    metrics["accuracy"] = float(accs[best])
    metrics["best_threshold"] = float(thresholds[best])
    return metrics


def print_summary(per_race: dict[str, dict]) -> None:
    """Print a formatted per-race summary table to stdout."""
    header = (
        f"{'Race':<12}{'AUC':>8}{'Acc':>8}"
        f"{'TAR@0.1%':>12}{'TAR@1%':>12}{'TAR@10%':>12}{'n_pairs':>10}"
    )
    print(header)
    print("-" * len(header))
    for race, m in per_race.items():
        print(
            f"{race:<12}{m['AUC']:>8.4f}{m['accuracy']:>8.4f}"
            f"{m['TAR@FAR=0.001']:>12.4f}{m['TAR@FAR=0.01']:>12.4f}"
            f"{m['TAR@FAR=0.1']:>12.4f}{m['n_pairs']:>10d}"
        )


def parse_args() -> argparse.Namespace:
    """Build CLI parser."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="Path to model .ckpt")
    p.add_argument(
        "--adaface-repo",
        required=True,
        help="Path to AdaFace repo containing net.py",
    )
    p.add_argument(
        "--rfw-root",
        required=True,
        help="Path to RFW test root (containing data/ and txts/)",
    )
    p.add_argument("--arch", default="ir_101", help="Architecture for net.build_model")
    p.add_argument("--output-dir", required=True, help="Directory for results")
    p.add_argument(
        "--model-name",
        default="model",
        help="Identifier used in output filenames and JSON",
    )
    p.add_argument(
        "--races",
        nargs="+",
        default=list(RACES),
        choices=list(RACES),
        help="Subset of races to evaluate (default: all 4)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_model(args.checkpoint, args.adaface_repo, args.arch)
    logger.info("Model loaded on %s", device)

    rfw_root = Path(args.rfw_root)
    results: dict = {
        "model": args.model_name,
        "checkpoint": str(args.checkpoint),
        "per_race": {},
    }

    for race in args.races:
        pairs_file = rfw_root / "txts" / race / f"{race}_pairs.txt"
        if not pairs_file.exists():
            logger.warning("Pairs file missing for %s: %s", race, pairs_file)
            continue
        pairs = parse_pairs_file(pairs_file, race, rfw_root)
        logger.info("%s: evaluating %d pairs", race, len(pairs))
        scores, labels, kept_paths, missing = compute_pair_scores(model, device, pairs)
        if missing:
            logger.warning("%s: %d images missing/unreadable", race, missing)
        if len(scores) == 0:
            logger.error("%s: no valid pairs evaluated, skipping metrics", race)
            continue
        m = compute_metrics(scores, labels)
        m["n_pairs"] = int(len(labels))
        m["n_missing"] = int(missing)
        results["per_race"][race] = m

        np.savez(
            out_dir / f"{args.model_name}_{race}_scores.npz",
            scores=scores,
            labels=labels,
        )
        with open(out_dir / f"{args.model_name}_{race}_pairs.json", "w") as f:
            json.dump(kept_paths, f)

    json_path = out_dir / f"{args.model_name}_rfw_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", json_path)

    if results["per_race"]:
        print_summary(results["per_race"])


if __name__ == "__main__":
    main()
