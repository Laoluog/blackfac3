"""Compute ITA (Individual Typology Angle) for RFW images and bin model
verification results by skin-tone bin.

ITA is computed from the center 60% crop of each image, in CIELab space:

    ITA = arctan((L - 50) / b) * (180 / pi)

Each pair is assigned a pair-ITA = mean of the two image ITAs. Pairs are
partitioned into 5 equal-population (percentile) bins. Per-bin TAR@FAR=1%
is reported for every supplied model, and a plot is produced overlaying
all models on (mean-ITA, TAR@FAR=1%) axes.

Expects per-model score files written by evaluate_rfw.py:
  {scores_dir}/{model_name}_{race}_scores.npz   (scores, labels)
  {scores_dir}/{model_name}_{race}_pairs.json   (list of (p1, p2))
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm

logger = logging.getLogger(__name__)

RACES = ("African", "Asian", "Caucasian", "Indian")
DEFAULT_N_BINS = 5
DEFAULT_FAR = 0.01


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def compute_ita(image_path: str | Path) -> float | None:
    """Compute ITA from the center 60% crop of an image.

    Returns None if the image cannot be read or the mean ``b`` channel is
    too close to zero for arctan to be meaningful.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    ch, cw = int(h * 0.6), int(w * 0.6)
    if ch < 2 or cw < 2:
        return None
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    crop = img[y0 : y0 + ch, x0 : x0 + cw].astype(np.float32) / 255.0
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    L = float(lab[..., 0].mean())
    b = float(lab[..., 2].mean())
    if abs(b) < 1e-6:
        return None
    return float(np.degrees(np.arctan((L - 50.0) / b)))


def collect_image_paths(rfw_root: Path, races: list[str]) -> list[Path]:
    """Walk the RFW test tree and return all .jpg image paths under the given races."""
    paths: list[Path] = []
    for race in races:
        race_dir = rfw_root / "data" / race
        if not race_dir.exists():
            logger.warning("Race dir missing: %s", race_dir)
            continue
        paths.extend(race_dir.rglob("*.jpg"))
    return paths


def compute_all_ita(
    rfw_root: Path, races: list[str], out_json: Path
) -> dict[str, float]:
    """Compute ITA for every image in the chosen races and persist as JSON.

    Returns the dict keyed by absolute image path.
    """
    images = collect_image_paths(rfw_root, races)
    logger.info("Computing ITA for %d images", len(images))
    ita: dict[str, float] = {}
    skipped = 0
    for p in tqdm(images, desc="ita"):
        v = compute_ita(p)
        if v is None:
            skipped += 1
            continue
        ita[str(p)] = v
    logger.info("Computed ITA for %d images (%d skipped)", len(ita), skipped)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(ita, f)
    return ita


def load_or_compute_ita(
    ita_json: Path, rfw_root: Path, races: list[str], force: bool
) -> dict[str, float]:
    """Load ITA JSON if present (and not forced to recompute); else compute it."""
    if ita_json.exists() and not force:
        with open(ita_json) as f:
            ita = json.load(f)
        logger.info("Loaded %d ITA values from %s", len(ita), ita_json)
        return ita
    return compute_all_ita(rfw_root, races, ita_json)


def load_model_scores(
    scores_dir: Path, model_name: str, races: list[str]
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Concatenate per-race score npz + pair JSON for a single model."""
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_pairs: list[tuple[str, str]] = []
    for race in races:
        npz_path = scores_dir / f"{model_name}_{race}_scores.npz"
        pairs_path = scores_dir / f"{model_name}_{race}_pairs.json"
        if not npz_path.exists() or not pairs_path.exists():
            logger.warning(
                "Missing score files for model=%s race=%s (looked for %s)",
                model_name, race, npz_path,
            )
            continue
        data = np.load(npz_path)
        all_scores.append(data["scores"])
        all_labels.append(data["labels"])
        with open(pairs_path) as f:
            all_pairs.extend(tuple(p) for p in json.load(f))
    if not all_scores:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            [],
        )
    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels),
        all_pairs,
    )


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far: float) -> float:
    """TAR at the target FAR (largest fpr <= far)."""
    if len(np.unique(labels)) < 2 or len(scores) == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = int(np.searchsorted(fpr, far, side="right") - 1)
    idx = max(idx, 0)
    return float(tpr[idx])


def bin_pairs_by_ita(
    pairs: list[tuple[str, str]],
    ita: dict[str, float],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pair mean ITA and bin assignments by percentile.

    Pairs with either image missing from the ITA dict get NaN and are
    excluded from bin assignments (bin = -1).

    Returns (pair_ita, bin_idx) — both length-N arrays.
    """
    pair_ita = np.array(
        [
            (ita[p1] + ita[p2]) / 2.0 if p1 in ita and p2 in ita else np.nan
            for p1, p2 in pairs
        ],
        dtype=np.float64,
    )
    valid = ~np.isnan(pair_ita)
    bin_idx = np.full(len(pair_ita), -1, dtype=np.int64)
    if valid.sum() == 0:
        return pair_ita, bin_idx
    quantiles = np.quantile(
        pair_ita[valid], np.linspace(0, 1, n_bins + 1)
    )
    quantiles[-1] += 1e-9
    bin_idx_valid = np.digitize(pair_ita[valid], quantiles[1:-1])
    bin_idx[valid] = bin_idx_valid
    return pair_ita, bin_idx


def evaluate_model_by_bin(
    scores: np.ndarray,
    labels: np.ndarray,
    pair_ita: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
    far: float,
) -> list[dict]:
    """For each bin, compute TAR@FAR, n_pairs, mean ITA, ITA range."""
    out: list[dict] = []
    for b in range(n_bins):
        sel = bin_idx == b
        n = int(sel.sum())
        if n == 0:
            out.append(
                {
                    "bin": b,
                    "n_pairs": 0,
                    "mean_ita": float("nan"),
                    "ita_min": float("nan"),
                    "ita_max": float("nan"),
                    f"TAR@FAR={far}": float("nan"),
                }
            )
            continue
        out.append(
            {
                "bin": b,
                "n_pairs": n,
                "mean_ita": float(pair_ita[sel].mean()),
                "ita_min": float(pair_ita[sel].min()),
                "ita_max": float(pair_ita[sel].max()),
                f"TAR@FAR={far}": tar_at_far(scores[sel], labels[sel], far),
            }
        )
    return out


def plot_models_by_ita(
    per_model: dict[str, list[dict]], far: float, out_path: Path
) -> None:
    """Save a line plot of TAR@FAR vs. mean per-bin ITA, one line per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, bins in per_model.items():
        xs = [b["mean_ita"] for b in bins]
        ys = [b[f"TAR@FAR={far}"] for b in bins]
        ax.plot(xs, ys, marker="o", label=model)
    ax.set_xlabel("Mean ITA per bin (deg)")
    ax.set_ylabel(f"TAR @ FAR={far}")
    ax.set_title("Verification performance vs. skin tone (ITA)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Build CLI parser."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rfw-root", required=True, help="Path to RFW test root")
    p.add_argument(
        "--scores-dir",
        required=True,
        help="Directory containing per-model score npz/pairs json from evaluate_rfw.py",
    )
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names (must match those used in evaluate_rfw.py --model-name)",
    )
    p.add_argument("--output-dir", required=True, help="Where to write results+plot")
    p.add_argument(
        "--races",
        nargs="+",
        default=list(RACES),
        choices=list(RACES),
        help="Subset of races to include",
    )
    p.add_argument(
        "--ita-json",
        default=None,
        help="Path to ITA JSON (defaults to <output-dir>/ita.json)",
    )
    p.add_argument(
        "--recompute-ita",
        action="store_true",
        help="Recompute ITA even if --ita-json exists",
    )
    p.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    p.add_argument("--far", type=float, default=DEFAULT_FAR)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging()
    np.random.seed(args.seed)

    rfw_root = Path(args.rfw_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ita_json = Path(args.ita_json) if args.ita_json else out_dir / "ita.json"

    ita = load_or_compute_ita(ita_json, rfw_root, args.races, args.recompute_ita)

    per_model: dict[str, list[dict]] = {}
    for model in args.models:
        scores, labels, pairs = load_model_scores(
            Path(args.scores_dir), model, args.races
        )
        if len(scores) == 0:
            logger.warning("No scores loaded for model %s; skipping", model)
            continue
        pair_ita, bin_idx = bin_pairs_by_ita(pairs, ita, args.n_bins)
        bins = evaluate_model_by_bin(
            scores, labels, pair_ita, bin_idx, args.n_bins, args.far
        )
        per_model[model] = bins

    results = {
        "n_bins": args.n_bins,
        "far": args.far,
        "races": args.races,
        "per_model": per_model,
    }
    out_json = out_dir / "ita_binned_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_json)

    plot_path = out_dir / "ita_vs_tar.png"
    plot_models_by_ita(per_model, args.far, plot_path)
    logger.info("Wrote %s", plot_path)

    header = (
        f"{'model':<20}{'bin':>4}{'n':>8}{'mean_ITA':>10}"
        f"{'ITA_range':>20}{'TAR@FAR=' + str(args.far):>16}"
    )
    print(header)
    print("-" * len(header))
    for model, bins in per_model.items():
        for b in bins:
            rng = f"[{b['ita_min']:.1f},{b['ita_max']:.1f}]"
            print(
                f"{model:<20}{b['bin']:>4d}{b['n_pairs']:>8d}"
                f"{b['mean_ita']:>10.2f}{rng:>20}"
                f"{b[f'TAR@FAR={args.far}']:>16.4f}"
            )


if __name__ == "__main__":
    main()
