"""Precompute ITA (skin tone) for every training image and cache to JSON.

The ITA-reweighting experiment up-weights darker-skinned training images. Doing
that requires an ITA value per image, and computing ITA over the full
BUPT-Balancedface corpus (~1.25M images) takes a while single-threaded, so this
script does it once, in parallel, and writes a cache that every subsequent
``train.py`` / ``distill.py --reweight ita`` run reuses via ``--ita-json``.

The cache is a JSON dict keyed by the *exact* path string that
``torchvision.datasets.ImageFolder`` stores in ``dataset.samples`` (so the keys
line up with what ``train.build_sampler`` looks up). Values are the ITA angle in
degrees, or ``null`` where the image could not be read. Run with the same
``--data-root`` you will later pass to ``train.py``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from evaluate_ita import compute_ita

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def list_imagefolder_paths(root: str) -> list[str]:
    """Return image paths exactly as ImageFolder would enumerate them.

    Uses torchvision's ImageFolder so the cache keys match the paths that
    ``train.build_sampler`` reads from ``dataset.samples``.
    """
    from torchvision import datasets

    ds = datasets.ImageFolder(root)
    logger.info("ImageFolder: %d images across %d classes", len(ds.samples), len(ds.classes))
    return [p for p, _ in ds.samples]


def _ita_one(path: str) -> tuple[str, float | None]:
    """Worker: compute ITA for a single path (None if unreadable)."""
    return path, compute_ita(path)


def compute_cache(paths: list[str], workers: int) -> dict[str, float | None]:
    """Compute ITA for all paths in parallel, returning a path->ita dict."""
    out: dict[str, float | None] = {}
    with Pool(processes=workers) as pool:
        for path, v in tqdm(
            pool.imap_unordered(_ita_one, paths, chunksize=256),
            total=len(paths),
            desc="ita",
        ):
            out[path] = v
    n_ok = sum(1 for v in out.values() if v is not None)
    logger.info("Computed ITA for %d/%d images (%d unreadable)", n_ok, len(out), len(out) - n_ok)
    return out


def parse_args() -> argparse.Namespace:
    """Build CLI parser."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", required=True, help="ImageFolder root (identity subfolders)")
    p.add_argument("--out", required=True, help="Output ITA JSON cache path")
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2)),
        help="Parallel worker processes (default: all CPUs)",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = parse_args()
    paths = list_imagefolder_paths(args.data_root)
    cache = compute_cache(paths, args.workers)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cache, f)
    logger.info("Wrote ITA cache for %d images to %s", len(cache), out_path)


if __name__ == "__main__":
    main()
