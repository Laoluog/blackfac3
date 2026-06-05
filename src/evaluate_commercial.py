"""Evaluate AWS Rekognition and Azure Face API on RFW pairs.

For each (image_a, image_b, label) pair, calls the requested cloud
verification APIs, persists raw similarity scores incrementally so a
crash does not lose progress, then computes TAR@FAR, accuracy, and AUC.

Credentials:
  - AWS: standard boto3 chain (AWS_PROFILE / AWS_ACCESS_KEY_ID etc.).
  - Azure: env vars AZURE_FACE_ENDPOINT and AZURE_FACE_KEY.

Note: Azure Face API verification is gated to approved customers as of 2023.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve
from tqdm import tqdm

from evaluate_rfw import RACES, parse_pairs_file

logger = logging.getLogger(__name__)

DEFAULT_RPS = 20
PROGRESS_INTERVAL = 500
CHECKPOINT_INTERVAL = 100


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class RateLimiter:
    """Token-bucket-ish limiter capping requests-per-second."""

    def __init__(self, rps: int) -> None:
        self.interval = 1.0 / max(rps, 1)
        self.times: deque[float] = deque(maxlen=max(rps, 1))

    def wait(self) -> None:
        """Block until issuing one more request would respect the RPS cap."""
        now = time.monotonic()
        if len(self.times) == self.times.maxlen:
            earliest = self.times[0]
            elapsed = now - earliest
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
                now = time.monotonic()
        self.times.append(now)


def load_checkpoint(path: Path) -> dict[str, list[dict]]:
    """Load prior partial results from a checkpoint JSON file, if any."""
    if not path.exists():
        return {"aws": [], "azure": []}
    with open(path) as f:
        data = json.load(f)
    data.setdefault("aws", [])
    data.setdefault("azure", [])
    return data


def save_checkpoint(path: Path, data: dict) -> None:
    """Atomically persist checkpoint via tmp+rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)


def already_done(entries: Iterable[dict]) -> set[tuple[str, str]]:
    """Index already-evaluated (p1, p2) pairs from prior checkpoint entries."""
    return {(e["p1"], e["p2"]) for e in entries}


def aws_compare(client, p1: Path, p2: Path) -> float | None:
    """Call Rekognition compare_faces. Returns similarity in [0, 100] or None."""
    try:
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            resp = client.compare_faces(
                SourceImage={"Bytes": f1.read()},
                TargetImage={"Bytes": f2.read()},
                SimilarityThreshold=0,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("AWS error on %s vs %s: %s", p1, p2, e)
        return None
    matches = resp.get("FaceMatches", [])
    if not matches:
        return 0.0
    return float(max(m["Similarity"] for m in matches))


def azure_detect(endpoint: str, key: str, image_path: Path) -> str | None:
    """Run Azure Face detect, return faceId or None."""
    url = endpoint.rstrip("/") + "/face/v1.0/detect"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream",
    }
    params = {"returnFaceId": "true", "detectionModel": "detection_03"}
    try:
        with open(image_path, "rb") as f:
            r = requests.post(url, params=params, headers=headers, data=f.read(), timeout=30)
        r.raise_for_status()
        faces = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("Azure detect error on %s: %s", image_path, e)
        return None
    if not faces:
        return None
    return faces[0].get("faceId")


def azure_verify(endpoint: str, key: str, face_id_1: str, face_id_2: str) -> float | None:
    """Run Azure Face verify, return confidence in [0, 1] or None."""
    url = endpoint.rstrip("/") + "/face/v1.0/verify"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/json",
    }
    body = {"faceId1": face_id_1, "faceId2": face_id_2}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        out = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("Azure verify error on %s vs %s: %s", face_id_1, face_id_2, e)
        return None
    return float(out.get("confidence", 0.0))


def run_aws(
    pairs: list[tuple[Path, Path, int]],
    checkpoint_path: Path,
    rps: int,
) -> list[dict]:
    """Evaluate all pairs through Rekognition with checkpointing + rate limiting."""
    import boto3  # local import so the module loads without AWS deps for --azure-only

    client = boto3.client("rekognition")
    state = load_checkpoint(checkpoint_path)
    done = already_done(state["aws"])
    limiter = RateLimiter(rps)
    new_count = 0
    running = state["aws"]
    correct = total = 0
    for i, (p1, p2, label) in enumerate(tqdm(pairs, desc="aws")):
        key = (str(p1), str(p2))
        if key in done:
            continue
        limiter.wait()
        score = aws_compare(client, p1, p2)
        running.append({"p1": str(p1), "p2": str(p2), "label": int(label), "score": score})
        new_count += 1
        if score is not None:
            total += 1
            pred = 1 if score >= 80.0 else 0  # default Rekognition recommended threshold
            correct += int(pred == label)
        if new_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(checkpoint_path, state)
        if total and total % PROGRESS_INTERVAL == 0:
            logger.info("[AWS] processed=%d running_acc=%.4f", total, correct / total)
    save_checkpoint(checkpoint_path, state)
    return running


def run_azure(
    pairs: list[tuple[Path, Path, int]],
    checkpoint_path: Path,
    rps: int,
) -> list[dict]:
    """Evaluate all pairs through Azure Face API with checkpointing."""
    endpoint = os.environ.get("AZURE_FACE_ENDPOINT")
    key = os.environ.get("AZURE_FACE_KEY")
    if not endpoint or not key:
        raise RuntimeError("AZURE_FACE_ENDPOINT and AZURE_FACE_KEY must be set")
    state = load_checkpoint(checkpoint_path)
    done = already_done(state["azure"])
    limiter = RateLimiter(rps)
    new_count = 0
    running = state["azure"]
    correct = total = 0
    for p1, p2, label in tqdm(pairs, desc="azure"):
        k = (str(p1), str(p2))
        if k in done:
            continue
        limiter.wait()
        fid1 = azure_detect(endpoint, key, p1)
        limiter.wait()
        fid2 = azure_detect(endpoint, key, p2) if fid1 else None
        score: float | None
        if fid1 and fid2:
            limiter.wait()
            score = azure_verify(endpoint, key, fid1, fid2)
        else:
            score = None
        running.append({"p1": str(p1), "p2": str(p2), "label": int(label), "score": score})
        new_count += 1
        if score is not None:
            total += 1
            pred = 1 if score >= 0.5 else 0
            correct += int(pred == label)
        if new_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(checkpoint_path, state)
        if total and total % PROGRESS_INTERVAL == 0:
            logger.info("[Azure] processed=%d running_acc=%.4f", total, correct / total)
    save_checkpoint(checkpoint_path, state)
    return running


def compute_metrics(entries: list[dict], far_target: float = 0.01) -> dict:
    """Compute AUC, best-threshold accuracy, and TAR@FAR from checkpoint entries."""
    valid = [e for e in entries if e["score"] is not None]
    if len(valid) < 2:
        return {"n": len(valid), "error": "not enough valid scores"}
    scores = np.array([e["score"] for e in valid], dtype=np.float64)
    labels = np.array([e["label"] for e in valid], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return {"n": len(valid), "error": "only one label present"}
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = int(np.searchsorted(fpr, far_target, side="right") - 1)
    idx = max(idx, 0)
    accs = (tpr + (1.0 - fpr)) / 2.0
    best = int(np.argmax(accs))
    return {
        "n": len(valid),
        "n_skipped": len(entries) - len(valid),
        "AUC": float(sk_auc(fpr, tpr)),
        "accuracy": float(accs[best]),
        "best_threshold": float(thresholds[best]),
        f"TAR@FAR={far_target}": float(tpr[idx]),
    }


def gather_pairs(
    rfw_root: Path, races: list[str]
) -> list[tuple[Path, Path, int]]:
    """Concatenate all RFW pairs across the chosen race subsets."""
    all_pairs: list[tuple[Path, Path, int]] = []
    for race in races:
        pf = rfw_root / "txts" / race / f"{race}_pairs.txt"
        if not pf.exists():
            logger.warning("Pairs file missing for %s: %s", race, pf)
            continue
        all_pairs.extend(parse_pairs_file(pf, race, rfw_root))
    return all_pairs


def parse_args() -> argparse.Namespace:
    """Build CLI parser."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rfw-root", required=True, help="Path to RFW test root")
    p.add_argument("--output-dir", required=True, help="Where to write outputs")
    p.add_argument(
        "--races",
        nargs="+",
        default=["African"],
        choices=list(RACES),
        help="Subset of races (default: African)",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--aws-only", action="store_true")
    group.add_argument("--azure-only", action="store_true")
    p.add_argument("--rps", type=int, default=DEFAULT_RPS, help="Max requests/sec")
    p.add_argument("--far", type=float, default=0.01)
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

    pairs = gather_pairs(rfw_root, args.races)
    logger.info("Loaded %d total pairs across %s", len(pairs), args.races)

    checkpoint_path = out_dir / "commercial_checkpoint.json"
    results: dict = {"races": args.races, "n_pairs": len(pairs), "providers": {}}

    if not args.azure_only:
        aws_entries = run_aws(pairs, checkpoint_path, args.rps)
        results["providers"]["aws"] = compute_metrics(aws_entries, args.far)

    if not args.aws_only:
        azure_entries = run_azure(pairs, checkpoint_path, args.rps)
        results["providers"]["azure"] = compute_metrics(azure_entries, args.far)

    out_json = out_dir / "commercial_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_json)

    header = f"{'provider':<10}{'n':>8}{'AUC':>10}{'acc':>10}{'TAR@FAR':>14}"
    print(header)
    print("-" * len(header))
    for prov, m in results["providers"].items():
        if "error" in m:
            print(f"{prov:<10}{m.get('n', 0):>8d}  {m['error']}")
            continue
        print(
            f"{prov:<10}{m['n']:>8d}{m['AUC']:>10.4f}{m['accuracy']:>10.4f}"
            f"{m[f'TAR@FAR={args.far}']:>14.4f}"
        )


if __name__ == "__main__":
    main()
