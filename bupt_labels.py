"""Parse BUPT-Balancedface label list and build demographic/ITA sampling weights.

The BUPT ``train_balancedface.lst`` is order-matched to ``train.rec`` and has
three whitespace/tab-separated columns:

    <relative_image_path>   <id_label>   <race_label>

Race label convention (BUPT):  0=Caucasian, 1=Indian, 2=Asian, 3=African.
NOTE this index order differs from the alphabetical RACES tuple used by the
RFW evaluation code, so always map through ``BUPT_RACE_LABELS``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# BUPT .lst race index -> name
BUPT_RACE_LABELS: dict[int, str] = {0: "Caucasian", 1: "Indian", 2: "Asian", 3: "African"}


def parse_lst(lst_path: str | Path) -> list[tuple[str, int, int]]:
    """Parse a BUPT label list into (rel_path, id_label, race_label) tuples.

    Order is preserved so the i-th entry corresponds to record index i in the
    matching ``train.rec``. Lines that do not have at least 3 columns are
    skipped with a debug log.
    """
    entries: list[tuple[str, int, int]] = []
    with open(lst_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                if line.strip():
                    logger.debug("Skipping malformed lst line: %r", line)
                continue
            rel_path = parts[0]
            try:
                id_label = int(parts[1])
                race_label = int(parts[2])
            except ValueError:
                logger.debug("Non-integer labels in lst line: %r", line)
                continue
            entries.append((rel_path, id_label, race_label))
    logger.info("Parsed %d entries from %s", len(entries), lst_path)
    return entries


def race_array(entries: list[tuple[str, int, int]]) -> np.ndarray:
    """Return an int array of race labels in record order."""
    return np.array([r for _, _, r in entries], dtype=np.int64)


def num_classes_from_entries(entries: list[tuple[str, int, int]]) -> int:
    """Infer the number of identity classes (max id_label + 1)."""
    if not entries:
        return 0
    return int(max(i for _, i, _ in entries) + 1)


def inverse_frequency_weights(group: np.ndarray) -> np.ndarray:
    """Per-sample weights inversely proportional to that sample's group frequency.

    Up-weights under-represented groups so a WeightedRandomSampler draws them
    more often. Returns weights normalized to mean 1.0.
    """
    groups, counts = np.unique(group, return_counts=True)
    freq = {g: c for g, c in zip(groups, counts)}
    w = np.array([1.0 / freq[g] for g in group], dtype=np.float64)
    w *= len(w) / w.sum()  # normalize to mean 1
    return w


def race_sample_weights(entries: list[tuple[str, int, int]]) -> np.ndarray:
    """Inverse-frequency weights by race label (BUPT is ~balanced, so near-uniform)."""
    return inverse_frequency_weights(race_array(entries))


def ita_bin_weights(
    ita_values: np.ndarray, n_bins: int = 5
) -> np.ndarray:
    """Inverse-frequency weights over percentile ITA bins.

    Args:
        ita_values: per-sample ITA (NaN allowed; NaN samples get weight 1.0).
        n_bins: number of percentile bins.

    Returns:
        Per-sample weights (mean ~1.0). Under-populated skin-tone bins get
        higher weight, which is the intended fairness rebalancing.
    """
    w = np.ones(len(ita_values), dtype=np.float64)
    valid = ~np.isnan(ita_values)
    if valid.sum() == 0:
        return w
    q = np.quantile(ita_values[valid], np.linspace(0, 1, n_bins + 1))
    q[-1] += 1e-9
    bins = np.digitize(ita_values[valid], q[1:-1])
    w[valid] = inverse_frequency_weights(bins)
    return w


def compute_ita_for_paths(
    image_paths: list[str | Path],
) -> np.ndarray:
    """Compute ITA for a list of image paths (NaN where unreadable).

    Thin wrapper over ``evaluate_ita.compute_ita`` so the training code and
    the RFW evaluation share the exact same skin-tone definition.
    """
    from evaluate_ita import compute_ita  # local import to avoid hard dep at import time

    out = np.empty(len(image_paths), dtype=np.float64)
    for i, p in enumerate(image_paths):
        v = compute_ita(p)
        out[i] = np.nan if v is None else v
    return out
