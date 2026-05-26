"""Unit tests for the RFW/ITA evaluation pipeline.

Covers:
  - ITA computation on synthetic single-color images
  - RFW pairs file parsing (3-col genuine, 4-col imposter)
  - TAR@FAR + AUC metric computation on known scores/labels
  - Embedding L2-normalization via a mocked model
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch


# ---------- ITA ----------

def _write_solid(path: Path, bgr: tuple[int, int, int], size: int = 112) -> None:
    """Write a solid-color BGR JPEG of `size` x `size`."""
    arr = np.full((size, size, 3), bgr, dtype=np.uint8)
    cv2.imwrite(str(path), arr)


def test_ita_returns_float_for_realistic_skin(tmp_path: Path) -> None:
    """A light-skin-ish BGR color should yield a finite ITA in a plausible range."""
    from evaluate_ita import compute_ita

    img = tmp_path / "skin.jpg"
    _write_solid(img, (180, 200, 230))  # B G R
    ita = compute_ita(img)
    assert ita is not None
    assert np.isfinite(ita)
    assert -90.0 < ita < 90.0


def test_ita_lighter_image_has_higher_ita(tmp_path: Path) -> None:
    """Lighter skin tones should produce a larger ITA than darker ones."""
    from evaluate_ita import compute_ita

    light = tmp_path / "light.jpg"
    dark = tmp_path / "dark.jpg"
    _write_solid(light, (210, 220, 235))
    _write_solid(dark, (60, 70, 90))
    ita_light = compute_ita(light)
    ita_dark = compute_ita(dark)
    assert ita_light is not None and ita_dark is not None
    assert ita_light > ita_dark


def test_ita_missing_image_returns_none(tmp_path: Path) -> None:
    """compute_ita returns None for an unreadable path."""
    from evaluate_ita import compute_ita

    assert compute_ita(tmp_path / "does_not_exist.jpg") is None


# ---------- pair parsing ----------

def test_parse_pairs_file_handles_both_formats(tmp_path: Path) -> None:
    """Genuine (3 col) and imposter (4 col) lines map to label 1 and 0 respectively."""
    from evaluate_rfw import parse_pairs_file

    pairs_txt = tmp_path / "African_pairs.txt"
    pairs_txt.write_text(
        "Alice\t1\t2\n"
        "Bob\t3\tCarol\t4\n"
    )
    rfw_root = tmp_path / "test"
    pairs = parse_pairs_file(pairs_txt, "African", rfw_root)
    assert len(pairs) == 2
    p1, p2, label = pairs[0]
    assert label == 1
    assert p1.name == "Alice_0001.jpg"
    assert p2.name == "Alice_0002.jpg"
    assert p1.parent == rfw_root / "data" / "African" / "Alice"
    p1, p2, label = pairs[1]
    assert label == 0
    assert p1.name == "Bob_0003.jpg"
    assert p2.name == "Carol_0004.jpg"


# ---------- metrics ----------

def test_compute_metrics_perfect_separation() -> None:
    """Perfectly separable scores should give AUC=1.0, accuracy=1.0, TAR@all FARs=1.0."""
    from evaluate_rfw import compute_metrics

    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    m = compute_metrics(scores, labels)
    assert m["AUC"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["TAR@FAR=0.001"] == pytest.approx(1.0)
    assert m["TAR@FAR=0.01"] == pytest.approx(1.0)


def test_compute_metrics_tar_at_far_monotone() -> None:
    """TAR@FAR should be non-decreasing in FAR."""
    from evaluate_rfw import compute_metrics

    rng = np.random.default_rng(0)
    pos = rng.normal(0.7, 0.2, 500)
    neg = rng.normal(0.3, 0.2, 500)
    scores = np.concatenate([pos, neg])
    labels = np.array([1] * 500 + [0] * 500)
    m = compute_metrics(scores, labels)
    assert m["TAR@FAR=0.001"] <= m["TAR@FAR=0.01"] <= m["TAR@FAR=0.1"]
    assert 0.5 < m["AUC"] <= 1.0


# ---------- embedding normalization ----------

def test_get_embedding_is_unit_norm() -> None:
    """get_embedding must return an L2-normalized vector regardless of raw scale."""
    from evaluate_rfw import get_embedding

    model = MagicMock()
    raw = torch.tensor([[3.0, 4.0, 0.0]])  # raw norm = 5
    model.return_value = (raw, torch.tensor([5.0]))
    out = get_embedding(model, torch.zeros(1, 3, 112, 112), torch.device("cpu"))
    assert out.shape == (3,)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)
    assert out == pytest.approx(np.array([0.6, 0.8, 0.0]), abs=1e-6)


def test_get_embedding_handles_non_tuple_output() -> None:
    """A model that returns a raw tensor (not (emb, norm)) should also work."""
    from evaluate_rfw import get_embedding

    model = MagicMock()
    model.return_value = torch.tensor([[1.0, 0.0, 0.0]])
    out = get_embedding(model, torch.zeros(1, 3, 112, 112), torch.device("cpu"))
    assert np.linalg.norm(out) == pytest.approx(1.0)
