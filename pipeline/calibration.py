"""
pipeline/calibration.py
=======================
Phase 5b — probability-calibration metrics.

A classifier can be accurate yet *miscalibrated*: when it says "90% sure this is
player X", that should be right 90% of the time. For an anti-cheat decision —
ban / flag at a fixed false-positive budget — calibrated probabilities are what
let you pick a threshold that *means* something. These helpers quantify it:

- ``expected_calibration_error`` — ECE: the average gap between confidence and
  accuracy, binned by confidence (the single headline number).
- ``reliability_curve`` — the per-bin (confidence, accuracy) points behind a
  reliability diagram.
- ``multiclass_brier`` — Brier score generalised to K classes (mean squared
  error between the predicted probability vector and the one-hot truth).

scikit-learn ships ``CalibratedClassifierCV`` for the *fix* (isotonic / Platt);
these cover the *measurement*, which sklearn does not provide for multiclass.
Used by ``notebooks/13_calibration.ipynb``; tested in
``tests/test_calibration.py``.
"""

from __future__ import annotations

import numpy as np


def reliability_curve(confidences, correct, n_bins: int = 10):
    """Per-bin reliability points for a reliability diagram.

    ``confidences`` is each prediction's top-label probability; ``correct`` is
    the matching 0/1 (was the top label right?). Returns four equal-width-bin
    arrays — ``(bin_center, bin_accuracy, bin_confidence, bin_count)`` — with
    NaN accuracy/confidence for empty bins.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    acc = np.full(n_bins, np.nan)
    conf = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    # np.digitize → bin index in [1, n_bins]; clip the right edge into the last bin.
    idx = np.clip(np.digitize(confidences, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        count[b] = int(m.sum())
        if count[b]:
            acc[b] = correct[m].mean()
            conf[b] = confidences[m].mean()
    return centers, acc, conf, count


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """Expected Calibration Error: Σ_bin (count/N) · |accuracy − confidence|.

    0 = perfectly calibrated. Computed on the top-label confidence (the standard
    multiclass ECE).
    """
    confidences = np.asarray(confidences, dtype=float)
    if confidences.size == 0:
        return float("nan")
    _, acc, conf, count = reliability_curve(confidences, correct, n_bins)
    mask = count > 0
    weights = count[mask] / count.sum()
    return float(np.sum(weights * np.abs(acc[mask] - conf[mask])))


def multiclass_brier(y_true_idx, proba) -> float:
    """Multiclass Brier score: mean over samples of Σ_k (p_k − onehot_k)².

    ``y_true_idx`` are integer class labels (0..K-1); ``proba`` is ``(n, K)``.
    Ranges 0 (perfect) to 2 (worst); equals the familiar binary Brier when K=2
    and you pass both columns.
    """
    proba = np.asarray(proba, dtype=float)
    y_true_idx = np.asarray(y_true_idx, dtype=int)
    n, k = proba.shape
    onehot = np.zeros((n, k), dtype=float)
    onehot[np.arange(n), y_true_idx] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())
