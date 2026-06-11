"""
pipeline/verification.py
========================
Verification metrics for the Phase-6 reframe (docs/ROADMAP.md): closed-set
"which of N players" → **"is this really player X?"** (account-sharing /
smurf detection) and **open-set** "none of the enrolled players".

Conventions: higher score = more genuine. ``genuine`` are scores from true
(user, sample) pairs; ``impostor`` from mismatched pairs. EER is the point
where false-accept rate equals false-reject rate — the standard biometric
headline (lower is better; 0.5 = chance).
"""

from __future__ import annotations

import numpy as np


def _stack(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("need at least one genuine and one impostor score")
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    s = np.r_[genuine, impostor]
    return y, s


def det_curve(
    genuine: np.ndarray, impostor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(far, frr, thresholds) — the detection-error-tradeoff curve."""
    from sklearn.metrics import det_curve as _det

    y, s = _stack(genuine, impostor)
    far, frr, thr = _det(y, s)
    return far, frr, thr


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and its threshold.

    Computed from the DET curve by linear interpolation of the FAR/FRR
    crossing. Returns ``(eer, threshold)``.
    """
    far, frr, thr = det_curve(genuine, impostor)
    diff = far - frr
    idx = int(np.argmin(np.abs(diff)))
    # interpolate between the two points bracketing the sign change, if any
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change):
        i = int(sign_change[0])
        # linear interp of both curves to the crossing of far(t) and frr(t)
        d0, d1 = diff[i], diff[i + 1]
        w = 0.0 if d1 == d0 else -d0 / (d1 - d0)
        eer_val = float(far[i] + w * (far[i + 1] - far[i]))
        thr_val = float(thr[i] + w * (thr[i + 1] - thr[i]))
        return eer_val, thr_val
    return float((far[idx] + frr[idx]) / 2.0), float(thr[idx])


def far_at_frr(
    genuine: np.ndarray, impostor: np.ndarray, frr_target: float = 0.05
) -> float:
    """False-accept rate at the threshold where FRR first ≤ ``frr_target``.

    The open-set operating point: "rejecting at most ``frr_target`` of real
    users, how many impostors get through?"
    """
    far, frr, _ = det_curve(genuine, impostor)
    ok = np.where(frr <= frr_target)[0]
    if len(ok) == 0:
        return 1.0  # cannot reach the target rejection rate
    return float(far[ok].min())


def verification_scores(
    proba: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiclass probabilities → (genuine, impostor) verification scores.

    Every (sample, enrolled-class) pair becomes one verification trial with
    score ``proba[sample, class]`` — genuine when ``class == y_true[sample]``,
    impostor otherwise. This converts a closed-set classifier into the
    pairwise verification protocol without retraining.
    """
    proba = np.asarray(proba, dtype=np.float64)
    y_true = np.asarray(y_true)
    n, k = proba.shape
    mask = np.zeros((n, k), dtype=bool)
    mask[np.arange(n), y_true] = True
    return proba[mask], proba[~mask]
