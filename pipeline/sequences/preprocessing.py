"""
pipeline/sequences/preprocessing.py
===================================
Convert a BehaviorDNA recorder session JSON into a fixed-width event tensor,
and fit/apply per-feature normalization stats.

Each event becomes one row of an ``(N, 8)`` float32 array:

    [dt_ms, dx_norm, dy_norm,
     is_mouse_move, is_mouse_click_press, is_mouse_scroll,
     is_key_press, is_key_release]

- ``dt_ms`` is ``log1p(ms_since_previous_event)`` to compress dynamic range
  (idle gaps can be huge; mouse-move bursts are sub-ms).
- ``dx_norm`` and ``dy_norm`` are divided by ``norm_factor = sensitivity * dpi / 800.0``
  so the same physical mouse movement looks the same across hardware setups
  (mirrors the convention in ``pipeline.features.run.compute_mouse_kinematics``).
- The 5 one-hot indicators cover the event types that survive
  ``pipeline.ingestion.run.VALID_EVENT_TYPES``; mouse_click ``pressed=False``
  events (button releases) drop into the all-zeros bucket since they carry
  little signal on their own.

Normalization is **fit on the training fold only** and saved alongside the
model (same train/test-leakage discipline as ``StandardScaler`` in
``pipeline/training/run.py``). At inference, ``apply_normalizer`` consumes the
saved stats to scale new sessions.
"""

from __future__ import annotations

import numpy as np

EVENT_FEATURE_DIM = 8

# Column indices (also used by tests + the dataset module)
COL_DT = 0
COL_DX = 1
COL_DY = 2
COL_IS_MOUSE_MOVE = 3
COL_IS_MOUSE_CLICK_PRESS = 4
COL_IS_MOUSE_SCROLL = 5
COL_IS_KEY_PRESS = 6
COL_IS_KEY_RELEASE = 7


def session_to_event_tensor(session: dict) -> np.ndarray:
    """Convert a BehaviorDNA session dict into a ``(N, 8)`` float32 tensor.

    ``session`` follows the recorder JSON schema (see ``data/raw/*.json``).
    Returns an empty ``(0, 8)`` array if the session has no events.
    """
    events = session.get("events", [])
    if not events:
        return np.zeros((0, EVENT_FEATURE_DIM), dtype=np.float32)

    sensitivity = float(session.get("sensitivity", 1.0))
    dpi = float(session.get("dpi", 800.0))
    norm_factor = max(sensitivity * dpi / 800.0, 1e-6)

    out = np.zeros((len(events), EVENT_FEATURE_DIM), dtype=np.float32)
    prev_t: float | None = None

    for i, ev in enumerate(events):
        t = float(ev.get("t", 0.0))
        dt_ms = 0.0 if prev_t is None else max(t - prev_t, 0.0)
        out[i, COL_DT] = np.log1p(dt_ms)
        prev_t = t

        ev_type = ev.get("type", "")
        if ev_type == "mouse_move":
            out[i, COL_DX] = float(ev.get("dx") or 0.0) / norm_factor
            out[i, COL_DY] = float(ev.get("dy") or 0.0) / norm_factor
            out[i, COL_IS_MOUSE_MOVE] = 1.0
        elif ev_type == "mouse_click":
            if ev.get("pressed") is True:
                out[i, COL_IS_MOUSE_CLICK_PRESS] = 1.0
            # button-up events leave all one-hots at 0
        elif ev_type == "mouse_scroll":
            out[i, COL_DX] = float(ev.get("dx") or 0.0) / norm_factor
            out[i, COL_DY] = float(ev.get("dy") or 0.0) / norm_factor
            out[i, COL_IS_MOUSE_SCROLL] = 1.0
        elif ev_type == "key_press":
            out[i, COL_IS_KEY_PRESS] = 1.0
        elif ev_type == "key_release":
            out[i, COL_IS_KEY_RELEASE] = 1.0
        # unknown types fall through with dt set but everything else zero

    return out


def fit_normalizer(tensors: list[np.ndarray]) -> dict:
    """Compute per-channel mean and std across a list of session tensors.

    Channels that have ~zero variance (e.g. an unused one-hot in the training
    fold) get ``std = 1.0`` so ``apply_normalizer`` doesn't blow up.

    Returns
    -------
    dict
        ``{"mean": np.ndarray (8,), "std": np.ndarray (8,)}``. JSON-serialisable
        once cast to lists, so it can ride along with the model artifact.
    """
    if not tensors:
        return {
            "mean": np.zeros(EVENT_FEATURE_DIM, dtype=np.float32),
            "std": np.ones(EVENT_FEATURE_DIM, dtype=np.float32),
        }

    stacked = np.concatenate([t for t in tensors if len(t) > 0], axis=0)
    if stacked.size == 0:
        return {
            "mean": np.zeros(EVENT_FEATURE_DIM, dtype=np.float32),
            "std": np.ones(EVENT_FEATURE_DIM, dtype=np.float32),
        }

    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0  # avoid divide-by-zero in apply
    return {"mean": mean, "std": std}


def apply_normalizer(tensor: np.ndarray, stats: dict) -> np.ndarray:
    """Z-score-normalise an event tensor with the saved training stats."""
    if tensor.size == 0:
        return tensor.astype(np.float32, copy=False)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((tensor - mean) / std).astype(np.float32)


def serialize_stats(stats: dict) -> dict:
    """Convert normalizer stats to a JSON-serialisable dict."""
    return {k: np.asarray(v).tolist() for k, v in stats.items()}


def deserialize_stats(stats: dict) -> dict:
    """Inverse of ``serialize_stats``."""
    return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}
