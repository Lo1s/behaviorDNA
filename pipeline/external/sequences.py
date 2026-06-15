"""
pipeline/external/sequences.py
==============================
Turn a public mouse-dynamics corpus session (recorder-schema dict) into the
**8-D event-tensor chunks** the sequence encoder consumes — the bridge between
the Phase-6 external adapters and the Phase-8.2 contrastive machinery.

Desktop mouse captures are long and idle-dominated, so each session is first
split into continuous-activity bursts (``base.split_on_idle`` — the same
treatment Phase 6's windowed pipeline uses), each burst is encoded to the shared
8-D schema (``session_to_event_tensor``), and the bursts are sliced into
fixed-length chunks (``sequences.dataset._chunk_indices``). Every step is reused
from existing, tested code; this module only composes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.external.base import split_on_idle
from pipeline.sequences.dataset import _chunk_indices
from pipeline.sequences.preprocessing import EVENT_FEATURE_DIM, session_to_event_tensor


def session_to_segment_tensors(session: dict) -> list[np.ndarray]:
    """One recorder-schema session → a list of ``(N_i, 8)`` per-burst tensors.

    Splits at idle gaps first (``split_on_idle``) so each tensor is one
    continuous activity burst (dt resets at each burst start, matching how the
    encoder saw GTA sessions). Returns ``[]`` for an empty / event-less session.
    """
    events = session.get("events", [])
    if not events:
        return []
    df = pd.DataFrame(events)
    if "t" not in df.columns:
        return []
    sens = session.get("sensitivity", 1.0)
    dpi = session.get("dpi", 800.0)
    out: list[np.ndarray] = []
    for seg in split_on_idle(df):
        seg_session = {
            "sensitivity": sens,
            "dpi": dpi,
            "events": seg.to_dict("records"),
        }
        t = session_to_event_tensor(seg_session)
        if len(t):
            out.append(t)
    return out


def session_to_chunks(
    session: dict, chunk_length: int = 64, stride: int = 32
) -> np.ndarray:
    """One session → ``(n_chunks, chunk_length, 8)`` raw (un-normalised) chunks.

    Chunks are sliced from each idle-split burst, so a chunk never straddles an
    idle gap. Normalise with the corpus stats before embedding.
    """
    chunks: list[np.ndarray] = []
    for t in session_to_segment_tensors(session):
        for start in _chunk_indices(len(t), chunk_length, stride):
            chunks.append(t[start : start + chunk_length])
    if not chunks:
        return np.empty((0, chunk_length, EVENT_FEATURE_DIM), dtype=np.float32)
    return np.stack(chunks).astype(np.float32)
