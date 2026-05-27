"""
pipeline/sequences/dataset.py
=============================
PyTorch ``Dataset`` that chunks one or more session event tensors into
fixed-length overlapping windows for the LSTM autoencoder.

A "chunk" is a contiguous slice of ``L`` consecutive events from a single
session. We slide a window of length ``L`` across each session with stride
``S`` and emit each resulting slice as one training example. Sessions
shorter than ``L`` are skipped (no padding) — we'd rather drop a tiny
session than pollute the training set with mostly-zero rows.

Example:
    >>> tensors = [session_to_event_tensor(s) for s in sessions]
    >>> ds = EventSequenceDataset(tensors, chunk_length=64, stride=32)
    >>> len(ds)        # number of chunks across all sessions
    >>> ds[0].shape    # torch.Size([64, 8])
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.sequences.preprocessing import EVENT_FEATURE_DIM


def _chunk_indices(n_events: int, chunk_length: int, stride: int) -> list[int]:
    """Return the start indices for each chunk of length ``chunk_length``."""
    if n_events < chunk_length or chunk_length <= 0 or stride <= 0:
        return []
    last_start = n_events - chunk_length
    return list(range(0, last_start + 1, stride))


class EventSequenceDataset(Dataset):
    """Sliding-window dataset over a list of session event tensors.

    Parameters
    ----------
    tensors:
        List of ``(N_i, 8)`` ``np.ndarray`` (one per session). All must have
        the same feature dimension. May be already z-score-normalised or raw
        — the dataset doesn't know either way.
    chunk_length:
        Length of each output sequence (``L``).
    stride:
        Step between consecutive chunk starts. ``stride == chunk_length`` →
        no overlap; ``stride == chunk_length // 2`` → 50 % overlap (default
        for training).
    session_ids:
        Optional list of identifiers (one per input tensor) so the dataset
        can return which session a chunk came from. Useful for per-session
        anomaly aggregation downstream.
    """

    def __init__(
        self,
        tensors: list[np.ndarray],
        chunk_length: int = 64,
        stride: int = 32,
        session_ids: list[str] | None = None,
    ) -> None:
        if chunk_length <= 0:
            raise ValueError(f"chunk_length must be > 0, got {chunk_length}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")

        for i, t in enumerate(tensors):
            if t.ndim != 2 or t.shape[1] != EVENT_FEATURE_DIM:
                raise ValueError(
                    f"tensors[{i}] must have shape (N, {EVENT_FEATURE_DIM}); "
                    f"got {t.shape}"
                )

        if session_ids is not None and len(session_ids) != len(tensors):
            raise ValueError(
                f"session_ids length {len(session_ids)} must match "
                f"tensors length {len(tensors)}"
            )

        self.chunk_length = chunk_length
        self.stride = stride
        self.tensors = tensors
        self.session_ids = session_ids or [str(i) for i in range(len(tensors))]

        # Build a flat lookup table: chunk_index → (tensor_index, start)
        self._index: list[tuple[int, int]] = []
        for ti, t in enumerate(tensors):
            for start in _chunk_indices(len(t), chunk_length, stride):
                self._index.append((ti, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ti, start = self._index[idx]
        chunk = self.tensors[ti][start : start + self.chunk_length]
        return torch.from_numpy(chunk).float()

    def chunk_origin(self, idx: int) -> tuple[str, int]:
        """Return ``(session_id, chunk_start_event_idx)`` for chunk ``idx``."""
        ti, start = self._index[idx]
        return self.session_ids[ti], start

    def session_chunk_counts(self) -> dict[str, int]:
        """Return ``{session_id: n_chunks}`` for downstream session aggregation."""
        counts: dict[str, int] = {sid: 0 for sid in self.session_ids}
        for ti, _ in self._index:
            counts[self.session_ids[ti]] += 1
        return counts
