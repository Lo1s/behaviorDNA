"""
pipeline/pretraining/masking.py
===============================
Masked-denoising dataset for self-supervised pretraining.

The pretext task: randomly **zero out a fraction of timesteps** in an input
chunk and train the autoencoder to reconstruct the *clean* chunk. Forcing the
model to fill in masked steps from surrounding context makes the bottleneck
learn the human-motion manifold (the structure of how a real hand moves a
mouse), which is exactly the prior we want to transfer into the downstream
legit-manifold anomaly detector.

Masking is **deterministic per chunk** (seeded by ``base_seed + chunk_index``)
so a pretraining run is fully reproducible; the same mask is reused across
epochs, which is sufficient at this corpus scale.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.sequences.dataset import _chunk_indices
from pipeline.sequences.preprocessing import EVENT_FEATURE_DIM


def mask_chunk(chunk: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of ``chunk`` (L, F) with ``frac`` of its timesteps zeroed.

    At least one step is masked whenever ``frac > 0`` and the chunk is
    non-empty, so the model always has something to denoise.
    """
    if frac <= 0:
        return chunk.copy()
    masked = chunk.copy()
    n = len(chunk)
    k = max(1, int(round(n * frac)))
    idx = rng.choice(n, size=min(k, n), replace=False)
    masked[idx] = 0.0
    return masked


class MaskedDenoisingDataset(Dataset):
    """Sliding-window dataset yielding ``(masked, clean)`` chunk pairs.

    Mirrors :class:`pipeline.sequences.dataset.EventSequenceDataset` (same
    chunk-index logic, same 8-D guard) but returns a *pair*: the masked input
    the model sees and the clean target it must reconstruct.

    Parameters
    ----------
    tensors:
        List of normalised ``(N_i, 8)`` arrays.
    chunk_length, stride:
        Window length and step (default 50 % overlap, like the AE training set).
    mask_frac:
        Fraction of timesteps to zero per chunk.
    seed:
        Base seed; chunk ``i`` is masked with ``default_rng(seed + i)``.
    """

    def __init__(
        self,
        tensors: list[np.ndarray],
        chunk_length: int = 64,
        stride: int = 32,
        mask_frac: float = 0.15,
        seed: int = 42,
    ) -> None:
        if chunk_length <= 0 or stride <= 0:
            raise ValueError("chunk_length and stride must be > 0")
        if not 0.0 <= mask_frac < 1.0:
            raise ValueError(f"mask_frac must be in [0, 1), got {mask_frac}")
        for i, t in enumerate(tensors):
            if t.ndim != 2 or t.shape[1] != EVENT_FEATURE_DIM:
                raise ValueError(
                    f"tensors[{i}] must have shape (N, {EVENT_FEATURE_DIM}); got {t.shape}"
                )

        self.tensors = tensors
        self.chunk_length = chunk_length
        self.stride = stride
        self.mask_frac = mask_frac
        self.seed = seed

        self._index: list[tuple[int, int]] = []
        for ti, t in enumerate(tensors):
            for start in _chunk_indices(len(t), chunk_length, stride):
                self._index.append((ti, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ti, start = self._index[idx]
        clean = self.tensors[ti][start : start + self.chunk_length]
        rng = np.random.default_rng(self.seed + idx)
        masked = mask_chunk(clean, self.mask_frac, rng)
        return (
            torch.from_numpy(masked).float(),
            torch.from_numpy(clean.copy()).float(),
        )
