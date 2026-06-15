"""
pipeline/pretraining/augment.py
===============================
Stochastic time-series augmentation for **contrastive** pretraining (Phase 8.2).

Phase 8 / 8.1 pretrained with masked-denoising *reconstruction* and got a null;
the reconstruction MSE is magnitude-dominated, so the learned representation is
biased toward motion *magnitude* (the Phase 8.1 "near-separable at random init"
caveat). A SimCLR / TS2Vec-style contrastive objective instead asks the encoder
to map **two augmented views of the same chunk** close together and different
chunks apart — learning features that are *invariant* to the augmentations
(notably motion scale) rather than reconstructing raw magnitude.

This module supplies those augmentations on a normalised ``(L, 8)`` event chunk
(the shared schema from :mod:`pipeline.sequences.preprocessing`). Each transform
takes ``(chunk, rng, ...)`` and returns a fresh ``(L, 8)`` array, so a single
:class:`Augmenter` call with one ``rng`` yields one random view; calling it twice
on the same chunk (advancing the rng) yields the two views a contrastive pair
needs. The continuous motion channels ``dx``/``dy`` are the only ones jittered /
scaled — the one-hot event-type channels are left to ``time_mask`` / ``crop``,
which act on whole timesteps.

The pieces are intentionally small + deterministic-given-the-rng so the contrastive
dataset can seed reproducibly (``np.random.default_rng([seed, epoch, idx])``) and
the unit tests can assert exact behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.pretraining.masking import mask_chunk
from pipeline.sequences.preprocessing import COL_DX, COL_DY, EVENT_FEATURE_DIM

# The continuous motion channels — the only ones it makes sense to jitter/scale
# (the other six are dt / one-hot indicators handled by time_mask + crop_resize).
MOTION_CHANNELS: tuple[int, ...] = (COL_DX, COL_DY)


def jitter(
    chunk: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.1,
    channels: tuple[int, ...] = MOTION_CHANNELS,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise (std ``sigma``) to the motion channels.

    Operates in normalised space (chunks are z-scored upstream), so ``sigma`` is
    in units of the per-channel std. Returns a copy; other channels untouched.
    """
    out = chunk.copy()
    if sigma <= 0 or len(out) == 0:
        return out
    noise = rng.normal(0.0, sigma, size=(len(out), len(channels))).astype(out.dtype)
    out[:, list(channels)] += noise
    return out


def scale(
    chunk: np.ndarray,
    rng: np.random.Generator,
    low: float = 0.7,
    high: float = 1.3,
    channels: tuple[int, ...] = MOTION_CHANNELS,
) -> np.ndarray:
    """Multiply the motion channels by one random scalar drawn from ``U(low, high)``.

    This is the **magnitude-invariance** the whole phase is about: forcing two
    differently-scaled views of a chunk to the same embedding stops the encoder
    from keying on raw motion magnitude (the reconstruction objective's bias).
    """
    out = chunk.copy()
    if len(out) == 0:
        return out
    factor = float(rng.uniform(low, high))
    out[:, list(channels)] *= factor
    return out


def time_mask(
    chunk: np.ndarray, rng: np.random.Generator, frac: float = 0.15
) -> np.ndarray:
    """Zero a random ``frac`` of whole timesteps (reuses :func:`masking.mask_chunk`)."""
    return mask_chunk(chunk, frac, rng)


def crop_resize(
    chunk: np.ndarray, rng: np.random.Generator, min_frac: float = 0.5
) -> np.ndarray:
    """Random contiguous crop ∈ ``[min_frac·L, L]`` linearly resized back to ``L``.

    A time-warp augmentation: keeps the output shape ``(L, F)`` (so batches stack)
    while changing the local time scale, which the encoder must become invariant
    to. Linear interpolation per channel; a no-op when the crop spans the full
    chunk or the chunk is too short to crop.
    """
    out = chunk.copy()
    n = len(out)
    if n < 4:
        return out
    min_len = max(2, int(round(min_frac * n)))
    if min_len >= n:
        return out
    crop_len = int(rng.integers(min_len, n))  # [min_len, n-1]
    start = int(rng.integers(0, n - crop_len + 1))
    sub = out[start : start + crop_len]
    xp = np.linspace(0.0, 1.0, num=crop_len)
    x = np.linspace(0.0, 1.0, num=n)
    resized = np.empty_like(out)
    for f in range(out.shape[1]):
        resized[:, f] = np.interp(x, xp, sub[:, f])
    return resized.astype(chunk.dtype, copy=False)


@dataclass
class Augmenter:
    """Compose the transforms SimCLR-style: each applied with its own probability.

    One ``__call__(chunk, rng)`` returns a single random view. ``jitter`` defaults
    to probability 1.0 so a view always differs from the clean chunk (and the two
    views of a pair differ from each other); the rest are applied stochastically.
    All randomness flows through the passed ``rng`` → deterministic given its seed.
    """

    jitter_sigma: float = 0.1
    jitter_prob: float = 1.0
    scale_low: float = 0.7
    scale_high: float = 1.3
    scale_prob: float = 0.8
    mask_frac: float = 0.15
    mask_prob: float = 0.5
    crop_min_frac: float = 0.5
    crop_prob: float = 0.5
    channels: tuple[int, ...] = field(default=MOTION_CHANNELS)

    def __call__(self, chunk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if chunk.ndim != 2 or chunk.shape[1] != EVENT_FEATURE_DIM:
            raise ValueError(
                f"chunk must have shape (L, {EVENT_FEATURE_DIM}); got {chunk.shape}"
            )
        out = chunk
        # crop_resize first (changes the whole timeline), then per-channel noise.
        if rng.random() < self.crop_prob:
            out = crop_resize(out, rng, self.crop_min_frac)
        if rng.random() < self.mask_prob:
            out = time_mask(out, rng, self.mask_frac)
        if rng.random() < self.scale_prob:
            out = scale(out, rng, self.scale_low, self.scale_high, self.channels)
        if rng.random() < self.jitter_prob:
            out = jitter(out, rng, self.jitter_sigma, self.channels)
        # always return a distinct array (out may still alias chunk if every op skipped)
        return out if out is not chunk else chunk.copy()
