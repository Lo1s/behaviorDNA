"""
pipeline/pretraining/contrastive.py
===================================
Phase 8.2 — **contrastive** self-supervised pretraining of the sequence encoder.

The Phase 8 / 8.1 masked-denoising objective returned a rigorous null; this swaps
the *objective* (not the architecture or data) for a SimCLR / TS2Vec-style
contrastive one — the last untested lever on ``docs/PRETRAINING.md``'s "what would
change the verdict" list.

Pipeline of one training step:

    clean chunk (L, 8)
        │  Augmenter (pipeline.pretraining.augment) — twice, advancing the rng
        ▼
    view₁, view₂ (L, 8)
        │  LSTMAutoencoder.encode  (the SAME 16-D bottleneck Phase 8/8.1 used)
        ▼
    h₁, h₂ (16)
        │  ProjectionHead (discarded after pretraining — standard SimCLR)
        ▼
    z₁, z₂ (proj_dim)
        │  NT-Xent / InfoNCE: pull a chunk's two views together, push other
        ▼     chunks' views apart

Only the **backbone** ``LSTMAutoencoder`` is persisted (via ``save_lstm_ae``), so
the artifact format is byte-identical to the 8.1 encoders and the frozen-embedding
eval (:mod:`pipeline.pretraining.embed_eval`) shares one code path across objectives.
The decoder is never trained (eval uses ``encode`` only) — it stays at init in the
saved state_dict, which still loads cleanly into a fresh ``LSTMAutoencoder``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pipeline.models.lstm_ae import (
    EVENT_FEATURE_DIM,
    LSTMAutoencoder,
    TrainingHistory,
    _select_device,
)
from pipeline.pretraining.augment import Augmenter
from pipeline.pretraining.cs2cd_full import CS2CDShardChunkDataset
from pipeline.sequences.dataset import _chunk_indices

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projection head + NT-Xent loss
# ---------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    """MLP ``in_dim → hidden → out_dim`` mapping the bottleneck to the contrastive
    space. Discarded after pretraining (only the backbone transfers)."""

    def __init__(self, in_dim: int = 16, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    """Normalised temperature-scaled cross-entropy (SimCLR NT-Xent).

    ``z1``/``z2`` are the two views' projections ``(B, D)``. For each of the ``2B``
    embeddings the positive is its paired view and the negatives are the other
    ``2B-2`` embeddings — implemented as cross-entropy over the cosine-similarity
    matrix with the self-similarity diagonal masked out.
    """
    if z1.shape != z2.shape or z1.dim() != 2:
        raise ValueError(f"z1, z2 must be matching (B, D); got {z1.shape}, {z2.shape}")
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature  # (2B, 2B) cosine sims
    # mask self-comparisons so they can't be picked as the positive
    self_mask = torch.eye(2 * b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    # positive of i in [0,B) is i+B; positive of i in [B,2B) is i-B
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)


# ---------------------------------------------------------------------------
# Datasets — yield TWO augmented views per chunk
# ---------------------------------------------------------------------------
def _two_views(
    window: np.ndarray, augment: Augmenter, seed: int, epoch: int, idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two augmented views of ``window`` — reproducible per ``(seed, epoch, idx)``.

    A single rng is advanced across both calls so the views differ; seeding by a
    ``SeedSequence`` list keeps it worker-safe *and* epoch-varying (call
    ``set_epoch`` each epoch for fresh augmentations, the SimCLR norm).
    """
    rng = np.random.default_rng([seed, epoch, idx])
    v1 = augment(window, rng)
    v2 = augment(window, rng)
    return torch.from_numpy(v1).float(), torch.from_numpy(v2).float()


class ContrastiveSequenceDataset(Dataset):
    """In-memory two-view dataset over a list of normalised ``(N_i, 8)`` tensors.

    Mirrors :class:`pipeline.pretraining.masking.MaskedDenoisingDataset`'s chunk
    indexing but returns ``(view1, view2)`` instead of ``(masked, clean)``. Used
    for the captcha corpus (loaded fully into memory).
    """

    def __init__(
        self,
        tensors: list[np.ndarray],
        chunk_length: int = 64,
        stride: int = 64,
        *,
        augment: Augmenter | None = None,
        seed: int = 42,
    ) -> None:
        if chunk_length <= 0 or stride <= 0:
            raise ValueError("chunk_length and stride must be > 0")
        for i, t in enumerate(tensors):
            if t.ndim != 2 or t.shape[1] != EVENT_FEATURE_DIM:
                raise ValueError(
                    f"tensors[{i}] must have shape (N, {EVENT_FEATURE_DIM}); got {t.shape}"
                )
        self.tensors = tensors
        self.chunk_length = chunk_length
        self.stride = stride
        self.augment = augment or Augmenter()
        self.seed = seed
        self.epoch = 0
        self._index: list[tuple[int, int]] = []
        for ti, t in enumerate(tensors):
            for start in _chunk_indices(len(t), chunk_length, stride):
                self._index.append((ti, start))

    def set_epoch(self, epoch: int) -> None:
        """Vary augmentation per epoch (call before each training epoch)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ti, start = self._index[idx]
        window = self.tensors[ti][start : start + self.chunk_length]
        return _two_views(window, self.augment, self.seed, self.epoch, idx)


class CS2CDContrastiveShardDataset(CS2CDShardChunkDataset):
    """Two-view dataset over the cached CS2CD per-match shards.

    Reuses the parent's LRU + global chunk index + ``shard_index_groups`` (so
    :class:`pipeline.pretraining.cs2cd_full.ShardGroupedSampler` works unchanged);
    only the per-item transform changes from masking to two contrastive views.
    """

    def __init__(self, *args, augment: Augmenter | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.augment = augment or Augmenter()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self._clean_window(idx)
        return _two_views(window, self.augment, self.seed, self.epoch, idx)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _contrastive_epoch(
    backbone: LSTMAutoencoder,
    head: ProjectionHead,
    loader: DataLoader,
    temperature: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """One epoch over ``(view1, view2)`` batches — train if optimizer given."""
    is_train = optimizer is not None
    backbone.train(is_train)
    head.train(is_train)
    total_loss = 0.0
    n = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for v1, v2 in loader:
            if v1.size(0) < 2:  # NT-Xent needs ≥2 chunks to form negatives
                continue
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            z1 = head(backbone.encode(v1))
            z2 = head(backbone.encode(v2))
            loss = nt_xent_loss(z1, z2, temperature)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * v1.size(0)
            n += v1.size(0)
    return total_loss / max(n, 1)


def pretrain_contrastive(
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    feature_dim: int = EVENT_FEATURE_DIM,
    hidden_dim: int = 64,
    bottleneck_dim: int = 16,
    num_layers: int = 2,
    dropout: float = 0.2,
    proj_hidden_dim: int = 64,
    proj_dim: int = 32,
    temperature: float = 0.5,
    lr: float = 1e-3,
    epochs: int = 20,
    weight_decay: float = 1e-6,
    device: str = "auto",
    log_every: int = 1,
    early_stopping_patience: int | None = 5,
    set_epoch_fn=None,
) -> tuple[LSTMAutoencoder, ProjectionHead, TrainingHistory]:
    """Contrastively pretrain an ``LSTMAutoencoder`` backbone with NT-Xent.

    Optimises the encoder + bottleneck + projection head (the decoder is left at
    init — eval uses ``encode`` only). Returns the backbone with best-val-loss
    weights restored, the (discardable) head, and the loss history. ``set_epoch_fn``
    is called with the epoch number before each train epoch so the dataset can
    refresh its augmentations.
    """
    torch_device = _select_device(device)
    log.info("Contrastive pretraining (NT-Xent) on device=%s", torch_device)

    backbone = LSTMAutoencoder(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(torch_device)
    head = ProjectionHead(bottleneck_dim, proj_hidden_dim, proj_dim).to(torch_device)

    # Decoder gets no gradient from the contrastive loss → exclude it from the
    # optimiser (kept at init in the saved state_dict; eval never decodes).
    params = (
        list(backbone.encoder.parameters())
        + list(backbone.to_bottleneck.parameters())
        + list(head.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    history = TrainingHistory(device=str(torch_device))

    best_state: dict | None = None
    stale = 0
    for epoch in range(1, epochs + 1):
        if set_epoch_fn is not None:
            set_epoch_fn(epoch)
        train_loss = _contrastive_epoch(
            backbone, head, train_loader, temperature, torch_device, optimizer
        )
        history.train_loss.append(train_loss)

        if val_loader is not None:
            val_loss = _contrastive_epoch(
                backbone, head, val_loader, temperature, torch_device, None
            )
            history.val_loss.append(val_loss)
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in backbone.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
        else:
            val_loss = float("nan")

        if epoch % log_every == 0:
            log.info(
                "epoch %3d/%d  train=%.5f  val=%.5f%s",
                epoch,
                epochs,
                train_loss,
                val_loss,
                (
                    "  *best"
                    if val_loader is not None and history.best_epoch == epoch
                    else ""
                ),
            )

        if (
            val_loader is not None
            and early_stopping_patience is not None
            and stale >= early_stopping_patience
        ):
            log.info("Early stopping at epoch %d", epoch)
            break

    if best_state is not None:
        backbone.load_state_dict(best_state)
        log.info(
            "Restored best backbone from epoch %d (val_loss=%.5f)",
            history.best_epoch,
            history.best_val_loss,
        )
    return backbone, head, history
