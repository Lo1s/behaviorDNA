"""
pipeline/pretraining/pretrain.py
================================
Masked-denoising pretraining loop for the :class:`LSTMAutoencoder`.

This is the supervised-reconstruction loop of ``pipeline.models.lstm_ae`` with
one change: each batch is a ``(masked, clean)`` pair and the loss is
``MSE(model(masked), clean)`` instead of ``MSE(model(x), x)``. Everything else —
the model, device selection, ``TrainingHistory``, best-weight restore, and the
``save_lstm_ae`` artifact format — is reused so the pretrained weights load
straight into the downstream fine-tuning autoencoder.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline.models.lstm_ae import (
    EVENT_FEATURE_DIM,
    LSTMAutoencoder,
    TrainingHistory,
    _select_device,
)

log = logging.getLogger(__name__)


def _masked_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """One epoch over ``(masked, clean)`` batches — train if optimizer given."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for masked, clean in loader:
            masked = masked.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            recon = model(masked)
            loss = loss_fn(recon, clean)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * masked.size(0)
            n += masked.size(0)
    return total_loss / max(n, 1)


def pretrain_masked_ae(
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    feature_dim: int = EVENT_FEATURE_DIM,
    hidden_dim: int = 64,
    bottleneck_dim: int = 16,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 30,
    weight_decay: float = 0.0,
    device: str = "auto",
    log_every: int = 1,
    early_stopping_patience: int | None = 5,
) -> tuple[LSTMAutoencoder, TrainingHistory]:
    """Pretrain an ``LSTMAutoencoder`` with the masked-denoising objective.

    Returns the model with best-val-loss weights restored and the loss history.
    Same signature shape as ``pipeline.models.lstm_ae.train_lstm_ae`` so the two
    are interchangeable downstream.
    """
    torch_device = _select_device(device)
    log.info("Pretraining (masked-denoising) on device=%s", torch_device)

    model = LSTMAutoencoder(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = TrainingHistory(device=str(torch_device))

    best_state: dict | None = None
    stale = 0
    for epoch in range(1, epochs + 1):
        train_loss = _masked_epoch(
            model, train_loader, loss_fn, torch_device, optimizer
        )
        history.train_loss.append(train_loss)

        if val_loader is not None:
            val_loss = _masked_epoch(model, val_loader, loss_fn, torch_device, None)
            history.val_loss.append(val_loss)
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
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
        model.load_state_dict(best_state)
        log.info(
            "Restored best weights from epoch %d (val_loss=%.5f)",
            history.best_epoch,
            history.best_val_loss,
        )
    return model, history
