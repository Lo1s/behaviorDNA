"""
pipeline/models/lstm_ae.py
==========================
Bidirectional LSTM autoencoder for unsupervised behavioural anomaly detection.

Trained on raw event sequences from legit sessions only, the model learns to
reconstruct the normal millisecond-by-millisecond input stream. At inference
time, sequences containing cheats (especially short aimbot snaps that survive
none of Phase 1's window aggregations) produce elevated reconstruction error
that pops out clearly against the legit baseline.

Architecture (~80k params with the default config):

    Input  (B, L, 8)
       │
       ▼
    Bidirectional LSTM encoder (hidden=64, num_layers=2, dropout=0.2)
       │   final hidden h_T → (B, num_directions * num_layers * hidden)
       ▼
    Linear → bottleneck (16-D)
       │
       ▼  broadcast across L timesteps as decoder input
    LSTM decoder (hidden=64, num_layers=2)
       │
       ▼
    Linear → reconstruction (B, L, 8)
       │
       ▼
    Loss = MSE(input, reconstruction)

Three public entry points:

- ``LSTMAutoencoder(nn.Module)`` — the model
- ``train_lstm_ae(...)`` — full training loop with val-set early stopping
- ``score_sequences(...)`` — per-chunk reconstruction MSE for inference

The training loop is GPU-aware via ``torch.cuda.is_available()`` and falls
back to CPU silently. See ``docs/LSTM_AE.md`` for the full write-up.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline.sequences.preprocessing import EVENT_FEATURE_DIM

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder.

    Encoder is bidirectional (hidden_size doubled internally). The bottleneck
    flattens the encoder's last hidden state across direction × layer × hidden
    and projects it through a linear layer to ``bottleneck_dim``. The decoder
    is a one-directional LSTM whose input at every timestep is the broadcast
    bottleneck embedding — a simple "vector → sequence" pattern that avoids
    the complexity of teacher-forcing schedules while still producing a
    reconstruction the encoder is forced to compress through.
    """

    def __init__(
        self,
        feature_dim: int = EVENT_FEATURE_DIM,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if hidden_dim < 1 or bottleneck_dim < 1:
            raise ValueError("hidden_dim and bottleneck_dim must be positive")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bidirectional × num_layers × hidden, flattened
        encoder_out_dim = 2 * num_layers * hidden_dim
        self.to_bottleneck = nn.Linear(encoder_out_dim, bottleneck_dim)

        self.decoder = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.from_decoder = nn.Linear(hidden_dim, feature_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, L, F)`` → bottleneck embedding ``(B, bottleneck_dim)``."""
        _, (h_n, _) = self.encoder(x)
        # h_n shape: (num_layers * num_directions, B, hidden_dim)
        # Move batch to front and flatten the rest
        h_n = h_n.permute(1, 0, 2).reshape(x.size(0), -1)
        return self.to_bottleneck(h_n)

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode bottleneck ``z (B, bottleneck_dim)`` → sequence ``(B, L, F)``."""
        # Broadcast z across L timesteps as the decoder input
        decoder_in = z.unsqueeze(1).expand(-1, seq_len, -1)
        decoded, _ = self.decoder(decoder_in)
        return self.from_decoder(decoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z, seq_len=x.size(1))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean MSE across the sequence + feature axes.

        Returns a ``(B,)`` tensor — one scalar score per chunk. Used at
        inference time to rank chunks by anomaly score.
        """
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=(1, 2))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    device: str = "cpu"


def _select_device(prefer: str = "auto") -> torch.device:
    """Pick CUDA if available, otherwise CPU. Set ``prefer='cpu'`` to force CPU."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("CUDA requested but unavailable — falling back to CPU")
    return torch.device("cpu")


def _epoch_pass(
    model: LSTMAutoencoder,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one epoch — train when ``optimizer`` is given, else eval."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_chunks = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.size(0)
            n_chunks += batch.size(0)
    return total_loss / max(n_chunks, 1)


def train_lstm_ae(
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
    early_stopping_patience: int | None = 10,
) -> tuple[LSTMAutoencoder, TrainingHistory]:
    """Train an LSTM autoencoder on a DataLoader of event-sequence chunks.

    Returns the trained model with the best-val-loss weights restored, and a
    history object containing per-epoch losses (for plotting in the tutorial).

    If ``val_loader`` is None, no early stopping / best-weight selection
    happens — the final epoch's weights are returned as-is.
    """
    torch_device = _select_device(device)
    log.info("Training LSTM-AE on device=%s", torch_device)

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
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = _epoch_pass(model, train_loader, loss_fn, torch_device, optimizer)
        history.train_loss.append(train_loss)

        if val_loader is not None:
            val_loss = _epoch_pass(
                model, val_loader, loss_fn, torch_device, optimizer=None
            )
            history.val_loss.append(val_loss)
            improved = val_loss < history.best_val_loss
            if improved:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
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
            and epochs_without_improvement >= early_stopping_patience
        ):
            log.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch,
                early_stopping_patience,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(
            "Restored best weights from epoch %d (val_loss=%.5f)",
            history.best_epoch,
            history.best_val_loss,
        )

    return model, history


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_sequences(
    model: LSTMAutoencoder,
    sequences: torch.Tensor | np.ndarray,
    batch_size: int = 256,
    device: str = "auto",
) -> np.ndarray:
    """Compute per-chunk reconstruction MSE for a stack of sequences.

    Parameters
    ----------
    sequences:
        Tensor or array of shape ``(N, L, F)``.
    batch_size:
        Inference batch size.
    device:
        See ``_select_device``.

    Returns
    -------
    np.ndarray of shape ``(N,)`` — one float score per chunk.
    """
    torch_device = _select_device(device)
    model = model.to(torch_device)
    model.eval()

    if isinstance(sequences, np.ndarray):
        sequences = torch.from_numpy(sequences).float()
    elif not torch.is_tensor(sequences):
        raise TypeError(f"sequences must be Tensor or ndarray, got {type(sequences)}")

    scores: list[float] = []
    for start in range(0, sequences.size(0), batch_size):
        batch = sequences[start : start + batch_size].to(
            torch_device, non_blocking=True
        )
        batch_scores = model.reconstruction_error(batch)
        scores.extend(batch_scores.detach().cpu().tolist())

    return np.asarray(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Persistence — save and reload trained models for the streaming API
# ---------------------------------------------------------------------------

LSTM_AE_WEIGHTS_NAME = "lstm_ae.pt"
LSTM_AE_META_NAME = "lstm_ae_meta.json"


def save_lstm_ae(
    model: LSTMAutoencoder,
    normalizer_stats: dict,
    out_dir: Path,
    *,
    config: dict | None = None,
    history: TrainingHistory | None = None,
) -> tuple[Path, Path]:
    """Persist a trained LSTM-AE to ``out_dir``.

    Writes two files:

    - ``lstm_ae.pt``  — state_dict (architecture-agnostic weights)
    - ``lstm_ae_meta.json`` — architecture config, normaliser stats, optional
      training metrics. Everything ``load_lstm_ae`` needs to rebuild the
      model and prepare new inputs.

    Returns the two paths in the order above.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / LSTM_AE_WEIGHTS_NAME
    meta_path = out_dir / LSTM_AE_META_NAME

    torch.save(model.state_dict(), weights_path)

    meta: dict = {
        "feature_dim": int(model.feature_dim),
        "hidden_dim": int(model.hidden_dim),
        "bottleneck_dim": int(model.bottleneck_dim),
        "num_layers": int(model.num_layers),
        "normalizer": {
            "mean": np.asarray(normalizer_stats["mean"]).tolist(),
            "std": np.asarray(normalizer_stats["std"]).tolist(),
        },
    }
    if config is not None:
        meta["config"] = config
    if history is not None:
        meta["training"] = {
            "train_loss": [float(x) for x in history.train_loss],
            "val_loss": [float(x) for x in history.val_loss],
            "best_val_loss": float(history.best_val_loss),
            "best_epoch": int(history.best_epoch),
            "device": history.device,
        }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return weights_path, meta_path


def load_lstm_ae(
    out_dir: Path, *, device: str = "auto"
) -> tuple[LSTMAutoencoder, dict, dict]:
    """Load a persisted LSTM-AE produced by :func:`save_lstm_ae`.

    Returns
    -------
    (model, normalizer_stats, meta)
        - ``model`` — restored ``LSTMAutoencoder``, in eval mode, moved to the
          selected device.
        - ``normalizer_stats`` — ``{"mean": np.ndarray(8,), "std": np.ndarray(8,)}``
          suitable for ``pipeline.sequences.preprocessing.apply_normalizer``.
        - ``meta`` — the full metadata dict (config, training history, etc.).
    """
    out_dir = Path(out_dir)
    weights_path = out_dir / LSTM_AE_WEIGHTS_NAME
    meta_path = out_dir / LSTM_AE_META_NAME

    if not weights_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Could not find both {LSTM_AE_WEIGHTS_NAME} and {LSTM_AE_META_NAME} "
            f"in {out_dir}"
        )

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    torch_device = _select_device(device)
    model = LSTMAutoencoder(
        feature_dim=meta["feature_dim"],
        hidden_dim=meta["hidden_dim"],
        bottleneck_dim=meta["bottleneck_dim"],
        num_layers=meta["num_layers"],
        # dropout doesn't affect inference — restore to 0 by convention
        dropout=0.0,
    )
    state_dict = torch.load(weights_path, map_location=torch_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(torch_device).eval()

    norm = meta["normalizer"]
    normalizer_stats = {
        "mean": np.asarray(norm["mean"], dtype=np.float32),
        "std": np.asarray(norm["std"], dtype=np.float32),
    }
    return model, normalizer_stats, meta
