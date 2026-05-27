"""
scripts/train_lstm_ae.py
========================
Train the LSTM autoencoder on every legit recording in ``data/raw/`` and
persist the resulting model + normaliser to ``models/`` for downstream use
by the streaming API and the adversarial benchmark.

Usage:
    python -m scripts.train_lstm_ae [--epochs 30] [--device auto] [--val-fraction 0.15]

The script is idempotent: re-running it overwrites the artifact in place.
The benchmark (``pipeline.adversarial.benchmark``) loads this artifact when
present instead of retraining on every run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.models.lstm_ae import save_lstm_ae, train_lstm_ae
from pipeline.sequences.dataset import EventSequenceDataset
from pipeline.sequences.preprocessing import (
    apply_normalizer,
    fit_normalizer,
    session_to_event_tensor,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"


def _load_legit_tensors() -> tuple[list[np.ndarray], list[str]]:
    """Convert every JSON in ``data/raw/`` to an (N, 8) event tensor."""
    tensors: list[np.ndarray] = []
    names: list[str] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tensor = session_to_event_tensor(data)
        if len(tensor) == 0:
            log.warning("Skipping %s — no parseable events", path.name)
            continue
        tensors.append(tensor)
        names.append(path.name)
    return tensors, names


def main() -> int:
    parser = argparse.ArgumentParser(description="Train + persist the LSTM-AE")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MODELS_DIR,
        help="Where to write lstm_ae.pt + lstm_ae_meta.json (default: models/)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tensors, names = _load_legit_tensors()
    if not tensors:
        log.error("No legit sessions found in %s", RAW_DIR)
        return 1
    log.info(
        "Loaded %d legit sessions, %d total events",
        len(tensors),
        sum(len(t) for t in tensors),
    )

    # Train/val split on the session level
    n_val = max(1, int(round(len(tensors) * args.val_fraction)))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tensors))
    val_idx = set(perm[:n_val].tolist())
    train_tensors = [tensors[i] for i in range(len(tensors)) if i not in val_idx]
    val_tensors = [tensors[i] for i in range(len(tensors)) if i in val_idx]
    train_names = [names[i] for i in range(len(names)) if i not in val_idx]
    val_names = [names[i] for i in range(len(names)) if i in val_idx]
    log.info(
        "Train: %d sessions | Val: %d sessions", len(train_tensors), len(val_tensors)
    )
    log.info("Val held out: %s", val_names)

    stats = fit_normalizer(train_tensors)
    train_norm = [apply_normalizer(t, stats) for t in train_tensors]
    val_norm = [apply_normalizer(t, stats) for t in val_tensors]

    train_ds = EventSequenceDataset(
        train_norm, chunk_length=args.chunk_length, stride=args.stride
    )
    val_ds = EventSequenceDataset(
        val_norm, chunk_length=args.chunk_length, stride=args.chunk_length
    )
    log.info("Train chunks: %d | Val chunks: %d", len(train_ds), len(val_ds))

    if len(train_ds) == 0:
        log.error("No training chunks — sessions are all shorter than chunk_length")
        return 2

    pin = args.device != "cpu" and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin,
        num_workers=0,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=pin,
            num_workers=0,
        )
        if len(val_ds) > 0
        else None
    )

    model, history = train_lstm_ae(
        train_loader,
        val_loader,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        log_every=5,
    )

    config = dict(
        chunk_length=args.chunk_length,
        stride=args.stride,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        train_sessions=train_names,
        val_sessions=val_names,
    )
    weights, meta = save_lstm_ae(
        model, stats, args.out_dir, config=config, history=history
    )
    log.info("Saved model weights → %s", weights)
    log.info("Saved metadata     → %s", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
