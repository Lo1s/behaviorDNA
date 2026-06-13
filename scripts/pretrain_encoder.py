"""
scripts/pretrain_encoder.py
===========================
Phase 8 — self-supervised pretraining of the sequence encoder.

Pretrains the canonical :class:`LSTMAutoencoder` with a **masked-denoising**
objective (``pipeline.pretraining``) on the unlabelled **CaptchaSolve30k**
human-mouse corpus, then persists the full encoder+decoder weights to
``models/pretrained_encoder.pt`` (+ ``_meta.json``) via the same ``save_lstm_ae``
format the downstream AE uses — so the weights load straight into fine-tuning.

Usage (CUDA desktop):
    python -m scripts.pretrain_encoder --max-sessions 6000 --epochs 30

The artifact is DVC-tracked (``dvc add models/pretrained_encoder.pt``). Idempotent
— re-running overwrites in place. Seeds fixed for reproducibility.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.models.lstm_ae import save_lstm_ae
from pipeline.pretraining.corpora import captcha_to_tensors
from pipeline.pretraining.masking import MaskedDenoisingDataset
from pipeline.pretraining.pretrain import pretrain_masked_ae
from pipeline.sequences.preprocessing import apply_normalizer, fit_normalizer

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PRETRAINED_WEIGHTS = "pretrained_encoder.pt"
PRETRAINED_META = "pretrained_encoder_meta.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Masked-denoising pretraining on CaptchaSolve30k"
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=6000,
        help="Subsample of mouse sessions to pretrain on (None=all)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mask-frac", type=float, default=0.15)
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
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    max_sessions = (
        None
        if (args.max_sessions is not None and args.max_sessions <= 0)
        else args.max_sessions
    )
    tensors = captcha_to_tensors(max_sessions=max_sessions, seed=args.seed)
    if not tensors:
        log.error("No captcha sessions loaded")
        return 1
    log.info(
        "Loaded %d captcha sessions, %d total ticks",
        len(tensors),
        sum(len(t) for t in tensors),
    )

    # Session-level train/val split (best-weight selection on held-out sessions).
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tensors))
    n_val = max(1, int(round(len(tensors) * args.val_fraction)))
    val_idx = set(perm[:n_val].tolist())
    train = [tensors[i] for i in range(len(tensors)) if i not in val_idx]
    val = [tensors[i] for i in range(len(tensors)) if i in val_idx]

    # Normaliser fit on the pretraining train fold (z-scores each channel).
    stats = fit_normalizer(train)
    train_norm = [apply_normalizer(t, stats) for t in train]
    val_norm = [apply_normalizer(t, stats) for t in val]

    train_ds = MaskedDenoisingDataset(
        train_norm, args.chunk_length, args.stride, args.mask_frac, seed=args.seed
    )
    val_ds = MaskedDenoisingDataset(
        val_norm, args.chunk_length, args.chunk_length, args.mask_frac, seed=args.seed
    )
    log.info("Pretrain chunks: %d train | %d val", len(train_ds), len(val_ds))
    if len(train_ds) == 0:
        log.error("No training chunks — sessions shorter than chunk_length")
        return 2

    pin = args.device != "cpu" and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin)
        if len(val_ds) > 0
        else None
    )

    model, history = pretrain_masked_ae(
        train_loader,
        val_loader,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        log_every=1,
    )

    config = dict(
        objective="masked_denoising",
        corpus="captcha30k",
        max_sessions=max_sessions,
        mask_frac=args.mask_frac,
        chunk_length=args.chunk_length,
        stride=args.stride,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        n_sessions=len(tensors),
        n_train_chunks=len(train_ds),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # save_lstm_ae writes the *fixed* names lstm_ae.pt/_meta.json — save into a
    # temp dir first, then move to the pretrained names, so we never clobber the
    # canonical (DVC-tracked) models/lstm_ae.pt artifact sitting next to us.
    with tempfile.TemporaryDirectory() as tmp:
        weights, meta = save_lstm_ae(
            model, stats, Path(tmp), config=config, history=history
        )
        shutil.move(str(weights), args.out_dir / PRETRAINED_WEIGHTS)
        shutil.move(str(meta), args.out_dir / PRETRAINED_META)
    log.info("Saved pretrained encoder → %s", args.out_dir / PRETRAINED_WEIGHTS)
    log.info("Best val_loss=%.5f @ epoch %d", history.best_val_loss, history.best_epoch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
