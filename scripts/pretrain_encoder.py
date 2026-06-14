"""
scripts/pretrain_encoder.py
===========================
Phase 8 / 8.1 — self-supervised pretraining of the sequence encoder.

Pretrains the canonical :class:`LSTMAutoencoder` with a **masked-denoising**
objective (``pipeline.pretraining``), then persists the full encoder+decoder
weights via ``save_lstm_ae`` (same format the downstream AE uses → the weights
load straight into fine-tuning).

Corpora (``--corpus``):
  * ``captcha30k`` (Phase 8, default) — out-of-domain human-mouse corpus, loaded
    fully into memory.
  * ``cs2cd_full`` (Phase 8.1) — **in-domain** CS2 release, streamed lazily from
    the per-match shard cache (:mod:`pipeline.pretraining.cs2cd_full`).
    **LEGIT-only** (``no_cheater_present``); match-disjoint train/val from the
    manifest; the *volume* axis is ``--max-matches`` (a manifest subset key). The
    experiment ``--source`` controls the ``dt`` channel: ``s1`` = native CS2 tick
    ``dt``; ``s2`` = **dt-neutralised** (zeroed, in both domains) to isolate the
    temporal mismatch the Phase 8 null blamed. The shard cache is encoded once
    (native); ``s2`` reuses those shards with ``dt`` zeroed at load — no re-encode.

Usage:
    python -m scripts.pretrain_encoder                          # captcha (Phase 8)
    python -m scripts.pretrain_encoder --corpus cs2cd_full --source s1 \
        --max-matches 50 --out-name pretrained_cs2cd_s1_50      # Phase 8.1

Artifacts are DVC-tracked. Idempotent per ``--out-name``; seeds fixed.
"""

from __future__ import annotations

import argparse
import json
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
from pipeline.pretraining.cs2cd_full import (
    MANIFEST,
    CS2CDShardChunkDataset,
    ShardGroupedSampler,
    fit_shard_normalizer,
    shards_for_matches,
)
from pipeline.pretraining.masking import MaskedDenoisingDataset
from pipeline.pretraining.pretrain import pretrain_masked_ae
from pipeline.sequences.preprocessing import apply_normalizer, fit_normalizer

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# experiment source → dt-channel transform (None = native CS2 tick dt; 0.0 = neutralised)
_SOURCE_DT = {"s1": None, "s2": 0.0}


def _captcha_data(args):
    """Build ``(train_ds, val_ds, stats, meta)`` from the in-memory captcha corpus."""
    max_sessions = (
        None
        if (args.max_sessions is not None and args.max_sessions <= 0)
        else args.max_sessions
    )
    tensors = captcha_to_tensors(max_sessions=max_sessions, seed=args.seed)
    if not tensors:
        raise SystemExit("No captcha sessions loaded")
    log.info(
        "Loaded %d captcha sessions, %d ticks",
        len(tensors),
        sum(len(t) for t in tensors),
    )
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tensors))
    n_val = max(1, int(round(len(tensors) * args.val_fraction)))
    val_idx = set(perm[:n_val].tolist())
    train = [tensors[i] for i in range(len(tensors)) if i not in val_idx]
    val = [tensors[i] for i in range(len(tensors)) if i in val_idx]
    stats = fit_normalizer(train)
    train_norm = [apply_normalizer(t, stats) for t in train]
    val_norm = [apply_normalizer(t, stats) for t in val]
    train_ds = MaskedDenoisingDataset(
        train_norm, args.chunk_length, args.stride, args.mask_frac, seed=args.seed
    )
    val_ds = MaskedDenoisingDataset(
        val_norm, args.chunk_length, args.chunk_length, args.mask_frac, seed=args.seed
    )
    meta = dict(
        corpus="captcha30k",
        source="n/a",
        max_sessions=max_sessions,
        n_units=len(tensors),
    )
    return train_ds, val_ds, stats, meta


def _cs2cd_data(args):
    """Build ``(train_ds, val_ds, stats, meta)`` lazily from the CS2CD shard cache."""
    if not MANIFEST.exists():
        raise SystemExit(
            f"manifest missing: {MANIFEST}; run "
            "`python -m pipeline.pretraining.cs2cd_full --step manifest`"
        )
    manifest = json.loads(MANIFEST.read_text())
    dt = _SOURCE_DT[args.source]  # s1=None (native), s2=0.0 (neutralised)

    if args.max_matches:
        subsets = manifest["diversity_subsets"]
        key = str(args.max_matches)
        if key not in subsets:
            raise SystemExit(
                f"--max-matches {key} not a manifest subset {list(subsets)}"
            )
        train_ids = subsets[key]
    else:
        train_ids = manifest["pretrain_matches"]

    # Shards are encoded once (native, cache dir "s1"); s2 reuses them via dt-neutralisation.
    train_shards = shards_for_matches(
        train_ids, subdir="no_cheater_present", source="s1"
    )
    val_shards = shards_for_matches(
        manifest["val_matches"], subdir="no_cheater_present", source="s1"
    )
    if not train_shards:
        raise SystemExit(
            "no train shards — run "
            "`python -m pipeline.pretraining.cs2cd_full --step encode-legit`"
        )
    log.info(
        "CS2CD pretrain: %d train / %d val shards (source=%s, dt_override=%s, matches=%s)",
        len(train_shards),
        len(val_shards),
        args.source,
        dt,
        args.max_matches or "all-pretrain",
    )
    stats = fit_shard_normalizer(train_shards, seed=args.seed, dt_override_ms=dt)
    train_ds = CS2CDShardChunkDataset(
        train_shards,
        stats=stats,
        chunk_length=args.chunk_length,
        stride=args.stride,
        mask_frac=args.mask_frac,
        seed=args.seed,
        dt_override_ms=dt,
    )
    val_ds = CS2CDShardChunkDataset(
        val_shards,
        stats=stats,
        chunk_length=args.chunk_length,
        stride=args.chunk_length,  # non-overlapping val windows
        mask_frac=args.mask_frac,
        seed=args.seed,
        dt_override_ms=dt,
    )
    meta = dict(
        corpus="cs2cd_full",
        source=args.source,
        dt_override_ms=dt,
        max_matches=args.max_matches,
        n_units=len(train_shards),
        split_unit=manifest["split_unit"],
        manifest_seed=manifest["seed"],
        branch=manifest.get("branch"),
    )
    return train_ds, val_ds, stats, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Masked-denoising pretraining (Phase 8 / 8.1)"
    )
    parser.add_argument(
        "--corpus", choices=["captcha30k", "cs2cd_full"], default="captcha30k"
    )
    parser.add_argument(
        "--source",
        choices=["s1", "s2"],
        default="s1",
        help="cs2cd_full dt encoding: s1 native tick dt, s2 dt-neutralised",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="cs2cd_full volume axis: a manifest diversity-subset key (e.g. 50)",
    )
    parser.add_argument(
        "--max-sessions", type=int, default=6000, help="captcha subsample (None=all)"
    )
    parser.add_argument(
        "--out-name", default="pretrained_encoder", help="artifact stem under --out-dir"
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
    parser.add_argument("--num-workers", type=int, default=0)
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

    if args.corpus == "captcha30k":
        train_ds, val_ds, stats, corpus_meta = _captcha_data(args)
        train_sampler = None
    else:
        train_ds, val_ds, stats, corpus_meta = _cs2cd_data(args)
        # Shard-grouped shuffle keeps the LRU from thrashing over large shards.
        train_sampler = ShardGroupedSampler(train_ds, shuffle=True, seed=args.seed)

    log.info("Pretrain chunks: %d train | %d val", len(train_ds), len(val_ds))
    if len(train_ds) == 0:
        log.error("No training chunks — units shorter than chunk_length")
        return 2

    pin = args.device != "cpu" and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
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
        n_train_chunks=len(train_ds),
        **corpus_meta,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # save_lstm_ae writes the *fixed* names lstm_ae.pt/_meta.json — save into a temp
    # dir first, then move to the --out-name names, so we never clobber the canonical
    # (DVC-tracked) models/lstm_ae.pt artifact sitting next to us.
    with tempfile.TemporaryDirectory() as tmp:
        weights, meta = save_lstm_ae(
            model, stats, Path(tmp), config=config, history=history
        )
        out_weights = args.out_dir / f"{args.out_name}.pt"
        out_meta = args.out_dir / f"{args.out_name}_meta.json"
        shutil.move(str(weights), out_weights)
        shutil.move(str(meta), out_meta)
    log.info("Saved pretrained encoder → %s", out_weights)
    log.info("Best val_loss=%.5f @ epoch %d", history.best_val_loss, history.best_epoch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
