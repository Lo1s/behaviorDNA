"""
scripts/contrastive_transfer.py
===============================
Phase 8.2 headline — does a **contrastive** pretext task (the last untested lever
after the Phase 8 / 8.1 masked-denoising nulls) buy transferable structure for GTA
cheat detection, measured **contrastive-natively** on the frozen embedding?

Two phases (mirrors ``scripts/indomain_transfer.py``):

  * ``--phase pretrain`` — contrastively pretrain the LSTM-AE backbone (NT-Xent over
    two augmented views) on CS2CD shards at volumes 50/200/382 + on captcha, saving
    ``models/pretrained_contrastive_{cs2cd_{vol},captcha}.pt``. Idempotent per
    out-name (skips existing → per-encoder checkpointing; resumable).
  * ``--phase eval`` — freeze each encoder, embed GTA legit/cheat chunks through the
    16-D bottleneck, and score the representation directly with one-class detectors
    (Mahalanobis / OCSVM / kNN) + a CV linear probe, over budget × seed (mean±std).

The decisive rows are **random-init** (did contrastive learn anything above a random
projection? — kills Phase 8.1's "near-separable at random init" caveat) and the
Phase-8.1 **reconstruction** encoder (is contrastive a *better objective* under the
identical frozen-embedding eval?).

Reuses verbatim: the CS2CD shard pipeline (``pipeline.pretraining.cs2cd_full``), the
Phase-8 GTA pool + chunkers (``scripts.data_efficiency``), the captcha adapter, and
``save_lstm_ae``. New machinery is ``pipeline.pretraining.{augment,contrastive,embed_eval}``.

Usage (CUDA desktop):
    python -m scripts.contrastive_transfer --phase pretrain   # 4 encoders (long pole)
    python -m scripts.contrastive_transfer --phase eval       # frozen-embedding matrix
    python -m scripts.contrastive_transfer --phase all
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
from torch.utils.data import DataLoader, Sampler

from pipeline.models.lstm_ae import LSTMAutoencoder, save_lstm_ae
from pipeline.pretraining.augment import Augmenter
from pipeline.pretraining.contrastive import (
    ContrastiveSequenceDataset,
    CS2CDContrastiveShardDataset,
    pretrain_contrastive,
)
from pipeline.pretraining.corpora import captcha_to_tensors
from pipeline.pretraining.cs2cd_full import (
    MANIFEST,
    ShardGroupedSampler,
    fit_shard_normalizer,
    shards_for_matches,
)
from pipeline.pretraining.embed_eval import (
    SCORERS,
    embed_chunks,
    linear_probe_auc,
    oneclass_auc,
)
from pipeline.sequences.preprocessing import apply_normalizer, fit_normalizer
from scripts.data_efficiency import (
    VAL_FRAC,
    _eval_chunks_from_raw,
    _gta_pool,
    _strided_chunks,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUT_JSON = ROOT / "reports" / "contrastive_transfer.json"
OUT_FIG = ROOT / "reports" / "figures" / "phase8_2_contrastive_transfer_gta.png"
RECON_ENCODER = (
    MODELS_DIR / "pretrained_cs2cd_s1_382.pt"
)  # Phase 8.1, objective baseline

VOLUMES = [50, 200, 382]
GTA_BUDGETS = [2, 5, 10, 15]
SEEDS = [0, 1, 2]
METRICS = ["mahalanobis", "ocsvm", "knn", "linear_probe"]
N_ONECLASS_FIT = 8000  # cap legit one-class fit set (OCSVM RBF is O(n²))

# Architecture — identical to the 8.1 encoders so the embedding is comparable.
ENC = dict(hidden_dim=64, bottleneck_dim=16, num_layers=2)


# ---------------------------------------------------------------------------
# Capped sampler — bound chunks/epoch (382 matches ≈ 12.5M chunks otherwise)
# ---------------------------------------------------------------------------
class CappedSampler(Sampler[int]):
    """Yield at most ``max_chunks`` of the base sampler's per-epoch stream.

    Wraps :class:`ShardGroupedSampler`, whose shard order reshuffles each epoch, so
    the truncated prefix covers different shards across epochs → broad coverage at
    a bounded wall-time. ``max_chunks=None`` is a pass-through.
    """

    def __init__(self, base: Sampler[int], max_chunks: int | None):
        self.base = base
        self.max_chunks = max_chunks

    def __len__(self) -> int:
        n = len(self.base)
        return min(n, self.max_chunks) if self.max_chunks else n

    def __iter__(self):
        if not self.max_chunks:
            yield from self.base
            return
        for i, idx in enumerate(self.base):
            if i >= self.max_chunks:
                break
            yield idx


# ---------------------------------------------------------------------------
# Pretrain phase
# ---------------------------------------------------------------------------
def _save_encoder(backbone, stats, out_name: str, config: dict, history) -> Path:
    """Persist the backbone under ``models/{out_name}.pt`` (+ ``_meta.json``).

    ``save_lstm_ae`` writes the fixed ``lstm_ae.pt`` name → save to a temp dir then
    move, so we never clobber the canonical DVC-tracked artifact (pretrain_encoder
    uses the same trick).
    """
    with tempfile.TemporaryDirectory() as tmp:
        weights, meta = save_lstm_ae(
            backbone, stats, Path(tmp), config=config, history=history
        )
        out_weights = MODELS_DIR / f"{out_name}.pt"
        out_meta = MODELS_DIR / f"{out_name}_meta.json"
        shutil.move(str(weights), out_weights)
        shutil.move(str(meta), out_meta)
    return out_weights


def _train_and_save(out_name, train_ds, val_ds, sampler, stats, corpus_meta, args):
    pin = args.device != "cpu" and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    log.info("  %s: %d train chunks (capped/epoch)", out_name, len(train_ds))

    backbone, _head, history = pretrain_contrastive(
        train_loader,
        val_loader,
        **ENC,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        set_epoch_fn=train_ds.set_epoch,
    )
    config = dict(
        objective="contrastive_ntxent",
        temperature=args.temperature,
        proj_dim=args.proj_dim,
        chunk_length=args.chunk_length,
        stride=args.stride,
        epochs=args.epochs,
        max_chunks_per_epoch=args.max_chunks_per_epoch,
        seed=args.seed,
        **ENC,
        **corpus_meta,
    )
    path = _save_encoder(backbone, stats, out_name, config, history)
    log.info(
        "  saved %s (best val_loss=%.5f @ %d)",
        path,
        history.best_val_loss,
        history.best_epoch,
    )


def _pretrain_cs2cd(volume: int, args) -> None:
    out_name = f"pretrained_contrastive_cs2cd_{volume}"
    if (MODELS_DIR / f"{out_name}.pt").exists():
        log.info("present, skip: %s", out_name)
        return
    manifest = json.loads(MANIFEST.read_text())
    train_ids = manifest["diversity_subsets"][str(volume)]
    train_shards = shards_for_matches(
        train_ids, subdir="no_cheater_present", source="s1"
    )
    val_shards = shards_for_matches(
        manifest["val_matches"], subdir="no_cheater_present", source="s1"
    )
    if not train_shards:
        raise SystemExit("no train shards — run pipeline.pretraining.cs2cd_full first")
    stats = fit_shard_normalizer(train_shards, seed=args.seed)
    aug = Augmenter()
    train_ds = CS2CDContrastiveShardDataset(
        train_shards,
        stats=stats,
        chunk_length=args.chunk_length,
        stride=args.stride,
        augment=aug,
        seed=args.seed,
    )
    val_ds = CS2CDContrastiveShardDataset(
        val_shards,
        stats=stats,
        chunk_length=args.chunk_length,
        stride=args.chunk_length,
        augment=aug,
        seed=args.seed,
    )
    sampler = CappedSampler(
        ShardGroupedSampler(train_ds, shuffle=True, seed=args.seed),
        args.max_chunks_per_epoch,
    )
    corpus_meta = dict(corpus="cs2cd_full", source="s1", volume=volume)
    log.info(
        "contrastive cs2cd @ %d: %d/%d train/val shards",
        volume,
        len(train_shards),
        len(val_shards),
    )
    _train_and_save(out_name, train_ds, val_ds, sampler, stats, corpus_meta, args)


def _pretrain_captcha(args) -> None:
    out_name = "pretrained_contrastive_captcha"
    if (MODELS_DIR / f"{out_name}.pt").exists():
        log.info("present, skip: %s", out_name)
        return
    tensors = captcha_to_tensors(max_sessions=args.captcha_sessions, seed=args.seed)
    if not tensors:
        raise SystemExit("no captcha sessions loaded")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tensors))
    n_val = max(1, int(round(len(tensors) * VAL_FRAC)))
    val_idx = set(perm[:n_val].tolist())
    train = [tensors[i] for i in range(len(tensors)) if i not in val_idx]
    val = [tensors[i] for i in range(len(tensors)) if i in val_idx]
    stats = fit_normalizer(train)
    aug = Augmenter()
    train_ds = ContrastiveSequenceDataset(
        [apply_normalizer(t, stats) for t in train],
        args.chunk_length,
        args.stride,
        augment=aug,
        seed=args.seed,
    )
    val_ds = ContrastiveSequenceDataset(
        [apply_normalizer(t, stats) for t in val],
        args.chunk_length,
        args.chunk_length,
        augment=aug,
        seed=args.seed,
    )
    corpus_meta = dict(corpus="captcha30k", n_units=len(tensors))
    log.info("contrastive captcha: %d sessions", len(tensors))
    _train_and_save(out_name, train_ds, val_ds, None, stats, corpus_meta, args)


def _pretrain_grid(args) -> None:
    for volume in args.volumes:
        _pretrain_cs2cd(volume, args)
    if not args.no_captcha:
        _pretrain_captcha(args)


# ---------------------------------------------------------------------------
# Eval phase — frozen-embedding matrix on the GTA pool
# ---------------------------------------------------------------------------
def _load_backbone(path: Path) -> LSTMAutoencoder:
    model = LSTMAutoencoder(**ENC)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def _random_backbone(seed: int) -> LSTMAutoencoder:
    torch.manual_seed(seed)
    return LSTMAutoencoder(**ENC)


def _eval_sources(args) -> list[tuple[str, Path | None]]:
    """(label, encoder path | None for random) — skipping missing artifacts."""
    srcs: list[tuple[str, Path | None]] = [("random", None)]
    if RECON_ENCODER.exists():
        srcs.append(("recon_cs2cd_382", RECON_ENCODER))
    for v in args.volumes:
        srcs.append(
            (
                f"contrastive_cs2cd_{v}",
                MODELS_DIR / f"pretrained_contrastive_cs2cd_{v}.pt",
            )
        )
    if not args.no_captcha:
        srcs.append(
            ("contrastive_captcha", MODELS_DIR / "pretrained_contrastive_captcha.pt")
        )
    return [(lab, p) for lab, p in srcs if p is None or p.exists()]


def _metrics_for_model(model, train_legit_emb, eval_legit_emb, eval_cheat_emb, seed):
    """All four AUCs for one frozen encoder's embeddings."""
    out = {}
    for name in ("mahalanobis", "ocsvm", "knn"):
        out[name] = oneclass_auc(
            train_legit_emb, eval_legit_emb, eval_cheat_emb, SCORERS[name]
        )
    emb = np.r_[eval_legit_emb, eval_cheat_emb]
    y = np.r_[np.zeros(len(eval_legit_emb)), np.ones(len(eval_cheat_emb))]
    out["linear_probe"] = linear_probe_auc(emb, y, seed=seed)
    return out


def _eval_grid(args) -> list[dict]:
    units, eval_legit, eval_cheat = _gta_pool()
    log.info(
        "GTA pool: %d legit units | eval legit/cheat %d/%d",
        len(units),
        len(eval_legit),
        len(eval_cheat),
    )
    sources = _eval_sources(args)
    log.info("sources: %s", [s[0] for s in sources])
    budgets = [b for b in GTA_BUDGETS if b <= len(units)] or [len(units)]

    runs: list[dict] = []
    for budget in budgets:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(units), size=min(budget, len(units)), replace=False)
            train_units = [units[i] for i in pick]
            stats = fit_normalizer(train_units)
            train_chunks = _strided_chunks(train_units, stats)
            # Cap the one-class fit set: OCSVM (RBF) is O(n²) and a one-class fit
            # needs nowhere near tens-of-thousands of points. Seeded subsample →
            # bounded cost + identical fit set for every source (comparable).
            if len(train_chunks) > N_ONECLASS_FIT:
                sub = rng.choice(len(train_chunks), size=N_ONECLASS_FIT, replace=False)
                train_chunks = train_chunks[sub]
            legit_eval, cheat_eval = _eval_chunks_from_raw(
                eval_legit, eval_cheat, stats
            )
            if len(train_chunks) < 4 or len(legit_eval) == 0 or len(cheat_eval) == 0:
                continue
            for label, path in sources:
                model = _random_backbone(seed) if path is None else _load_backbone(path)
                tr = embed_chunks(model, train_chunks, device=args.device)
                el = embed_chunks(model, legit_eval, device=args.device)
                ec = embed_chunks(model, cheat_eval, device=args.device)
                m = _metrics_for_model(model, tr, el, ec, seed)
                runs.append(dict(source=label, budget=int(budget), seed=int(seed), **m))
                log.info(
                    "  %-22s budget=%2d seed=%d → maha %.3f ocsvm %.3f knn %.3f probe %.3f",
                    label,
                    budget,
                    seed,
                    m["mahalanobis"],
                    m["ocsvm"],
                    m["knn"],
                    m["linear_probe"],
                )
        _save_json(runs)  # checkpoint after each budget
    return runs


def _summarize(runs: list[dict]) -> dict:
    """Mean±std per (source, metric) over all budget×seed runs."""
    out: dict = {}
    for source in sorted({r["source"] for r in runs}):
        out[source] = {}
        for metric in METRICS:
            vals = [
                r[metric]
                for r in runs
                if r["source"] == source and not np.isnan(r[metric])
            ]
            if vals:
                out[source][metric] = dict(
                    mean=float(np.mean(vals)), std=float(np.std(vals)), n=len(vals)
                )
    return out


def _save_json(runs: list[dict]) -> dict:
    summary = _summarize(runs)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            dict(
                target="gta",
                eval="frozen_embedding_oneclass+linearprobe",
                objective="contrastive_ntxent",
                note="contrastive-native eval; compare each source to random-init "
                "(did it learn?) and to recon_cs2cd_382 (objective head-to-head). "
                "Legit eval baseline includes the one-class fit's own sessions → "
                "mildly optimistic but identical for every source.",
                volumes=VOLUMES,
                metrics=METRICS,
                n_runs=len(runs),
                runs=runs,
                summary=summary,
            ),
            indent=2,
        )
        + "\n"
    )
    return summary


def _render_figure(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = (
        ["random", "recon_cs2cd_382"]
        + [f"contrastive_cs2cd_{v}" for v in VOLUMES]
        + ["contrastive_captcha"]
    )
    sources = [s for s in order if s in summary]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: grouped bars — every metric per source (mean ± std over budget×seed).
    x = np.arange(len(sources))
    w = 0.8 / len(METRICS)
    cmap = plt.get_cmap("tab10")
    for i, metric in enumerate(METRICS):
        means = [summary[s].get(metric, {}).get("mean", np.nan) for s in sources]
        stds = [summary[s].get(metric, {}).get("std", 0.0) for s in sources]
        ax1.bar(
            x + (i - (len(METRICS) - 1) / 2) * w,
            means,
            w,
            yerr=stds,
            capsize=2,
            color=cmap(i),
            label=metric,
        )
    ax1.axhline(0.5, color="#8892a4", ls=":", lw=1.2, label="chance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sources, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("GTA cheat-detection ROC AUC (frozen embedding)")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_title("Frozen-embedding eval per source (mean ± std, budget×seed)")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel 2: Mahalanobis AUC vs contrastive pretraining volume (+ baselines).
    vols, vmeans, vstds = [], [], []
    for v in VOLUMES:
        s = f"contrastive_cs2cd_{v}"
        if s in summary and "mahalanobis" in summary[s]:
            vols.append(v)
            vmeans.append(summary[s]["mahalanobis"]["mean"])
            vstds.append(summary[s]["mahalanobis"]["std"])
    if vols:
        ax2.errorbar(
            vols,
            vmeans,
            yerr=vstds,
            fmt="-o",
            capsize=3,
            color=cmap(0),
            label="contrastive cs2cd (Mahalanobis)",
        )
    for base, color in (("random", "#8892a4"), ("recon_cs2cd_382", "#e94560")):
        if base in summary and "mahalanobis" in summary[base]:
            ax2.axhline(
                summary[base]["mahalanobis"]["mean"],
                ls="--",
                color=color,
                lw=1.3,
                label=f"{base} (Mahalanobis)",
            )
    ax2.set_xlabel("# CS2CD pretraining matches (stream volume)")
    ax2.set_ylabel("ROC AUC")
    ax2.set_ylim(0.4, 1.0)
    ax2.set_title("Contrastive in-domain volume axis")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Phase 8.2 — contrastive (NT-Xent) pretraining, frozen-embedding transfer to GTA cheat detection"
    )
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 8.2 contrastive transfer")
    parser.add_argument("--phase", choices=["pretrain", "eval", "all"], default="all")
    parser.add_argument("--volumes", type=int, nargs="+", default=VOLUMES)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proj-dim", type=int, default=32)
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max-chunks-per-epoch", type=int, default=250_000)
    parser.add_argument("--captcha-sessions", type=int, default=6000)
    parser.add_argument("--no-captcha", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.phase in ("pretrain", "all"):
        log.info(
            "=== PRETRAIN: contrastive cs2cd %s + captcha=%s ===",
            args.volumes,
            not args.no_captcha,
        )
        _pretrain_grid(args)

    if args.phase in ("eval", "all"):
        log.info("=== EVAL: frozen-embedding matrix on GTA ===")
        runs = _eval_grid(args)
        if not runs:
            log.error("no eval runs produced")
            return 1
        summary = _save_json(runs)
        _render_figure(summary)
        log.info("Wrote %s + %s", OUT_JSON, OUT_FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
