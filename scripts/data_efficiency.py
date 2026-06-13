"""
scripts/data_efficiency.py
==========================
Phase 8 headline — **data-efficiency curve**: does captcha-pretraining buy
chunk-level cheat-detection performance at *low* fine-tuning budgets?

For a budget axis (number of legit fine-tuning units) × {pretrained-init,
scratch-init} × seeds, fine-tune the autoencoder on that legit subset and score
a **fixed** legit-vs-cheat chunk-AUC eval set. The headline figure plots
AUC(budget) for both inits with bootstrap-style mean±std bands. Pretrained
winning at low budget = the foundation-model line; no gap = the domain gap
dominates (an honest result either way — see ``docs/PRETRAINING.md``).

Reuses: the 8-D corpus adapters (``pipeline.pretraining.corpora``), the shared
reconstruction-AE loop (``scripts.compare_architectures.train_ae``),
``score_sequences``, and the GTA chunk-labelling helpers
(``pipeline.adversarial.benchmark``).

Usage (CUDA desktop):
    python -m scripts.data_efficiency --domain cs2cd
    python -m scripts.data_efficiency --domain gta
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pipeline.adversarial.benchmark import _chunk_cheat_labels, _typed_segments
from pipeline.models.lstm_ae import LSTMAutoencoder, score_sequences
from pipeline.pretraining.corpora import cs2cd_to_tensors_8d
from pipeline.sequences.preprocessing import (
    apply_normalizer,
    fit_normalizer,
    session_to_event_tensor,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PRETRAINED = ROOT / "models" / "pretrained_encoder.pt"
CHUNK, STRIDE, VAL_FRAC = 64, 32, 0.15
BUDGETS = {"cs2cd": [1, 2, 5, 10], "gta": [2, 5, 10, 15]}
SEEDS = [0, 1, 2, 3, 4]


def _nonoverlap_chunks(tensors: list[np.ndarray], stats: dict) -> np.ndarray:
    """Normalise + slice each tensor into non-overlapping (CHUNK, 8) windows."""
    out = []
    for t in tensors:
        norm = apply_normalizer(t, stats)
        n = len(norm) // CHUNK
        for i in range(n):
            out.append(norm[i * CHUNK : (i + 1) * CHUNK])
    if not out:
        return np.empty((0, CHUNK, 8), np.float32)
    return np.stack(out).astype(np.float32)


def _strided_chunks(tensors: list[np.ndarray], stats: dict) -> np.ndarray:
    """Overlapping (stride) chunks for training (more samples from little data)."""
    out = []
    for t in tensors:
        norm = apply_normalizer(t, stats)
        n = (len(norm) - CHUNK) // STRIDE + 1
        for i in range(max(n, 0)):
            out.append(norm[i * STRIDE : i * STRIDE + CHUNK])
    if not out:
        return np.empty((0, CHUNK, 8), np.float32)
    return np.stack(out).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-domain pools: (legit training units, fixed eval-legit, fixed eval-cheat)
# ---------------------------------------------------------------------------


def _cs2cd_pool():
    streams = cs2cd_to_tensors_8d()
    legit = [t for lab, t in streams if lab == 0]
    cheat = [t for lab, t in streams if lab == 1]
    # Training units = the substantial legit streams (one ~per player); tiny
    # 1-chunk runs stay in the eval-legit pool but make poor training units.
    units = sorted([t for t in legit if len(t) >= 4 * CHUNK], key=len, reverse=True)
    return units, legit, cheat


def _gta_pool():
    import json as _json

    legit_units: list[np.ndarray] = []
    eval_legit: list[np.ndarray] = []
    eval_cheat: list[np.ndarray] = []
    paths = sorted(RAW.glob("*.json"))
    cheat_dir = RAW / "cheat"
    if cheat_dir.is_dir():
        paths += sorted(cheat_dir.glob("*.json"))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = _json.load(f)
        t = session_to_event_tensor(d)
        if len(t) < CHUNK:
            continue
        if _typed_segments(d):
            # cheat session: split its non-overlapping chunks by label
            n = len(t) // CHUNK
            labels = _chunk_cheat_labels(d, CHUNK, n)
            for i in range(n):
                (eval_cheat if labels[i] else eval_legit).append(
                    t[i * CHUNK : (i + 1) * CHUNK]
                )
        else:
            legit_units.append(t)
            eval_legit.append(t)
    return legit_units, eval_legit, eval_cheat


def _eval_chunks_from_raw(eval_legit, eval_cheat, stats):
    """Normalise + slice both eval pools into (M, CHUNK, 8) stacks.

    Inputs are lists of variable-length ``(N>=CHUNK, 8)`` arrays — full streams
    (CS2CD), whole legit sessions, or single pre-cut cheat chunks (GTA, where a
    ``(CHUNK, 8)`` array yields exactly one window). ``_nonoverlap_chunks``
    handles all cases uniformly with the run's normaliser ``stats``.
    """
    return (
        _nonoverlap_chunks(eval_legit, stats),
        _nonoverlap_chunks(eval_cheat, stats),
    )


# ---------------------------------------------------------------------------
# One fine-tune + eval run
# ---------------------------------------------------------------------------


def _run_one(units, eval_legit, eval_cheat, budget, pretrained, seed, epochs, lr):
    from scripts.compare_architectures import _device, train_ae

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(units), size=min(budget, len(units)), replace=False)
    train_units = [units[i] for i in pick]

    stats = fit_normalizer(train_units)
    chunks = _strided_chunks(train_units, stats)
    if len(chunks) < 4:
        return None
    n_val = max(1, int(round(len(chunks) * VAL_FRAC)))
    perm = rng.permutation(len(chunks))
    va, tr = chunks[perm[:n_val]], chunks[perm[n_val:]]

    from scripts.benchmark_cs2cd_ae import _ChunkDataset

    pin = _device() == "cuda"
    tl = DataLoader(_ChunkDataset(tr), batch_size=256, shuffle=True, pin_memory=pin)
    vl = DataLoader(_ChunkDataset(va), batch_size=256, shuffle=False, pin_memory=pin)

    torch.manual_seed(seed)
    model = LSTMAutoencoder(hidden_dim=64, bottleneck_dim=16, num_layers=2)
    if pretrained:
        sd = torch.load(PRETRAINED, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
    train_ae(model, tl, vl, epochs, lr)

    legit_eval, cheat_eval = _eval_chunks_from_raw(eval_legit, eval_cheat, stats)
    if len(legit_eval) == 0 or len(cheat_eval) == 0:
        return None
    ls = score_sequences(model, legit_eval, device=_device())
    cs = score_sequences(model, cheat_eval, device=_device())
    y = np.r_[np.zeros(len(ls)), np.ones(len(cs))]
    return {
        "budget": int(budget),
        "init": "pretrained" if pretrained else "scratch",
        "seed": int(seed),
        "auc": float(roc_auc_score(y, np.r_[ls, cs])),
        "n_train_chunks": int(len(tr)),
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 8 data-efficiency curve")
    parser.add_argument("--domain", choices=["cs2cd", "gta"], required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    if not PRETRAINED.exists():
        log.error(
            "Pretrained encoder not found: %s — run scripts.pretrain_encoder first",
            PRETRAINED,
        )
        return 1

    units, eval_legit, eval_cheat = (
        _cs2cd_pool if args.domain == "cs2cd" else _gta_pool
    )()
    log.info(
        "%s pool: %d legit units | eval legit/cheat streams-or-chunks: %d / %d",
        args.domain,
        len(units),
        len(eval_legit),
        len(eval_cheat),
    )

    seeds = SEEDS[: args.seeds]
    budgets = [b for b in BUDGETS[args.domain] if b <= len(units)] or [len(units)]
    runs = []
    for budget in budgets:
        for pretrained in (True, False):
            for seed in seeds:
                r = _run_one(
                    units,
                    eval_legit,
                    eval_cheat,
                    budget,
                    pretrained,
                    seed,
                    args.epochs,
                    args.lr,
                )
                if r is not None:
                    runs.append(r)
                    log.info(
                        "  budget=%2d %-10s seed=%d → AUC %.3f",
                        budget,
                        r["init"],
                        seed,
                        r["auc"],
                    )

    summary = _summarize(runs, budgets)
    out_json = ROOT / "reports" / f"data_efficiency_{args.domain}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {"domain": args.domain, "runs": runs, "summary": summary}, f, indent=2
        )
    _render_figure(args.domain, summary, budgets)
    log.info("Wrote %s", out_json)
    return 0


def _summarize(runs, budgets):
    out = {"pretrained": [], "scratch": []}
    for init in ("pretrained", "scratch"):
        for b in budgets:
            aucs = [r["auc"] for r in runs if r["init"] == init and r["budget"] == b]
            if aucs:
                out[init].append(
                    {
                        "budget": b,
                        "mean": float(np.mean(aucs)),
                        "std": float(np.std(aucs)),
                        "n": len(aucs),
                    }
                )
    return out


def _render_figure(domain, summary, budgets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"pretrained": "#4c78a8", "scratch": "#e94560"}
    for init in ("pretrained", "scratch"):
        pts = summary[init]
        if not pts:
            continue
        xs = [p["budget"] for p in pts]
        ms = np.array([p["mean"] for p in pts])
        ss = np.array([p["std"] for p in pts])
        ax.plot(xs, ms, "-o", color=colors[init], label=f"{init}-init")
        ax.fill_between(xs, ms - ss, ms + ss, color=colors[init], alpha=0.18)
    ax.axhline(0.5, color="#8892a4", ls=":", lw=1.2, label="chance")
    unit = "legit streams (≈players)" if domain == "cs2cd" else "legit sessions"
    ax.set_xlabel(f"# fine-tuning {unit}")
    ax.set_ylabel("chunk-level cheat-detection ROC AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(
        f"Phase 8 data-efficiency — {domain.upper()} "
        f"({'real cheats, 10 players' if domain == 'cs2cd' else 'synthetic cheats, N=18'})\n"
        "captcha-pretrained vs from-scratch (mean ± std over seeds)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_fig = ROOT / "reports" / "figures" / f"phase8_data_efficiency_{domain}.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run())
