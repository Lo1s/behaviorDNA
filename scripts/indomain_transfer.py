"""
scripts/indomain_transfer.py
============================
Phase 8.1 headline — does **in-domain** (CS2CD) pretraining rescue the Phase 8
transfer null, and is the ``dt`` mismatch the culprit?

Grid over **arm × source × pretraining-volume × fine-tune-budget × seed**, scored
by GTA chunk-level cheat-detection ROC AUC (the real transfer test). Arms:

  * **A — scratch**       : random init, fine-tuned on the GTA budget.
  * **B — frozen**        : load a pretrained encoder, FREEZE ``encoder`` +
    ``to_bottleneck`` (``requires_grad=False``), train only the decoder. Tests
    whether the pretrained embedding is good enough that a thin readout suffices.
  * **C — fine-tune**     : load a pretrained encoder, fine-tune all weights.

Sources (the pretrained init): **captcha** (Phase 8, out-of-domain),
**cs2cd_s1** (in-domain, native CS2 tick ``dt``), **cs2cd_s2** (in-domain,
``dt``-neutralised). s1-vs-s2 isolates the temporal (``dt``) gap the Phase 8 null
blamed; captcha-vs-cs2cd_s1 isolates the in-domain effect. Pretraining volume
(``50/200/382`` legit matches) is the Branch-THIN stream-volume axis (NOT player
diversity — the release is per-match-anonymised, see ``cs2cd_diversity_probe``).

**GTA target** = Phase 8's chunk-level cheat-detection pool verbatim (train on all
legit, eval cheat-vs-legit chunks) — informative and directly comparable to the
Phase 8 null. A player-disjoint variant (hold out hydra) was tried but floored
every arm at chance (cross-player shift swamps the signal at N=2 training
players); kept as a caveat. (3 players / 18 sessions — the standing small-N caveat.)
The CS2CD cheat-detection sanity arm is intentionally **omitted**: the full
release carries no recoverable per-player cheat label (no ``cheater_present``
column; the match-level "not-cheater" label is ~55.6% precise per the dataset
card), and the balanced sample would leak into pretraining.

Reuses the Phase 8 machinery verbatim: ``_strided_chunks``/``_eval_chunks_from_raw``
(data_efficiency), ``train_ae`` + the ``trainable_params`` frozen hook
(compare_architectures), ``score_sequences``, and ``pretrain_encoder.main`` for
the pretraining phase.

Usage (CUDA desktop):
    python -m scripts.indomain_transfer --phase all          # pretrain + transfer
    python -m scripts.indomain_transfer --phase transfer     # transfer only (encoders cached)
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

from pipeline.models.lstm_ae import LSTMAutoencoder, score_sequences
from pipeline.sequences.preprocessing import COL_DT, fit_normalizer
from scripts.data_efficiency import (
    VAL_FRAC,
    _eval_chunks_from_raw,
    _strided_chunks,
)
from scripts.data_efficiency import _gta_pool as _p8_gta_pool

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
OUT_JSON = ROOT / "reports" / "phase8_1_indomain_transfer.json"
OUT_FIG = ROOT / "reports" / "figures" / "phase8_1_indomain_transfer_gta.png"
CAPTCHA_ENCODER = MODELS_DIR / "pretrained_encoder.pt"

VOLUMES = [50, 200, 382]  # CS2CD pretraining stream-volume axis (manifest subset keys)
SOURCES = ["s1", "s2"]  # native dt vs dt-neutralised
GTA_BUDGETS = [2, 5, 10, 15]  # fine-tune legit sessions (matches Phase 8)
SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# GTA target pool (NON-disjoint — identical to Phase 8, for comparability)
# ---------------------------------------------------------------------------
def _zero_dt(t: np.ndarray) -> np.ndarray:
    """dt-neutralise (s2): zero the dt channel so it can't transfer-mismatch."""
    out = t.copy()
    out[:, COL_DT] = 0.0
    return out


def _gta_pool(*, neutralize_dt: bool = False):
    """GTA cheat-detection pool ``(legit train units, eval legit, eval cheat)``.

    Reuses Phase 8's ``data_efficiency._gta_pool`` verbatim (train on all legit,
    eval cheat-vs-legit chunks) so the in-domain-vs-captcha-vs-scratch comparison
    is informative *and* directly comparable to the Phase 8 null. For ``s2`` the
    ``dt`` channel is zeroed in every tensor.

    NB: a *player-disjoint* variant (hold out hydra) was tried and floored every
    arm at chance — the cross-player shift swamps the cheat signal at N=2 training
    players, so it can't test the pretraining question. Reported as a caveat, not
    used as the target. (3 players / 18 sessions — the standing small-N caveat.)
    """
    units, eval_legit, eval_cheat = _p8_gta_pool()
    if neutralize_dt:
        units = [_zero_dt(t) for t in units]
        eval_legit = [_zero_dt(t) for t in eval_legit]
        eval_cheat = [_zero_dt(t) for t in eval_cheat]
    return units, eval_legit, eval_cheat


# ---------------------------------------------------------------------------
# Arms + one fine-tune/eval run
# ---------------------------------------------------------------------------
def _apply_arm(model: LSTMAutoencoder, init: str, pretrained_path: Path | None):
    """Set up a model for arm ``init``; return the param list to optimise.

    scratch → all params; finetune → load pretrained, all params; frozen → load
    pretrained, freeze encoder + to_bottleneck, optimise the decoder only.
    """
    if init in ("frozen", "finetune"):
        sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
    if init == "frozen":
        for module in (model.encoder, model.to_bottleneck):
            for prm in module.parameters():
                prm.requires_grad_(False)
        return [prm for prm in model.parameters() if prm.requires_grad]
    return None  # all params trainable


def _run_one(
    units, eval_legit, eval_cheat, *, init, pretrained_path, budget, seed, epochs, lr
):
    """One fine-tune+eval run → result dict (or None if too little data)."""
    from scripts.benchmark_cs2cd_ae import _ChunkDataset
    from scripts.compare_architectures import _device, train_ae

    if len(units) == 0:
        return None
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
    pin = _device() == "cuda"
    tl = DataLoader(_ChunkDataset(tr), batch_size=256, shuffle=True, pin_memory=pin)
    vl = DataLoader(_ChunkDataset(va), batch_size=256, shuffle=False, pin_memory=pin)

    torch.manual_seed(seed)
    model = LSTMAutoencoder(hidden_dim=64, bottleneck_dim=16, num_layers=2)
    trainable = _apply_arm(model, init, pretrained_path)
    train_ae(model, tl, vl, epochs, lr, trainable_params=trainable)

    legit_eval, cheat_eval = _eval_chunks_from_raw(eval_legit, eval_cheat, stats)
    if len(legit_eval) == 0 or len(cheat_eval) == 0:
        return None
    ls = score_sequences(model, legit_eval, device=_device())
    cs = score_sequences(model, cheat_eval, device=_device())
    y = np.r_[np.zeros(len(ls)), np.ones(len(cs))]
    return {
        "auc": float(roc_auc_score(y, np.r_[ls, cs])),
        "budget": int(budget),
        "seed": int(seed),
        "n_train_chunks": int(len(tr)),
    }


# ---------------------------------------------------------------------------
# Pretraining phase (the long pole) — produces the source × volume encoders
# ---------------------------------------------------------------------------
def _ensure_pretrained(
    source: str, volume: int, *, epochs, num_workers, seed, device
) -> Path:
    name = f"pretrained_cs2cd_{source}_{volume}"
    path = MODELS_DIR / f"{name}.pt"
    if path.exists():
        log.info("pretrained encoder present: %s", path)
        return path
    from scripts.pretrain_encoder import main as pretrain_main

    rc = pretrain_main(
        [
            "--corpus",
            "cs2cd_full",
            "--source",
            source,
            "--max-matches",
            str(volume),
            "--out-name",
            name,
            "--epochs",
            str(epochs),
            "--num-workers",
            str(num_workers),
            "--device",
            device,
            "--seed",
            str(seed),
        ]
    )
    if rc != 0 or not path.exists():
        raise RuntimeError(f"pretraining failed for {name} (rc={rc})")
    return path


def _pretrain_grid(args) -> None:
    for source in SOURCES:
        for volume in VOLUMES:
            _ensure_pretrained(
                source,
                volume,
                epochs=args.pretrain_epochs,
                num_workers=args.num_workers,
                seed=args.seed,
                device=args.device,
            )


# ---------------------------------------------------------------------------
# Transfer phase — the arm × source × volume × budget × seed grid
# ---------------------------------------------------------------------------
def _configs():
    """Enumerate (label, init, source, path, neutralize_dt, volume) configs."""
    cfgs = []
    # scratch — one per input encoding (native vs dt-neutralised)
    cfgs.append(("scratch·native", "scratch", "native", None, False, None))
    cfgs.append(("scratch·dt0", "scratch", "dt0", None, True, None))
    # captcha (out-of-domain, native input) — only if the Phase-8 artifact is present
    if CAPTCHA_ENCODER.exists():
        for init in ("frozen", "finetune"):
            cfgs.append(
                (f"captcha·{init}", init, "captcha", CAPTCHA_ENCODER, False, None)
            )
    # in-domain cs2cd, per source × volume
    for volume in VOLUMES:
        for source in SOURCES:
            neutralize = source == "s2"
            path = MODELS_DIR / f"pretrained_cs2cd_{source}_{volume}.pt"
            for init in ("frozen", "finetune"):
                cfgs.append(
                    (
                        f"cs2cd_{source}·{init}@{volume}",
                        init,
                        f"cs2cd_{source}",
                        path,
                        neutralize,
                        volume,
                    )
                )
    return cfgs


def _transfer_grid(args) -> list[dict]:
    # Precompute both input-encoding pools once.
    pools = {
        False: _gta_pool(neutralize_dt=False),
        True: _gta_pool(neutralize_dt=True),
    }
    for neut, (units, el, ec) in pools.items():
        log.info(
            "GTA pool (neutralize_dt=%s): %d train units | eval legit/cheat %d/%d",
            neut,
            len(units),
            len(el),
            len(ec),
        )
    budgets = [b for b in GTA_BUDGETS if b <= len(pools[False][0])] or [
        len(pools[False][0])
    ]

    runs = []
    for label, init, source, path, neutralize, volume in _configs():
        if path is not None and not Path(path).exists():
            log.warning("skip %s — missing encoder %s", label, path)
            continue
        units, el, ec = pools[neutralize]
        for budget in budgets:
            for seed in SEEDS:
                r = _run_one(
                    units,
                    el,
                    ec,
                    init=init,
                    pretrained_path=path,
                    budget=budget,
                    seed=seed,
                    epochs=args.ft_epochs,
                    lr=args.lr,
                )
                if r is None:
                    continue
                r.update(label=label, init=init, source=source, volume=volume)
                runs.append(r)
                log.info(
                    "  %-26s budget=%2d seed=%d → AUC %.3f",
                    label,
                    budget,
                    seed,
                    r["auc"],
                )
        if runs:  # checkpoint after each config so a kill never discards completed runs
            _save_json(runs)
    return runs


def _summarize(runs):
    """Mean ± std AUC per (label, budget)."""
    out = {}
    labels = sorted({r["label"] for r in runs})
    for label in labels:
        pts = []
        for b in sorted({r["budget"] for r in runs if r["label"] == label}):
            aucs = [r["auc"] for r in runs if r["label"] == label and r["budget"] == b]
            pts.append(
                {
                    "budget": b,
                    "mean": float(np.mean(aucs)),
                    "std": float(np.std(aucs)),
                    "n": len(aucs),
                }
            )
        out[label] = pts
    return out


def _save_json(runs: list[dict]) -> dict:
    """Write the results JSON (incl. summary) and return the summary.

    Called after every config so a kill mid-grid never discards completed runs —
    the partial JSON is always valid and re-readable.
    """
    summary = _summarize(runs)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "target": "gta",
                "player_disjoint": False,
                "target_note": "Phase 8 non-disjoint GTA pool (comparable to the Phase 8 null); a player-disjoint variant floored every arm at chance",
                "branch": "THIN",
                "volumes": VOLUMES,
                "sources": SOURCES,
                "n_runs": len(runs),
                "runs": runs,
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    return summary


def _render_figure(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Panel 1: AUC vs budget at the top volume (382) — the in-domain-vs-captcha-vs-scratch story.
    top = max(VOLUMES)
    keep = [
        "scratch·native",
        "scratch·dt0",
        "captcha·frozen",
        "captcha·finetune",
        f"cs2cd_s1·frozen@{top}",
        f"cs2cd_s1·finetune@{top}",
        f"cs2cd_s2·frozen@{top}",
        f"cs2cd_s2·finetune@{top}",
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, label in enumerate([k for k in keep if k in summary]):
        pts = summary[label]
        xs = [p["budget"] for p in pts]
        ms = np.array([p["mean"] for p in pts])
        ss = np.array([p["std"] for p in pts])
        ls = "--" if "frozen" in label else "-"
        ax1.plot(xs, ms, ls, marker="o", color=cmap(i % 10), label=label)
        ax1.fill_between(xs, ms - ss, ms + ss, color=cmap(i % 10), alpha=0.12)
    ax1.axhline(0.5, color="#8892a4", ls=":", lw=1.2, label="chance")
    ax1.set_xlabel("# GTA fine-tune sessions (Phase-8 target)")
    ax1.set_ylabel("chunk-level cheat-detection ROC AUC")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_title(f"Transfer @ pretraining volume {top} (mean ± std, 3 seeds)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: AUC vs pretraining volume at a fixed mid budget — the THIN volume axis.
    mid_budget = sorted({p["budget"] for pts in summary.values() for p in pts})[
        len(GTA_BUDGETS) // 2
    ]
    for i, (src, init) in enumerate(
        [("s1", "finetune"), ("s2", "finetune"), ("s1", "frozen"), ("s2", "frozen")]
    ):
        xs, ms = [], []
        for vol in VOLUMES:
            label = f"cs2cd_{src}·{init}@{vol}"
            pt = next(
                (p for p in summary.get(label, []) if p["budget"] == mid_budget), None
            )
            if pt:
                xs.append(vol)
                ms.append(pt["mean"])
        if xs:
            ax2.plot(
                xs,
                ms,
                "--o" if init == "frozen" else "-o",
                color=cmap(i % 10),
                label=f"cs2cd_{src}·{init}",
            )
    ax2.axhline(0.5, color="#8892a4", ls=":", lw=1.2, label="chance")
    ax2.set_xlabel("# CS2CD pretraining matches (stream volume, not players)")
    ax2.set_ylabel("ROC AUC")
    ax2.set_ylim(0.4, 1.0)
    ax2.set_title(f"In-domain volume axis @ budget {mid_budget}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Phase 8.1 — in-domain (CS2CD) vs out-of-domain (captcha) pretraining transfer to GTA cheat detection"
    )
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 8.1 in-domain transfer grid")
    parser.add_argument(
        "--phase", choices=["pretrain", "transfer", "all"], default="all"
    )
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--ft-epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.phase in ("pretrain", "all"):
        log.info(
            "=== PRETRAIN PHASE: %d encoders (sources %s × volumes %s) ===",
            len(SOURCES) * len(VOLUMES),
            SOURCES,
            VOLUMES,
        )
        _pretrain_grid(args)

    if args.phase in ("transfer", "all"):
        log.info("=== TRANSFER PHASE: GTA player-disjoint ===")
        runs = _transfer_grid(args)
        if not runs:
            log.error("no transfer runs produced")
            return 1
        summary = _save_json(runs)
        _render_figure(summary)
        log.info("Wrote %s + %s", OUT_JSON, OUT_FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
