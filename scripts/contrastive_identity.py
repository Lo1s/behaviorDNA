"""
scripts/contrastive_identity.py
===============================
Phase 6.1 — does the Phase-8.2 **contrastive** lever transfer from cheat detection
to **player identity**? Pretrain the same LSTM-AE backbone contrastively on the
Balabit mouse-dynamics corpus, freeze it, embed sessions through the 16-D
bottleneck, and report the challenge's **session-level verification EER** ("is this
test session really the claimed user?") — directly comparable to Phase 6's
hand-crafted-feature baseline (session-EER ≈ 0.144, 10 users).

The scientific hook: 8.2's augmentations induce **scale-invariance** (good for
cheat detection), but a user's characteristic speed/scale *is* identity signal — so
a scale-invariant embedding may discard what identity needs. The ``noscale`` arm
(``Augmenter(scale_prob=0)``) tests that directly.

Encoders compared (frozen): ``random`` (3 seeds → band), ``contrastive_balabit``,
``contrastive_balabit_noscale``, and the 8.2 ``contrastive_cs2cd_382`` (a free
cross-domain transfer point). Two scoring routes per encoder:
  * **cosine** — enroll = mean training-session embedding; score = cosine to the
    claimed user (contrastive-native, no classifier);
  * **classifier** — LightGBM on per-chunk embeddings, per-test-session mean
    P(claimed) (apples-to-apples with Phase 6's protocol, hand-features→embedding).

Reuses: ``pipeline.external.{balabit,sequences,base}``, ``pipeline.verification.eer``,
8.2's ``augment``/``contrastive``/``embed_eval``, and ``run_external_identification``
for the hand-feature baseline. Offline; reads the public corpus only.

Usage (CUDA desktop):
    python -m scripts.contrastive_identity --phase pretrain   # 2 encoders
    python -m scripts.contrastive_identity --phase eval       # EER matrix + baseline
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

from pipeline.external.balabit import BalabitAdapter
from pipeline.external.sequences import session_to_chunks, session_to_segment_tensors
from pipeline.models.lstm_ae import LSTMAutoencoder, save_lstm_ae
from pipeline.pretraining.augment import Augmenter
from pipeline.pretraining.contrastive import (
    ContrastiveSequenceDataset,
    pretrain_contrastive,
)
from pipeline.pretraining.embed_eval import embed_chunks
from pipeline.sequences.preprocessing import apply_normalizer, fit_normalizer
from pipeline.verification import eer

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
BALABIT_DIR = ROOT / "data" / "external" / "balabit"
MODELS_DIR = ROOT / "models"
OUT_JSON = ROOT / "reports" / "contrastive_identity.json"
OUT_FIG = ROOT / "reports" / "figures" / "phase6_1_contrastive_identity.png"
CS2CD_ENCODER = MODELS_DIR / "pretrained_contrastive_cs2cd_382.pt"  # 8.2, cross-domain

ENC = dict(hidden_dim=64, bottleneck_dim=16, num_layers=2)
CHUNK, STRIDE = 64, 32  # pretrain stride 32 (more views); eval uses non-overlap
EVAL_STRIDE = 64
SEEDS = [0, 1, 2]  # random-encoder band
VAL_FRAC = 0.15


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _train_sessions() -> list[dict]:
    return list(BalabitAdapter(BALABIT_DIR).iter_sessions())


def _test_sessions() -> list[tuple[dict, str, bool]]:
    return list(BalabitAdapter(BALABIT_DIR).iter_test_sessions())


def _segment_tensors(sessions: list[dict]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for s in sessions:
        out.extend(session_to_segment_tensors(s))
    return out


def _corpus_stats(train_sessions: list[dict]) -> dict:
    """Deterministic corpus normaliser (fit on all training bursts) — shared by
    pretraining and every eval encoder so the input scaling is identical."""
    return fit_normalizer(_segment_tensors(train_sessions))


# ---------------------------------------------------------------------------
# Pretrain
# ---------------------------------------------------------------------------
def _pretrain(noscale: bool, args) -> None:
    name = "pretrained_contrastive_balabit" + ("_noscale" if noscale else "")
    out_path = MODELS_DIR / f"{name}.pt"
    if out_path.exists() and not args.overwrite:
        log.info("present, skip: %s", name)
        return
    sessions = _train_sessions()
    tensors = _segment_tensors(sessions)
    if not tensors:
        raise SystemExit(
            "no Balabit training bursts — is data/external/balabit populated?"
        )
    stats = fit_normalizer(tensors)
    norm = [apply_normalizer(t, stats) for t in tensors]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(norm))
    n_val = max(1, int(round(len(norm) * VAL_FRAC)))
    val_idx = set(perm[:n_val].tolist())
    train_t = [norm[i] for i in range(len(norm)) if i not in val_idx]
    val_t = [norm[i] for i in range(len(norm)) if i in val_idx]

    aug = Augmenter(scale_prob=0.0) if noscale else Augmenter()
    train_ds = ContrastiveSequenceDataset(
        train_t, CHUNK, STRIDE, augment=aug, seed=args.seed
    )
    val_ds = ContrastiveSequenceDataset(
        val_t, CHUNK, CHUNK, augment=aug, seed=args.seed
    )
    pin = args.device != "cpu" and torch.cuda.is_available()
    tl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    vl = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        if len(val_ds) > 0
        else None
    )
    log.info(
        "contrastive balabit%s: %d train / %d val bursts, %d train chunks",
        " (noscale)" if noscale else "",
        len(train_t),
        len(val_t),
        len(train_ds),
    )

    backbone, _head, history = pretrain_contrastive(
        tl,
        vl,
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
        corpus="balabit",
        noscale=noscale,
        temperature=args.temperature,
        proj_dim=args.proj_dim,
        chunk_length=CHUNK,
        stride=STRIDE,
        epochs=args.epochs,
        seed=args.seed,
        **ENC,
    )
    with tempfile.TemporaryDirectory() as tmp:
        w, m = save_lstm_ae(backbone, stats, Path(tmp), config=config, history=history)
        shutil.move(str(w), out_path)
        shutil.move(str(m), MODELS_DIR / f"{name}_meta.json")
    log.info(
        "saved %s (best val_loss=%.4f @ %d)",
        out_path,
        history.best_val_loss,
        history.best_epoch,
    )


# ---------------------------------------------------------------------------
# Embedding + verification
# ---------------------------------------------------------------------------
def _norm_chunks(session: dict, stats: dict) -> np.ndarray:
    ch = session_to_chunks(session, CHUNK, EVAL_STRIDE)
    if len(ch) == 0:
        return ch
    mean = np.asarray(stats["mean"], np.float32)
    std = np.asarray(stats["std"], np.float32)
    return ((ch - mean) / std).astype(np.float32)


def _session_embedding(model, session, stats, device) -> np.ndarray | None:
    ch = _norm_chunks(session, stats)
    if len(ch) == 0:
        return None
    return embed_chunks(model, ch, device=device).mean(axis=0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _cosine_eer(model, train_sessions, test_triples, stats, device) -> dict:
    """Enroll each user = mean training-session embedding; score test sessions by
    cosine to the claimed user → session EER."""
    enroll: dict[str, list[np.ndarray]] = {}
    for s in train_sessions:
        emb = _session_embedding(model, s, stats, device)
        if emb is not None:
            enroll.setdefault(s["player"], []).append(emb)
    centroids = {u: np.mean(v, axis=0) for u, v in enroll.items()}
    genuine, impostor = [], []
    for session, claimed, is_impostor in test_triples:
        if claimed not in centroids:
            continue
        emb = _session_embedding(model, session, stats, device)
        if emb is None:
            continue
        score = _cosine(emb, centroids[claimed])
        (impostor if is_impostor else genuine).append(score)
    if not genuine or not impostor:
        return {
            "session_eer": float("nan"),
            "n_genuine": len(genuine),
            "n_impostor": len(impostor),
        }
    val, _ = eer(np.array(genuine), np.array(impostor))
    return {
        "session_eer": float(val),
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
    }


def _classifier_eer(model, train_sessions, test_triples, stats, device) -> dict:
    """LightGBM on per-chunk embeddings; per test session mean P(claimed) → EER."""
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import LabelEncoder

    X, y = [], []
    for s in train_sessions:
        ch = _norm_chunks(s, stats)
        if len(ch) == 0:
            continue
        X.append(embed_chunks(model, ch, device=device))
        y.extend([s["player"]] * len(ch))
    if not X:
        return {"session_eer": float("nan")}
    X = np.concatenate(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = LGBMClassifier(
        num_leaves=31,
        n_estimators=200,
        learning_rate=0.1,
        min_child_samples=5,
        class_weight="balanced",
        verbose=-1,
    )
    clf.fit(X, y_enc)

    classes = set(le.classes_)
    genuine, impostor = [], []
    for session, claimed, is_impostor in test_triples:
        if claimed not in classes:
            continue
        ch = _norm_chunks(session, stats)
        if len(ch) == 0:
            continue
        proba = clf.predict_proba(embed_chunks(model, ch, device=device)).mean(axis=0)
        idx = int(np.where(le.classes_ == claimed)[0][0])
        (impostor if is_impostor else genuine).append(float(proba[idx]))
    if not genuine or not impostor:
        return {"session_eer": float("nan")}
    val, _ = eer(np.array(genuine), np.array(impostor))
    return {
        "session_eer": float(val),
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
    }


def _eval_encoder(model, train_sessions, test_triples, stats, device) -> dict:
    return {
        "cosine": _cosine_eer(model, train_sessions, test_triples, stats, device),
        "classifier": _classifier_eer(
            model, train_sessions, test_triples, stats, device
        ),
    }


# ---------------------------------------------------------------------------
# Eval phase
# ---------------------------------------------------------------------------
def _load(path: Path) -> LSTMAutoencoder:
    m = LSTMAutoencoder(**ENC)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m


def _hand_feature_baseline() -> dict:
    """Re-run Phase 6's windowed-feature LightGBM on the fresh data → session EER."""
    from scripts.run_external_identification import run_balabit

    r = run_balabit(BALABIT_DIR)
    return {
        "session_eer": r["verification"]["session_eer"],
        "closed_set_accuracy": r["closed_set"]["accuracy"],
        "n_users": r["n_users"],
    }


def _eval(args) -> dict:
    device = args.device
    train_sessions = _train_sessions()
    test_triples = _test_sessions()
    stats = _corpus_stats(train_sessions)
    log.info(
        "Balabit: %d train sessions | %d labelled test sessions",
        len(train_sessions),
        len(test_triples),
    )

    results: dict = {}

    # random-init band (3 seeds)
    rand_runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = LSTMAutoencoder(**ENC)
        rand_runs.append(_eval_encoder(m, train_sessions, test_triples, stats, device))
        log.info(
            "  random seed=%d → cosine %.3f | classifier %.3f",
            seed,
            rand_runs[-1]["cosine"]["session_eer"],
            rand_runs[-1]["classifier"]["session_eer"],
        )
    results["random"] = {
        route: {
            "session_eer_mean": float(
                np.nanmean([r[route]["session_eer"] for r in rand_runs])
            ),
            "session_eer_std": float(
                np.nanstd([r[route]["session_eer"] for r in rand_runs])
            ),
        }
        for route in ("cosine", "classifier")
    }

    # pretrained encoders
    encoders = {
        "contrastive_balabit": MODELS_DIR / "pretrained_contrastive_balabit.pt",
        "contrastive_balabit_noscale": MODELS_DIR
        / "pretrained_contrastive_balabit_noscale.pt",
        "contrastive_cs2cd_382": CS2CD_ENCODER,
    }
    for label, path in encoders.items():
        if not path.exists():
            log.warning("skip %s — missing %s", label, path)
            continue
        res = _eval_encoder(_load(path), train_sessions, test_triples, stats, device)
        results[label] = res
        log.info(
            "  %-28s → cosine %.3f | classifier %.3f",
            label,
            res["cosine"]["session_eer"],
            res["classifier"]["session_eer"],
        )

    results["hand_feature_baseline"] = _hand_feature_baseline()
    log.info(
        "  hand-feature baseline → session_eer %.3f",
        results["hand_feature_baseline"]["session_eer"],
    )
    return results


def _save(results: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "corpus": "balabit",
                "task": "session-level verification EER (challenge protocol)",
                "note": "compare contrastive vs random (did it learn identity?) and vs the "
                "hand-feature baseline (0.144). noscale tests whether scale-invariant "
                "augmentation discards identity's magnitude cues. Lower EER = better.",
                "eval_stride": EVAL_STRIDE,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


def _render_figure(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [
        "random",
        "contrastive_balabit",
        "contrastive_balabit_noscale",
        "contrastive_cs2cd_382",
    ]
    labels = [x for x in order if x in results]

    def _eer(lbl, route):
        r = results[lbl][route]
        return r.get("session_eer", r.get("session_eer_mean", float("nan")))

    def _err(lbl, route):
        return results[lbl][route].get("session_eer_std", 0.0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    w = 0.38
    for i, route in enumerate(("cosine", "classifier")):
        means = [_eer(lbl, route) for lbl in labels]
        errs = [_err(lbl, route) for lbl in labels]
        ax.bar(
            x + (i - 0.5) * w,
            means,
            w,
            yerr=errs,
            capsize=3,
            label=f"{route} (session-EER)",
        )
    base = results.get("hand_feature_baseline", {}).get("session_eer")
    if base is not None:
        ax.axhline(
            base,
            color="#4c78a8",
            ls="--",
            lw=1.4,
            label=f"hand-feature baseline ({base:.3f})",
        )
    ax.axhline(0.5, color="#8892a4", ls=":", lw=1.2, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Balabit session verification EER (lower = better)")
    ax.set_ylim(0.0, 0.6)
    ax.set_title(
        "Phase 6.1 — contrastive embeddings for player identity (Balabit, 10 users)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6.1 contrastive identity")
    parser.add_argument("--phase", choices=["pretrain", "eval", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proj-dim", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.phase in ("pretrain", "all"):
        log.info("=== PRETRAIN: contrastive balabit (scale + noscale) ===")
        _pretrain(noscale=False, args=args)
        _pretrain(noscale=True, args=args)

    if args.phase in ("eval", "all"):
        log.info("=== EVAL: Balabit session verification EER ===")
        results = _eval(args)
        _save(results)
        _render_figure(results)
        log.info("Wrote %s + %s", OUT_JSON, OUT_FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
