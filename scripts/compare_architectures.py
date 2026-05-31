"""
scripts/compare_architectures.py
================================
Head-to-head of sequence-autoencoder architectures for chunk-level cheat
detection: **LSTM-AE vs TCN-AE vs Transformer-AE**, all trained with the *same*
loop on the *same* legit chunks and evaluated with the *same* chunk-AUC metric.

Reports, per model: parameter count, train time, best val reconstruction loss,
the train/val overfit gap, and chunk ROC AUC per cheat type. The point is the
honest comparison at this data scale (18 sessions) — capacity is not the
bottleneck — not to crown a winner. Re-run after real cheat recordings land to
see whether the ranking changes.

Outputs `reports/architecture_comparison.json` + `reports/figures/arch_comparison.png`.

Usage:
    python -m scripts.compare_architectures --epochs 25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from pipeline.models.lstm_ae import LSTMAutoencoder, score_sequences
from pipeline.models.tcn_ae import TCNAutoencoder
from pipeline.models.transformer_ae import TransformerAutoencoder
from pipeline.sequences.dataset import EventSequenceDataset
from pipeline.sequences.preprocessing import (
    apply_normalizer,
    fit_normalizer,
    session_to_event_tensor,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SYN = ROOT / "data" / "synthetic"
OUT_JSON = ROOT / "reports" / "architecture_comparison.json"
OUT_FIG = ROOT / "reports" / "figures" / "arch_comparison.png"
CHUNK, STRIDE, SEED, VAL_FRAC = 64, 32, 42, 0.15
CHEATS = ["aimbot", "triggerbot", "macro"]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_legit_tensors() -> list[np.ndarray]:
    out = []
    for p in sorted(RAW.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            t = session_to_event_tensor(json.load(f))
        if len(t):
            out.append(t)
    return out


def _build_loaders(batch: int = 256):
    tensors = _load_legit_tensors()
    if not tensors:
        raise RuntimeError("No legit sessions in data/raw/")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tensors))
    n_val = max(1, int(round(len(tensors) * VAL_FRAC)))
    val_idx = set(perm[:n_val].tolist())
    tr = [tensors[i] for i in range(len(tensors)) if i not in val_idx]
    va = [tensors[i] for i in range(len(tensors)) if i in val_idx]
    stats = fit_normalizer(tr)
    tr_ds = EventSequenceDataset(
        [apply_normalizer(t, stats) for t in tr], CHUNK, STRIDE
    )
    va_ds = EventSequenceDataset([apply_normalizer(t, stats) for t in va], CHUNK, CHUNK)
    pin = _device() == "cuda"
    tl = DataLoader(tr_ds, batch_size=batch, shuffle=True, pin_memory=pin)
    vl = DataLoader(va_ds, batch_size=batch, shuffle=False, pin_memory=pin)
    return tl, vl, stats


def train_ae(
    model: nn.Module, tl: DataLoader, vl: DataLoader, epochs: int, lr: float
) -> dict:
    """Generic reconstruction-AE training loop shared by all architectures."""
    dev = _device()
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val, best_state, last_train = float("inf"), None, float("nan")
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        tot = n = 0.0
        for (
            batch
        ) in tl:  # EventSequenceDataset yields (L, F) tensors → batched (B, L, F)
            batch = batch.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            tot += loss.item() * len(batch)
            n += len(batch)
        last_train = tot / max(n, 1)
        model.eval()
        with torch.no_grad():
            vtot = vn = 0.0
            for vb in vl:
                vb = vb.to(dev)
                vtot += loss_fn(model(vb), vb).item() * len(vb)
                vn += len(vb)
        val = vtot / max(vn, 1)
        if val < best_val:
            best_val = val
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "params": int(sum(p.numel() for p in model.parameters())),
        "train_time_s": round(time.perf_counter() - t0, 1),
        "train_loss": round(last_train, 5),
        "val_loss": round(best_val, 5),
        "overfit_gap": round(best_val - last_train, 5),
    }


def _synthetic_chunks(stats: dict):
    """Per cheat type: (cheat-flagged chunks, legit chunks) pooled across files."""
    legit, cheat = [], {c: [] for c in CHEATS}
    for p in sorted(SYN.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        label = d.get("cheat_label", "legit")
        t = session_to_event_tensor(d)
        if len(t) < CHUNK:
            continue
        norm = apply_normalizer(t, stats)
        n = len(norm) // CHUNK
        chunks = np.stack([norm[i * CHUNK : (i + 1) * CHUNK] for i in range(n)])
        flags = np.zeros(n, dtype=bool)
        segs = d.get("cheat_segments") or []
        if segs and d.get("events"):
            times = np.array([e.get("t", 0.0) for e in d["events"]])
            in_seg = np.zeros(len(times), dtype=bool)
            for a, b in segs:
                in_seg |= (times >= a) & (times <= b)
            for i in range(n):
                flags[i] = in_seg[i * CHUNK : (i + 1) * CHUNK].any()
        legit.append(chunks[~flags])
        if label in cheat:
            cheat[label].append(chunks[flags])
    legit = np.concatenate(legit) if legit else np.empty((0, CHUNK, 8), np.float32)
    cheat = {
        c: (np.concatenate(v) if v else np.empty((0, CHUNK, 8), np.float32))
        for c, v in cheat.items()
    }
    return legit, cheat


def _chunk_auc(model, legit, cheat) -> dict:
    legit_scores = score_sequences(model, legit, device=_device())
    out = {}
    for c in CHEATS:
        if len(cheat[c]) == 0:
            out[c] = float("nan")
            continue
        cs = score_sequences(model, cheat[c], device=_device())
        y = np.r_[np.zeros(len(legit_scores)), np.ones(len(cs))]
        s = np.r_[legit_scores, cs]
        out[c] = float(roc_auc_score(y, s))
    return out


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare sequence-AE architectures")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log.info("Building loaders (device=%s)…", _device())
    tl, vl, stats = _build_loaders()
    legit, cheat = _synthetic_chunks(stats)
    log.info(
        "Eval chunks: %d legit | %s", len(legit), {c: len(cheat[c]) for c in CHEATS}
    )

    builders = {
        "LSTM-AE": lambda: LSTMAutoencoder(
            hidden_dim=64, bottleneck_dim=16, num_layers=2
        ),
        "TCN-AE": lambda: TCNAutoencoder(
            seq_len=CHUNK, hidden_dim=32, bottleneck_dim=16
        ),
        "Transformer-AE": lambda: TransformerAutoencoder(
            seq_len=CHUNK, d_model=64, nhead=4, num_layers=2, bottleneck_dim=16
        ),
    }

    results = {}
    for name, build in builders.items():
        log.info("Training %s…", name)
        torch.manual_seed(SEED)
        model = build()
        stats_train = train_ae(model, tl, vl, args.epochs, args.lr)
        auc = _chunk_auc(model, legit, cheat)
        results[name] = {**stats_train, "chunk_auc": auc}
        log.info(
            "  %s: params=%d time=%ss val_loss=%.4f gap=%.4f | AUC %s",
            name,
            stats_train["params"],
            stats_train["train_time_s"],
            stats_train["val_loss"],
            stats_train["overfit_gap"],
            {c: round(auc[c], 3) for c in CHEATS},
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    _render_figure(results)
    log.info("Wrote %s and %s", OUT_JSON, OUT_FIG)
    return 0


def _render_figure(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    x = np.arange(len(CHEATS))
    w = 0.8 / len(names)
    colors = {"LSTM-AE": "#4c78a8", "TCN-AE": "#54a24b", "Transformer-AE": "#e94560"}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, name in enumerate(names):
        auc = results[name]["chunk_auc"]
        ax.bar(
            x + (i - (len(names) - 1) / 2) * w,
            [auc[c] for c in CHEATS],
            w,
            label=f"{name} ({results[name]['params'] / 1e3:.0f}k params, {results[name]['train_time_s']}s)",
            color=colors.get(name),
        )
    ax.axhline(0.5, color="#8892a4", linestyle=":", linewidth=1.2, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(CHEATS)
    ax.set_ylabel("chunk-level ROC AUC")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Sequence-AE architecture comparison — chunk-level cheat detection\n(same training loop, same eval; 18 real legit sessions + synthetic cheats)"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run())
