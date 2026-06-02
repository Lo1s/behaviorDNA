"""
scripts/benchmark_cs2cd_ae.py
=============================
Second-dataset architecture check: train **LSTM-AE / TCN-AE / Transformer-AE** on
the EXTERNAL **CS2CD** (Counter-Strike 2 cheat-detection) dataset's legit mouse
stream and score chunk-level cheat AUC. This is an independent, 10-player
cross-dataset sanity check on `scripts/compare_architectures.py`, whose own cheat
data is single-game / single-source (GTA, one player).

The labelled file (`data/external/cs2cd/cs2cd_balanced_25000.parquet`, 25k legit +
25k cheat ticks, 10 players) **interleaves each player's cheat-match and
clean-match by tick**, so a naive sort-by-tick mixes labels within every window.
We instead group by ``(steamid, cheater_present)`` to recover **contiguous
same-label streams**, encode a compact mouse-input tensor (dx, dy, fire,
rightclick), chunk into 64-tick windows, and run the *same* reconstruction-AE
training loop + chunk-AUC metric as the GTA comparison.

Methodology matches `compare_architectures.py` (train on legit tr/va split, eval
chunk AUC on legit vs cheat) so the two are directly comparable. As there, the
legit eval baseline overlaps the AE's training distribution → absolute AUCs are
mildly optimistic, but the *ranking* across backbones is unaffected.

Outputs `reports/architecture_comparison_cs2cd.json` +
`reports/figures/arch_comparison_cs2cd.png`.

Usage:
    python -m scripts.benchmark_cs2cd_ae --epochs 25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from pipeline.models.lstm_ae import LSTMAutoencoder, score_sequences
from pipeline.models.tcn_ae import TCNAutoencoder
from pipeline.models.transformer_ae import TransformerAutoencoder
from pipeline.sequences.preprocessing import apply_normalizer, fit_normalizer
from scripts.compare_architectures import _device, train_ae

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "external" / "cs2cd" / "cs2cd_balanced_25000.parquet"
OUT_JSON = ROOT / "reports" / "architecture_comparison_cs2cd.json"
OUT_FIG = ROOT / "reports" / "figures" / "arch_comparison_cs2cd.png"

# Compact CS2 mouse-input encoding (per tick). dx/dy are the raw aim deltas; the
# two click flags carry fire/scope timing. The AE learns the legit manifold and
# flags superhuman snaps / inhuman fire timing as high reconstruction error.
CS2_FEATURES = ["usercmd_mouse_dx", "usercmd_mouse_dy", "FIRE", "RIGHTCLICK"]
CHUNK, STRIDE, SEED, VAL_FRAC, GAP = 64, 32, 42, 0.15, 2


def _streams_from_df(df: pd.DataFrame) -> list[tuple[int, np.ndarray]]:
    """Contiguous same-label streams as ``(label, (N, F) float32)`` tuples.

    Groups by ``(steamid, cheater_present)`` (the balanced file interleaves a
    player's cheat- and clean-match by tick), drops duplicate ticks, and splits
    each group into runs wherever the tick gap exceeds ``GAP``. Only runs long
    enough for at least one chunk are kept.
    """
    df = df.copy()
    df["steamid"] = df["steamid"].astype(str)
    streams: list[tuple[int, np.ndarray]] = []
    for (_sid, lab), g in df.groupby(["steamid", "cheater_present"]):
        g = g.drop_duplicates("tick").sort_values("tick")
        ticks = g["tick"].to_numpy()
        if len(ticks) == 0:
            continue
        run_id = np.concatenate([[0], (np.diff(ticks) > GAP).cumsum()])
        feats = g[CS2_FEATURES].to_numpy().astype(np.float32)
        # ~0.04% of usercmd_mouse_dx/dy are null in the raw data; a missing delta
        # = no movement that tick. Left unfilled, one NaN poisons the normalizer.
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        for rid in np.unique(run_id):
            arr = feats[run_id == rid]
            if len(arr) >= CHUNK:
                streams.append((int(lab), arr))
    return streams


def _load_streams() -> list[tuple[int, np.ndarray]]:
    if not PARQUET.exists():
        raise FileNotFoundError(
            f"CS2CD parquet not found: {PARQUET}. See notebooks/05_external_datasets.ipynb."
        )
    df = pd.read_parquet(
        PARQUET, columns=["tick", "steamid", "cheater_present", *CS2_FEATURES]
    )
    return _streams_from_df(df)


def _stack_chunks(tensors: list[np.ndarray], stride: int) -> np.ndarray:
    """Stack (possibly strided) chunks from normalized streams → (M, CHUNK, F)."""
    out = []
    for t in tensors:
        n = (len(t) - CHUNK) // stride + 1
        for i in range(n):
            out.append(t[i * stride : i * stride + CHUNK])
    if not out:
        return np.empty((0, CHUNK, len(CS2_FEATURES)), np.float32)
    return np.stack(out)


class _ChunkDataset(Dataset):
    """Minimal dataset over pre-chunked ``(M, CHUNK, F)`` arrays.

    Yields a single ``(CHUNK, F)`` float tensor per item — the same contract the
    shared ``train_ae`` loop expects — without the BehaviorDNA-specific
    feature-dim-8 guard in ``EventSequenceDataset`` (CS2 uses F=4).
    """

    def __init__(self, chunks: np.ndarray) -> None:
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self.chunks[i]).float()


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CS2CD external-dataset AE comparison")
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

    log.info("Loading CS2CD streams (device=%s)…", _device())
    streams = _load_streams()
    legit = [t for lab, t in streams if lab == 0]
    cheat = [t for lab, t in streams if lab == 1]
    if not legit or not cheat:
        raise RuntimeError(
            f"Need both classes; got {len(legit)} legit / {len(cheat)} cheat streams"
        )

    # Train AE on legit only (tr/va split for best-model selection), mirroring
    # compare_architectures._build_loaders.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(legit))
    n_val = max(1, int(round(len(legit) * VAL_FRAC)))
    val_idx = set(perm[:n_val].tolist())
    tr = [legit[i] for i in range(len(legit)) if i not in val_idx]
    va = [legit[i] for i in range(len(legit)) if i in val_idx]
    stats = fit_normalizer(tr)

    tr_chunks = _stack_chunks([apply_normalizer(t, stats) for t in tr], STRIDE)
    va_chunks = _stack_chunks([apply_normalizer(t, stats) for t in va], CHUNK)
    pin = _device() == "cuda"
    tl = DataLoader(
        _ChunkDataset(tr_chunks), batch_size=256, shuffle=True, pin_memory=pin
    )
    vl = DataLoader(
        _ChunkDataset(va_chunks), batch_size=256, shuffle=False, pin_memory=pin
    )

    # Non-overlapping eval chunks: all legit vs all cheat.
    legit_eval = _stack_chunks([apply_normalizer(t, stats) for t in legit], CHUNK)
    cheat_eval = _stack_chunks([apply_normalizer(t, stats) for t in cheat], CHUNK)
    log.info(
        "Streams: %d legit / %d cheat | eval chunks: %d legit / %d cheat",
        len(legit),
        len(cheat),
        len(legit_eval),
        len(cheat_eval),
    )

    F = len(CS2_FEATURES)
    builders = {
        "LSTM-AE": lambda: LSTMAutoencoder(
            feature_dim=F, hidden_dim=64, bottleneck_dim=16, num_layers=2
        ),
        "TCN-AE": lambda: TCNAutoencoder(
            feature_dim=F, seq_len=CHUNK, hidden_dim=32, bottleneck_dim=16
        ),
        "Transformer-AE": lambda: TransformerAutoencoder(
            feature_dim=F,
            seq_len=CHUNK,
            d_model=64,
            nhead=4,
            num_layers=2,
            bottleneck_dim=16,
        ),
    }

    results = {}
    for name, build in builders.items():
        log.info("Training %s…", name)
        torch.manual_seed(SEED)
        model = build()
        stats_train = train_ae(model, tl, vl, args.epochs, args.lr)
        ls = score_sequences(model, legit_eval, device=_device())
        cs = score_sequences(model, cheat_eval, device=_device())
        y = np.r_[np.zeros(len(ls)), np.ones(len(cs))]
        s = np.r_[ls, cs]
        auc = float(roc_auc_score(y, s))
        results[name] = {**stats_train, "chunk_auc": auc}
        log.info(
            "  %s: params=%d time=%ss val_loss=%.4f gap=%.4f | cheat AUC=%.3f",
            name,
            stats_train["params"],
            stats_train["train_time_s"],
            stats_train["val_loss"],
            stats_train["overfit_gap"],
            auc,
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
    colors = {"LSTM-AE": "#4c78a8", "TCN-AE": "#54a24b", "Transformer-AE": "#e94560"}
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = ax.bar(
        names,
        [results[n]["chunk_auc"] for n in names],
        color=[colors.get(n) for n in names],
        width=0.6,
    )
    for n, b in zip(names, bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.01,
            f"{results[n]['chunk_auc']:.3f}\n{results[n]['params'] / 1e3:.0f}k · {results[n]['train_time_s']}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(0.5, color="#8892a4", linestyle=":", linewidth=1.2, label="chance")
    ax.set_ylabel("chunk-level ROC AUC")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Sequence-AE comparison on EXTERNAL CS2CD\n"
        "(Counter-Strike 2 cheat detection; 10 players, mouse dx/dy + fire)"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run())
