"""
scripts/evasion_frontier.py
===========================
Phase 7 headline — the **detection-vs-evasion frontier**.

Sweeps the humanisation strength ``λ ∈ [0, 1]`` (``pipeline/adversarial/humanizer.py``)
over the three cheats on the real GTA legit recordings and, for each λ, measures:

* **detector AUC(λ)** — the chunk-level LSTM-AE (the Phase-2 model that actually
  detects aimbot/triggerbot), plus a reference classical window detector
  (OneClassSVM session-max), and
* **utility(λ)** — the cheat's closed-form residual advantage over an unaided human.

Writes ``reports/evasion_frontier.json`` + ``reports/figures/phase7_evasion_frontier.png``
(AUC(λ) & utility(λ) curves + the parametric detection-vs-utility frontier with the
equilibrium region marked). Inference-only on the committed ``models/lstm_ae.pt`` —
no training, CPU-fine. Follows the ``scripts/data_efficiency.py`` pattern.

Usage:
    python -m scripts.evasion_frontier                 # full sweep (3 seeds)
    python -m scripts.evasion_frontier --seeds 1 --lambdas 0,1   # quick dev pass
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from pipeline.adversarial.benchmark import _chunk_cheat_labels
from pipeline.adversarial.humanizer import (
    VALID_CHEATS,
    cheat_utility,
    humanize,
    player_baseline,
)
from pipeline.features.run import (
    CHEAT_FEATURE_COLS,
    polling_rate_norm,
    process_session_windows,
)
from pipeline.ingestion.run import parse_events, parse_session_metadata
from pipeline.models.lstm_ae import load_lstm_ae, score_sequences
from pipeline.sequences.preprocessing import apply_normalizer, session_to_event_tensor

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
DEFAULT_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_SEEDS = [0, 1, 2]
CHANCE = 0.5


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Window-feature helper (one row per 30 s window, from a session dict)
# ---------------------------------------------------------------------------


def _session_windows(data: dict) -> pd.DataFrame:
    meta = parse_session_metadata(data, Path(data.get("session_id", "x") + ".json"))
    edf = parse_events(data)
    if edf.empty:
        return pd.DataFrame()
    nf = (float(meta["sensitivity"]) * float(meta["dpi"])) / 800.0
    rn = polling_rate_norm(meta.get("polling_rate"))
    windows = process_session_windows(edf.sort_values("t"), nf, rn)
    return pd.DataFrame(windows) if windows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Detector scoring
# ---------------------------------------------------------------------------


class FrontierScorer:
    """Holds the fixed legit baselines so every λ is scored against the same
    negatives (and the OneClassSVM is fit once on legit window features)."""

    def __init__(self, sessions: list[dict], device: str):
        self.sessions = sessions
        self.device = device
        self.model, self.stats, meta = load_lstm_ae(MODELS, device=device)
        self.chunk_len = int((meta.get("config") or {}).get("chunk_length", 64))

        # Fixed legit chunk pool (negatives for chunk AUC) — scored once.
        legit_chunks = []
        for d in sessions:
            sc = self._chunk_scores(d, only_cheat=False)
            if sc.size:
                legit_chunks.append(sc)
        self.legit_chunk_scores = np.concatenate(legit_chunks)

        # Fixed legit window detector (OneClassSVM session-max) — fit once.
        legit_feats = [
            w.assign(_sid=i)
            for i, w in enumerate(map(_session_windows, sessions))
            if not w.empty
        ]
        feats = pd.concat(legit_feats, ignore_index=True)
        X = feats[CHEAT_FEATURE_COLS].fillna(0.0).to_numpy()
        self.scaler = StandardScaler().fit(X)
        self.ocsvm = OneClassSVM(kernel="rbf", nu=0.05).fit(self.scaler.transform(X))
        anom = -self.ocsvm.score_samples(self.scaler.transform(X))
        self.legit_session_scores = (
            pd.DataFrame({"sid": feats["_sid"], "s": anom})
            .groupby("sid")["s"]
            .max()
            .to_numpy()
        )

    def _chunk_scores(self, data: dict, *, only_cheat: bool) -> np.ndarray:
        tensor = session_to_event_tensor(data)
        if len(tensor) < self.chunk_len:
            return np.empty(0, np.float32)
        norm = apply_normalizer(tensor, self.stats)
        n = len(norm) // self.chunk_len
        if only_cheat:
            labels = _chunk_cheat_labels(data, self.chunk_len, n)
            idx = [i for i in range(n) if labels[i]]
        else:
            idx = list(range(n))
        if not idx:
            return np.empty(0, np.float32)
        chunks = np.stack(
            [norm[i * self.chunk_len : (i + 1) * self.chunk_len] for i in idx]
        )
        return score_sequences(self.model, chunks, device=self.device)

    def lstm_chunk_auc(self, cheat_sessions: list[dict]) -> tuple[float, int]:
        pos = [self._chunk_scores(d, only_cheat=True) for d in cheat_sessions]
        pos = (
            np.concatenate([p for p in pos if p.size])
            if any(p.size for p in pos)
            else np.empty(0)
        )
        if pos.size == 0:
            return float("nan"), 0
        y = np.r_[np.zeros(len(self.legit_chunk_scores)), np.ones(len(pos))]
        s = np.r_[self.legit_chunk_scores, pos]
        return float(roc_auc_score(y, s)), int(pos.size)

    def window_auc(self, cheat_sessions: list[dict]) -> float:
        cheat_max = []
        for d in cheat_sessions:
            w = _session_windows(d)
            if w.empty:
                continue
            X = self.scaler.transform(w[CHEAT_FEATURE_COLS].fillna(0.0).to_numpy())
            cheat_max.append(float(np.max(-self.ocsvm.score_samples(X))))
        if not cheat_max:
            return float("nan")
        y = np.r_[np.zeros(len(self.legit_session_scores)), np.ones(len(cheat_max))]
        s = np.r_[self.legit_session_scores, np.asarray(cheat_max)]
        return float(roc_auc_score(y, s))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7 detection-vs-evasion frontier"
    )
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument(
        "--lambdas",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LAMBDAS),
        help="comma-separated λ grid",
    )
    parser.add_argument(
        "--no-window-detector",
        action="store_true",
        help="skip the (slow) classical window detector reference curve",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = list(range(args.seeds))
    device = _device()

    paths = sorted(RAW.glob("*.json"))
    if not paths:
        log.error("No legit recordings in %s — run `dvc pull` first.", RAW)
        return 1
    sessions: list[dict] = [json.loads(p.read_text()) for p in paths]
    # One baseline per player; humanise each session toward its own player.
    by_player: dict[str, list[dict]] = {}
    for s in sessions:
        by_player.setdefault(s.get("player", "?"), []).append(s)
    baselines = {pl: player_baseline(ss) for pl, ss in by_player.items()}
    log.info(
        "Loaded %d legit sessions / %d players; λ=%s; seeds=%s; device=%s",
        len(sessions),
        len(by_player),
        lambdas,
        seeds,
        device,
    )

    scorer = FrontierScorer(sessions, device)
    log.info(
        "Fixed legit baseline: %d chunks, %d sessions (window detector)",
        len(scorer.legit_chunk_scores),
        len(scorer.legit_session_scores),
    )

    runs: list[dict] = []
    for cheat in VALID_CHEATS:
        for lam in lambdas:
            util = float(
                np.mean([cheat_utility(cheat, lam, baselines[pl]) for pl in by_player])
            )
            window_done = False
            for seed in seeds:
                cheat_sessions = [
                    humanize(s, cheat, lam, baselines[s.get("player", "?")], seed=seed)
                    for s in sessions
                ]
                auc, n_pos = scorer.lstm_chunk_auc(cheat_sessions)
                rec = {
                    "cheat": cheat,
                    "lambda": lam,
                    "seed": seed,
                    "lstm_chunk_auc": auc,
                    "utility": util,
                    "n_pos_chunks": n_pos,
                }
                # window detector is seed-robust (30 s aggregates) → seed 0 only
                if not args.no_window_detector and not window_done:
                    rec["window_auc"] = scorer.window_auc(cheat_sessions)
                    window_done = True
                runs.append(rec)
                log.info(
                    "  %-11s λ=%.2f seed=%d → LSTM AUC %.3f  util %.2f  (n_pos=%d)",
                    cheat,
                    lam,
                    seed,
                    auc,
                    util,
                    n_pos,
                )

    summary = _summarize(runs, lambdas)
    out = {
        "meta": {
            "n_sessions": len(sessions),
            "n_players": len(by_player),
            "lambdas": lambdas,
            "seeds": seeds,
            "device": device,
            "chunk_length": scorer.chunk_len,
            "detector": "LSTMAutoencoder/chunk + OneClassSVM/window",
            "chance": CHANCE,
        },
        "summary": summary,
        "runs": runs,
    }
    out_json = ROOT / "reports" / "evasion_frontier.json"
    out_json.write_text(json.dumps(out, indent=2))
    _render_figure(summary, lambdas)
    log.info("Wrote %s", out_json)
    _log_equilibrium(summary)
    return 0


def _summarize(runs: list[dict], lambdas: list[float]) -> dict:
    out: dict[str, dict] = {}
    df = pd.DataFrame(runs)
    for cheat in VALID_CHEATS:
        sub = df[df["cheat"] == cheat]
        auc_pts, util_pts, win_pts = [], [], []
        for lam in lambdas:
            g = sub[sub["lambda"] == lam]
            aucs = g["lstm_chunk_auc"].dropna().to_numpy()
            if len(aucs):
                auc_pts.append(
                    {
                        "lambda": lam,
                        "mean": float(np.mean(aucs)),
                        "std": float(np.std(aucs)),
                        "n": int(len(aucs)),
                    }
                )
            util_pts.append({"lambda": lam, "utility": float(g["utility"].iloc[0])})
            wa = (
                g["window_auc"].dropna().to_numpy()
                if "window_auc" in g
                else np.array([])
            )
            if len(wa):
                win_pts.append({"lambda": lam, "auc": float(wa[0])})
        out[cheat] = {
            "lstm_chunk_auc": auc_pts,
            "window_auc": win_pts,
            "utility": util_pts,
        }
    return out


def _log_equilibrium(summary: dict) -> None:
    """Print, per cheat, the detection AUC once utility has decayed to ~0.2
    ('barely worth running') — the headline equilibrium reading."""
    for cheat, s in summary.items():
        aucs = {p["lambda"]: p["mean"] for p in s["lstm_chunk_auc"]}
        utils = {p["lambda"]: p["utility"] for p in s["utility"]}
        # smallest λ whose utility ≤ 0.2
        evade = [lam for lam, u in sorted(utils.items()) if u <= 0.2]
        if evade and evade[0] in aucs:
            lam = evade[0]
            log.info(
                "EQUILIBRIUM %-11s: utility≤0.2 first at λ=%.2f → detection AUC %.3f",
                cheat,
                lam,
                aucs[lam],
            )


def _render_figure(summary: dict, lambdas: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"aimbot": "#e94560", "triggerbot": "#4c78a8", "macro": "#5a9e6f"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A — detection AUC(λ) (solid) + utility(λ) (dotted, twin axis)
    axU = axL.twinx()
    for cheat, s in summary.items():
        c = colors[cheat]
        xs = [p["lambda"] for p in s["lstm_chunk_auc"]]
        ms = np.array([p["mean"] for p in s["lstm_chunk_auc"]])
        ss = np.array([p["std"] for p in s["lstm_chunk_auc"]])
        axL.plot(xs, ms, "-o", color=c, label=f"{cheat} — detection AUC")
        axL.fill_between(xs, ms - ss, ms + ss, color=c, alpha=0.15)
        ux = [p["lambda"] for p in s["utility"]]
        uy = [p["utility"] for p in s["utility"]]
        axU.plot(ux, uy, ":", color=c, alpha=0.8, label=f"{cheat} — utility")
    axL.axhline(CHANCE, color="#8892a4", ls="--", lw=1, label="detection chance")
    axL.set_xlabel("humanisation strength λ")
    axL.set_ylabel("chunk-level detection ROC AUC")
    axU.set_ylabel("cheat utility (residual advantage)")
    axL.set_ylim(0.4, 1.02)
    axU.set_ylim(0.0, 1.02)
    axL.set_title("Detection AUC and utility vs humanisation λ")
    axL.legend(loc="lower left", fontsize=8)
    axL.grid(True, alpha=0.3)

    # Panel B — parametric detection-vs-utility frontier
    for cheat, s in summary.items():
        c = colors[cheat]
        util = {p["lambda"]: p["utility"] for p in s["utility"]}
        auc = {p["lambda"]: p["mean"] for p in s["lstm_chunk_auc"]}
        lams = [lam for lam in lambdas if lam in auc]
        xs = [util[lam] for lam in lams]
        ys = [auc[lam] for lam in lams]
        axR.plot(xs, ys, "-o", color=c, label=cheat)
        for lam in lams:
            axR.annotate(
                f"λ={lam:g}",
                (util[lam], auc[lam]),
                fontsize=7,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    axR.axhline(CHANCE, color="#8892a4", ls="--", lw=1)
    axR.axvspan(
        0.0, 0.2, color="#cfe8cf", alpha=0.5, label="utility ≤ 0.2 (not worth running)"
    )
    axR.set_xlabel("cheat utility →  (more advantage)")
    axR.set_ylabel("detection AUC →  (easier to catch)")
    axR.set_xlim(-0.02, 1.02)
    axR.set_ylim(0.4, 1.02)
    axR.invert_xaxis()
    axR.set_title("The frontier: detectable ↔ worth running")
    axR.legend(loc="lower left", fontsize=8)
    axR.grid(True, alpha=0.3)

    fig.suptitle(
        "Phase 7 — detection-vs-evasion frontier (synthetic cheats on 18 real GTA sessions)",
        fontsize=12,
    )
    fig.tight_layout()
    out_fig = ROOT / "reports" / "figures" / "phase7_evasion_frontier.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run())
