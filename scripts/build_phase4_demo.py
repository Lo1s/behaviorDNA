"""
scripts/build_phase4_demo.py
============================
Render the Phase 4 demo artifacts (PNG + animated GIF) **programmatically**.

Picks a session from ``data/raw/`` (or the path given on the command line),
injects a synthetic aimbot at a configured timestamp, drives the full
streaming pipeline offline, then renders the risk-score timeline as:

- ``reports/figures/phase4_live_replay.png`` — static annotated line plot
- ``reports/figures/phase4_live_demo.gif``    — animated growing timeline

Both figures share the same data, so they always tell the same story.

Usage:
    python -m scripts.build_phase4_demo
    python -m scripts.build_phase4_demo --session data/raw/<file>.json --inject-at 30 --cheat aimbot

Dependencies: matplotlib + Pillow (already in requirements.txt via streamlit).
No screen capture, no manual editing — re-runnable against any session or
model checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from pipeline.inference.streaming import build_stream_state
from scripts.replay_session import _maybe_inject_cheat, replay_offline

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "reports" / "figures"
# Default to a session that was held out from LSTM-AE training (in the val
# split), so the model has never seen this player's exact patterns. Gives a
# cleaner demo: the risk score stays low through the warm-up, then rises
# when the synthetic cheat is injected. If this file is missing, the CLI
# falls back to the first JSON in data/raw/.
DEFAULT_SESSION = (
    ROOT / "data" / "raw" / "20260516T053654_hydRa_arc_raiders_7f80d8fa.json"
)


# ---------------------------------------------------------------------------
# Static figure
# ---------------------------------------------------------------------------


def render_static(
    updates: list[dict],
    out_path: Path,
    *,
    cheat_type: str | None,
    inject_at_s: float | None,
    risk_threshold: float = 0.5,
) -> None:
    """Render the risk-score timeline + per-detector contributions."""
    if not updates:
        log.warning("No updates to render")
        return

    ts = np.array([u["t"] / 1000.0 for u in updates])
    risk = np.array([u["session_risk"] for u in updates])

    # Per-detector logit contributions stacked
    detector_names = sorted({k for u in updates for k in u.get("detector_logits", {})})
    logits = {
        name: np.array(
            [u["detector_logits"].get(name, 0.0) for u in updates], dtype=np.float64
        )
        for name in detector_names
    }

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1])
    )

    # Top: risk-score timeline
    axes[0].plot(ts, risk, color="#e94560", linewidth=2.2, label="combined risk")
    axes[0].axhline(
        risk_threshold,
        color="#8892a4",
        linestyle="--",
        linewidth=1.2,
        label=f"alert threshold = {risk_threshold:.2f}",
    )
    if cheat_type is not None and inject_at_s is not None:
        axes[0].axvline(
            inject_at_s,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"{cheat_type} injected at t={inject_at_s:.0f}s",
        )
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("session risk")
    axes[0].set_title("Phase 4 — live session risk over time")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Bottom: per-detector logit contributions
    colours = {
        "IsolationForest": "#4ecca3",
        "LocalOutlierFactor": "#f5a623",
        "OneClassSVM": "#6a4c93",
        "LSTMAutoencoder": "#e94560",
    }
    for name in detector_names:
        axes[1].plot(
            ts,
            logits[name],
            label=name,
            linewidth=1.4,
            color=colours.get(name, None),
            alpha=0.85,
        )
    axes[1].axhline(0, color="grey", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("logit contribution")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Animated GIF
# ---------------------------------------------------------------------------


def render_gif(
    updates: list[dict],
    out_path: Path,
    *,
    cheat_type: str | None,
    inject_at_s: float | None,
    risk_threshold: float = 0.5,
    n_frames: int = 40,
    fps: int = 8,
) -> None:
    """Same data as render_static, but as a growing animated timeline."""
    if not updates:
        return

    ts = np.array([u["t"] / 1000.0 for u in updates])
    risk = np.array([u["session_risk"] for u in updates])

    # Subsample to ``n_frames`` evenly distributed snapshots
    if len(ts) > n_frames:
        idx = np.linspace(0, len(ts) - 1, n_frames).astype(int)
        ts = ts[idx]
        risk = risk[idx]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    (line,) = ax.plot([], [], color="#e94560", linewidth=2.2)
    ax.axhline(
        risk_threshold,
        color="#8892a4",
        linestyle="--",
        linewidth=1.2,
        label=f"alert threshold = {risk_threshold:.2f}",
    )
    if cheat_type is not None and inject_at_s is not None:
        ax.axvline(
            inject_at_s,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"{cheat_type} injected at t={inject_at_s:.0f}s",
        )

    ax.set_xlim(ts.min(), ts.max() if ts.max() > ts.min() else ts.min() + 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("session risk")
    ax.set_title("BehaviorDNA — live cheat-risk score (mock data)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    def init():
        line.set_data([], [])
        return (line,)

    def update_frame(i: int):
        line.set_data(ts[: i + 1], risk[: i + 1])
        return (line,)

    anim = FuncAnimation(
        fig,
        update_frame,
        init_func=init,
        frames=len(ts),
        interval=1000 / fps,
        blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 4 demo PNG + GIF")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument(
        "--cheat",
        choices=["aimbot", "triggerbot", "macro"],
        default="aimbot",
    )
    parser.add_argument(
        "--inject-at",
        type=float,
        default=30.0,
        help="When (seconds) to inject the cheat (default 30)",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=FIGURES_DIR / "phase4_live_replay.png",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=FIGURES_DIR / "phase4_live_demo.gif",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    if not args.session.exists():
        log.warning(
            "Session not found at %s — falling back to the first JSON in data/raw/",
            args.session,
        )
        candidates = sorted((ROOT / "data" / "raw").glob("*.json"))
        if not candidates:
            log.error("No JSON files in data/raw/")
            return 1
        args.session = candidates[0]
        log.info("Using fallback session %s", args.session.name)

    with open(args.session, encoding="utf-8") as f:
        session = json.load(f)
    log.info(
        "Loaded session: %s  (%d events)",
        args.session.name,
        len(session.get("events", [])),
    )

    session = _maybe_inject_cheat(session, args.cheat, args.inject_at)
    log.info(
        "Injected %s at t=%.1fs  → %d cheat segments",
        args.cheat,
        args.inject_at,
        len(session.get("cheat_segments", [])),
    )

    log.info("Building streaming engine (this is the slow part — ~45 s)…")
    state = build_stream_state()
    log.info("Replaying session through the engine…")
    updates = replay_offline(session, state=state)
    log.info("Captured %d ScoreUpdate snapshots", len(updates))

    args.png.parent.mkdir(parents=True, exist_ok=True)
    args.gif.parent.mkdir(parents=True, exist_ok=True)

    render_static(
        updates,
        args.png,
        cheat_type=args.cheat,
        inject_at_s=args.inject_at,
        risk_threshold=args.threshold,
    )
    render_gif(
        updates,
        args.gif,
        cheat_type=args.cheat,
        inject_at_s=args.inject_at,
        risk_threshold=args.threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
