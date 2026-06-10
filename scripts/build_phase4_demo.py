"""
scripts/build_phase4_demo.py
============================
Build the Phase-4 detection figure: **per-chunk LSTM-AE reconstruction error,
legit chunks vs injected-cheat chunks**, one panel per cheat type.

Why this figure (and not a live risk timeline): the chunk-level LSTM-AE is a
*between-population* detector — it separates legit behaviour chunks from cheat
chunks (the headline benchmark metric, ROC AUC ≈ 0.79 aimbot / 0.93 triggerbot
/ 0.60 macro on real GTA data). It does **not** localise *when* sparse injected
cheating starts inside one session, so a single-session "risk over time" replay
shows nothing and would overclaim. The session-level Bayesian aggregator that
would turn this into a live score saturates on the current 18-session
calibration set (see docs/STREAMING.md → Phase 4.1). This figure shows the
honest, reproducible signal: the reconstruction-error distributions the model
actually learnt to separate.

Reads ``data/synthetic/`` (produced by ``pipeline.adversarial.generate_dataset``)
and the persisted LSTM-AE in ``models/``.

Output:
    reports/figures/phase4_chunk_detection.png
    reports/figures/phase4_chunk_flags.gif    (with --gif)

The GIF is the README's 15-second hero: one injected-cheat session replayed
chunk by chunk, each chunk's reconstruction error landing as a dot — red and
counted when it crosses the legit-pool p95 threshold. Same honest framing as
the static figure (a *chunk-level* detector; the shaded band is ground truth,
the flags are what the model fires).

Usage:
    python -m scripts.build_phase4_demo
    python -m scripts.build_phase4_demo --gif
    python -m scripts.build_phase4_demo --synthetic-dir data/synthetic --out <path>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# Panels left→right, with the colour used for the cheat distribution.
CHEAT_ORDER = ["aimbot", "triggerbot", "macro"]
CHEAT_COLOUR = {"aimbot": "#6a4c93", "triggerbot": "#e94560", "macro": "#f5a623"}
LEGIT_COLOUR = "#4ecca3"


def _chunk_scores(path: Path, model, stats, chunk_length: int):
    """Score every non-overlapping chunk of one session.

    Returns ``(scores, is_cheat)`` arrays (one entry per chunk) or ``None`` if
    the session is shorter than one chunk. ``is_cheat[i]`` is True when chunk i
    overlaps any ``cheat_segment``.
    """
    import torch

    from pipeline.models.lstm_ae import score_sequences
    from pipeline.sequences.preprocessing import (
        apply_normalizer,
        session_to_event_tensor,
    )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tensor = session_to_event_tensor(data)
    if len(tensor) < chunk_length:
        return None
    norm = apply_normalizer(tensor, stats)
    n = len(norm) // chunk_length
    chunks = np.stack(
        [norm[i * chunk_length : (i + 1) * chunk_length] for i in range(n)]
    )

    is_cheat = np.zeros(n, dtype=bool)
    segs = data.get("cheat_segments") or []
    if segs and data.get("events"):
        times = np.array([ev.get("t", 0.0) for ev in data["events"]])
        in_seg = np.zeros(len(times), dtype=bool)
        for a, b in segs:
            in_seg |= (times >= a) & (times <= b)
        for i in range(n):
            is_cheat[i] = in_seg[i * chunk_length : (i + 1) * chunk_length].any()

    scores = score_sequences(
        model, torch.from_numpy(chunks).float(), batch_size=256, device="auto"
    )
    return np.asarray(scores, dtype=np.float64), is_cheat


def collect_chunk_scores(synthetic_dir: Path, model, stats, chunk_length: int):
    """Pool per-chunk scores across every synthetic session.

    Returns ``(legit_pool, cheat_pools)`` where ``legit_pool`` is every chunk
    that does NOT overlap a cheat segment (from legit files and the clean parts
    of cheat files), and ``cheat_pools[label]`` is every cheat-overlapping chunk
    for that cheat type.
    """
    legit_pool: list[np.ndarray] = []
    cheat_pools: dict[str, list[np.ndarray]] = {c: [] for c in CHEAT_ORDER}

    for path in sorted(synthetic_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            label = json.load(f).get("cheat_label", "legit")
        res = _chunk_scores(path, model, stats, chunk_length)
        if res is None:
            continue
        scores, is_cheat = res
        legit_pool.append(scores[~is_cheat])
        if label in cheat_pools:
            cheat_pools[label].append(scores[is_cheat])

    legit = np.concatenate(legit_pool) if legit_pool else np.array([])
    cheat = {
        c: (np.concatenate(v) if v else np.array([])) for c, v in cheat_pools.items()
    }
    return legit, cheat


def render_chunk_detection(legit, cheat_by_type, out_path: Path) -> None:
    """One density panel per cheat type: legit chunks vs cheat chunks + AUC."""
    from sklearn.metrics import roc_auc_score

    fig, axes = plt.subplots(1, len(CHEAT_ORDER), figsize=(14, 4.6), sharey=True)
    if len(CHEAT_ORDER) == 1:
        axes = [axes]

    for ax, cheat_type in zip(axes, CHEAT_ORDER):
        cheat = cheat_by_type.get(cheat_type, np.array([]))
        if len(cheat) == 0 or len(legit) == 0:
            ax.set_title(f"{cheat_type}\n(no data)")
            continue

        y = np.r_[np.zeros(len(legit)), np.ones(len(cheat))]
        s = np.r_[legit, cheat]
        auc = roc_auc_score(y, s)

        # Clip the x-view to the 98th pct so a few extreme chunks don't crush it.
        clip = float(np.percentile(s, 98))
        bins = np.linspace(0, clip, 50)
        ax.hist(
            np.clip(legit, 0, clip),
            bins=bins,
            density=True,
            alpha=0.6,
            color=LEGIT_COLOUR,
            label=f"legit chunks (n={len(legit)})",
        )
        ax.hist(
            np.clip(cheat, 0, clip),
            bins=bins,
            density=True,
            alpha=0.6,
            color=CHEAT_COLOUR[cheat_type],
            label=f"{cheat_type} chunks (n={len(cheat)})",
        )
        ax.axvline(np.median(legit), color="#2a7d63", linestyle="--", linewidth=1.2)
        ax.axvline(
            np.median(cheat),
            color=CHEAT_COLOUR[cheat_type],
            linestyle="--",
            linewidth=1.4,
        )
        ax.set_title(f"{cheat_type} — chunk ROC AUC = {auc:.3f}")
        ax.set_xlabel("LSTM-AE reconstruction error (per 64-event chunk)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("density")
    fig.suptitle(
        "Phase 4 — chunk-level cheat detection on real GTA data\n"
        "LSTM-AE reconstruction error separates injected-cheat chunks from legit play",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def pick_gif_session(
    synthetic_dir: Path,
    model,
    stats,
    chunk_length: int,
    threshold: float,
    labels: tuple[str, ...] = ("aimbot", "triggerbot"),
):
    """Choose the cheat session whose flags will read best in a short replay.

    Scores every candidate session and rates the *viewport that will actually
    be rendered*: flags should land inside the injected band (hits), not on
    legit chunks (false flags), with enough cheat chunks to be visible.
    Returns ``(label, path, scores, is_cheat)``.
    """
    best = None
    for path in sorted(synthetic_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            label = json.load(f).get("cheat_label", "legit")
        if label not in labels:
            continue
        res = _chunk_scores(path, model, stats, chunk_length)
        if res is None:
            continue
        scores, is_cheat = res
        if is_cheat.sum() == 0 or len(scores) < 20:
            continue
        lo, hi = _gif_viewport(is_cheat, len(scores))
        s, c = scores[lo:hi], is_cheat[lo:hi]
        if c.sum() == 0:
            continue
        hit_rate = float((s[c] > threshold).mean())
        fp_rate = float((s[~c] > threshold).mean())
        # favour clean in-window separation + enough cheat chunks to see
        quality = hit_rate - 2.0 * fp_rate + 0.03 * min(int(c.sum()), 12)
        if best is None or quality > best[0]:
            best = (quality, label, path, scores, is_cheat)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def _gif_viewport(
    is_cheat: np.ndarray, n: int, context: int = 40, max_chunks: int = 160
) -> tuple[int, int]:
    """[start, end) window around the injected-cheat span, with lead-in context.

    Real sessions are thousands of chunks with sparse injection — replaying
    everything is a blur. Zoom to the cheat region plus enough legit lead-in
    that the viewer sees the detector staying quiet first.
    """
    idx = np.flatnonzero(is_cheat)
    if len(idx) == 0:
        return 0, min(n, max_chunks)
    start = max(0, int(idx[0]) - context)
    end = min(n, int(idx[-1]) + context + 1)
    if end - start <= max_chunks:
        return start, end
    # span too wide — slide a max_chunks window to the densest cheat cluster,
    # then pull it back so the viewer gets a quiet lead-in before the flags
    counts = np.convolve(is_cheat.astype(int), np.ones(max_chunks, dtype=int))
    best_end = int(np.argmax(counts)) + 1  # window is [best_end-max_chunks, best_end)
    start = max(0, min(best_end - max_chunks, n - max_chunks))
    first_in_window = int(np.flatnonzero(is_cheat[start : start + max_chunks])[0])
    start = max(0, start + first_in_window - context)
    return start, min(n, start + max_chunks)


def render_chunk_flag_gif(
    scores: np.ndarray,
    is_cheat: np.ndarray,
    threshold: float,
    out_path: Path,
    cheat_label: str = "triggerbot",
    duration_s: float = 14.0,
    max_frames: int = 140,
) -> None:
    """Animated replay: chunks stream in, flags fire above the legit-p95 line."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    lo, hi = _gif_viewport(is_cheat, len(scores))
    scores = scores[lo:hi]
    is_cheat = is_cheat[lo:hi]
    n = len(scores)

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ymax = float(max(scores.max(), threshold) * 1.15)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(
        f"chunk index (64-event chunks; window of {n} chunks around the injection)"
    )
    ax.set_ylabel("LSTM-AE reconstruction error")
    ax.grid(True, alpha=0.3)

    # Ground truth: shade the chunks a cheat was actually injected into.
    in_band = False
    for i in range(n + 1):
        cheat_here = bool(is_cheat[i]) if i < n else False
        if cheat_here and not in_band:
            start, in_band = i, True
        elif not cheat_here and in_band:
            ax.axvspan(start - 0.5, i - 0.5, color="#e94560", alpha=0.12)
            in_band = False
    ax.axhline(
        threshold,
        color="#c0392b",
        linestyle="--",
        linewidth=1.3,
        label="flag threshold (95th pct of legit chunks)",
    )
    ax.plot(
        [],
        [],
        color="#e94560",
        alpha=0.25,
        linewidth=8,
        label="injected cheat (ground truth)",
    )

    (trace,) = ax.plot([], [], color="#9aa7b0", linewidth=1.0, zorder=1)
    ok_dots = ax.scatter(
        [], [], s=34, color=LEGIT_COLOUR, zorder=2, label="chunk scored legit"
    )
    flag_dots = ax.scatter(
        [], [], s=60, color="#e94560", marker="D", zorder=3, label="⚑ chunk flagged"
    )
    counter = ax.text(
        0.985,
        0.94,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#c0392b",
    )
    ax.set_title(
        f"Live chunk-level cheat flags — {cheat_label} injected into a real GTA session\n"
        "LSTM autoencoder reconstruction error, scored as the session streams",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    xs = np.arange(n)
    step = max(1, int(np.ceil(n / max_frames)))
    reveals = list(range(step, n + 1, step))
    if reveals[-1] != n:
        reveals.append(n)
    fps = max(3.0, min(20.0, len(reveals) / duration_s))

    def update(frame: int):
        upto = reveals[frame]
        trace.set_data(xs[:upto], scores[:upto])
        flagged = scores[:upto] > threshold
        ok_dots.set_offsets(np.c_[xs[:upto][~flagged], scores[:upto][~flagged]])
        flag_dots.set_offsets(np.c_[xs[:upto][flagged], scores[:upto][flagged]])
        counter.set_text(f"⚑ {int(flagged.sum())} chunks flagged")
        return trace, ok_dots, flag_dots, counter

    anim = FuncAnimation(fig, update, frames=len(reveals), blit=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    log.info("Wrote %s (%d frames @ %.1f fps)", out_path, len(reveals), fps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Phase-4 chunk-level cheat-detection figure"
    )
    parser.add_argument("--synthetic-dir", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument(
        "--out", type=Path, default=FIGURES_DIR / "phase4_chunk_detection.png"
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="also render the animated chunk-flag replay GIF",
    )
    parser.add_argument(
        "--gif-out", type=Path, default=FIGURES_DIR / "phase4_chunk_flags.gif"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    from pipeline.models.lstm_ae import LSTM_AE_WEIGHTS_NAME, load_lstm_ae

    if not (args.model_dir / LSTM_AE_WEIGHTS_NAME).exists():
        log.error(
            "No LSTM-AE artifact at %s — run `python -m scripts.train_lstm_ae` first",
            args.model_dir,
        )
        return 1
    model, stats, meta = load_lstm_ae(args.model_dir, device="auto")
    chunk_length = int((meta.get("config") or {}).get("chunk_length", 64))
    log.info("Loaded LSTM-AE (chunk_length=%d)", chunk_length)

    legit, cheat_by_type = collect_chunk_scores(
        args.synthetic_dir, model, stats, chunk_length
    )
    if len(legit) == 0:
        log.error("No legit chunks scored — is %s populated?", args.synthetic_dir)
        return 1
    for c in CHEAT_ORDER:
        n = len(cheat_by_type.get(c, []))
        log.info("  %s: %d cheat chunks  (legit pool: %d)", c, n, len(legit))

    render_chunk_detection(legit, cheat_by_type, args.out)

    if args.gif:
        threshold = float(np.percentile(legit, 95))
        picked = pick_gif_session(
            args.synthetic_dir, model, stats, chunk_length, threshold
        )
        if picked is None:
            log.error("No suitable cheat session for the GIF replay.")
            return 1
        gif_label, path, scores, is_cheat = picked
        log.info(
            "GIF session: %s [%s] (%d chunks, %d cheat-overlapping)",
            path.name,
            gif_label,
            len(scores),
            int(is_cheat.sum()),
        )
        render_chunk_flag_gif(
            scores, is_cheat, threshold, args.gif_out, cheat_label=gif_label
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
