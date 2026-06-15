"""
scripts/domain_gap_report.py
============================
Phase 8 / 8.1 — quantify the pretraining→game **domain gap** before claiming
transfer.

Encodes the corpora into the shared 8-D event-tensor schema and runs the
project's KS + PSI drift tooling (``pipeline.monitoring.drift``) on the
per-channel distributions, relative to a chosen **reference** corpus:

  * ``--reference captcha`` (Phase 8, default): CaptchaSolve30k vs {CS2CD, GTA}
    → ``reports/pretraining_domain_gap.json``.
  * ``--reference cs2cd`` (Phase 8.1): in-domain CS2CD-legit vs {GTA, captcha}
    → ``reports/pretraining_domain_gap_cs2cd_ref.json``. Quantifies whether
    in-domain pretraining is actually *closer* to GTA than captcha was — read
    this **before** interpreting the 8.1 transfer result. (The CS2CD channel
    distribution is the same whether drawn from the balanced sample or the full
    release — it's the per-tick usercmd motion — so the balanced adapter is a
    faithful, cheap reference here.)

Usage:
    python -m scripts.domain_gap_report                      # captcha reference (Phase 8)
    python -m scripts.domain_gap_report --reference cs2cd    # CS2CD reference (Phase 8.1)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline.ingestion.run import _is_cheat_session
from pipeline.monitoring.drift import compute_drift_report
from pipeline.pretraining.corpora import (
    DRIFT_CHANNELS,
    captcha_to_tensors,
    channel_summary_frame,
    cs2cd_to_tensors_8d,
)
from pipeline.sequences.preprocessing import session_to_event_tensor

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# (json, figure) per reference corpus — separate paths so neither clobbers the other.
OUT = {
    "captcha": (
        ROOT / "reports" / "pretraining_domain_gap.json",
        ROOT / "reports" / "figures" / "phase8_domain_gap.png",
    ),
    "cs2cd": (
        ROOT / "reports" / "pretraining_domain_gap_cs2cd_ref.json",
        ROOT / "reports" / "figures" / "phase8_1_domain_gap_cs2cd_ref.png",
    ),
}
# reference → ordered list of (comparison-name, target-corpus-key)
COMPARISONS = {
    "captcha": [("captcha_vs_cs2cd", "cs2cd"), ("captcha_vs_gta", "gta")],
    "cs2cd": [("cs2cd_vs_gta", "gta"), ("cs2cd_vs_captcha", "captcha")],
}


def _gta_legit_tensors() -> list:
    import json as _json

    out = []
    for p in sorted(RAW.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            d = _json.load(f)
        if _is_cheat_session(d):
            continue
        t = session_to_event_tensor(d)
        if len(t):
            out.append(t)
    return out


def _frame(key: str, args):
    """Channel-summary frame for a corpus key (raises FileNotFoundError if absent)."""
    if key == "captcha":
        return channel_summary_frame(
            captcha_to_tensors(max_sessions=args.max_captcha, seed=args.seed)
        )
    if key == "cs2cd":
        return channel_summary_frame(
            [t for lab, t in cs2cd_to_tensors_8d() if lab == 0]
        )
    if key == "gta":
        return channel_summary_frame(_gta_legit_tensors())
    raise ValueError(f"unknown corpus key {key}")


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretraining→game domain-gap report")
    parser.add_argument("--reference", choices=["captcha", "cs2cd"], default="captcha")
    parser.add_argument("--max-captcha", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    ref = args.reference
    comps = COMPARISONS[ref]
    needed = {ref} | {tgt for _, tgt in comps}

    log.info("Encoding corpora to 8-D (reference=%s)…", ref)
    frames: dict = {}
    for key in needed:
        try:
            frames[key] = _frame(key, args)
            log.info("  %-8s rows=%d", key, len(frames[key]))
        except FileNotFoundError as exc:
            log.warning("  %-8s unavailable — %s", key, exc)
            frames[key] = None

    ref_frame = frames.get(ref)
    if ref_frame is None or ref_frame.empty:
        log.error("reference corpus %s unavailable — cannot compute gap", ref)
        return 1

    report = {}
    for name, tgt in comps:
        cur = frames.get(tgt)
        if cur is None or cur.empty:
            log.warning("%s: target %s unavailable — skipping", name, tgt)
            continue
        df = compute_drift_report(ref_frame, cur, DRIFT_CHANNELS)
        report[name] = df.to_dict(orient="records")
        log.info("\n%s\n%s", name, df.to_string(index=False))

    out_json, out_fig = OUT[ref]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    _render_figure(report, [c[0] for c in comps], ref, out_fig)
    log.info("Wrote %s and %s", out_json, out_fig)
    return 0


def _render_figure(
    report: dict, comp_order: list[str], ref: str, out_fig: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    comps = [c for c in comp_order if c in report]
    if not comps:
        return
    x = np.arange(len(DRIFT_CHANNELS))
    w = 0.8 / len(comps)
    palette = ["#4c78a8", "#e94560", "#54a24b", "#b279a2"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(comps):
        by_feat = {r["feature"]: r["psi"] for r in report[c]}
        vals = [by_feat.get(f, float("nan")) for f in DRIFT_CHANNELS]
        ax.bar(
            x + (i - (len(comps) - 1) / 2) * w,
            vals,
            w,
            label=c.replace("_", " "),
            color=palette[i % len(palette)],
        )
    ax.axhline(0.25, color="#8892a4", ls="--", lw=1, label="PSI 0.25 (significant)")
    ax.axhline(0.10, color="#c0c6d0", ls=":", lw=1, label="PSI 0.10 (moderate)")
    ax.set_xticks(x)
    ax.set_xticklabels(DRIFT_CHANNELS, rotation=20, ha="right")
    ax.set_ylabel(f"PSI (vs {ref} reference corpus)")
    phase = "8" if ref == "captcha" else "8.1"
    ax.set_title(f"Phase {phase} — {ref}→game domain gap (per 8-D channel)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run())
