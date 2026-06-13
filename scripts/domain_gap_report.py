"""
scripts/domain_gap_report.py
============================
Phase 8 — quantify the **captcha → game** domain gap before claiming transfer.

Encodes all three corpora into the shared 8-D event-tensor schema, then runs the
project's existing KS + PSI drift tooling (``pipeline.monitoring.drift``) on the
per-channel distributions, comparing the pretraining corpus (CaptchaSolve30k)
against each fine-tuning target (CS2CD legit, GTA legit). De-risks the transfer
result: a large gap predicts limited transfer (an honest finding either way).

Writes ``reports/pretraining_domain_gap.json`` + ``reports/figures/phase8_domain_gap.png``.

Usage:
    python -m scripts.domain_gap_report [--max-captcha 4000]
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
OUT_JSON = ROOT / "reports" / "pretraining_domain_gap.json"
OUT_FIG = ROOT / "reports" / "figures" / "phase8_domain_gap.png"


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


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 8 captcha→game domain-gap report"
    )
    parser.add_argument("--max-captcha", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    log.info("Encoding corpora to 8-D…")
    captcha = channel_summary_frame(
        captcha_to_tensors(max_sessions=args.max_captcha, seed=args.seed)
    )
    cs2_legit = channel_summary_frame(
        [t for lab, t in cs2cd_to_tensors_8d() if lab == 0]
    )
    gta = channel_summary_frame(_gta_legit_tensors())
    log.info(
        "rows — captcha=%d cs2cd=%d gta=%d", len(captcha), len(cs2_legit), len(gta)
    )

    report = {}
    for name, cur in [("captcha_vs_cs2cd", cs2_legit), ("captcha_vs_gta", gta)]:
        if cur.empty:
            log.warning("%s: target corpus empty — skipping", name)
            continue
        df = compute_drift_report(captcha, cur, DRIFT_CHANNELS)
        report[name] = df.to_dict(orient="records")
        log.info("\n%s\n%s", name, df.to_string(index=False))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    _render_figure(report)
    log.info("Wrote %s and %s", OUT_JSON, OUT_FIG)
    return 0


def _render_figure(report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    comps = [c for c in ("captcha_vs_cs2cd", "captcha_vs_gta") if c in report]
    if not comps:
        return
    x = np.arange(len(DRIFT_CHANNELS))
    w = 0.8 / len(comps)
    colors = {"captcha_vs_cs2cd": "#4c78a8", "captcha_vs_gta": "#e94560"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(comps):
        by_feat = {r["feature"]: r["psi"] for r in report[c]}
        vals = [by_feat.get(f, float("nan")) for f in DRIFT_CHANNELS]
        ax.bar(
            x + (i - (len(comps) - 1) / 2) * w,
            vals,
            w,
            label=c.replace("_", " "),
            color=colors.get(c),
        )
    ax.axhline(0.25, color="#8892a4", ls="--", lw=1, label="PSI 0.25 (significant)")
    ax.axhline(0.10, color="#c0c6d0", ls=":", lw=1, label="PSI 0.10 (moderate)")
    ax.set_xticks(x)
    ax.set_xticklabels(DRIFT_CHANNELS, rotation=20, ha="right")
    ax.set_ylabel("PSI (vs CaptchaSolve30k pretraining corpus)")
    ax.set_title("Phase 8 — captcha→game domain gap (per 8-D channel)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run())
