"""
scripts/parse_cs2_demo.py
=========================
Phase 9 feasibility-spike CLI (docs/ROADMAP.md): turn a Counter-Strike 2
``.dem`` into per-window **outcome features** (:data:`OUTCOME_FEATURE_COLS`),
and — when given a simultaneously-recorded recorder session — estimate the
demo<->recorder **clock offset** by motion cross-correlation.

The spike question this answers: *can demoparser2 pull per-tick view-angles +
damage/kill events from a demo, and clock-sync it to a recorder run?* The
extraction half is validated against a real public demo; the sync half is a
marker-free cross-correlation whose ``peak_corr`` self-reports whether it worked.

Usage
-----
    # parse only (windows are demo-relative; pick the most-active player):
    python -m scripts.parse_cs2_demo --demo data/external/cs2_demo/test_demo.dem

    # name a player + write the per-window table:
    python -m scripts.parse_cs2_demo --demo my_match.dem --player "myname" \
        --out reports/outcome_demo.json

    # dual-capture: sync to the recorder session recorded at the same time:
    python -m scripts.parse_cs2_demo --demo my_match.dem --player "myname" \
        --recorder data/raw/my_session.json --out reports/outcome_demo.json

Nothing here is online or touches a live game — it parses a recorded file
offline (docs/ETHICS.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from pipeline.outcome import (
    CS2_DEFAULT_TICKRATE,
    aggregate_outcome_windows,
    angular_speed_series,
    estimate_offset_by_xcorr,
    parse_demo_outcomes,
    recorder_mouse_speed_series,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parse_cs2_demo")


def _most_active_player(outcomes) -> str:
    """Player with the most shots fired (the human behind the recorder, usually)."""
    fires = outcomes.fires
    if "user_name" in fires.columns and len(fires):
        counts = fires["user_name"].value_counts()
        if len(counts):
            return str(counts.index[0])
    return outcomes.players[0] if outcomes.players else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse a CS2 demo into per-window outcome features."
    )
    ap.add_argument("--demo", required=True, help="path to a CS2 .dem file")
    ap.add_argument("--player", default=None, help="player name (default: most-active)")
    ap.add_argument(
        "--tickrate",
        type=float,
        default=CS2_DEFAULT_TICKRATE,
        help="server tickrate (64 MM / 128 FACEIT)",
    )
    ap.add_argument(
        "--recorder",
        default=None,
        help="recorder session JSON recorded at the same time (enables clock-sync)",
    )
    ap.add_argument(
        "--grid-hz",
        type=float,
        default=16.0,
        help="resample grid for the sync cross-correlation",
    )
    ap.add_argument(
        "--out", default=None, help="write the per-window table + summary to this JSON"
    )
    args = ap.parse_args(argv)

    if not Path(args.demo).exists():
        log.error("demo not found: %s", args.demo)
        return 1

    log.info("parsing %s (tickrate=%g) …", args.demo, args.tickrate)
    outcomes = parse_demo_outcomes(args.demo, tickrate=args.tickrate)
    log.info(
        "players=%d  kills=%d  hurts=%d  fires=%d  angle-ticks=%d",
        len(outcomes.players),
        len(outcomes.kills),
        len(outcomes.hurts),
        len(outcomes.fires),
        outcomes.angles["tick"].nunique() if len(outcomes.angles) else 0,
    )

    player = args.player or _most_active_player(outcomes)
    log.info("player = %r", player)

    # --- clock-sync (only with a recorder session) ---
    offset_s = 0.0
    sync = None
    if args.recorder:
        rec = json.loads(Path(args.recorder).read_text())
        events = rec.get("events", [])
        t_demo, v_demo = angular_speed_series(outcomes, player, grid_hz=args.grid_hz)
        t_rec, v_rec = recorder_mouse_speed_series(events, grid_hz=args.grid_hz)
        sync = estimate_offset_by_xcorr(
            t_demo, v_demo, t_rec, v_rec, grid_hz=args.grid_hz
        )
        offset_s = sync["offset_s"]
        verdict = "STRONG" if sync["peak_corr"] >= 0.5 else "WEAK — do not trust"
        log.info(
            "sync: offset=%.3fs  peak_corr=%.3f  [%s]",
            sync["offset_s"],
            sync["peak_corr"],
            verdict,
        )

    windows = aggregate_outcome_windows(outcomes, player, offset_s=offset_s)
    log.info("emitted %d outcome windows", len(windows))
    if len(windows):
        head = windows.head(8).to_string(index=False)
        log.info("first windows:\n%s", head)
        # quick sanity: a high-skill window's headshot ratio / accuracy
        agg = {
            "kills": int(windows["kills"].sum()),
            "deaths": int(windows["deaths"].sum()),
            "shots_fired": int(windows["shots_fired"].sum()),
            "hits_dealt": int(windows["hits_dealt"].sum()),
            "session_headshot_ratio": round(
                float(
                    np.average(
                        windows["headshot_ratio"],
                        weights=windows["hits_dealt"].clip(lower=1),
                    )
                ),
                3,
            ),
            "session_accuracy": round(
                _safe_div(windows["hits_dealt"].sum(), windows["shots_fired"].sum()), 3
            ),
        }
        log.info("session totals: %s", agg)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "demo": str(args.demo),
            "player": player,
            "tickrate": args.tickrate,
            "n_windows": int(len(windows)),
            "sync": sync,
            "windows": windows.to_dict(orient="records"),
        }
        out.write_text(json.dumps(payload, indent=2))
        log.info("wrote %s", out)
    return 0


def _safe_div(a, b) -> float:
    return float(a) / float(b) if b else 0.0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
