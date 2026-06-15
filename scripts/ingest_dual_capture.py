"""
scripts/ingest_dual_capture.py
==============================
Phase 9 — ingest a **dual-capture** session (recorder JSON + CS2 ``.dem`` recorded
simultaneously) into one clock-synced, window-joined feature table:
**input features ⨝ outcome features** on ``(session_id, window_idx)``.

This is the entrypoint a real dual-capture session runs through once captured. The
recorder↔demo clock-sync is marker-free (motion cross-correlation) and
self-validating: the printed ``peak_corr`` / verdict tells you whether the
alignment (and therefore the join) can be trusted.

Usage
-----
    python -m scripts.ingest_dual_capture \
        --recorder data/raw/my_session.json \
        --demo my_match.dem \
        --player "myname" --tickrate 64 \
        --out reports/dual_capture_demo.parquet

``--player`` defaults to the most-active player (most shots). ``--tickrate`` is
64 (Valve MM) or 128 (FACEIT) — the demo header doesn't expose it reliably.
Offline only; nothing here touches a live game (docs/ETHICS.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline.outcome import CS2_DEFAULT_TICKRATE, ingest_dual_capture

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest_dual_capture")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ingest a dual-capture (recorder + CS2 demo) session into a "
        "window-joined input⨝outcome feature table."
    )
    ap.add_argument("--recorder", required=True, help="recorder session JSON")
    ap.add_argument("--demo", required=True, help="CS2 .dem recorded at the same time")
    ap.add_argument("--player", default=None, help="player name (default: most-active)")
    ap.add_argument(
        "--tickrate",
        type=float,
        default=CS2_DEFAULT_TICKRATE,
        help="64 MM / 128 FACEIT",
    )
    ap.add_argument("--grid-hz", type=float, default=16.0, help="sync resample grid")
    ap.add_argument("--out", required=True, help="output joined table (.parquet)")
    ap.add_argument(
        "--out-meta", default=None, help="write the SyncResult to this JSON (optional)"
    )
    args = ap.parse_args(argv)

    for label, p in (("recorder", args.recorder), ("demo", args.demo)):
        if not Path(p).exists():
            log.error("%s not found: %s", label, p)
            return 1

    log.info("ingesting recorder=%s + demo=%s …", args.recorder, args.demo)
    joined, sync = ingest_dual_capture(
        args.recorder,
        args.demo,
        player=args.player,
        tickrate=args.tickrate,
        grid_hz=args.grid_hz,
    )

    log.info(
        "player=%r  sync: offset=%.3fs  peak_corr=%.3f  [%s]",
        sync.player,
        sync.offset_s,
        sync.peak_corr,
        sync.verdict,
    )
    log.info(
        "windows: %d input | %d outcome | %d joined (combat)",
        sync.n_input_windows,
        sync.n_outcome_windows,
        sync.n_joined_windows,
    )
    if sync.peak_corr < 0.5:
        log.warning(
            "WEAK sync — the join may be window-shifted. Re-capture with a large "
            "deliberate flick near the start, or verify the player/tickrate."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out, index=False)
    log.info("wrote %s (%d rows × %d cols)", out, len(joined), joined.shape[1])

    if args.out_meta:
        meta = Path(args.out_meta)
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(sync.as_dict(), indent=2) + "\n")
        log.info("wrote %s", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
