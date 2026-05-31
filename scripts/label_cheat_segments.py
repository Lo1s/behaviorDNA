"""
scripts/label_cheat_segments.py
===============================
Turn a session recorded *with* the live harness (``collector/cheat_sim.py``)
into a labelled cheat session the pipeline can ingest.

The harness's toggle hotkeys (F8/F9/F10) are ordinary key events, so the
recorder captures them **in-band** in the session's own event stream — no
cross-process clock sync needed. This script:

1. derives ``cheat_label`` + ``cheat_segments`` from those toggle presses
   (``pipeline.adversarial.live_cheat.toggle_log_to_segments``), then
2. **strips the toggle/quit control keys** from the events so they don't show up
   as spurious keystrokes in the features, and
3. writes the result with the same schema the synthetic generator produces, so
   ``pipeline.adversarial.benchmark`` / the LSTM-AE / streaming ingest it unchanged.

Usage:
    python -m scripts.label_cheat_segments data/raw/<session>.json
    python -m scripts.label_cheat_segments <in>.json --out <out>.json --keep-toggle-keys
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.adversarial.live_cheat import (
    CHEAT_AIMBOT,
    CHEAT_MACRO,
    CHEAT_TRIGGERBOT,
    toggle_log_to_segments,
)

# Keys as the recorder logs special keys (pynput → str(key) == "Key.f8").
RECORDED_TOGGLE_KEYS = {
    "Key.f8": CHEAT_AIMBOT,
    "Key.f9": CHEAT_TRIGGERBOT,
    "Key.f10": CHEAT_MACRO,
}
QUIT_KEY = "Key.f12"
CONTROL_KEYS = set(RECORDED_TOGGLE_KEYS) | {QUIT_KEY}


def label_session(session: dict, strip_toggle_keys: bool = True) -> dict:
    """Return a copy of ``session`` with ``cheat_label`` + ``cheat_segments`` set.

    Pure (no I/O). Derives labels from the in-band toggle presses; with
    ``strip_toggle_keys`` the control-key events are removed and ``event_count``
    is recomputed.
    """
    out = dict(session)
    events = list(session.get("events", []))
    end_ms = session.get("duration_ms") or max(
        (float(e.get("t", 0.0)) for e in events), default=0.0
    )

    label, segments = toggle_log_to_segments(events, RECORDED_TOGGLE_KEYS, end_ms)

    if strip_toggle_keys:
        events = [e for e in events if e.get("key") not in CONTROL_KEYS]

    out["events"] = events
    out["event_count"] = len(events)
    out["cheat_label"] = label
    out["cheat_segments"] = segments
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label a harness-recorded session")
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: in place")
    parser.add_argument(
        "--keep-toggle-keys",
        action="store_true",
        help="do not strip the F8/F9/F10/F12 control-key events",
    )
    args = parser.parse_args(argv)

    if not args.session.exists():
        print(f"[ERROR] not found: {args.session}")
        return 1

    with open(args.session, encoding="utf-8") as f:
        session = json.load(f)

    labelled = label_session(session, strip_toggle_keys=not args.keep_toggle_keys)
    out_path = args.out or args.session
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labelled, f, separators=(",", ":"))

    print(
        f"{out_path.name}: cheat_label={labelled['cheat_label']} "
        f"segments={len(labelled['cheat_segments'])} events={labelled['event_count']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
