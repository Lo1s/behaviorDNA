"""
scripts/label_cheat_segments.py
===============================
Turn a session recorded *with* the live harness (``collector/cheat_sim.py``)
into a labelled cheat session the pipeline can ingest.

The harness's toggle hotkeys (F8/F9/F10) are ordinary key events, so the
recorder captures them **in-band** in the session's own event stream — no
cross-process clock sync needed. This script:

1. derives **typed** ``cheat_segments_typed`` (one ``{start_ms, end_ms, cheat}``
   per span) + the untyped ``cheat_segments`` union + ``cheat_labels`` +
   ``cheat_label`` from those toggle presses
   (``pipeline.adversarial.live_cheat.toggle_log_to_typed_segments``), then
2. **strips the toggle/quit control keys** from the events so they don't show up
   as spurious keystrokes in the features, and
3. writes the result with the same schema the synthetic generator produces, so
   ``pipeline.adversarial.benchmark`` / the LSTM-AE / streaming ingest it unchanged.

Because the type of every span is captured here, multi-cheat recordings (e.g.
aimbot → triggerbot → macro in one session) are labelled correctly with no
manual reconstruction. Pass ``--difficulty`` to tag the cheat-sim difficulty.

Usage:
    python -m scripts.label_cheat_segments data/raw/<session>.json --difficulty obvious
    python -m scripts.label_cheat_segments <in>.json --out <out>.json --keep-toggle-keys
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.adversarial.live_cheat import (
    CHEAT_AIMBOT,
    CHEAT_LEGIT,
    CHEAT_MACRO,
    CHEAT_TRIGGERBOT,
    toggle_log_to_typed_segments,
)

# Keys as the recorder logs special keys (pynput → str(key) == "Key.f8").
RECORDED_TOGGLE_KEYS = {
    "Key.f8": CHEAT_AIMBOT,
    "Key.f9": CHEAT_TRIGGERBOT,
    "Key.f10": CHEAT_MACRO,
}
QUIT_KEY = "Key.f12"
CONTROL_KEYS = set(RECORDED_TOGGLE_KEYS) | {QUIT_KEY}


def label_session(
    session: dict,
    strip_toggle_keys: bool = True,
    difficulty: str | None = None,
) -> dict:
    """Return a copy of ``session`` with typed cheat labels derived from toggles.

    Pure (no I/O). Reads the in-band F8/F9/F10 toggle presses and writes:

    - ``cheat_segments_typed`` — ``[{start_ms, end_ms, cheat}]``, one entry per
      cheat-active span **tagged with its own type** (so a recording that
      toggled aimbot → triggerbot → macro keeps all three distinctly);
    - ``cheat_segments`` — the untyped ``[[start_ms, end_ms]]`` union, kept for
      backward-compatible consumers;
    - ``cheat_labels`` — sorted unique cheat types present;
    - ``cheat_label`` — the single type if exactly one was used, else
      ``"mixed"`` (``"legit"`` if nothing was toggled). The old single-argmax
      label is intentionally dropped — it mis-described multi-cheat sessions.

    With ``strip_toggle_keys`` the control-key events are removed and
    ``event_count`` recomputed. ``difficulty`` (if given) is stored verbatim.
    """
    out = dict(session)
    events = list(session.get("events", []))
    end_ms = session.get("duration_ms") or max(
        (float(e.get("t", 0.0)) for e in events), default=0.0
    )

    typed = toggle_log_to_typed_segments(events, RECORDED_TOGGLE_KEYS, end_ms)

    if strip_toggle_keys:
        events = [e for e in events if e.get("key") not in CONTROL_KEYS]

    types = sorted({ct for _, _, ct in typed})
    out["events"] = events
    out["event_count"] = len(events)
    out["cheat_segments"] = sorted([s, e] for s, e, _ in typed)
    out["cheat_segments_typed"] = [
        {"start_ms": s, "end_ms": e, "cheat": ct} for s, e, ct in typed
    ]
    out["cheat_labels"] = types
    if not types:
        out["cheat_label"] = CHEAT_LEGIT
    elif len(types) == 1:
        out["cheat_label"] = types[0]
    else:
        out["cheat_label"] = "mixed"
    if difficulty:
        out["difficulty"] = difficulty.lower()
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
    parser.add_argument(
        "--difficulty",
        choices=["soft", "medium", "obvious"],
        default=None,
        help="cheat-sim difficulty for this recording (stored on the session)",
    )
    args = parser.parse_args(argv)

    if not args.session.exists():
        print(f"[ERROR] not found: {args.session}")
        return 1

    with open(args.session, encoding="utf-8") as f:
        session = json.load(f)

    labelled = label_session(
        session,
        strip_toggle_keys=not args.keep_toggle_keys,
        difficulty=args.difficulty,
    )
    out_path = args.out or args.session
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labelled, f, separators=(",", ":"))

    diff = labelled.get("difficulty", "-")
    print(
        f"{out_path.name}: cheat_label={labelled['cheat_label']} "
        f"types={labelled['cheat_labels']} difficulty={diff} "
        f"segments={len(labelled['cheat_segments'])} events={labelled['event_count']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
