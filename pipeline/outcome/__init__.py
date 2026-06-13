"""
pipeline/outcome/
=================
Outcome-labelled telemetry (docs/ROADMAP.md **Phase 9**).

The *causally strongest* cheat signals — headshot ratio, damage/shot, accuracy,
view-angle aim dynamics — cannot be derived from the recorder's mouse/keyboard
stream (docs/SIGNALS.md). They live in the game's own event log. This package
parses a Counter-Strike 2 ``.dem`` (SourceTV demo) into those features, aligned
onto the **same 30 s window grid** the input-feature pipeline uses, so the two
can be joined once *dual-capture* sessions (recorder + demo, recorded
simultaneously) exist.

The one piece that genuinely needs real dual-capture data is the **clock-sync**
between demo-tick-time and recorder-time — see ``estimate_offset_by_xcorr``.
"""

from pipeline.outcome.cs2_demo import (
    CS2_DEFAULT_TICKRATE,
    FLICK_ANGVEL_DEG_S,
    DemoOutcomes,
    aggregate_outcome_windows,
    angular_speed_series,
    estimate_offset_by_xcorr,
    parse_demo_outcomes,
    recorder_mouse_speed_series,
    tick_to_seconds,
    view_angle_kinematics,
)

__all__ = [
    "CS2_DEFAULT_TICKRATE",
    "FLICK_ANGVEL_DEG_S",
    "DemoOutcomes",
    "aggregate_outcome_windows",
    "angular_speed_series",
    "estimate_offset_by_xcorr",
    "parse_demo_outcomes",
    "recorder_mouse_speed_series",
    "tick_to_seconds",
    "view_angle_kinematics",
]
