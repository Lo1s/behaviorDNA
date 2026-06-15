"""
tests/test_dual_capture.py
==========================
Unit + integration tests for the Phase 9 dual-capture ingest
(``pipeline/outcome/dual_capture.py``).

Most tests run on **synthetic** frames + a synthetic recorder dict — no
``demoparser2`` and no ``.dem`` needed (``ingest_dual_capture`` accepts a
pre-parsed ``DemoOutcomes``). The thing that must be right is the
**window-grid alignment**: the input side and the outcome side must count 30 s
windows from the *same* anchor so the ``(session_id, window_idx)`` join is exact —
including when the recorder session opens with a non-mouse event before the first
mouse-move. One integration test runs the full path against the real public demo
when it (and the native parser) are present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.constants import WINDOW_MS
from pipeline.features.run import OUTCOME_FEATURE_COLS
from pipeline.outcome import DemoOutcomes, ingest_dual_capture
from pipeline.outcome.dual_capture import (
    _recorder_anchor_ms,
    join_input_outcome,
    recorder_input_windows,
)

TR = 64.0
GRID_HZ = 16.0


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _make_outcomes(*, kills=None, hurts=None, fires=None, angles=None, tickrate=TR):
    kills = pd.DataFrame(
        kills or [],
        columns=["tick", "attacker_name", "user_name", "headshot", "hitgroup"],
    )
    hurts = pd.DataFrame(
        hurts or [], columns=["tick", "attacker_name", "dmg_health", "hitgroup"]
    )
    fires = pd.DataFrame(fires or [], columns=["tick", "user_name", "weapon"])
    angles = (
        angles
        if angles is not None
        else pd.DataFrame([], columns=["name", "tick", "pitch", "yaw"])
    )
    players = sorted(set(angles["name"]) | {"P"}) if len(angles) else ["P"]
    return DemoOutcomes(kills, hurts, fires, angles, tickrate, players)


def _dual_capture_fixture(*, offset_s=5.0, seconds=70.0, anchor_pre_ms=0.0, seed=0):
    """Build a self-consistent (recorder dict, DemoOutcomes) pair.

    Recorder mouse speed and demo view-angle speed both track one structured
    profile, so the cross-correlation locks on; the demo clock leads the recorder
    by ``offset_s``. Combat (3 shots, 1 headshot kill, 1 hit) is placed in recorder
    window 1 (30-60 s). ``anchor_pre_ms`` optionally prepends a key-press before the
    first mouse-move so the window anchor (min event t) precedes it.
    """
    rng = np.random.default_rng(seed)
    dt_s = 1.0 / GRID_HZ
    n = int(seconds * GRID_HZ)
    speed = (
        60.0
        + 120.0 * np.abs(rng.normal(size=n))
        + 40.0 * np.abs(np.sin(np.arange(n) * 0.15))
    )

    # recorder: one mouse-move per grid step, dx encodes speed (px in dt → px/s).
    # anchor_pre_ms optionally prepends a key-press at t=0 so the window anchor
    # (min event t = 0) precedes the first mouse-move (which starts at mouse_start).
    events = []
    if anchor_pre_ms:
        events.append({"type": "key_press", "t": 0.0, "key": "w"})
    mouse_start_ms = float(anchor_pre_ms)
    mouse_start_s = mouse_start_ms / 1000.0
    for k in range(n):
        events.append(
            {
                "type": "mouse_move",
                "t": mouse_start_ms + k * (1000.0 / GRID_HZ),
                "dx": int(round(speed[k] * dt_s)),
                "dy": 0,
            }
        )
    recorder = {
        "session_id": "dc_test",
        "sensitivity": 1.0,
        "dpi": 800,
        "polling_rate": 1000,
        "events": events,
    }

    # demo angles: yaw increments track the SAME speed; demo leads the recorder
    # ANCHOR (t=0) by offset_s, so the motion sample at recorder-anchor time
    # (mouse_start_s + k*dt) sits at demo time (mouse_start_s + k*dt + offset_s).
    yaw = 0.0
    rows = []
    for k in range(n):
        demo_time = mouse_start_s + k * dt_s + offset_s
        yaw += speed[k] * dt_s
        rows.append(("P", int(round(demo_time * TR)), 0.0, yaw))
    angles = pd.DataFrame(rows, columns=["name", "tick", "pitch", "yaw"])

    # combat in recorder window 1 (rec seconds 30-60) → demo tick (rec+offset)*TR
    def _tick(rec_s):
        return int(round((rec_s + offset_s) * TR))

    fires = [(_tick(s), "P", "ak47") for s in (35.0, 40.0, 45.0)]
    kills = [(_tick(40.0), "P", "V", True, "head")]
    hurts = [(_tick(35.0), "P", 50.0, "head")]
    outcomes = _make_outcomes(angles=angles, fires=fires, kills=kills, hurts=hurts)
    return recorder, outcomes


# --------------------------------------------------------------------------- #
# input-feature side + anchor
# --------------------------------------------------------------------------- #
def test_recorder_input_windows_grid_and_schema():
    recorder, _ = _dual_capture_fixture()
    df = recorder_input_windows(recorder)
    assert {"session_id", "window_idx"}.issubset(df.columns)
    assert df.shape[1] > 5  # input features present
    assert (df["session_id"] == "dc_test").all()
    # 70 s of motion → windows 0,1,2 on the 30 s grid
    assert sorted(df["window_idx"].unique().tolist()) == [0, 1, 2]


def test_anchor_is_min_event_not_first_mousemove():
    # key-press at t=0, mouse-moves start at t=20 s → anchor must be 0, not 20 s
    recorder, _ = _dual_capture_fixture(anchor_pre_ms=20_000.0)
    assert _recorder_anchor_ms(recorder, "dc_test") == 0.0


# --------------------------------------------------------------------------- #
# join semantics
# --------------------------------------------------------------------------- #
def test_join_left_fills_missing_outcome_with_zero():
    in_df = pd.DataFrame(
        {"session_id": ["s", "s"], "window_idx": [0, 1], "feat_a": [1.0, 2.0]}
    )
    out_df = pd.DataFrame(
        [{"session_id": "s", "window_idx": 1, **{c: 3.0 for c in OUTCOME_FEATURE_COLS}}]
    )
    joined = join_input_outcome(in_df, out_df)
    assert len(joined) == 2
    w0 = joined[joined.window_idx == 0].iloc[0]
    w1 = joined[joined.window_idx == 1].iloc[0]
    assert not w0["has_outcome"] and w0["shots_fired"] == 0.0  # filled
    assert w1["has_outcome"] and w1["kills"] == 3.0
    assert "feat_a" in joined.columns  # input columns preserved


def test_join_empty_input_returns_schema():
    out_df = pd.DataFrame(
        [{"session_id": "s", "window_idx": 0, **{c: 1.0 for c in OUTCOME_FEATURE_COLS}}]
    )
    joined = join_input_outcome(
        pd.DataFrame(columns=["session_id", "window_idx"]), out_df
    )
    assert len(joined) == 0
    assert "has_outcome" in joined.columns


# --------------------------------------------------------------------------- #
# full ingest (synthetic, no demoparser2)
# --------------------------------------------------------------------------- #
def test_ingest_dual_capture_aligns_and_joins():
    offset = 5.0
    recorder, outcomes = _dual_capture_fixture(offset_s=offset)
    joined, sync = ingest_dual_capture(recorder, outcomes, player="P", grid_hz=GRID_HZ)

    # sync recovered + self-validated
    assert sync.peak_corr > 0.5 and sync.verdict.startswith("STRONG")
    assert abs(sync.offset_s - offset) <= 3.0 / GRID_HZ  # within a few grid samples

    # schema: input cols + outcome cols + join key + flag
    for c in OUTCOME_FEATURE_COLS:
        assert c in joined.columns
    assert {"session_id", "window_idx", "has_outcome"}.issubset(joined.columns)

    # combat landed in window 1 (the alignment payoff)
    w1 = joined[joined.window_idx == 1].iloc[0]
    assert w1["has_outcome"]
    assert w1["shots_fired"] == 3
    assert w1["kills"] == 1
    assert w1["headshot_ratio"] == 1.0  # the one hit was a headshot

    # no shots outside window 1
    assert joined.loc[joined.window_idx != 1, "shots_fired"].sum() == 0


def test_ingest_alignment_survives_pre_mousemove_anchor():
    # the anchor fix: a key-press 20 s before the first mouse-move must NOT shift
    # the outcome grid relative to the input grid → combat still lands in window 1
    offset = 4.0
    recorder, outcomes = _dual_capture_fixture(offset_s=offset, anchor_pre_ms=20_000.0)
    joined, sync = ingest_dual_capture(recorder, outcomes, player="P", grid_hz=GRID_HZ)
    assert sync.peak_corr > 0.5
    w1 = joined[joined.window_idx == 1]
    assert len(w1) == 1 and w1.iloc[0]["shots_fired"] == 3


def test_ingest_weak_sync_flagged():
    # recorder motion uncorrelated with the demo angle motion → WEAK verdict
    recorder, outcomes = _dual_capture_fixture(seed=1)
    rng = np.random.default_rng(99)
    for e in recorder["events"]:
        if e["type"] == "mouse_move":
            e["dx"] = int(rng.integers(-50, 50))  # scramble → destroy correlation
    _joined, sync = ingest_dual_capture(recorder, outcomes, player="P", grid_hz=GRID_HZ)
    assert sync.peak_corr < 0.5
    assert sync.verdict.startswith("WEAK")
    assert WINDOW_MS == 30_000  # guards the grid constant the join assumes


# --------------------------------------------------------------------------- #
# real public demo (skipped without the .dem + native parser)
# --------------------------------------------------------------------------- #
def _real_demo_available():
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data/external/cs2_demo/test_demo.dem"
    if not p.exists():
        return None
    try:
        import demoparser2  # noqa: F401
    except Exception:
        return None
    return p


@pytest.mark.skipif(
    _real_demo_available() is None, reason="public .dem / demoparser2 not present"
)
def test_ingest_on_real_demo_with_synthetic_recorder():
    """Full path on the real demo: parse it, synthesise a recorder mirroring its own
    view-angle motion, and confirm the ingest aligns (high peak_corr) and produces a
    populated joined table with the outcome columns."""
    from pipeline.outcome import angular_speed_series, parse_demo_outcomes
    from pipeline.outcome.dual_capture import most_active_player

    dem = _real_demo_available()
    outcomes = parse_demo_outcomes(str(dem), tickrate=TR)
    player = most_active_player(outcomes)
    _t_demo, v_demo = angular_speed_series(outcomes, player, grid_hz=GRID_HZ)
    assert v_demo.size > GRID_HZ * 30  # enough motion to align

    # recorder mouse motion mirrors the demo's view-angle motion on a 0-based clock
    events = [
        {
            "type": "mouse_move",
            "t": k * (1000.0 / GRID_HZ),
            "dx": int(round(float(v) * 0.05)),
            "dy": 0,
        }
        for k, v in enumerate(v_demo)
    ]
    recorder = {"session_id": "real_demo_synth_rec", "events": events}

    joined, sync = ingest_dual_capture(
        recorder, outcomes, player=player, grid_hz=GRID_HZ
    )
    assert sync.peak_corr > 0.5  # real-demo motion aligns to the synthetic recorder
    assert len(joined) > 0
    assert set(OUTCOME_FEATURE_COLS).issubset(joined.columns)
