"""
tests/test_outcome.py
=====================
Unit tests for the Phase 9 outcome-telemetry spike (pipeline/outcome/cs2_demo.py).

These run on **synthetic** pandas frames — no ``demoparser2`` native dep and no
``.dem`` file are needed (only ``parse_demo_outcomes`` touches those, and it is
exercised against a real demo manually, see docs/CHEAT_DATA_COLLECTION.md). The
two things that must be right are tested here: the per-window aggregation onto the
``WINDOW_MS`` grid, and the cross-correlation clock-sync (offset recovery +
self-validation against mismatched signals).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.constants import WINDOW_MS
from pipeline.features.run import OUTCOME_FEATURE_COLS
from pipeline.outcome import (
    DemoOutcomes,
    aggregate_outcome_windows,
    estimate_offset_by_xcorr,
    recorder_mouse_speed_series,
    tick_to_seconds,
    view_angle_kinematics,
)
from pipeline.outcome.cs2_demo import FLICK_ANGVEL_DEG_S, _resample_uniform, _wrap_deg

TR = 64.0  # tickrate used throughout


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _angles(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["name", "tick", "pitch", "yaw"])


def _make_outcomes(
    *, kills=None, hurts=None, fires=None, angles=None, tickrate=TR
) -> DemoOutcomes:
    kills = pd.DataFrame(
        kills or [],
        columns=["tick", "attacker_name", "user_name", "headshot", "hitgroup"],
    )
    hurts = pd.DataFrame(
        hurts or [], columns=["tick", "attacker_name", "dmg_health", "hitgroup"]
    )
    fires = pd.DataFrame(fires or [], columns=["tick", "user_name", "weapon"])
    angles = angles if angles is not None else _angles([])
    players = sorted(set(angles["name"]) | {"P"}) if len(angles) else ["P"]
    return DemoOutcomes(
        kills=kills,
        hurts=hurts,
        fires=fires,
        angles=angles,
        tickrate=tickrate,
        players=players,
    )


# --------------------------------------------------------------------------- #
# tick / angle math
# --------------------------------------------------------------------------- #
def test_tick_to_seconds_scalar_and_array():
    assert tick_to_seconds(64, TR) == 1.0
    np.testing.assert_allclose(tick_to_seconds([0, 64, 128], TR), [0.0, 1.0, 2.0])


def test_wrap_deg_wraps_across_180():
    # 179 -> -179 is a 2 deg move, not 358; the boundary wraps to -180 (== +180)
    np.testing.assert_allclose(
        _wrap_deg([2.0, 358.0, -358.0, 180.0]), [2.0, -2.0, 2.0, -180.0]
    )


def test_view_angle_kinematics_known_velocity():
    # one player, yaw advances 10 deg every 64 ticks (=1 s) -> 10 deg/s
    rows = [("P", t, 0.0, 10.0 * i) for i, t in enumerate(range(1, 1 + 64 * 5, 64))]
    out = view_angle_kinematics(_angles(rows), "P", TR)
    assert out["angvel"].size == 4
    np.testing.assert_allclose(out["angvel"], 10.0, rtol=1e-6)


def test_view_angle_kinematics_uses_wrapped_yaw():
    # yaw 179 -> -179 over 1 s is a 2 deg/s move, not 358
    rows = [("P", 1, 0.0, 179.0), ("P", 1 + 64, 0.0, -179.0)]
    out = view_angle_kinematics(_angles(rows), "P", TR)
    np.testing.assert_allclose(out["angvel"], 2.0, rtol=1e-6)


# --------------------------------------------------------------------------- #
# per-window aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_columns_match_contract():
    out = _make_outcomes(fires=[(1, "P", "ak47")])
    df = aggregate_outcome_windows(out, "P")
    assert list(df.columns) == ["session_id", "window_idx", *OUTCOME_FEATURE_COLS]


def test_aggregate_empty_returns_empty_with_schema():
    df = aggregate_outcome_windows(_make_outcomes(), "P")
    assert len(df) == 0
    assert list(df.columns) == ["session_id", "window_idx", *OUTCOME_FEATURE_COLS]


def test_aggregate_basic_combat_stats():
    # window 0 (ticks 0..30s*64). 4 shots, 2 hits (one head, 75 dmg total), 1 headshot kill.
    sec = int(64)
    out = _make_outcomes(
        fires=[(t, "P", "ak47") for t in (sec, 2 * sec, 3 * sec, 4 * sec)],
        hurts=[(sec, "P", 50.0, "head"), (2 * sec, "P", 25.0, "chest")],
        kills=[(sec, "P", "V", True, "head")],
    )
    df = aggregate_outcome_windows(out, "P").set_index("window_idx")
    row = df.loc[0]
    assert row["shots_fired"] == 4
    assert row["hits_dealt"] == 2
    assert row["kills"] == 1
    assert row["damage_dealt"] == 75.0
    assert row["accuracy"] == 0.5  # 2/4
    assert row["damage_per_shot"] == 75.0 / 4
    assert row["headshot_ratio"] == 0.5  # 1 head hit of 2 hits
    assert row["kills_per_shot"] == 0.25


def test_aggregate_window_binning_and_offset():
    # an event at demo t = 40 s lands in window 1 (30 s windows) with offset 0,
    # but in window 0 once we shift recorder t=0 to demo t=15 s (40-15=25 < 30).
    tick_40s = int(40 * TR)
    out = _make_outcomes(fires=[(tick_40s, "P", "ak47")])
    assert int(aggregate_outcome_windows(out, "P")["window_idx"].iloc[0]) == 1
    shifted = aggregate_outcome_windows(out, "P", offset_s=15.0)
    assert int(shifted["window_idx"].iloc[0]) == 0
    assert WINDOW_MS == 30_000  # guards the constant the binning assumes


def test_aggregate_flick_count_uses_threshold():
    # one fast move above the flick threshold, one slow move below it, same window
    big = (FLICK_ANGVEL_DEG_S + 200.0) / TR  # deg moved in 1 tick to exceed threshold
    rows = [("P", 1, 0.0, 0.0), ("P", 2, 0.0, big), ("P", 2 + 64, 0.0, big + 1.0)]
    out = _make_outcomes(angles=_angles(rows))
    df = aggregate_outcome_windows(out, "P")
    assert int(df["flick_count"].sum()) == 1


# --------------------------------------------------------------------------- #
# clock-sync
# --------------------------------------------------------------------------- #
def test_resample_uniform_bins_by_mean():
    t = np.array([0.0, 0.04, 0.5, 1.0])
    v = np.array([2.0, 4.0, 9.0, 1.0])
    grid, out = _resample_uniform(t, v, grid_hz=2.0)  # 0.5 s bins
    assert out[0] == 3.0  # mean(2,4) in bin [0,0.5)


def test_estimate_offset_recovers_known_lag():
    rng = np.random.default_rng(0)
    grid_hz = 16.0
    n = 600
    t = np.arange(n) / grid_hz
    v = np.abs(rng.normal(size=n)) + np.sin(t)  # some structure
    start = 80  # recorder starts 80 samples (5 s) into the demo signal
    t_rec = np.arange(n - start) / grid_hz
    v_rec = v[start:] * 1.5 + rng.normal(0, 0.02, n - start)  # scaled + noisy
    res = estimate_offset_by_xcorr(t, v, t_rec, v_rec, grid_hz=grid_hz)
    assert res["peak_corr"] > 0.9
    assert abs(res["offset_s"] - start / grid_hz) <= 1.0 / grid_hz  # within one sample


def test_estimate_offset_rejects_unrelated_signals():
    rng = np.random.default_rng(1)
    grid_hz = 16.0
    a = np.abs(rng.normal(size=500))
    b = np.abs(rng.normal(size=500))  # independent -> no real alignment
    res = estimate_offset_by_xcorr(
        np.arange(500) / grid_hz, a, np.arange(500) / grid_hz, b, grid_hz=grid_hz
    )
    assert res["peak_corr"] < 0.5  # self-validation flags the mismatch


def test_estimate_offset_handles_empty():
    res = estimate_offset_by_xcorr(
        np.array([]), np.array([]), np.array([1.0, 2.0]), np.array([1.0, 2.0])
    )
    assert res == {"offset_s": 0.0, "peak_corr": 0.0, "lag_samples": 0}


def test_recorder_mouse_speed_series_from_events():
    # two moves 100 px apart, 0.5 s apart -> 200 px/s; non-move events ignored
    events = [
        {"type": "mouse_move", "t": 0.0, "dx": 0, "dy": 0},
        {"type": "key_press", "t": 250.0},
        {"type": "mouse_move", "t": 500.0, "dx": 100, "dy": 0},
    ]
    t, v = recorder_mouse_speed_series(events, grid_hz=8.0)
    assert v.size > 0
    assert np.isclose(np.max(v), 200.0)
