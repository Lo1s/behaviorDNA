"""
tests/test_features.py
======================
Unit tests for pipeline/features/run.py
"""

import math

import pandas as pd

from pipeline.features.run import (
    FEATURE_COLS,
    REFERENCE_POLLING_RATE,
    compute_keyboard_patterns,
    compute_keystroke_periodicity,
    compute_mouse_kinematics,
    compute_reaction_features,
    compute_session_aggregates,
    compute_trajectory_features,
    polling_rate_norm,
    process_session_windows,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_mm(n=10, t_start=0.0, dx=5, dy=0, x0=100, y0=200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": [t_start + i * 10.0 for i in range(n)],
            "event_type": "mouse_move",
            "x": [x0 + i * dx for i in range(n)],
            "y": y0,
            "dx": dx,
            "dy": dy,
            "pressed": None,
            "key": None,
        }
    )


def make_clicks(times: list, pressed=True) -> pd.DataFrame:
    return pd.DataFrame(
        {"t": times, "event_type": "mouse_click", "pressed": pressed, "x": 0, "y": 0}
    )


def make_key_events(keys: list, times: list, event_type: str) -> pd.DataFrame:
    return pd.DataFrame({"t": times, "event_type": event_type, "key": keys})


def make_window_df(mm=None, mc=None, kp=None, kr=None) -> pd.DataFrame:
    parts = [df for df in [mm, mc, kp, kr] if df is not None and not df.empty]
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(
        columns=["t", "event_type", "x", "y", "dx", "dy", "pressed", "key"]
    )


EMPTY_MC = pd.DataFrame(columns=["t", "event_type", "pressed", "x", "y"])
EMPTY_KP = pd.DataFrame(columns=["t", "event_type", "key"])
EMPTY_KR = pd.DataFrame(columns=["t", "event_type", "key"])


# ---------------------------------------------------------------------------
# TestComputeMouseKinematics
# ---------------------------------------------------------------------------


class TestComputeMouseKinematics:
    def test_empty_mm_all_nan(self):
        empty_mm = pd.DataFrame(columns=["t", "event_type", "x", "y", "dx", "dy"])
        r = compute_mouse_kinematics(empty_mm, EMPTY_MC, norm_factor=1.0)
        for k in ("speed_mean", "speed_std", "accel_mean", "accel_std", "jitter"):
            assert math.isnan(r[k]), f"{k} should be NaN"

    def test_single_mm_all_nan(self):
        r = compute_mouse_kinematics(make_mm(n=1), EMPTY_MC, norm_factor=1.0)
        assert math.isnan(r["speed_mean"])
        assert math.isnan(r["jitter"])

    def test_speed_normalized_by_norm_factor(self):
        mm = make_mm(n=5, dx=10, dy=0)
        r1 = compute_mouse_kinematics(mm, EMPTY_MC, norm_factor=1.0)
        r2 = compute_mouse_kinematics(mm, EMPTY_MC, norm_factor=2.0)
        assert abs(r1["speed_mean"] - r2["speed_mean"] * 2) < 1e-9

    def test_straight_line_jitter_equals_1(self):
        # dx[0]=0 (no prior movement to first point); dx[i]=10 for i>=1.
        # total_path = sum(|dx_i|) = 90, euclidean = x[9]-x[0] = 90 → jitter=1.0
        rows = [
            {
                "t": float(i * 10),
                "event_type": "mouse_move",
                "x": float(i * 10),
                "y": 0.0,
                "dx": 0.0 if i == 0 else 10.0,
                "dy": 0.0,
                "pressed": None,
                "key": None,
            }
            for i in range(10)
        ]
        mm = pd.DataFrame(rows)
        r = compute_mouse_kinematics(mm, EMPTY_MC, norm_factor=1.0)
        assert abs(r["jitter"] - 1.0) < 1e-6

    def test_curved_path_jitter_greater_than_1(self):
        # Zigzag: total path > euclidean distance
        rows = []
        for i in range(10):
            dy = 5 if i % 2 == 0 else -5
            rows.append(
                {
                    "t": float(i * 10),
                    "event_type": "mouse_move",
                    "x": float(i * 10),
                    "y": float(100 + dy),
                    "dx": 10,
                    "dy": dy,
                }
            )
        mm = pd.DataFrame(rows)
        r = compute_mouse_kinematics(mm, EMPTY_MC, norm_factor=1.0)
        assert r["jitter"] > 1.0

    def test_click_interval_from_presses_only(self):
        mc = make_clicks([0.0, 100.0, 200.0], pressed=True)
        empty_mm = pd.DataFrame(columns=["t", "event_type", "x", "y", "dx", "dy"])
        r = compute_mouse_kinematics(empty_mm, mc, norm_factor=1.0)
        assert abs(r["click_interval_mean"] - 100.0) < 1e-9

    def test_no_clicks_returns_nan_click_interval(self):
        r = compute_mouse_kinematics(make_mm(n=5), EMPTY_MC, norm_factor=1.0)
        assert math.isnan(r["click_interval_mean"])


# ---------------------------------------------------------------------------
# TestComputeKeyboardPatterns
# ---------------------------------------------------------------------------


class TestComputeKeyboardPatterns:
    def test_empty_kp_returns_zero_burst_rate(self):
        r = compute_keyboard_patterns(EMPTY_KP, EMPTY_KR, window_duration_ms=30_000.0)
        assert r["burst_rate"] == 0.0
        assert math.isnan(r["iki_mean"])

    def test_burst_rate_correct(self):
        kp = make_key_events(
            ["a"] * 6, [float(i * 5000) for i in range(6)], "key_press"
        )
        r = compute_keyboard_patterns(kp, EMPTY_KR, window_duration_ms=30_000.0)
        assert abs(r["burst_rate"] - 0.2) < 1e-9

    def test_iki_computed_for_2plus_presses(self):
        kp = make_key_events(["a", "b", "c"], [0.0, 100.0, 300.0], "key_press")
        r = compute_keyboard_patterns(kp, EMPTY_KR, window_duration_ms=30_000.0)
        assert abs(r["iki_mean"] - 150.0) < 1e-9

    def test_single_press_nan_iki(self):
        kp = make_key_events(["a"], [0.0], "key_press")
        r = compute_keyboard_patterns(kp, EMPTY_KR, window_duration_ms=30_000.0)
        assert math.isnan(r["iki_mean"])

    def test_hold_duration_paired_correctly(self):
        kp = make_key_events(["a"], [0.0], "key_press")
        kr = make_key_events(["a"], [50.0], "key_release")
        r = compute_keyboard_patterns(kp, kr, window_duration_ms=30_000.0)
        assert abs(r["hold_mean"] - 50.0) < 1e-9

    def test_wasd_rhythm_nan_for_single_wasd_press(self):
        kp = make_key_events(["w"], [0.0], "key_press")
        r = compute_keyboard_patterns(kp, EMPTY_KR, window_duration_ms=30_000.0)
        assert math.isnan(r["wasd_rhythm"])


# ---------------------------------------------------------------------------
# TestComputeSessionAggregates
# ---------------------------------------------------------------------------


class TestComputeSessionAggregates:
    def test_event_rate_correct(self):
        mm = make_mm(n=60, t_start=0.0, dx=1, dy=0)
        window = make_window_df(mm=mm)
        r = compute_session_aggregates(window, w_start=0.0, window_duration_ms=30_000.0)
        assert abs(r["event_rate"] - 2.0) < 1e-9

    def test_mouse_key_ratio(self):
        mm = make_mm(n=10, dx=1)
        kp = make_key_events(
            ["a"] * 5, [float(i * 1000) for i in range(5)], "key_press"
        )
        window = make_window_df(mm=mm, kp=kp)
        r = compute_session_aggregates(window, w_start=0.0, window_duration_ms=30_000.0)
        # 10 mouse events / (5 key events + 1e-9) ≈ 2.0
        assert abs(r["mouse_key_ratio"] - 2.0) < 0.01

    def test_active_time_pct_full_coverage(self):
        # One event per second for 30s → every bucket occupied
        times = [float(i * 1000) for i in range(30)]
        rows = [
            {
                "t": t,
                "event_type": "mouse_move",
                "x": 0,
                "y": 0,
                "dx": 1,
                "dy": 0,
                "pressed": None,
                "key": None,
            }
            for t in times
        ]
        window = pd.DataFrame(rows)
        r = compute_session_aggregates(window, w_start=0.0, window_duration_ms=30_000.0)
        assert abs(r["active_time_pct"] - 1.0) < 1e-6

    def test_scroll_count_zero_without_scrolls(self):
        window = make_window_df(mm=make_mm(n=5))
        r = compute_session_aggregates(window, w_start=0.0, window_duration_ms=30_000.0)
        assert r["scroll_count"] == 0

    def test_scroll_direction_nan_without_scrolls(self):
        window = make_window_df(mm=make_mm(n=5))
        r = compute_session_aggregates(window, w_start=0.0, window_duration_ms=30_000.0)
        assert math.isnan(r["scroll_direction_ratio"])


# ---------------------------------------------------------------------------
# TestComputeTrajectoryFeatures  (Phase 1)
# ---------------------------------------------------------------------------


class TestComputeTrajectoryFeatures:
    def test_too_few_points_all_nan(self):
        for n in (0, 1, 2):
            r = compute_trajectory_features(make_mm(n=n), window_duration_ms=30_000.0)
            for k in (
                "mouse_curvature_mean",
                "mouse_curvature_std",
                "path_efficiency",
                "direction_changes_per_sec",
            ):
                assert math.isnan(r[k]), f"n={n}: {k} should be NaN"

    def test_straight_line_zero_curvature_full_efficiency(self):
        # Pure horizontal motion: no turns → curvature ≈ 0; perfectly straight → efficiency = 1
        mm = make_mm(n=5, dx=10, dy=0)
        r = compute_trajectory_features(mm, window_duration_ms=30_000.0)
        assert r["mouse_curvature_mean"] < 1e-6
        assert r["mouse_curvature_std"] < 1e-6
        assert abs(r["path_efficiency"] - 1.0) < 1e-6

    def test_right_angle_turn_pi_over_two_curvature(self):
        # Three points = two vectors: east, then north. Single turn of π/2.
        rows = [
            {"t": 0.0, "event_type": "mouse_move", "x": 0.0, "y": 0.0},
            {"t": 10.0, "event_type": "mouse_move", "x": 10.0, "y": 0.0},
            {"t": 20.0, "event_type": "mouse_move", "x": 10.0, "y": 10.0},
        ]
        mm = pd.DataFrame(rows)
        r = compute_trajectory_features(mm, window_duration_ms=30_000.0)
        assert abs(r["mouse_curvature_mean"] - math.pi / 2) < 1e-6

    def test_back_and_forth_detects_direction_changes(self):
        # Move +x, -x, +x, -x → 3 direction flips on x axis
        rows = [
            {"t": float(i * 10), "event_type": "mouse_move", "x": x, "y": 0.0}
            for i, x in enumerate([0.0, 10.0, 0.0, 10.0, 0.0])
        ]
        mm = pd.DataFrame(rows)
        r = compute_trajectory_features(mm, window_duration_ms=1000.0)
        # 3 sign-flips in 1 second
        assert r["direction_changes_per_sec"] >= 3.0

    def test_path_efficiency_low_when_self_intersecting(self):
        # Travel out and back: large path, near-zero displacement
        rows = [
            {"t": float(i * 10), "event_type": "mouse_move", "x": x, "y": 0.0}
            for i, x in enumerate([0.0, 10.0, 20.0, 30.0, 20.0, 10.0, 0.0])
        ]
        mm = pd.DataFrame(rows)
        r = compute_trajectory_features(mm, window_duration_ms=30_000.0)
        assert r["path_efficiency"] < 0.1


# ---------------------------------------------------------------------------
# TestComputeReactionFeatures  (Phase 1)
# ---------------------------------------------------------------------------


class TestComputeReactionFeatures:
    def test_empty_window_all_nan(self):
        empty = pd.DataFrame(columns=["t", "event_type", "x", "y", "pressed"])
        r = compute_reaction_features(empty)
        assert math.isnan(r["click_reaction_mean"])
        assert math.isnan(r["inter_click_movement"])

    def test_no_clicks_returns_nan(self):
        window = make_window_df(mm=make_mm(n=5))
        r = compute_reaction_features(window)
        assert math.isnan(r["click_reaction_mean"])
        assert math.isnan(r["inter_click_movement"])

    def test_click_reaction_matches_gap(self):
        # mouse_move at t=100, click at t=250 → reaction = 150 ms
        rows = [
            {"t": 100.0, "event_type": "mouse_move", "x": 5, "y": 5, "pressed": None},
            {"t": 250.0, "event_type": "mouse_click", "x": 5, "y": 5, "pressed": True},
        ]
        window = pd.DataFrame(rows)
        r = compute_reaction_features(window)
        assert abs(r["click_reaction_mean"] - 150.0) < 1e-6

    def test_inter_click_movement_euclidean(self):
        rows = [
            {
                "t": 100.0,
                "event_type": "mouse_click",
                "x": 0.0,
                "y": 0.0,
                "pressed": True,
            },
            {
                "t": 200.0,
                "event_type": "mouse_click",
                "x": 3.0,
                "y": 4.0,
                "pressed": True,
            },
            {
                "t": 300.0,
                "event_type": "mouse_click",
                "x": 0.0,
                "y": 0.0,
                "pressed": True,
            },
        ]
        window = pd.DataFrame(rows)
        r = compute_reaction_features(window)
        # Mean of [5.0, 5.0]
        assert abs(r["inter_click_movement"] - 5.0) < 1e-6

    def test_ignores_click_releases(self):
        rows = [
            {"t": 100.0, "event_type": "mouse_move", "x": 0, "y": 0, "pressed": None},
            {"t": 150.0, "event_type": "mouse_click", "x": 0, "y": 0, "pressed": True},
            {"t": 200.0, "event_type": "mouse_click", "x": 0, "y": 0, "pressed": False},
        ]
        window = pd.DataFrame(rows)
        r = compute_reaction_features(window)
        # Only one press → no inter-click movement
        assert math.isnan(r["inter_click_movement"])


# ---------------------------------------------------------------------------
# TestComputeKeystrokePeriodicity  (Phase 1)
# ---------------------------------------------------------------------------


class TestComputeKeystrokePeriodicity:
    def test_too_few_presses_returns_nan(self):
        for n in (0, 1, 2):
            kp = pd.DataFrame(
                {"t": list(range(n)), "event_type": "key_press", "key": "w"}
            )
            assert math.isnan(
                compute_keystroke_periodicity(kp)["keystroke_periodicity"]
            )

    def test_perfectly_periodic_cv_near_zero(self):
        # Press every 200 ms exactly → CV = 0
        kp = pd.DataFrame(
            {"t": [200.0 * i for i in range(10)], "event_type": "key_press", "key": "w"}
        )
        r = compute_keystroke_periodicity(kp)
        assert r["keystroke_periodicity"] < 1e-6

    def test_irregular_presses_high_cv(self):
        kp = pd.DataFrame(
            {
                "t": [0.0, 100.0, 350.0, 360.0, 800.0, 1200.0],
                "event_type": "key_press",
                "key": "w",
            }
        )
        r = compute_keystroke_periodicity(kp)
        assert r["keystroke_periodicity"] > 0.3


# ---------------------------------------------------------------------------
# TestProcessSessionWindows
# ---------------------------------------------------------------------------


class TestProcessSessionWindows:
    def test_empty_events_returns_empty_list(self):
        empty = pd.DataFrame(
            columns=["t", "event_type", "x", "y", "dx", "dy", "pressed", "key"]
        )
        assert process_session_windows(empty, norm_factor=1.0) == []

    def test_window_count_for_65s_session(self):
        # Events at t=0 through t=64000 ms → windows [0-30s], [30-60s], [60-65s]
        times = [float(i * 1000) for i in range(65)]
        rows = [
            {
                "t": t,
                "event_type": "mouse_move",
                "x": i,
                "y": 0,
                "dx": 1,
                "dy": 0,
                "pressed": None,
                "key": None,
            }
            for i, t in enumerate(times)
        ]
        df = pd.DataFrame(rows)
        windows = process_session_windows(df, norm_factor=1.0)
        assert len(windows) == 3

    def test_window_idx_starts_at_zero(self):
        mm = make_mm(n=10, t_start=0.0)
        windows = process_session_windows(mm, norm_factor=1.0)
        assert windows[0]["window_idx"] == 0

    def test_each_window_has_all_feature_keys(self):
        times = [float(i * 1000) for i in range(35)]
        rows = [
            {
                "t": t,
                "event_type": "mouse_move",
                "x": i,
                "y": 0,
                "dx": 1,
                "dy": 0,
                "pressed": None,
                "key": None,
            }
            for i, t in enumerate(times)
        ]
        df = pd.DataFrame(rows)
        windows = process_session_windows(df, norm_factor=1.0)
        assert len(windows) >= 1
        for w in windows:
            for col in FEATURE_COLS:
                assert col in w, f"Missing feature key: {col}"


# ---------------------------------------------------------------------------
# TestPollingRateNormalization  (pre-recording readiness)
# ---------------------------------------------------------------------------


def _mouse_stream(rate_hz: int, duration_ms: float = 30_000.0) -> pd.DataFrame:
    """Dense mouse_move stream at a given polling rate over one 30s window."""
    dt = 1000.0 / rate_hz
    n = int(duration_ms / dt)
    rows = [
        {
            "t": i * dt,
            "event_type": "mouse_move",
            "x": 100 + (i % 50),
            "y": 200 + (i % 30),
            "dx": 1 if i % 2 == 0 else -1,
            "dy": 1 if i % 3 == 0 else -1,
            "pressed": None,
            "key": None,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


class TestPollingRateNormalization:
    def test_polling_rate_norm_factor(self):
        assert polling_rate_norm(1000) == 1.0
        assert polling_rate_norm(125) == REFERENCE_POLLING_RATE / 125
        assert abs(polling_rate_norm(500) - 2.0) < 1e-9

    def test_polling_rate_norm_missing_or_invalid_is_identity(self):
        assert polling_rate_norm(None) == 1.0
        assert polling_rate_norm(0) == 1.0
        assert polling_rate_norm(-1) == 1.0
        assert polling_rate_norm("garbage") == 1.0

    def test_event_rate_diverges_without_normalization(self):
        """Sanity: identical behaviour at 1000 vs 125 Hz gives ~8x different raw event_rate."""
        hi = process_session_windows(
            _mouse_stream(1000), norm_factor=1.0, rate_norm=1.0
        )
        lo = process_session_windows(_mouse_stream(125), norm_factor=1.0, rate_norm=1.0)
        ratio = hi[0]["event_rate"] / lo[0]["event_rate"]
        assert 6.0 < ratio < 10.0, f"expected ~8x, got {ratio:.1f}x"

    def test_event_rate_matches_after_normalization(self):
        """With polling-rate normalization, the same behaviour gives matching event_rate."""
        hi = process_session_windows(
            _mouse_stream(1000), norm_factor=1.0, rate_norm=polling_rate_norm(1000)
        )
        lo = process_session_windows(
            _mouse_stream(125), norm_factor=1.0, rate_norm=polling_rate_norm(125)
        )
        hi_rate = hi[0]["event_rate"]
        lo_rate = lo[0]["event_rate"]
        # Within 10% after normalization (vs ~8x off without it)
        assert (
            abs(hi_rate - lo_rate) / hi_rate < 0.10
        ), f"normalized event_rate should match: {hi_rate:.1f} vs {lo_rate:.1f}"

    def test_direction_changes_normalized(self):
        """direction_changes_per_sec should also be brought to a common scale."""
        hi = process_session_windows(
            _mouse_stream(1000), norm_factor=1.0, rate_norm=polling_rate_norm(1000)
        )
        lo = process_session_windows(
            _mouse_stream(125), norm_factor=1.0, rate_norm=polling_rate_norm(125)
        )
        hi_dc = hi[0]["direction_changes_per_sec"]
        lo_dc = lo[0]["direction_changes_per_sec"]
        # Normalized values land in the same ballpark (within 2x; the flip
        # pattern isn't perfectly polling-invariant but normalization closes
        # most of the 8x hardware gap).
        assert 0.5 < (hi_dc / lo_dc) < 2.0, f"{hi_dc:.1f} vs {lo_dc:.1f}"
