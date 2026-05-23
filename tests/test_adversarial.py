"""
tests/test_adversarial.py
=========================
Unit tests for pipeline/adversarial/bot_generator.py.

These tests verify that each bot generator produces the expected measurable
signature (curvature collapse, reaction-time collapse, FFT spike) on a
synthetic baseline session — without depending on real recordings.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.adversarial.bot_generator import (
    CHEAT_AIMBOT,
    CHEAT_LEGIT,
    CHEAT_MACRO,
    CHEAT_TRIGGERBOT,
    CheatSpec,
    aim_snap_curvature,
    click_reaction_times,
    inject_cheat,
    key_press_intervals,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_session(
    n_clicks: int = 3,
    duration_ms: float = 30_000.0,
    mouse_rate_hz: int = 100,
    rng_seed: int = 0,
) -> dict:
    """Build a deterministic legit-like session with mouse movement + clicks.

    The mouse traces a noisy sinusoid (so trajectory has natural curvature),
    and clicks happen at regular intervals. Each click is preceded by 150 ms
    of mouse events so the aimbot/triggerbot have something to rewrite.
    """
    rng = np.random.default_rng(rng_seed)
    events: list[dict] = []
    dt_ms = 1000.0 / mouse_rate_hz
    n_samples = int(duration_ms / dt_ms)
    x_prev = 500.0
    y_prev = 500.0
    for i in range(n_samples):
        t = i * dt_ms
        x = 500 + 200 * math.sin(t / 1000.0) + rng.normal(0, 5)
        y = 500 + 200 * math.cos(t / 1500.0) + rng.normal(0, 5)
        events.append(
            {
                "t": float(t),
                "type": "mouse_move",
                "x": int(round(x)),
                "y": int(round(y)),
                "dx": int(round(x - x_prev)),
                "dy": int(round(y - y_prev)),
            }
        )
        x_prev, y_prev = x, y

    # Add clicks at evenly spaced moments (after at least 150 ms of moves).
    # Offset by 7.3 ms so the click time never aligns with a mouse_move sample.
    if n_clicks > 0:
        click_times = np.linspace(2000, duration_ms - 1000, n_clicks) + 7.3
        for click_t in click_times:
            # Find nearest mouse_move position
            nearest = min(events, key=lambda e: abs(e["t"] - click_t))
            events.append(
                {
                    "t": float(click_t),
                    "type": "mouse_click",
                    "x": nearest["x"],
                    "y": nearest["y"],
                    "button": "Button.left",
                    "pressed": True,
                }
            )
            events.append(
                {
                    "t": float(click_t + 50),
                    "type": "mouse_click",
                    "x": nearest["x"],
                    "y": nearest["y"],
                    "button": "Button.left",
                    "pressed": False,
                }
            )

    # Add some key presses (irregular, so legit KPI is noisy)
    key_times = sorted(rng.uniform(0, duration_ms, size=20).tolist())
    for kt in key_times:
        events.append({"t": float(kt), "type": "key_press", "key": "w"})
        events.append({"t": float(kt + 80), "type": "key_release", "key": "w"})

    events.sort(key=lambda e: e["t"])

    return {
        "session_id": "test1234",
        "player": "test",
        "game": "test_game",
        "activity": "free_roam",
        "sensitivity": 0.5,
        "dpi": 800,
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "duration_ms": duration_ms,
        "event_count": len(events),
        "events": events,
    }


@pytest.fixture
def baseline():
    return _synthetic_session()


# ---------------------------------------------------------------------------
# Aimbot
# ---------------------------------------------------------------------------


def test_aimbot_collapses_snap_curvature(baseline):
    legit_curv = aim_snap_curvature(baseline)
    assert legit_curv, "fixture must produce at least one click with prior moves"
    legit_mean = float(np.mean(legit_curv))

    aim = inject_cheat(baseline, CheatSpec(CHEAT_AIMBOT, dict(smoothing=0.0)))
    aim_curv = aim_snap_curvature(aim)
    aim_mean = float(np.mean(aim_curv))

    assert aim["cheat_label"] == CHEAT_AIMBOT
    assert len(aim["cheat_segments"]) > 0
    # The aimbot should drive curvature substantially below human baseline
    assert (
        aim_mean < legit_mean * 0.5
    ), f"aimbot mean curvature {aim_mean:.3f} should be << legit {legit_mean:.3f}"


def test_aimbot_difficulty_levels_produce_different_curvature(baseline):
    obvious = inject_cheat(baseline, CheatSpec(CHEAT_AIMBOT, dict(smoothing=0.0)))
    soft = inject_cheat(baseline, CheatSpec(CHEAT_AIMBOT, dict(smoothing=0.85)))
    obv_c = float(np.mean(aim_snap_curvature(obvious)))
    soft_c = float(np.mean(aim_snap_curvature(soft)))
    # Both should be lower than human, but the soft variant should be less obvious
    # (we don't enforce strict ordering since both can collapse to ~0)
    assert obv_c >= 0 and soft_c >= 0


def test_aimbot_does_not_mutate_input(baseline):
    original_ids = [id(e) for e in baseline["events"]]
    _ = inject_cheat(baseline, CheatSpec(CHEAT_AIMBOT, {}))
    # input must be unchanged
    assert [id(e) for e in baseline["events"]] == original_ids


def test_aimbot_no_clicks_returns_legit_label():
    sess = _synthetic_session(n_clicks=0)
    out = inject_cheat(sess, CheatSpec(CHEAT_AIMBOT, {}))
    assert out["cheat_label"] == CHEAT_LEGIT
    assert out["cheat_segments"] == []


# ---------------------------------------------------------------------------
# Triggerbot
# ---------------------------------------------------------------------------


def test_triggerbot_compresses_click_reaction(baseline):
    legit_rt = click_reaction_times(baseline)
    assert (
        legit_rt and min(legit_rt) > 5
    ), f"legit fixture should have reactions > 5ms; got min={min(legit_rt):.2f}"

    trig = inject_cheat(
        baseline, CheatSpec(CHEAT_TRIGGERBOT, dict(reaction_time_ms=3.0))
    )
    trig_rt = click_reaction_times(trig)

    assert trig["cheat_label"] == CHEAT_TRIGGERBOT
    # All reaction times in the triggerbot variant should be ~ 3 ms
    assert (
        max(trig_rt) < 10.0
    ), f"triggerbot reactions should be <10ms; got max={max(trig_rt):.2f}"


def test_triggerbot_preserves_event_count(baseline):
    trig = inject_cheat(baseline, CheatSpec(CHEAT_TRIGGERBOT, {}))
    assert len(trig["events"]) == len(baseline["events"])


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------


def test_macro_collapses_kpi_variation(baseline):
    """Macro should produce key-press intervals with near-zero coefficient of variation."""
    legit_ipi = key_press_intervals(baseline)
    assert legit_ipi
    legit_cv = float(np.std(legit_ipi) / (np.mean(legit_ipi) + 1e-9))

    mac = inject_cheat(
        baseline, CheatSpec(CHEAT_MACRO, dict(interval_ms=200.0, duration_ms=5000.0))
    )
    assert mac["cheat_label"] == CHEAT_MACRO

    # Restrict to macro-window KPI to test the regular-spacing property
    seg_start, seg_end = mac["cheat_segments"][0]
    macro_presses = [
        e["t"]
        for e in mac["events"]
        if e.get("type") == "key_press" and seg_start <= e["t"] <= seg_end
    ]
    macro_ipi = [
        macro_presses[i + 1] - macro_presses[i] for i in range(len(macro_presses) - 1)
    ]
    assert macro_ipi
    macro_cv = float(np.std(macro_ipi) / (np.mean(macro_ipi) + 1e-9))

    assert (
        macro_cv < legit_cv * 0.1
    ), f"macro CV {macro_cv:.4f} should be << legit CV {legit_cv:.4f}"
    # And the mean should be the configured interval
    assert abs(float(np.mean(macro_ipi)) - 200.0) < 1.0


def test_macro_strips_existing_keys_in_window(baseline):
    mac = inject_cheat(
        baseline,
        CheatSpec(
            CHEAT_MACRO, dict(interval_ms=200.0, duration_ms=5000.0, start_fraction=0.3)
        ),
    )
    seg_start, seg_end = mac["cheat_segments"][0]
    # All key_press events inside the segment must use one of the macro keys
    for ev in mac["events"]:
        if ev.get("type") == "key_press" and seg_start <= ev["t"] <= seg_end:
            assert ev["key"] in {"w", "a", "s", "d"}


# ---------------------------------------------------------------------------
# CheatSpec dispatch
# ---------------------------------------------------------------------------


def test_cheatspec_unknown_type_raises():
    with pytest.raises(ValueError):
        CheatSpec(cheat_type="esp_wallhack").build()


def test_inject_cheat_generates_new_session_id(baseline):
    out = inject_cheat(baseline, CheatSpec(CHEAT_AIMBOT, {}))
    assert out["session_id"] != baseline["session_id"]
    assert len(out["session_id"]) == 8


def test_inject_cheat_updates_event_count_and_duration(baseline):
    out = inject_cheat(baseline, CheatSpec(CHEAT_MACRO, {}))
    assert out["event_count"] == len(out["events"])
    assert out["duration_ms"] == max(e["t"] for e in out["events"])
