"""
tests/test_live_cheat.py
========================
Unit tests for pipeline/adversarial/live_cheat.py — the pure planning layer of
the live cheat-signature harness (no Windows / pynput deps).
"""

from __future__ import annotations

import numpy as np

from pipeline.adversarial.live_cheat import (
    AIMBOT_PRESETS,
    MACRO_PRESET,
    TRIGGERBOT_PRESET,
    plan_aim_snap,
    plan_macro_tick,
    plan_trigger_burst,
    toggle_log_to_segments,
)


def _rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Aimbot snap
# ---------------------------------------------------------------------------


class TestPlanAimSnap:
    def test_obvious_is_single_step_teleport(self):
        actions = plan_aim_snap(_rng(), AIMBOT_PRESETS["obvious"], target=(100, -50))
        assert len(actions) == 1
        assert (actions[0].dx, actions[0].dy) == (100, -50)

    def test_locks_on_exactly_regardless_of_difficulty(self):
        # Final summed displacement must equal the (rounded) target for every
        # preset — easing/overshoot/jitter must not drift the lock-on.
        for name, preset in AIMBOT_PRESETS.items():
            actions = plan_aim_snap(_rng(), preset, target=(123, -77))
            assert sum(a.dx for a in actions) == 123, name
            assert sum(a.dy for a in actions) == -77, name

    def test_smoothing_adds_steps(self):
        obvious = plan_aim_snap(_rng(), AIMBOT_PRESETS["obvious"], target=(200, 0))
        soft = plan_aim_snap(_rng(), AIMBOT_PRESETS["soft"], target=(200, 0))
        assert len(soft) > len(obvious)

    def test_delays_sum_to_snap_duration(self):
        preset = AIMBOT_PRESETS["medium"]
        actions = plan_aim_snap(_rng(), preset, target=(150, 30))
        total = sum(a.delay_ms for a in actions)
        assert abs(total - preset.snap_duration_ms) < 1e-6

    def test_deterministic_under_seed(self):
        a = plan_aim_snap(np.random.default_rng(7), AIMBOT_PRESETS["soft"])
        b = plan_aim_snap(np.random.default_rng(7), AIMBOT_PRESETS["soft"])
        assert [(x.dx, x.dy) for x in a] == [(x.dx, x.dy) for x in b]

    def test_all_actions_are_moves(self):
        actions = plan_aim_snap(_rng(), AIMBOT_PRESETS["soft"], target=(80, 80))
        assert all(a.kind == "move" for a in actions)


# ---------------------------------------------------------------------------
# Triggerbot
# ---------------------------------------------------------------------------


class TestPlanTriggerBurst:
    def test_reaction_is_subhuman(self):
        for a in plan_trigger_burst(_rng(), n_clicks=20):
            assert (
                TRIGGERBOT_PRESET["reaction_ms_lo"]
                <= a.delay_ms
                <= TRIGGERBOT_PRESET["reaction_ms_hi"]
            )
            assert a.kind == "click"

    def test_n_clicks_respected(self):
        assert len(plan_trigger_burst(_rng(), n_clicks=3)) == 3
        assert len(plan_trigger_burst(_rng(), n_clicks=0)) == 1  # min 1


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------


class TestPlanMacroTick:
    def test_click_then_recoil_move(self):
        actions = plan_macro_tick(_rng())
        kinds = [a.kind for a in actions]
        assert kinds == ["click", "move"]
        assert actions[1].dy == MACRO_PRESET["recoil_dy"]  # downward recoil comp

    def test_interval_near_preset(self):
        # jitter is small relative to the interval
        intervals = [
            plan_macro_tick(np.random.default_rng(s))[0].delay_ms for s in range(30)
        ]
        assert abs(np.mean(intervals) - MACRO_PRESET["interval_ms"]) < 5.0


# ---------------------------------------------------------------------------
# Toggle-log → cheat_segments
# ---------------------------------------------------------------------------


def _kp(t, key):
    return {"t": float(t), "type": "key_press", "key": key}


class TestToggleLogToSegments:
    KEYS = {"Key.f8": "aimbot", "Key.f9": "triggerbot"}

    def test_paired_presses_form_a_span(self):
        events = [_kp(1000, "Key.f8"), _kp(5000, "Key.f8")]
        label, segs = toggle_log_to_segments(events, self.KEYS)
        assert label == "aimbot"
        assert segs == [[1000.0, 5000.0]]

    def test_odd_press_runs_to_session_end(self):
        events = [_kp(2000, "Key.f8")]
        label, segs = toggle_log_to_segments(events, self.KEYS, session_end_ms=10000)
        assert segs == [[2000.0, 10000.0]]
        assert label == "aimbot"

    def test_dominant_cheat_wins_label(self):
        events = [
            _kp(0, "Key.f9"),
            _kp(500, "Key.f9"),  # triggerbot 500ms
            _kp(1000, "Key.f8"),
            _kp(9000, "Key.f8"),  # aimbot 8000ms
        ]
        label, segs = toggle_log_to_segments(events, self.KEYS)
        assert label == "aimbot"
        assert len(segs) == 2

    def test_no_toggles_is_legit(self):
        events = [_kp(100, "w"), _kp(200, "a")]
        label, segs = toggle_log_to_segments(events, self.KEYS)
        assert label == "legit"
        assert segs == []
