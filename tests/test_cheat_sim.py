"""
tests/test_cheat_sim.py
=======================
Unit tests for the pure, OS-independent parts of collector/cheat_sim.py —
the self-test plan builder. (The SendInput actuator + run_selftest are
Windows-only and exercised on the host, like record_session.py.)
"""

from __future__ import annotations

import numpy as np

from collector.cheat_sim import build_selftest_plan


def _by_name(seed=0):
    return {c.name: c for c in build_selftest_plan(np.random.default_rng(seed))}


def test_plan_covers_every_actuator_path():
    names = set(_by_name())
    assert {
        "move-right",
        "move-left",
        "move-up",
        "aimbot-snap",
        "triggerbot-click",
        "macro-tick",
    } <= names


def test_move_cases_have_expected_direction():
    c = _by_name()
    assert c["move-right"].expect_dx > 0 and c["move-right"].expect_dy == 0
    assert c["move-left"].expect_dx < 0
    assert c["move-up"].expect_dy < 0
    assert all(c[n].expect_clicks == 0 for n in ("move-right", "move-left", "move-up"))


def test_aimbot_snap_sums_to_target():
    snap = _by_name()["aimbot-snap"]
    assert (snap.expect_dx, snap.expect_dy) == (150, -60)  # lock-on is exact
    assert snap.expect_clicks == 0


def test_triggerbot_is_one_click_no_move():
    c = _by_name()["triggerbot-click"]
    assert c.expect_clicks == 1
    assert (c.expect_dx, c.expect_dy) == (0, 0)


def test_macro_has_click_and_downward_recoil():
    c = _by_name()["macro-tick"]
    assert c.expect_clicks == 1
    assert c.expect_dy > 0  # recoil compensation pulls down
