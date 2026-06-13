"""
tests/test_humanizer.py
=======================
Unit tests for pipeline/adversarial/humanizer.py (Phase 7).

Verify the λ knob behaves: each lever is monotone in ``lam`` (aimbot snap gets
curvier/jittered, triggerbot reaction grows toward human, macro cadence jitters),
the closed-form utility is monotone non-increasing, outputs stay schema-valid and
deterministic, and ``lam=0`` reproduces the obvious-bot anchor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.adversarial.bot_generator import (
    CHEAT_AIMBOT,
    CHEAT_MACRO,
    CHEAT_TRIGGERBOT,
    aim_snap_curvature,
)
from pipeline.adversarial.humanizer import (
    HUMAN_RT_MS,
    aimbot_utility,
    cheat_utility,
    humanize,
    humanize_aimbot,
    humanize_macro,
    humanize_triggerbot,
    macro_utility,
    player_baseline,
    triggerbot_utility,
)
from pipeline.ingestion.run import parse_events

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_session(
    n_clicks: int = 8, duration_ms: float = 30_000.0, mouse_rate_hz: int = 100
) -> dict:
    """Deterministic legit-like session: noisy sinusoid mouse + clicks + keys."""
    rng = np.random.default_rng(0)
    events: list[dict] = []
    dt_ms = 1000.0 / mouse_rate_hz
    n_samples = int(duration_ms / dt_ms)
    x_prev, y_prev = 500.0, 500.0
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

    click_times = np.linspace(2000, duration_ms - 1000, n_clicks) + 7.3
    for click_t in click_times:
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

    key_times = sorted(rng.uniform(0, duration_ms, size=40).tolist())
    for kt in key_times:
        events.append({"t": float(kt), "type": "key_press", "key": "w"})
        events.append({"t": float(kt + 80), "type": "key_release", "key": "w"})

    events.sort(key=lambda e: e["t"])
    return {
        "session_id": "test1234",
        "player": "test",
        "game": "test_game",
        "sensitivity": 0.5,
        "dpi": 800,
        "duration_ms": duration_ms,
        "event_count": len(events),
        "events": events,
    }


@pytest.fixture
def session():
    return _synthetic_session()


@pytest.fixture
def baseline(session):
    return player_baseline([session])


# ---------------------------------------------------------------------------
# Player baseline
# ---------------------------------------------------------------------------


def test_player_baseline_is_sane(baseline):
    assert baseline.move_step_scale > 0.0
    assert baseline.key_interval_cv > 0.0
    assert baseline.human_rt_ms == HUMAN_RT_MS


def test_player_baseline_empty_uses_defaults():
    b = player_baseline([{"events": []}])
    assert b.move_step_scale > 0.0
    assert b.key_interval_cv > 0.0


# ---------------------------------------------------------------------------
# Schema + determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cheat", [CHEAT_AIMBOT, CHEAT_TRIGGERBOT, CHEAT_MACRO])
@pytest.mark.parametrize("lam", LAMBDAS)
def test_schema_preserved_and_ingestible(session, baseline, cheat, lam):
    out = humanize(session, cheat, lam, baseline, seed=1)
    assert out["cheat_label"] == cheat
    assert isinstance(out["cheat_segments"], list) and out["cheat_segments"]
    assert out["event_count"] == len(out["events"])
    # original is untouched (deep copy)
    assert session["event_count"] == len(session["events"])
    # drop-in for the ingestion pipeline
    df = parse_events(out)
    assert not df.empty


@pytest.mark.parametrize("cheat", [CHEAT_AIMBOT, CHEAT_TRIGGERBOT, CHEAT_MACRO])
def test_determinism(session, baseline, cheat):
    a = humanize(session, cheat, 0.6, baseline, seed=7)
    b = humanize(session, cheat, 0.6, baseline, seed=7)
    assert a["events"] == b["events"]


# ---------------------------------------------------------------------------
# λ monotonicity of each lever
# ---------------------------------------------------------------------------


def _seg_widths(out: dict) -> list[float]:
    return [float(b - a) for a, b in out["cheat_segments"]]


def test_triggerbot_reaction_grows_with_lambda(session, baseline):
    means = []
    for lam in LAMBDAS:
        out = humanize_triggerbot(session, lam, baseline, seed=3)
        means.append(float(np.mean(_seg_widths(out))))
    # reaction time increases monotonically sub-human → human
    assert all(b >= a - 1e-6 for a, b in zip(means, means[1:])), means
    assert means[0] < 10.0  # ~3 ms sub-human at lam=0
    assert means[-1] > 100.0  # approaching human RT at lam=1


def test_aimbot_curvature_increases_with_lambda(session, baseline):
    obv = humanize_aimbot(session, 0.0, baseline, seed=3)
    soft = humanize_aimbot(session, 1.0, baseline, seed=3)
    obv_c = float(np.mean(aim_snap_curvature(obv)))
    soft_c = float(np.mean(aim_snap_curvature(soft)))
    # lam=0 is a near-straight teleport; lam=1 is eased + jittered → curvier
    assert soft_c > obv_c


def test_aimbot_locks_onto_target(session, baseline):
    out = humanize_aimbot(session, 1.0, baseline, seed=3)
    events = out["events"]
    clicks = [
        (i, e)
        for i, e in enumerate(events)
        if e.get("type") == "mouse_click" and e.get("pressed") is True
    ]
    checked = 0
    for idx, click in clicks:
        # the mouse_move immediately preceding the click must sit on target
        for j in range(idx - 1, -1, -1):
            if events[j].get("type") == "mouse_move":
                assert events[j]["x"] == click["x"]
                assert events[j]["y"] == click["y"]
                checked += 1
                break
    assert checked > 0


def test_macro_cv_increases_with_lambda(session, baseline):
    def macro_cv(out: dict) -> float:
        # measure cadence only WITHIN the injected macro span (the session also
        # carries its own irregular legit keystrokes outside the window)
        ((lo, hi),) = out["cheat_segments"]
        ts = sorted(
            e["t"]
            for e in out["events"]
            if e.get("type") == "key_press" and lo <= e["t"] <= hi
        )
        iv = np.diff(np.asarray(ts, dtype=np.float64))
        iv = iv[iv > 0]
        return float(iv.std() / iv.mean()) if iv.size > 1 and iv.mean() > 0 else 0.0

    low = macro_cv(humanize_macro(session, 0.0, baseline, seed=3))
    high = macro_cv(humanize_macro(session, 1.0, baseline, seed=3))
    assert low == pytest.approx(0.0, abs=1e-6)  # perfectly periodic at lam=0
    assert high > low


# ---------------------------------------------------------------------------
# Utility(λ) — monotone non-increasing, bounded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cheat", [CHEAT_AIMBOT, CHEAT_TRIGGERBOT, CHEAT_MACRO])
def test_utility_monotone_non_increasing(baseline, cheat):
    us = [cheat_utility(cheat, lam, baseline) for lam in LAMBDAS]
    assert all(0.0 <= u <= 1.0 for u in us), us
    assert all(b <= a + 1e-9 for a, b in zip(us, us[1:])), us
    assert us[0] > us[-1]  # strictly worth less when fully humanised
    assert us[-1] == pytest.approx(0.0, abs=1e-6)


def test_utility_specific_values(baseline):
    assert triggerbot_utility(0.0, baseline) > 0.95
    assert aimbot_utility(0.0, baseline) > 0.0
    assert macro_utility(0.5, baseline) == pytest.approx(0.5)
