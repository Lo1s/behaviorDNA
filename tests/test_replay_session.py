"""
tests/test_replay_session.py
============================
Smoke tests for the replay client in scripts/replay_session.py.

We exercise the offline path (no server) plus the cheat-injection helper.
The WebSocket path is exercised end-to-end in tests/test_streaming.py via
the FastAPI TestClient — we don't double-test it here.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from pipeline.features.run import FEATURE_COLS
from pipeline.inference.aggregator import RiskAggregator
from pipeline.inference.streaming import SessionStreamState
from scripts.replay_session import inject_cheat_if_requested, replay_offline


def _tiny_stream_state(chunk_length: int = 8) -> SessionStreamState:
    """Build a minimal SessionStreamState without loading the synthetic dataset."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (50, len(FEATURE_COLS)))
    scaler = StandardScaler().fit(X)
    det = IsolationForest(n_estimators=20, contamination=0.05, random_state=0)
    det.fit(scaler.transform(X))

    legit = rng.normal(0, 1, 100)
    cheat = rng.normal(3, 1, 100)
    scores = np.concatenate([legit, cheat])
    labels = np.concatenate([np.zeros(100), np.ones(100)])
    agg = RiskAggregator(prior_cheat_rate=0.05).fit(
        {"IsolationForest": (scores, labels)}
    )
    return SessionStreamState(
        classical_detectors={"IsolationForest": det},
        feature_scaler=scaler,
        aggregator=agg,
        chunk_length=chunk_length,
    )


def _tiny_session(n_events: int = 200, with_clicks: bool = True) -> dict:
    """Build a synthetic session JSON dict good enough for the replay path."""
    events = []
    for i in range(n_events):
        events.append(
            {
                "t": float(i * 100.0),
                "type": "mouse_move",
                "x": 100 + i,
                "y": 100,
                "dx": 1,
                "dy": 0,
            }
        )
    if with_clicks:
        # Two click pairs so the aimbot/triggerbot generators have something to rewrite
        events.extend(
            [
                {
                    "t": 5000.0,
                    "type": "mouse_click",
                    "x": 200,
                    "y": 100,
                    "pressed": True,
                },
                {
                    "t": 5050.0,
                    "type": "mouse_click",
                    "x": 200,
                    "y": 100,
                    "pressed": False,
                },
                {
                    "t": 15000.0,
                    "type": "mouse_click",
                    "x": 300,
                    "y": 100,
                    "pressed": True,
                },
                {
                    "t": 15050.0,
                    "type": "mouse_click",
                    "x": 300,
                    "y": 100,
                    "pressed": False,
                },
            ]
        )
        events.sort(key=lambda e: e["t"])
    return {
        "session_id": "test1234",
        "player": "test",
        "game": "test_game",
        "sensitivity": 1.0,
        "dpi": 800.0,
        "duration_ms": events[-1]["t"],
        "event_count": len(events),
        "events": events,
    }


# ---------------------------------------------------------------------------
# Cheat injection helper
# ---------------------------------------------------------------------------


class TestCheatInjection:
    def test_no_injection_returns_passthrough(self):
        session = _tiny_session()
        out = inject_cheat_if_requested(session, cheat_type=None, inject_at_s=None)
        assert out["cheat_label"] == "legit"
        assert out["cheat_segments"] == []
        assert len(out["events"]) == len(session["events"])

    def test_aimbot_injection_at_middle_preserves_pre_events(self):
        session = _tiny_session(n_events=200)
        out = inject_cheat_if_requested(session, cheat_type="aimbot", inject_at_s=10.0)
        # Cheat label set + at least one cheat segment
        assert out["cheat_label"] == "aimbot"
        assert len(out["cheat_segments"]) >= 1
        # All pre-injection events preserved verbatim
        pre = [e for e in session["events"] if e.get("t", 0.0) < 10_000.0]
        out_pre = [e for e in out["events"] if e.get("t", 0.0) < 10_000.0]
        assert pre == out_pre

    def test_injection_past_session_end_is_noop(self):
        session = _tiny_session(n_events=10)  # ~1 second
        out = inject_cheat_if_requested(session, cheat_type="aimbot", inject_at_s=120.0)
        # Falls back to original; no cheat segments
        assert out == session  # exact dict equality


# ---------------------------------------------------------------------------
# Offline replay
# ---------------------------------------------------------------------------


class TestReplayOffline:
    def test_offline_replay_emits_updates_and_writes_jsonl(self, tmp_path):
        session = _tiny_session(n_events=400)  # spans > one 30s window
        out_path = tmp_path / "scores.jsonl"
        updates = replay_offline(session, out_path=out_path, state=_tiny_stream_state())

        # Either we crossed a window boundary or not — but the finalize() snapshot
        # at the end is guaranteed, so the list should be non-empty.
        assert len(updates) >= 1
        last = updates[-1]
        assert "session_risk" in last
        assert 0.0 <= last["session_risk"] <= 1.0

        # JSONL file mirrors the in-memory list
        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == len(updates)
        # Each line is a valid JSON dict with the same keys
        for line, update in zip(lines, updates):
            parsed = json.loads(line)
            assert parsed["session_risk"] == update["session_risk"]
