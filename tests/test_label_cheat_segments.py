"""
tests/test_label_cheat_segments.py
==================================
Unit tests for the pure labelling core of scripts/label_cheat_segments.py.
"""

from __future__ import annotations

from scripts.label_cheat_segments import label_session


def _session():
    events = [
        {"t": 0.0, "type": "mouse_move", "x": 1, "y": 1, "dx": 1, "dy": 0},
        {"t": 1000.0, "type": "key_press", "key": "Key.f8"},  # aimbot ON
        {"t": 1500.0, "type": "mouse_move", "x": 5, "y": 5, "dx": 4, "dy": 4},
        {"t": 4000.0, "type": "key_press", "key": "Key.f8"},  # aimbot OFF
        {"t": 4100.0, "type": "key_press", "key": "w"},  # real gameplay key
    ]
    return {
        "session_id": "abc12345",
        "player": "tester",
        "duration_ms": 5000.0,
        "event_count": len(events),
        "events": events,
    }


class TestLabelSession:
    def test_derives_label_and_segments(self):
        out = label_session(_session())
        assert out["cheat_label"] == "aimbot"
        assert out["cheat_segments"] == [[1000.0, 4000.0]]

    def test_strips_toggle_keys_but_keeps_gameplay(self):
        out = label_session(_session(), strip_toggle_keys=True)
        keys = [e.get("key") for e in out["events"] if e["type"] == "key_press"]
        assert "Key.f8" not in keys
        assert "w" in keys  # real gameplay keystroke preserved
        assert out["event_count"] == len(out["events"])

    def test_keep_toggle_keys_option(self):
        out = label_session(_session(), strip_toggle_keys=False)
        keys = [e.get("key") for e in out["events"] if e["type"] == "key_press"]
        assert "Key.f8" in keys

    def test_does_not_mutate_input(self):
        s = _session()
        n_before = len(s["events"])
        label_session(s)
        assert len(s["events"]) == n_before  # original untouched

    def test_legit_when_no_toggles(self):
        s = _session()
        s["events"] = [e for e in s["events"] if e.get("key") != "Key.f8"]
        out = label_session(s)
        assert out["cheat_label"] == "legit"
        assert out["cheat_segments"] == []
