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

    def test_single_cheat_emits_typed_segments(self):
        out = label_session(_session())
        assert out["cheat_segments_typed"] == [
            {"start_ms": 1000.0, "end_ms": 4000.0, "cheat": "aimbot"}
        ]
        assert out["cheat_labels"] == ["aimbot"]

    def test_difficulty_is_stored(self):
        out = label_session(_session(), difficulty="Obvious")
        assert out["difficulty"] == "obvious"


def _multi_cheat_session():
    """aimbot → triggerbot → macro in one recording (the real-batch protocol)."""
    events = [
        {"t": 0.0, "type": "mouse_move", "x": 1, "y": 1, "dx": 1, "dy": 0},
        {"t": 1000.0, "type": "key_press", "key": "Key.f8"},  # aimbot ON
        {"t": 3000.0, "type": "key_press", "key": "Key.f8"},  # aimbot OFF
        {"t": 5000.0, "type": "key_press", "key": "Key.f9"},  # triggerbot ON
        {"t": 7000.0, "type": "key_press", "key": "Key.f9"},  # triggerbot OFF
        {"t": 9000.0, "type": "key_press", "key": "Key.f10"},  # macro ON
        {"t": 11000.0, "type": "key_press", "key": "Key.f10"},  # macro OFF
        {"t": 11500.0, "type": "key_press", "key": "w"},  # real gameplay key
    ]
    return {"session_id": "multi1234", "player": "tester", "events": events}


class TestMultiCheatSession:
    def test_label_is_mixed_with_all_types(self):
        out = label_session(_multi_cheat_session())
        assert out["cheat_label"] == "mixed"
        assert out["cheat_labels"] == ["aimbot", "macro", "triggerbot"]

    def test_typed_segments_preserve_each_type_in_order(self):
        out = label_session(_multi_cheat_session())
        typed = [
            (s["cheat"], s["start_ms"], s["end_ms"])
            for s in out["cheat_segments_typed"]
        ]
        assert typed == [
            ("aimbot", 1000.0, 3000.0),
            ("triggerbot", 5000.0, 7000.0),
            ("macro", 9000.0, 11000.0),
        ]

    def test_untyped_union_still_present_for_back_compat(self):
        out = label_session(_multi_cheat_session())
        assert out["cheat_segments"] == [
            [1000.0, 3000.0],
            [5000.0, 7000.0],
            [9000.0, 11000.0],
        ]
