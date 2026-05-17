"""
tests/test_ingestion.py
=======================
Unit tests for pipeline/ingestion/run.py
"""

from pathlib import Path

from pipeline.ingestion.run import (
    parse_events,
    parse_session_metadata,
    validate_session,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_session(**overrides) -> dict:
    """Return a minimal valid session dict, with optional field overrides."""
    base = {
        "session_id": "abc12345",
        "player": "TestPlayer",
        "game": "cs2",
        "sensitivity": 1.0,
        "dpi": 800,
        "recorded_at": "2024-01-01T12:00:00+00:00",
        "duration_ms": 300_000.0,
        "event_count": 500,
        "events": [
            {
                "t": i * 10.0,
                "type": "mouse_move",
                "x": 100 + i,
                "y": 200,
                "dx": 1,
                "dy": 0,
            }
            for i in range(500)
        ],
    }
    base.update(overrides)
    return base


DUMMY_PATH = Path("dummy.json")


# ---------------------------------------------------------------------------
# validate_session
# ---------------------------------------------------------------------------


class TestValidateSession:
    def test_valid_session_has_no_errors(self):
        assert validate_session(make_session(), DUMMY_PATH) == []

    def test_missing_field_is_reported(self):
        data = make_session()
        del data["player"]
        errors = validate_session(data, DUMMY_PATH)
        assert any("player" in e for e in errors)

    def test_wrong_type_is_reported(self):
        data = make_session(dpi="not-an-int")
        errors = validate_session(data, DUMMY_PATH)
        assert any("dpi" in e for e in errors)

    def test_empty_events_is_rejected(self):
        data = make_session(events=[])
        errors = validate_session(data, DUMMY_PATH)
        assert any("Empty" in e for e in errors)

    def test_too_few_events_warns(self):
        data = make_session(
            events=[{"t": 1.0, "type": "mouse_move"}] * 10, event_count=10
        )
        errors = validate_session(data, DUMMY_PATH)
        assert any("few events" in e for e in errors)

    def test_too_short_session_is_rejected(self):
        data = make_session(duration_ms=5_000.0)
        errors = validate_session(data, DUMMY_PATH)
        assert any("short" in e for e in errors)


# ---------------------------------------------------------------------------
# parse_session_metadata
# ---------------------------------------------------------------------------


class TestParseSessionMetadata:
    def test_player_is_lowercased(self):
        data = make_session(player="  JIRI  ")
        meta = parse_session_metadata(data, DUMMY_PATH)
        assert meta["player"] == "jiri"

    def test_game_is_normalised(self):
        data = make_session(game="Arc Raiders")
        meta = parse_session_metadata(data, DUMMY_PATH)
        assert meta["game"] == "arc_raiders"

    def test_source_file_is_captured(self):
        path = Path("20240101T120000_jiri_cs2_abc12345.json")
        meta = parse_session_metadata(make_session(), path)
        assert meta["source_file"] == path.name

    def test_numeric_types_are_correct(self):
        meta = parse_session_metadata(make_session(), DUMMY_PATH)
        assert isinstance(meta["sensitivity"], float)
        assert isinstance(meta["dpi"], int)
        assert isinstance(meta["duration_ms"], float)


# ---------------------------------------------------------------------------
# parse_events
# ---------------------------------------------------------------------------


class TestParseEvents:
    def test_returns_dataframe_with_correct_columns(self):
        data = make_session()
        df = parse_events(data)
        assert not df.empty
        assert "session_id" in df.columns
        assert "t" in df.columns
        assert "event_type" in df.columns

    def test_all_events_parsed(self):
        data = make_session()
        df = parse_events(data)
        assert len(df) == 500

    def test_unknown_event_types_are_dropped(self):
        data = make_session(
            events=[
                {"t": 1.0, "type": "mouse_move", "x": 100, "y": 200, "dx": 1, "dy": 0},
                {"t": 2.0, "type": "unknown_type"},
            ]
        )
        df = parse_events(data)
        assert len(df) == 1
        assert df.iloc[0]["event_type"] == "mouse_move"

    def test_session_id_is_propagated(self):
        data = make_session(session_id="test1234")
        df = parse_events(data)
        assert (df["session_id"] == "test1234").all()

    def test_empty_events_returns_empty_dataframe(self):
        data = make_session(events=[])
        df = parse_events(data)
        assert df.empty

    def test_mixed_event_types_are_all_present(self):
        events = [
            {"t": 1.0, "type": "mouse_move", "x": 100, "y": 200, "dx": 1, "dy": 0},
            {
                "t": 2.0,
                "type": "mouse_click",
                "x": 100,
                "y": 200,
                "button": "Button.left",
                "pressed": True,
            },
            {"t": 3.0, "type": "key_press", "key": "w"},
            {"t": 4.0, "type": "key_release", "key": "w"},
        ]
        data = make_session(events=events, event_count=4)
        df = parse_events(data)
        assert set(df["event_type"].unique()) == {
            "mouse_move",
            "mouse_click",
            "key_press",
            "key_release",
        }
