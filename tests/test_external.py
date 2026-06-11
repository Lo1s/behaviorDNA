"""
tests/test_external.py
======================
Phase-6 scaffold tests: the concrete pieces (envelope assembly, the mouse-only
feature slice, adapter contract) are pinned now so filling in the per-corpus
CSV parsing later can't silently break the recorder-schema contract.
"""

import json

import pytest

from pipeline.external import build_mouse_session, write_sessions
from pipeline.external.balabit import BalabitAdapter
from pipeline.external.sapimouse import SapiMouseAdapter
from pipeline.features.run import (
    FEATURE_COLS,
    ID_FEATURE_COLS,
    KEYBOARD_FEATURE_COLS,
    MOUSE_ID_FEATURE_COLS,
)
from pipeline.ingestion.run import validate_session


def _mouse_events(n: int = 150, span_ms: float = 90_000.0) -> list[dict]:
    step = span_ms / (n - 1)
    evs = []
    for i in range(n):
        evs.append(
            {
                "t": i * step,
                "type": "mouse_move",
                "x": 800 + i,
                "y": 400 + (i % 7),
                "dx": 1,
                "dy": (i % 7) - 3,
            }
        )
    return evs


class TestBuildMouseSession:
    def test_passes_ingestion_validation(self):
        sess = build_mouse_session(
            session_id="balabit_u1_s1",
            player="user1",
            mouse_events=_mouse_events(),
            game="balabit",
            recorded_at="2026-01-01T00:00:00Z",
        )
        assert validate_session(sess, filepath=None) == []

    def test_duration_and_count_derived(self):
        evs = _mouse_events(n=120, span_ms=80_000.0)
        sess = build_mouse_session(
            session_id="s",
            player="u",
            mouse_events=evs,
            game="balabit",
            recorded_at="2026-01-01T00:00:00Z",
        )
        assert sess["event_count"] == 120
        assert sess["duration_ms"] == pytest.approx(80_000.0)

    def test_default_hardware_is_normalisation_reference(self):
        sess = build_mouse_session(
            session_id="s",
            player="u",
            mouse_events=_mouse_events(),
            game="balabit",
            recorded_at="2026-01-01T00:00:00Z",
        )
        # sensitivity*dpi/800 == 1.0 → no spurious sens/DPI scaling
        assert sess["sensitivity"] == 1.0 and sess["dpi"] == 800

    def test_write_sessions_roundtrip(self, tmp_path):
        sess = build_mouse_session(
            session_id="balabit_u1_s1",
            player="u",
            mouse_events=_mouse_events(),
            game="balabit",
            recorded_at="2026-01-01T00:00:00Z",
        )
        n = write_sessions(iter([sess]), tmp_path)
        assert n == 1
        loaded = json.loads((tmp_path / "balabit_u1_s1.json").read_text())
        assert loaded["session_id"] == "balabit_u1_s1"


class TestMouseFeatureSlice:
    def test_drops_exactly_the_keyboard_features(self):
        assert set(MOUSE_ID_FEATURE_COLS) == set(ID_FEATURE_COLS) - set(
            KEYBOARD_FEATURE_COLS
        )

    def test_is_strict_subset_in_bank_order(self):
        assert MOUSE_ID_FEATURE_COLS == [
            c for c in FEATURE_COLS if c in MOUSE_ID_FEATURE_COLS
        ]

    def test_no_keyboard_feature_survives(self):
        assert not (set(MOUSE_ID_FEATURE_COLS) & set(KEYBOARD_FEATURE_COLS))


class TestAdapterStubs:
    @pytest.mark.parametrize("cls", [BalabitAdapter, SapiMouseAdapter])
    def test_iter_sessions_not_implemented_yet(self, cls, tmp_path):
        with pytest.raises(NotImplementedError):
            list(cls(tmp_path).iter_sessions())

    def test_corpus_tags(self, tmp_path):
        assert BalabitAdapter(tmp_path).game == "balabit"
        assert SapiMouseAdapter(tmp_path).game == "sapimouse"
