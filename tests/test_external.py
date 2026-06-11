"""
tests/test_external.py
======================
Phase-6 scaffold tests: the concrete pieces (envelope assembly, the mouse-only
feature slice, adapter contract) are pinned now so filling in the per-corpus
CSV parsing later can't silently break the recorder-schema contract.
"""

import json

import numpy as np
import pytest

from pipeline.external import build_mouse_session, write_sessions
from pipeline.external.balabit import BalabitAdapter
from pipeline.external.base import rows_to_recorder_events, split_on_idle
from pipeline.external.sapimouse import SapiMouseAdapter
from pipeline.features.run import (
    FEATURE_COLS,
    ID_FEATURE_COLS,
    KEYBOARD_FEATURE_COLS,
    MOUSE_ID_FEATURE_COLS,
)
from pipeline.ingestion.run import validate_session
from pipeline.verification import eer, far_at_frr, verification_scores


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


class TestRowsToRecorderEvents:
    def test_moves_carry_position_deltas(self):
        evs = rows_to_recorder_events(
            [
                (0.0, "NoButton", "Move", 100, 200),
                (10.0, "NoButton", "Move", 103, 198),
                (20.0, "NoButton", "Drag", 110, 190),
            ]
        )
        assert [e["type"] for e in evs] == ["mouse_move"] * 3
        assert (evs[0]["dx"], evs[0]["dy"]) == (0, 0)
        assert (evs[1]["dx"], evs[1]["dy"]) == (3, -2)
        assert (evs[2]["dx"], evs[2]["dy"]) == (7, -8)

    def test_clicks_and_scrolls(self):
        evs = rows_to_recorder_events(
            [
                (0.0, "Left", "Pressed", 5, 5),
                (50.0, "Left", "Released", 5, 5),
                (60.0, "Scroll", "Down", 5, 5),
                (70.0, "Scroll", "Up", 5, 5),
            ]
        )
        assert evs[0] == {
            "t": 0.0,
            "type": "mouse_click",
            "x": 5,
            "y": 5,
            "button": "left",
            "pressed": True,
        }
        assert evs[1]["pressed"] is False
        assert (evs[2]["type"], evs[2]["dy"]) == ("mouse_scroll", -1)
        assert (evs[3]["type"], evs[3]["dy"]) == ("mouse_scroll", 1)

    def test_unknown_states_dropped(self):
        assert rows_to_recorder_events([(0.0, "NoButton", "Wiggle", 1, 1)]) == []


def _write_balabit_fixture(root, n_rows=150):
    """Tiny on-disk Balabit tree: 1 train user/session + 1 labelled test pair."""
    train = root / "training_files" / "user7"
    train.mkdir(parents=True)
    test = root / "test_files" / "user7"
    test.mkdir(parents=True)
    header = "record timestamp,client timestamp,button,state,x,y\n"

    def rows(n):
        out = [header]
        for i in range(n):
            out.append(f"{i*0.5},{i*0.5},NoButton,Move,{100+i},{200+(i%5)}\n")
        out.append(f"{n*0.5},{n*0.5},Left,Pressed,{100+n},200\n")
        out.append(f"{n*0.5+0.1},{n*0.5+0.1},Left,Released,{100+n},200\n")
        return "".join(out)

    (train / "session_0000000001").write_text(rows(n_rows))
    (test / "session_0000000002").write_text(rows(n_rows))
    (test / "session_0000000003").write_text(rows(n_rows))
    (root / "public_labels.csv").write_text(
        "filename,is_illegal\nsession_0000000002,0\nsession_0000000003,1\n"
    )


class TestBalabitAdapter:
    def test_training_sessions_parse_and_validate(self, tmp_path):
        _write_balabit_fixture(tmp_path)
        sessions = list(BalabitAdapter(tmp_path).iter_sessions())
        assert len(sessions) == 1
        s = sessions[0]
        assert s["player"] == "user7"
        assert s["game"] == "balabit"
        assert validate_session(s, filepath=None) == []
        # seconds → ms conversion: 150 rows at 0.5 s spacing ≈ 75 000 ms
        assert s["duration_ms"] == pytest.approx(75_100, rel=0.01)
        types = {e["type"] for e in s["events"]}
        assert types == {"mouse_move", "mouse_click"}

    def test_test_sessions_carry_impostor_labels(self, tmp_path):
        _write_balabit_fixture(tmp_path)
        triples = list(BalabitAdapter(tmp_path).iter_test_sessions())
        assert [(c, imp) for _, c, imp in triples] == [
            ("user7", False),
            ("user7", True),
        ]

    def test_short_sessions_skipped(self, tmp_path):
        _write_balabit_fixture(tmp_path, n_rows=10)
        assert list(BalabitAdapter(tmp_path).iter_sessions()) == []


class TestSapiMouseAdapter:
    def _write_fixture(self, root, n_rows=150):
        user = root / "sapimouse" / "user1"
        user.mkdir(parents=True)
        header = "client timestamp,button,state,x,y\n"

        def rows(n, t0):
            out = [header]
            for i in range(n):
                out.append(f"{t0 + i*400},NoButton,Move,{50+i},{60+(i%3)}\n")
            return "".join(out)

        (user / "session_2020_05_14_1min.csv").write_text(rows(n_rows, 15585))
        (user / "session_2020_05_14_3min.csv").write_text(rows(n_rows * 3, 29256))

    def test_sessions_parse_with_protocol_tags(self, tmp_path):
        self._write_fixture(tmp_path)
        sessions = list(SapiMouseAdapter(tmp_path).iter_sessions())
        assert [(s["player"], s["protocol"]) for s in sessions] == [
            ("user1", "1min"),
            ("user1", "3min"),
        ]
        # arbitrary start offset removed: t begins at 0
        assert sessions[0]["events"][0]["t"] == 0.0

    def test_works_when_pointed_at_unpacked_dir(self, tmp_path):
        self._write_fixture(tmp_path)
        sessions = list(SapiMouseAdapter(tmp_path / "sapimouse").iter_sessions())
        assert len(sessions) == 2

    def test_corpus_tags(self, tmp_path):
        assert BalabitAdapter(tmp_path).game == "balabit"
        assert SapiMouseAdapter(tmp_path).game == "sapimouse"


class TestSplitOnIdle:
    def _frame(self, ts):
        import pandas as pd

        return pd.DataFrame({"t": ts, "event_type": ["mouse_move"] * len(ts)})

    def test_splits_at_gap_and_keeps_long_segments(self):
        # two 40 s bursts separated by a 60 s idle gap
        burst1 = list(np.arange(0, 40_000, 100.0))
        burst2 = list(np.arange(100_000, 140_000, 100.0))
        segs = split_on_idle(self._frame(burst1 + burst2), gap_ms=10_000)
        assert len(segs) == 2
        assert segs[0]["t"].iloc[-1] < 40_000 <= segs[1]["t"].iloc[0]

    def test_short_segments_dropped(self):
        # 5 s burst, gap, 40 s burst → only the long one survives
        short = list(np.arange(0, 5_000, 100.0))
        long = list(np.arange(50_000, 90_000, 100.0))
        segs = split_on_idle(self._frame(short + long), gap_ms=10_000)
        assert len(segs) == 1
        assert segs[0]["t"].iloc[0] == 50_000

    def test_empty_frame(self):
        import pandas as pd

        assert split_on_idle(pd.DataFrame({"t": []})) == []


class TestVerificationMetrics:
    def test_eer_separable_distributions(self):
        rng = np.random.default_rng(0)
        genuine = rng.normal(0.9, 0.02, 500)
        impostor = rng.normal(0.1, 0.02, 500)
        e, thr = eer(genuine, impostor)
        assert e < 0.01
        assert 0.1 < thr < 0.9

    def test_eer_identical_distributions_is_chance(self):
        rng = np.random.default_rng(1)
        genuine = rng.normal(0.5, 0.1, 2000)
        impostor = rng.normal(0.5, 0.1, 2000)
        e, _ = eer(genuine, impostor)
        assert 0.45 < e < 0.55

    def test_far_at_frr_tradeoff(self):
        rng = np.random.default_rng(2)
        genuine = rng.normal(0.7, 0.1, 1000)
        impostor = rng.normal(0.3, 0.1, 1000)
        loose = far_at_frr(genuine, impostor, frr_target=0.20)
        strict = far_at_frr(genuine, impostor, frr_target=0.01)
        assert loose <= strict  # rejecting more genuines lets fewer impostors in

    def test_verification_scores_split(self):
        proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        genuine, impostor = verification_scores(proba, np.array([0, 1]))
        assert genuine.tolist() == [0.7, 0.8]
        assert sorted(impostor.tolist()) == [0.1, 0.1, 0.1, 0.2]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            eer(np.array([]), np.array([0.5]))
