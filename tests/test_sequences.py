"""
tests/test_sequences.py
=======================
Unit tests for pipeline.sequences (preprocessing + dataset).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.sequences import (
    EVENT_FEATURE_DIM,
    EventSequenceDataset,
    apply_normalizer,
    fit_normalizer,
    session_to_event_tensor,
)
from pipeline.sequences.preprocessing import (
    COL_DT,
    COL_DX,
    COL_DY,
    COL_IS_KEY_PRESS,
    COL_IS_KEY_RELEASE,
    COL_IS_MOUSE_CLICK_PRESS,
    COL_IS_MOUSE_MOVE,
    COL_IS_MOUSE_SCROLL,
    deserialize_stats,
    serialize_stats,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_session(
    events: list[dict], sensitivity: float = 1.0, dpi: float = 800.0
) -> dict:
    return {
        "session_id": "test1234",
        "player": "test",
        "game": "test_game",
        "sensitivity": sensitivity,
        "dpi": dpi,
        "duration_ms": (events[-1]["t"] if events else 0.0),
        "event_count": len(events),
        "events": events,
    }


@pytest.fixture
def mixed_session() -> dict:
    return make_session(
        [
            {"t": 0.0, "type": "mouse_move", "x": 0, "y": 0, "dx": 5, "dy": 3},
            {"t": 100.0, "type": "mouse_click", "x": 5, "y": 3, "pressed": True},
            {"t": 102.0, "type": "mouse_click", "x": 5, "y": 3, "pressed": False},
            {"t": 200.0, "type": "key_press", "key": "w"},
            {"t": 300.0, "type": "key_release", "key": "w"},
            {"t": 400.0, "type": "mouse_scroll", "x": 5, "y": 3, "dx": 0, "dy": -1},
            {"t": 500.0, "type": "mouse_move", "x": 10, "y": 6, "dx": 5, "dy": 3},
        ],
        sensitivity=1.0,
        dpi=800.0,
    )


# ---------------------------------------------------------------------------
# TestSessionToEventTensor
# ---------------------------------------------------------------------------


class TestSessionToEventTensor:
    def test_empty_session_returns_zero_rows(self):
        t = session_to_event_tensor(make_session([]))
        assert t.shape == (0, EVENT_FEATURE_DIM)
        assert t.dtype == np.float32

    def test_shape_matches_event_count(self, mixed_session):
        t = session_to_event_tensor(mixed_session)
        assert t.shape == (len(mixed_session["events"]), EVENT_FEATURE_DIM)

    def test_dt_first_event_is_zero(self, mixed_session):
        # log1p(0) = 0
        t = session_to_event_tensor(mixed_session)
        assert t[0, COL_DT] == 0.0

    def test_dt_log1p_compression(self, mixed_session):
        # Second event at t=100, prev at t=0 → dt=100, log1p(100) ≈ 4.615
        t = session_to_event_tensor(mixed_session)
        assert abs(t[1, COL_DT] - math.log1p(100.0)) < 1e-5

    def test_one_hot_indicators(self, mixed_session):
        t = session_to_event_tensor(mixed_session)
        # Event 0: mouse_move
        assert t[0, COL_IS_MOUSE_MOVE] == 1.0
        assert t[0, COL_IS_MOUSE_CLICK_PRESS] == 0.0
        # Event 1: mouse_click pressed=True
        assert t[1, COL_IS_MOUSE_CLICK_PRESS] == 1.0
        assert t[1, COL_IS_MOUSE_MOVE] == 0.0
        # Event 2: mouse_click pressed=False → all one-hots zero
        assert t[2, COL_IS_MOUSE_CLICK_PRESS] == 0.0
        assert t[2, COL_IS_KEY_PRESS] == 0.0
        # Event 3: key_press
        assert t[3, COL_IS_KEY_PRESS] == 1.0
        # Event 4: key_release
        assert t[4, COL_IS_KEY_RELEASE] == 1.0
        # Event 5: mouse_scroll
        assert t[5, COL_IS_MOUSE_SCROLL] == 1.0

    def test_dx_dy_normalized_by_dpi_sensitivity(self):
        session = make_session(
            [{"t": 0.0, "type": "mouse_move", "x": 0, "y": 0, "dx": 16, "dy": 8}],
            sensitivity=2.0,
            dpi=1600.0,
        )
        t = session_to_event_tensor(session)
        # norm_factor = 2.0 * 1600 / 800 = 4.0 → dx_norm = 16/4 = 4.0
        assert abs(t[0, COL_DX] - 4.0) < 1e-6
        assert abs(t[0, COL_DY] - 2.0) < 1e-6

    def test_non_mouse_events_have_zero_dx_dy(self, mixed_session):
        t = session_to_event_tensor(mixed_session)
        # Event 3 (key_press): dx/dy should be 0
        assert t[3, COL_DX] == 0.0
        assert t[3, COL_DY] == 0.0

    def test_handles_missing_dpi_gracefully(self):
        session = {"events": [{"t": 0.0, "type": "mouse_move", "dx": 8, "dy": 4}]}
        t = session_to_event_tensor(session)
        # Default sensitivity=1, dpi=800 → norm_factor=1 → dx_norm=8, dy_norm=4
        assert abs(t[0, COL_DX] - 8.0) < 1e-6


# ---------------------------------------------------------------------------
# TestNormalizer
# ---------------------------------------------------------------------------


class TestNormalizer:
    def test_empty_list_returns_unit_stats(self):
        stats = fit_normalizer([])
        assert np.allclose(stats["mean"], 0.0)
        assert np.allclose(stats["std"], 1.0)
        assert stats["mean"].shape == (EVENT_FEATURE_DIM,)

    def test_fit_matches_stacked_stats(self, mixed_session):
        t1 = session_to_event_tensor(mixed_session)
        t2 = session_to_event_tensor(mixed_session)  # same data twice
        stats = fit_normalizer([t1, t2])
        stacked = np.concatenate([t1, t2], axis=0)
        np.testing.assert_allclose(stats["mean"], stacked.mean(axis=0), rtol=1e-5)

    def test_zero_variance_channel_gets_unit_std(self):
        # Three identical events → all channels constant → std should be 1 not 0
        t = np.ones((3, EVENT_FEATURE_DIM), dtype=np.float32)
        stats = fit_normalizer([t])
        assert (stats["std"] >= 1.0).all()

    def test_apply_normalizer_zero_mean_unit_std(self, mixed_session):
        t = session_to_event_tensor(mixed_session)
        stats = fit_normalizer([t])
        normalized = apply_normalizer(t, stats)
        # After z-scoring on the same data, columns with variance have mean ≈ 0
        nonzero_var = stats["std"] > 1.0  # excludes our forced-unit-std cols
        if nonzero_var.any():
            assert abs(normalized[:, nonzero_var].mean()) < 1e-5

    def test_apply_on_empty_returns_empty(self):
        stats = fit_normalizer([np.ones((2, EVENT_FEATURE_DIM), dtype=np.float32)])
        empty = np.zeros((0, EVENT_FEATURE_DIM), dtype=np.float32)
        out = apply_normalizer(empty, stats)
        assert out.shape == (0, EVENT_FEATURE_DIM)

    def test_serialize_roundtrip(self, mixed_session):
        t = session_to_event_tensor(mixed_session)
        stats = fit_normalizer([t])
        roundtripped = deserialize_stats(serialize_stats(stats))
        np.testing.assert_allclose(stats["mean"], roundtripped["mean"])
        np.testing.assert_allclose(stats["std"], roundtripped["std"])


# ---------------------------------------------------------------------------
# TestEventSequenceDataset
# ---------------------------------------------------------------------------


class TestEventSequenceDataset:
    def test_short_session_yields_no_chunks(self):
        t = np.zeros((10, EVENT_FEATURE_DIM), dtype=np.float32)
        ds = EventSequenceDataset([t], chunk_length=64, stride=32)
        assert len(ds) == 0

    def test_exact_chunk_yields_one(self):
        t = np.zeros((64, EVENT_FEATURE_DIM), dtype=np.float32)
        ds = EventSequenceDataset([t], chunk_length=64, stride=32)
        assert len(ds) == 1

    def test_overlapping_chunks_count(self):
        # 200 events, chunk=64, stride=32 → start indices 0,32,64,96,128 → 5 chunks
        t = np.zeros((200, EVENT_FEATURE_DIM), dtype=np.float32)
        ds = EventSequenceDataset([t], chunk_length=64, stride=32)
        # last_start = 200 - 64 = 136; starts: 0,32,64,96,128 → 5 chunks
        assert len(ds) == 5

    def test_no_overlap_chunks_count(self):
        t = np.zeros((192, EVENT_FEATURE_DIM), dtype=np.float32)
        ds = EventSequenceDataset([t], chunk_length=64, stride=64)
        # 192 / 64 = 3 chunks
        assert len(ds) == 3

    def test_chunk_shape_and_dtype(self):
        t = np.random.RandomState(0).randn(100, EVENT_FEATURE_DIM).astype(np.float32)
        ds = EventSequenceDataset([t], chunk_length=32, stride=16)
        chunk = ds[0]
        assert tuple(chunk.shape) == (32, EVENT_FEATURE_DIM)
        assert chunk.dtype.is_floating_point

    def test_invalid_feature_dim_raises(self):
        bad = np.zeros((10, 5), dtype=np.float32)  # wrong dim
        with pytest.raises(ValueError, match="shape"):
            EventSequenceDataset([bad], chunk_length=5, stride=1)

    def test_multiple_sessions_concatenated(self):
        t1 = np.zeros((128, EVENT_FEATURE_DIM), dtype=np.float32)
        t2 = np.zeros((96, EVENT_FEATURE_DIM), dtype=np.float32)
        ds = EventSequenceDataset(
            [t1, t2], chunk_length=64, stride=32, session_ids=["a", "b"]
        )
        # t1: starts 0,32,64 → 3 chunks; t2: starts 0,32 → 2 chunks → 5 total
        assert len(ds) == 5
        # First chunk is from session a, last from session b
        assert ds.chunk_origin(0)[0] == "a"
        assert ds.chunk_origin(len(ds) - 1)[0] == "b"

    def test_session_chunk_counts(self):
        t1 = np.zeros((128, EVENT_FEATURE_DIM), dtype=np.float32)
        t2 = np.zeros((96, EVENT_FEATURE_DIM), dtype=np.float32)
        ds = EventSequenceDataset(
            [t1, t2], chunk_length=64, stride=32, session_ids=["a", "b"]
        )
        counts = ds.session_chunk_counts()
        assert counts == {"a": 3, "b": 2}

    def test_chunk_data_matches_underlying_tensor(self):
        t = np.arange(200 * EVENT_FEATURE_DIM, dtype=np.float32).reshape(
            200, EVENT_FEATURE_DIM
        )
        ds = EventSequenceDataset([t], chunk_length=64, stride=32)
        # chunk 0 = t[0:64], chunk 1 = t[32:96]
        np.testing.assert_array_equal(ds[0].numpy(), t[0:64])
        np.testing.assert_array_equal(ds[1].numpy(), t[32:96])

    def test_session_ids_length_must_match(self):
        t = np.zeros((100, EVENT_FEATURE_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="session_ids"):
            EventSequenceDataset([t, t], session_ids=["only_one"])
