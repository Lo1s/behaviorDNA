"""
tests/test_streaming.py
=======================
Tests for the streaming inference engine (pipeline.inference.streaming)
plus the WebSocket endpoint in api/streaming.py.

Engine tests build a ``SessionStreamState`` directly (no transport, no API),
feed it synthetic events, and assert the right callbacks fire at the
right boundaries. WebSocket tests use FastAPI's TestClient against the
real router with a hand-constructed stream_template (no model load).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from api.streaming import streaming_router
from pipeline.features.run import FEATURE_COLS
from pipeline.inference.aggregator import RiskAggregator
from pipeline.inference.streaming import (
    WINDOW_MS,
    ScoreUpdate,
    SessionStreamState,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fitted_aggregator() -> RiskAggregator:
    rng = np.random.default_rng(0)
    n = 100
    legit = rng.normal(0, 1, n)
    cheat = rng.normal(3, 1, n)
    scores = np.concatenate([legit, cheat])
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    return RiskAggregator(prior_cheat_rate=0.05).fit(
        {"IsolationForest": (scores, labels)}
    )


def _fitted_detector_and_scaler():
    rng = np.random.default_rng(0)
    # 50 legit windows × N features (track FEATURE_COLS so feature additions don't break the fixture)
    X = rng.normal(0, 1, (50, len(FEATURE_COLS)))
    scaler = StandardScaler().fit(X)
    det = IsolationForest(n_estimators=20, contamination=0.05, random_state=0)
    det.fit(scaler.transform(X))
    return det, scaler


def _stream_state(chunk_length: int = 8) -> SessionStreamState:
    det, scaler = _fitted_detector_and_scaler()
    return SessionStreamState(
        classical_detectors={"IsolationForest": det},
        feature_scaler=scaler,
        aggregator=_fitted_aggregator(),
        lstm_ae_model=None,  # skip LSTM in the unit test
        lstm_ae_stats=None,
        chunk_length=chunk_length,
    )


def _event(t: float, event_type: str = "mouse_move", **kw) -> dict:
    base = {"t": float(t), "type": event_type, "x": 100, "y": 100, "dx": 1, "dy": 1}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Basic state machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_no_update_until_first_boundary(self):
        state = _stream_state(chunk_length=8)
        for i in range(5):
            assert state.push_event(_event(i * 10.0)) is None

    def test_chunk_boundary_fires_update(self):
        # No LSTM model loaded so chunk flush is a no-op, but the state machine
        # still records chunks as 'consumed' — we don't get a triggered chunk.
        # Force a window flush instead by spanning WINDOW_MS.
        state = _stream_state(chunk_length=8)
        # Send 7 events well under WINDOW_MS → no triggers
        for i in range(7):
            assert state.push_event(_event(i * 10.0)) is None
        # 8th event still under WINDOW_MS, but it completes a chunk-length;
        # however the chunk flush is a no-op without the LSTM model. So no
        # update should fire from the chunk path.
        # Instead, send an event past the first window boundary → triggers window flush
        update = state.push_event(_event(WINDOW_MS + 1.0, "key_press", key="w"))
        assert isinstance(update, ScoreUpdate)
        assert update.triggered_by in ("window", "chunk")
        assert update.n_windows >= 1
        assert "IsolationForest" in update.per_detector

    def test_multiple_windows_accumulate(self):
        state = _stream_state(chunk_length=8)
        # Send events spanning 3 windows
        for sec in range(0, 90):  # 90 events at 1s spacing → spans 3×30s windows
            state.push_event(_event(sec * 1000.0))
        # Plus one more event well past the last window boundary
        last = state.push_event(_event(91_000.0))
        assert last is not None
        assert last.n_windows >= 2
        # Running max only goes up
        max_seen = last.per_detector["IsolationForest"]
        # Another event shouldn't lower it
        nxt = state.push_event(_event(120_000.0))
        if nxt is not None:
            assert nxt.per_detector["IsolationForest"] >= max_seen - 1e-9

    def test_session_risk_in_unit_interval(self):
        state = _stream_state(chunk_length=8)
        for sec in range(0, 60):
            state.push_event(_event(sec * 1000.0))
        update = state.push_event(_event(61_000.0))
        if update is not None:
            assert 0.0 <= update.session_risk <= 1.0

    def test_finalize_returns_snapshot_after_any_events(self):
        state = _stream_state(chunk_length=8)
        for i in range(10):
            state.push_event(_event(i * 100.0))
        final = state.finalize()
        assert final is not None
        assert final.triggered_by == "finalize"
        assert final.n_events == 10

    def test_finalize_returns_none_with_no_events(self):
        state = _stream_state(chunk_length=8)
        assert state.finalize() is None

    def test_score_update_to_dict(self):
        """ScoreUpdate must serialise cleanly to JSON-friendly types."""
        update = ScoreUpdate(
            t=1234.5,
            n_events=10,
            n_windows=1,
            n_chunks=2,
            per_detector={"IsolationForest": 0.42},
            session_risk=0.13,
            detector_logits={"IsolationForest": -1.0},
            triggered_by="window",
        )
        d = update.to_dict()
        assert d["t"] == 1234.5
        assert d["per_detector"]["IsolationForest"] == 0.42
        assert d["triggered_by"] == "window"


# ---------------------------------------------------------------------------
# Per-session hardware normalisation
# ---------------------------------------------------------------------------


class TestConfigureForSession:
    def test_sets_norm_factor_and_rate_norm(self):
        state = _stream_state()
        assert state.norm_factor == 1.0 and state.rate_norm == 1.0
        state.configure_for_session(sensitivity=25.0, dpi=800, polling_rate=500)
        assert state.norm_factor == pytest.approx(25.0)  # 25*800/800
        assert state.rate_norm == pytest.approx(2.0)  # 1000/500

    def test_missing_fields_leave_current_value(self):
        state = _stream_state()
        state.configure_for_session(sensitivity=2.0, dpi=1600)  # no polling_rate
        assert state.norm_factor == pytest.approx(4.0)  # 2*1600/800
        assert state.rate_norm == 1.0  # untouched


# ---------------------------------------------------------------------------
# finalize() partial-buffer policy
# ---------------------------------------------------------------------------


class TestFinalizePartialBuffers:
    def test_finalize_flushes_trailing_partial_window(self):
        # 63 events inside the first 30s window, chunk_length 64: nothing flushes
        # during push (no boundary crossed, chunk incomplete). finalize() must
        # score the trailing partial window; the partial chunk is discarded.
        state = _stream_state(chunk_length=64)
        for i in range(63):
            assert state.push_event(_event(i * 100.0)) is None
        final = state.finalize()
        assert final is not None
        assert final.n_windows == 1  # partial window now scored
        assert final.n_chunks == 0  # partial chunk discarded (LSTM needs full chunk)
        assert "IsolationForest" in final.per_detector

    def test_finalize_is_idempotent(self):
        state = _stream_state(chunk_length=64)
        for i in range(63):
            state.push_event(_event(i * 100.0))
        first = state.finalize()
        second = state.finalize()
        assert first.n_windows == second.n_windows == 1  # no double-count


# ---------------------------------------------------------------------------
# Determinism: same events in → same final scores
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_gives_same_output(self):
        events = [_event(sec * 1000.0) for sec in range(60)]

        state_a = _stream_state(chunk_length=8)
        for ev in events:
            state_a.push_event(ev)
        final_a = state_a.finalize()

        state_b = _stream_state(chunk_length=8)
        for ev in events:
            state_b.push_event(ev)
        final_b = state_b.finalize()

        assert final_a is not None and final_b is not None
        assert abs(final_a.session_risk - final_b.session_risk) < 1e-9
        for det in final_a.per_detector:
            assert abs(final_a.per_detector[det] - final_b.per_detector[det]) < 1e-9


# ---------------------------------------------------------------------------
# Chunk-length math
# ---------------------------------------------------------------------------


class TestChunkBookkeeping:
    def test_event_count_matches_pushes(self):
        state = _stream_state(chunk_length=8)
        for i in range(25):
            state.push_event(_event(i * 100.0))
        final = state.finalize()
        assert final.n_events == 25

    @pytest.mark.parametrize("chunk_length", [4, 16, 32])
    def test_no_lstm_chunk_scores_when_model_absent(self, chunk_length):
        state = _stream_state(chunk_length=chunk_length)
        for i in range(chunk_length * 3):
            state.push_event(_event(i * 100.0))
        final = state.finalize()
        # No LSTM model loaded → no chunk scores accumulated
        assert final.n_chunks == 0
        # Therefore no LSTMAutoencoder key in per_detector
        assert "LSTMAutoencoder" not in final.per_detector


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


def _ws_test_app() -> TestClient:
    """Spin up a FastAPI app with just the streaming router + a hand-crafted
    stream_template. Bypasses the real lifespan (no model files needed)."""
    app = FastAPI()
    app.include_router(streaming_router)
    app.state.stream_template = _stream_state(chunk_length=8)
    return TestClient(app)


class TestWebSocketEndpoint:
    def test_accepts_connection_and_handles_events(self):
        client = _ws_test_app()
        with client.websocket_connect("/stream") as ws:
            # Send 60 events to cross a 30s window boundary
            for sec in range(0, 60):
                ws.send_text(
                    json.dumps(
                        {
                            "t": sec * 1000.0,
                            "type": "mouse_move",
                            "x": 1,
                            "y": 1,
                            "dx": 1,
                            "dy": 1,
                        }
                    )
                )
            # Send the final event well past the boundary
            ws.send_text(json.dumps({"t": 61_000.0, "type": "key_press", "key": "w"}))
            # Send sentinel to flush a final snapshot
            ws.send_text(json.dumps({"type": "__end__"}))
            # Collect any messages until the socket closes
            messages: list[dict] = []
            while True:
                try:
                    msg = ws.receive_json()
                except Exception:
                    break
                messages.append(msg)
                if msg.get("triggered_by") == "finalize":
                    break
            # At least one update should have fired
            assert len(messages) >= 1
            assert all("session_risk" in m for m in messages)

    def test_session_metadata_message_is_accepted_and_not_an_event(self):
        client = _ws_test_app()
        with client.websocket_connect("/stream") as ws:
            # __session__ first message must configure, not be scored as an event
            ws.send_text(
                json.dumps(
                    {
                        "type": "__session__",
                        "sensitivity": 25.0,
                        "dpi": 800,
                        "polling_rate": 500,
                    }
                )
            )
            for sec in range(0, 61):
                ws.send_text(
                    json.dumps(
                        {
                            "t": sec * 1000.0,
                            "type": "mouse_move",
                            "x": 1,
                            "y": 1,
                            "dx": 1,
                            "dy": 1,
                        }
                    )
                )
            ws.send_text(json.dumps({"type": "__end__"}))
            messages: list[dict] = []
            while True:
                try:
                    msg = ws.receive_json()
                except Exception:
                    break
                messages.append(msg)
                if msg.get("triggered_by") == "finalize":
                    break
            assert len(messages) >= 1
            # 61 events pushed; the __session__ message is NOT counted as one
            assert messages[-1]["n_events"] == 61

    def test_invalid_json_returns_error_then_continues(self):
        client = _ws_test_app()
        with client.websocket_connect("/stream") as ws:
            ws.send_text("not valid json")
            response = ws.receive_json()
            assert "error" in response
            # Socket should still be open — send a valid sentinel and expect to close
            ws.send_text(json.dumps({"type": "__end__"}))
