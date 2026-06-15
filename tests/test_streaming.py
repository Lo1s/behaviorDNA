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
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from api.streaming import streaming_router
from pipeline.features.run import FEATURE_COLS, process_session_windows
from pipeline.inference.aggregator import RiskAggregator
from pipeline.inference.streaming import (
    WINDOW_MS,
    ScoreUpdate,
    SessionStreamState,
    compute_window_feature_row,
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
# Offline ↔ streaming feature parity (review finding H3)
# ---------------------------------------------------------------------------


def _offline_rows(events: list[dict], nf: float = 1.0, rn: float = 1.0) -> list[dict]:
    """Offline window rows for an event list (mirrors the ingestion frame)."""
    df = pd.DataFrame(
        [
            {
                "session_id": "s",
                "t": float(e["t"]),
                "event_type": e["type"],
                "x": e.get("x"),
                "y": e.get("y"),
                "dx": e.get("dx"),
                "dy": e.get("dy"),
                "pressed": e.get("pressed"),
                "key": e.get("key"),
            }
            for e in events
        ]
    )
    return process_session_windows(df, nf, rn)


def _streaming_rows(events: list[dict], nf: float = 1.0, rn: float = 1.0) -> list[dict]:
    """Reconstruct the feature rows the streaming engine produces, window by
    window, exactly as ``SessionStreamState`` drives ``compute_window_feature_row``:
    anchor at the first event, full_window=True for any window with a later event,
    full_window=False for the trailing partial window."""
    anchor = min(e["t"] for e in events)
    t_max = max(e["t"] for e in events)
    rows = []
    idx = 0
    while True:
        w_start = anchor + idx * WINDOW_MS
        if w_start > t_max:
            break
        w_end = w_start + WINDOW_MS
        has_later = any(e["t"] >= w_end for e in events)
        row = compute_window_feature_row(
            events, w_start, w_end, nf, rn, full_window=has_later
        )
        if row is not None:
            row = {**row, "window_idx": idx}
            rows.append(row)
        idx += 1
    return rows


def _assert_rows_equal(off: list[dict], strm: list[dict]) -> None:
    # Only compare windows the streaming path emits (it guards <2-event windows;
    # offline would otherwise include degenerate 1-event windows). Realistic test
    # sessions below keep every populated window at >=2 events, so the lists match.
    assert [r["window_idx"] for r in off] == [r["window_idx"] for r in strm]
    for r_off, r_str in zip(off, strm):
        assert set(r_off) == set(r_str)
        for k in r_off:
            assert r_str[k] == pytest.approx(
                r_off[k], nan_ok=True, abs=1e-9
            ), f"window {r_off['window_idx']} feature {k!r}: off={r_off[k]} strm={r_str[k]}"


class TestOfflineParity:
    """The streaming window features must equal the offline ones bit-for-bit;
    the prior re-anchored-mini-frame path inflated rate features on sparse
    windows (offline event_rate 1.0 vs streaming 1.034)."""

    def test_dense_session(self):
        # 100ms spacing across ~2.2 windows → 2 full + 1 partial-final window.
        events = [_event(i * 100.0) for i in range(660)]
        _assert_rows_equal(_offline_rows(events), _streaming_rows(events))

    def test_sparse_completed_window(self):
        # The reviewer's exact probe: 1 ev/s for 30s (sparse but COMPLETED). Two
        # boundary-crossing events make window 0 full-width and give window 1 the
        # >=2 events both paths require (avoids the degenerate 1-event window).
        events = [_event(i * 1000.0) for i in range(30)]
        events += [
            _event(WINDOW_MS + 5.0, "key_press", key="w"),
            _event(WINDOW_MS + 1005.0),
        ]
        off = _offline_rows(events)
        strm = _streaming_rows(events)
        _assert_rows_equal(off, strm)
        # And the headline: a sparse full window rates against 30s, not 29s
        # (pre-fix the streaming helper produced 1.034 here).
        assert off[0]["event_rate"] == pytest.approx(1.0, abs=1e-9)
        assert strm[0]["event_rate"] == pytest.approx(1.0, abs=1e-9)

    def test_idle_gap_session(self):
        # Window 0 populated, windows 1-2 empty (AFK), window 3 populated.
        events = [_event(i * 1000.0) for i in range(0, 25)]  # window 0
        events += [_event(3 * WINDOW_MS + i * 1000.0) for i in range(0, 25)]  # window 3
        off = _offline_rows(events)
        strm = _streaming_rows(events)
        assert [r["window_idx"] for r in off] == [0, 3]  # gap windows skipped
        _assert_rows_equal(off, strm)

    def test_partial_final_window(self):
        # Window 0 full, then a short partial window 1 (a few events, no boundary
        # crossed) → exercises the full_window=False (finalize) duration path.
        events = [_event(i * 500.0) for i in range(0, 61)]  # fills window 0 (0..30s)
        events += [_event(WINDOW_MS + i * 1000.0) for i in range(0, 5)]  # partial w1
        off = _offline_rows(events)
        strm = _streaming_rows(events)
        _assert_rows_equal(off, strm)


class TestOutOfOrderRejection:
    """Out-of-order events are rejected and counted (H3), not silently folded
    into the current window."""

    def test_decreasing_timestamp_is_dropped(self):
        state = _stream_state(chunk_length=8)
        assert state.push_event(_event(1000.0)) is None
        assert state.push_event(_event(2000.0)) is None
        # An event in the past is rejected (returns None, counter increments).
        assert state.push_event(_event(1500.0)) is None
        assert state.dropped_out_of_order == 1
        assert state.n_events == 2  # the late event was not ingested

    def test_equal_timestamp_is_kept(self):
        state = _stream_state(chunk_length=8)
        state.push_event(_event(1000.0))
        state.push_event(_event(1000.0))  # batched same-t event is fine
        assert state.dropped_out_of_order == 0
        assert state.n_events == 2


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


# ---------------------------------------------------------------------------
# Serving bundle: persist-and-load instead of fit-at-startup
# ---------------------------------------------------------------------------


class TestServingBundle:
    def test_save_load_roundtrip_scores_identically(self, tmp_path):
        from pipeline.inference.streaming import load_stream_state, save_stream_bundle

        state = _stream_state(chunk_length=8)
        bundle = tmp_path / "serving_bundle.pkl"
        save_stream_bundle(state, bundle, metadata={"test": True})

        # tmp_path has no lstm_ae.pt → loaded state has no LSTM (handled).
        loaded = load_stream_state(bundle, model_dir=tmp_path)
        assert loaded.lstm_ae_model is None
        assert sorted(loaded.classical_detectors) == sorted(state.classical_detectors)

        events = [_event(sec * 1000.0) for sec in range(60)]
        for ev in events:
            state.push_event(ev)
        for ev in events:
            loaded.push_event(ev)
        a, b = state.finalize(), loaded.finalize()
        assert abs(a.session_risk - b.session_risk) < 1e-9

    def test_schema_mismatch_raises(self, tmp_path):
        import pickle

        from pipeline.inference.streaming import load_stream_state

        bad = tmp_path / "bad.pkl"
        with open(bad, "wb") as f:
            pickle.dump({"schema_version": 999, "chunk_length": 8}, f)
        with pytest.raises(ValueError, match="schema"):
            load_stream_state(bad, model_dir=tmp_path)

    def test_load_or_build_prefers_bundle(self, tmp_path):
        # With a bundle present, the helper must LOAD it (1 detector here), never
        # fall back to build_stream_state (which fits 3 detectors from
        # data/synthetic). The detector count is the tell.
        from pipeline.inference.streaming import (
            load_or_build_stream_state,
            save_stream_bundle,
        )

        state = _stream_state(chunk_length=8)
        bundle = tmp_path / "serving_bundle.pkl"
        save_stream_bundle(state, bundle)
        loaded = load_or_build_stream_state(bundle_path=bundle)
        assert set(loaded.classical_detectors) == {"IsolationForest"}


# ---------------------------------------------------------------------------
# Bounded memory over a long (always-on) session
# ---------------------------------------------------------------------------


class TestBoundedBuffers:
    def test_window_buffer_stays_bounded_over_long_session(self):
        # 10 windows of events at 100ms spacing (~3000 events / ~300s). The
        # per-window buffer must stay ~one window's size, not grow with the whole
        # session (regression guard against the old unbounded self.events list).
        state = _stream_state(chunk_length=8)
        per_window = WINDOW_MS // 100  # events per 30s window at 100ms spacing
        n = int(per_window * 10)
        for i in range(n):
            state.push_event(_event(i * 100.0))
        assert len(state.window_buffer) <= per_window + 10
        final = state.finalize()
        assert final.n_events == n  # counter is exact even though events aren't kept
        assert final.n_windows >= 9
