"""
api/streaming.py
================
WebSocket endpoint that streams per-event scores from a live session.

Client protocol (JSON over WebSocket):

  Client → server:
    {"t": 12345.6, "type": "mouse_move", "x": 100, "y": 200, "dx": 1, "dy": 0}
    {"t": 12350.0, "type": "mouse_click", "x": 100, "y": 200, "pressed": true}
    ...
    {"type": "__end__"}      ← optional sentinel to flush final scores

  Server → client (only when a window or chunk boundary fires):
    {"t": 30000.0, "n_events": 1234, "n_windows": 1, "n_chunks": 3,
     "per_detector": {"IsolationForest": 0.42, "LSTMAutoencoder": 1.83},
     "session_risk": 0.18,
     "detector_logits": {"IsolationForest": -1.2, "LSTMAutoencoder": 0.8},
     "triggered_by": "window"}

The router exposes one route:

    /stream  — WebSocket endpoint

Mount with ``api/main.py: app.include_router(streaming_router)`` after the
existing ``/predict/*`` endpoints; the lifespan in ``api/main.py`` builds a
single shared ``SessionStreamState`` template at startup so each connection
gets a fresh stream wired to pre-loaded models.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pipeline.inference.streaming import SessionStreamState, build_stream_state

log = logging.getLogger(__name__)

streaming_router = APIRouter()


@streaming_router.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    """One WebSocket connection = one streaming session.

    Server pre-loads the LSTM-AE + classical detectors + calibrated
    aggregator once on app startup (see ``api/main.py``). Each client
    connection gets its own ``SessionStreamState`` initialised from the
    shared models, so concurrent sessions do not share state.
    """
    await websocket.accept()
    app_state = websocket.app.state

    # If the lifespan hook hasn't initialised the streaming components,
    # build them lazily here. This makes the endpoint usable even when the
    # API is started without the lifespan (e.g. in tests).
    template_state: SessionStreamState | None = getattr(
        app_state, "stream_template", None
    )
    if template_state is None:
        log.info("stream_template not found in app.state — building lazily")
        try:
            template_state = build_stream_state()
            app_state.stream_template = template_state
        except Exception as e:
            await websocket.send_json({"error": f"failed to build stream state: {e}"})
            await websocket.close()
            return

    # Each connection gets a fresh per-session state, but shares the
    # pre-loaded detectors and aggregator (read-only after fit).
    state = SessionStreamState(
        classical_detectors=template_state.classical_detectors,
        feature_scaler=template_state.feature_scaler,
        aggregator=template_state.aggregator,
        lstm_ae_model=template_state.lstm_ae_model,
        lstm_ae_stats=template_state.lstm_ae_stats,
        chunk_length=template_state.chunk_length,
        device=template_state.device,
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid JSON"})
                continue

            if payload.get("type") == "__end__":
                # Client signalled end of stream — emit a final snapshot
                final = state.finalize()
                if final is not None:
                    await websocket.send_json(final.to_dict())
                break

            update = state.push_event(payload)
            if update is not None:
                await websocket.send_json(update.to_dict())
    except WebSocketDisconnect:
        # Client disconnected — best-effort emit the final snapshot to
        # the server log, but don't send (the socket is gone).
        final = state.finalize()
        if final is not None:
            log.info(
                "Client disconnected — final session_risk=%.3f over %d events",
                final.session_risk,
                final.n_events,
            )
    except Exception as e:
        log.exception("Stream handler error: %s", e)
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
        except Exception:
            pass
