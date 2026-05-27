"""
pipeline/inference/streaming.py
===============================
Transport-independent streaming-inference engine.

The same ``SessionStreamState`` is driven by both the WebSocket API (in
production / live demos) and the replay client (in tests and the dashboard
demo). It maintains:

- a sliding **window buffer** for the 25-D classical features computed
  every 30 s, scored by every classical detector loaded at startup;
- a sliding **chunk buffer** for the LSTM-AE chunk-level reconstruction
  score, computed every ``chunk_length`` events;
- per-detector running session statistics (max-so-far for classical;
  p95-so-far for LSTM-AE chunks) plus the live ``RiskAggregator``
  combined score.

The engine is in-memory only and assumes events arrive monotonically by
``t`` within a session. Out-of-order events are tolerated but window
boundaries are decided by the timestamp of the *first* arrived event,
not wall-clock time — so this is fundamentally an event-driven engine.

Output: every ``push_event(event)`` call returns either ``None`` (no new
scores yet) or a ``ScoreUpdate`` carrying the latest per-detector and
session-level numbers. Callers (WebSocket API, replay loop, tests)
forward those to their downstream consumer.

See ``docs/STREAMING.md`` for the full architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.adversarial.benchmark import _build_detectors, load_synthetic_features
from pipeline.features.run import FEATURE_COLS, process_session_windows
from pipeline.inference.aggregator import RiskAggregator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for the public interface
# ---------------------------------------------------------------------------


@dataclass
class ScoreUpdate:
    """Snapshot emitted by ``SessionStreamState.push_event``."""

    t: float  # session timestamp (ms) of the event that triggered the update
    n_events: int
    n_windows: int
    n_chunks: int
    per_detector: dict[str, float] = field(default_factory=dict)
    """Running per-detector session score (max for classical, p95 for LSTM)."""
    session_risk: float = 0.0
    """Combined risk in [0, 1] from the aggregator."""
    detector_logits: dict[str, float] = field(default_factory=dict)
    """Per-detector logit contributions, useful for the dashboard's stacked-area chart."""
    triggered_by: str = ""  # "window" | "chunk" | ""

    def to_dict(self) -> dict:
        return {
            "t": float(self.t),
            "n_events": int(self.n_events),
            "n_windows": int(self.n_windows),
            "n_chunks": int(self.n_chunks),
            "per_detector": {k: float(v) for k, v in self.per_detector.items()},
            "session_risk": float(self.session_risk),
            "detector_logits": {k: float(v) for k, v in self.detector_logits.items()},
            "triggered_by": self.triggered_by,
        }


# ---------------------------------------------------------------------------
# Streaming engine
# ---------------------------------------------------------------------------


WINDOW_MS = 30_000


class SessionStreamState:
    """Online scorer that ingests events one at a time.

    Construct once per session, then call ``push_event(event)`` for every
    incoming event. The constructor wires up:

    - classical detectors (already fitted on legit-only window features),
    - the persisted LSTM-AE (loaded lazily on first chunk),
    - the fitted ``RiskAggregator`` (per-detector calibrators + prior).
    """

    def __init__(
        self,
        classical_detectors: dict[str, object],
        feature_scaler,
        aggregator: RiskAggregator,
        lstm_ae_model=None,
        lstm_ae_stats: dict | None = None,
        chunk_length: int = 64,
        norm_factor: float = 1.0,
        device: str = "auto",
    ) -> None:
        self.classical_detectors = classical_detectors
        self.feature_scaler = feature_scaler
        self.aggregator = aggregator
        self.lstm_ae_model = lstm_ae_model
        self.lstm_ae_stats = lstm_ae_stats
        self.chunk_length = chunk_length
        self.norm_factor = norm_factor
        self.device = device

        # Per-session running state
        self.events: list[dict] = []
        self.first_event_t: Optional[float] = None
        self.next_window_end_t: Optional[float] = None
        self.completed_windows: list[dict] = []
        self.chunk_buffer: list[dict] = []
        self.lstm_chunk_scores: list[float] = []

        # Running per-detector aggregates (max for classical, p95 for LSTM)
        self._classical_max: dict[str, float] = {
            name: -np.inf for name in classical_detectors
        }
        self._last_update: Optional[ScoreUpdate] = None

    # -------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------

    def push_event(self, event: dict) -> Optional[ScoreUpdate]:
        """Ingest one event. Return a ``ScoreUpdate`` only if scoring fired.

        ``event`` is the same dict shape the recorder emits:
        ``{"t": ..., "type": ..., "x": ..., "y": ..., "dx": ..., "dy": ..., ...}``.
        """
        t = float(event.get("t", 0.0))
        if self.first_event_t is None:
            self.first_event_t = t
            self.next_window_end_t = t + WINDOW_MS

        self.events.append(event)
        self.chunk_buffer.append(event)

        triggered: list[str] = []

        # Flush a window when we cross its end timestamp
        while self.next_window_end_t is not None and t >= self.next_window_end_t:
            self._flush_window(
                self.next_window_end_t - WINDOW_MS, self.next_window_end_t
            )
            self.next_window_end_t += WINDOW_MS
            triggered.append("window")

        # Flush a chunk when the buffer reaches chunk_length
        while len(self.chunk_buffer) >= self.chunk_length:
            self._flush_chunk()
            triggered.append("chunk")

        if not triggered:
            return None

        return self._build_update(t, triggered[-1])

    def finalize(self, final_t: Optional[float] = None) -> Optional[ScoreUpdate]:
        """Force a score update at the end of a session (e.g. on WS disconnect)."""
        if not self.events:
            return None
        t = (
            float(final_t)
            if final_t is not None
            else float(self.events[-1].get("t", 0.0))
        )
        # No new buffers to flush — just emit the latest aggregate snapshot.
        return self._build_update(t, triggered_by="finalize")

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _flush_window(self, w_start: float, w_end: float) -> None:
        """Compute classical-detector scores for the events in [w_start, w_end)."""
        import pandas as pd

        window_events = [e for e in self.events if w_start <= e.get("t", 0.0) < w_end]
        # process_session_windows divides by (t_max - t_anchor); a single-event
        # window has zero duration and triggers a divide-by-zero. Two events
        # is the minimum that produces a meaningful feature row.
        if len(window_events) < 2:
            return

        rows = []
        for ev in window_events:
            rows.append(
                {
                    "session_id": "stream",
                    "t": float(ev.get("t", 0.0)),
                    "event_type": ev.get("type", ""),
                    "x": ev.get("x"),
                    "y": ev.get("y"),
                    "dx": ev.get("dx"),
                    "dy": ev.get("dy"),
                    "pressed": ev.get("pressed"),
                    "key": ev.get("key"),
                }
            )
        df = pd.DataFrame(rows)
        windows = process_session_windows(df, self.norm_factor)
        if not windows:
            return

        # process_session_windows returns *zero-anchored* windows; since this
        # batch contains exactly one 30s span it will produce 1 row.
        feat_row = windows[0]
        # NaN can come from feature helpers when the window has e.g. no clicks
        # or no key events. The training pipeline does the same fillna(0.0).
        x_vals = []
        for col in FEATURE_COLS:
            v = feat_row.get(col, 0.0)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0.0
            x_vals.append(float(v))
        x = np.array(x_vals, dtype=np.float64).reshape(1, -1)
        x_scaled = self.feature_scaler.transform(x)

        per_detector_now: dict[str, float] = {}
        for name, det in self.classical_detectors.items():
            score = float(-det.score_samples(x_scaled)[0])
            per_detector_now[name] = score
            if score > self._classical_max[name]:
                self._classical_max[name] = score

        self.completed_windows.append(
            {"start": w_start, "end": w_end, **per_detector_now}
        )

    def _flush_chunk(self) -> None:
        """Score the next ``chunk_length`` events with the LSTM-AE."""
        if self.lstm_ae_model is None or self.lstm_ae_stats is None:
            # No LSTM — just drop the chunk's worth of events and continue
            del self.chunk_buffer[: self.chunk_length]
            return

        import torch

        from pipeline.models.lstm_ae import score_sequences
        from pipeline.sequences.preprocessing import (
            apply_normalizer,
            session_to_event_tensor,
        )

        chunk_events = self.chunk_buffer[: self.chunk_length]
        del self.chunk_buffer[: self.chunk_length]

        # Convert to (chunk_length, 8) tensor using the same path as offline
        tensor = session_to_event_tensor(
            {"events": chunk_events, "sensitivity": 1.0, "dpi": 800.0}
        )
        if len(tensor) < self.chunk_length:
            return
        tensor = tensor[: self.chunk_length]
        normalized = apply_normalizer(tensor, self.lstm_ae_stats)
        scores = score_sequences(
            self.lstm_ae_model,
            torch.from_numpy(normalized[None]).float(),
            batch_size=1,
            device=self.device,
        )
        self.lstm_chunk_scores.append(float(scores[0]))

    def _build_update(self, t: float, triggered_by: str) -> ScoreUpdate:
        per_detector = {
            name: (val if val != -np.inf else 0.0)
            for name, val in self._classical_max.items()
        }
        if self.lstm_chunk_scores:
            per_detector["LSTMAutoencoder"] = float(
                np.percentile(self.lstm_chunk_scores, 95)
            )

        explanation = self.aggregator.explain(per_detector)
        update = ScoreUpdate(
            t=t,
            n_events=len(self.events),
            n_windows=len(self.completed_windows),
            n_chunks=len(self.lstm_chunk_scores),
            per_detector=per_detector,
            session_risk=float(explanation["posterior_risk"]),
            detector_logits=dict(explanation["per_detector_logit"]),
            triggered_by=triggered_by,
        )
        self._last_update = update
        return update


# ---------------------------------------------------------------------------
# Factory: build a SessionStreamState ready to score
# ---------------------------------------------------------------------------


def build_stream_state(
    *,
    synthetic_dir: Path | None = None,
    model_dir: Path | None = None,
    prior_cheat_rate: float = 0.05,
    chunk_length: int = 64,
    device: str = "auto",
) -> SessionStreamState:
    """Wire up classical detectors + LSTM-AE + aggregator from disk artifacts.

    Used by the WebSocket API on startup and by the replay client when
    driving the engine offline.
    """
    from sklearn.preprocessing import StandardScaler

    if synthetic_dir is None:
        synthetic_dir = Path(__file__).resolve().parents[2] / "data" / "synthetic"
    if model_dir is None:
        model_dir = Path(__file__).resolve().parents[2] / "models"

    # Fit the classical detectors on legit-only window features
    feats = load_synthetic_features(synthetic_dir)
    X = feats[FEATURE_COLS].fillna(0.0).to_numpy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    legit_mask = feats["cheat_label"].eq("legit").to_numpy()
    legit_X = X_scaled[legit_mask]

    detectors = _build_detectors()
    for det in detectors.values():
        det.fit(legit_X)

    # Load LSTM-AE if persisted
    lstm_ae_model = None
    lstm_ae_stats = None
    try:
        from pipeline.models.lstm_ae import LSTM_AE_WEIGHTS_NAME, load_lstm_ae

        if (model_dir / LSTM_AE_WEIGHTS_NAME).exists():
            lstm_ae_model, lstm_ae_stats, meta = load_lstm_ae(model_dir, device=device)
            chunk_length = int(
                (meta.get("config") or {}).get("chunk_length", chunk_length)
            )
            log.info("Loaded LSTM-AE artifact (chunk_length=%d)", chunk_length)
        else:
            log.warning(
                "No LSTM-AE artifact at %s — streaming will skip the chunk-level signal",
                model_dir,
            )
    except ImportError:
        log.warning("torch unavailable — streaming will skip the chunk-level signal")

    # Fit the aggregator on the same training-half + LSTM signal we used in
    # the offline benchmark, so the streaming risk score uses the same
    # calibration as the post-hoc reports.
    from pipeline.adversarial.benchmark import (
        _collect_per_session_scores,
    )

    scores_by_session, label_by_session = _collect_per_session_scores(synthetic_dir)
    detector_names = sorted({d for s in scores_by_session.values() for d in s})
    train_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for det in detector_names:
        s, y = [], []
        for f in scores_by_session:
            if det in scores_by_session[f]:
                s.append(scores_by_session[f][det])
                y.append(0 if label_by_session[f] == "legit" else 1)
        if s:
            train_data[det] = (
                np.asarray(s, dtype=np.float64),
                np.asarray(y, dtype=np.int64),
            )
    aggregator = RiskAggregator(prior_cheat_rate=prior_cheat_rate).fit(train_data)

    return SessionStreamState(
        classical_detectors=detectors,
        feature_scaler=scaler,
        aggregator=aggregator,
        lstm_ae_model=lstm_ae_model,
        lstm_ae_stats=lstm_ae_stats,
        chunk_length=chunk_length,
        device=device,
    )
