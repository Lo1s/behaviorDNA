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
import pandas as pd

from pipeline.adversarial.benchmark import _build_detectors, load_synthetic_features
from pipeline.constants import WINDOW_MS
from pipeline.features.run import (
    CHEAT_FEATURE_COLS,
    polling_rate_norm,
    process_session_windows,
)
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
    lstm_chunk_score: Optional[float] = None
    """Instantaneous LSTM-AE reconstruction error of the most recent chunk
    (None until the first chunk fires). The running p95 of these is what lands
    in ``per_detector['LSTMAutoencoder']``; this raw per-chunk value is the
    chunk-level cheat signal the Phase-4 demo plots over time."""

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
            "lstm_chunk_score": (
                float(self.lstm_chunk_score)
                if self.lstm_chunk_score is not None
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Pure scoring helpers (no engine state) — separately unit-testable.
# The state machine below owns *when* to score (buffering + window/chunk
# boundaries); these own *how* to turn a completed unit into scores.
# ---------------------------------------------------------------------------


def compute_window_feature_row(
    events: list[dict],
    w_start: float,
    w_end: float,
    norm_factor: float,
    rate_norm: float,
) -> Optional[dict]:
    """Events in [w_start, w_end) → one classical feature-row dict (or None).

    Mirrors the offline feature pipeline exactly. Returns None when the window
    has fewer than 2 events: ``process_session_windows`` divides by the window
    span (t_max − t_anchor), so a single-event window has zero duration.
    """
    window_events = [e for e in events if w_start <= e.get("t", 0.0) < w_end]
    if len(window_events) < 2:
        return None
    rows = [
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
        for ev in window_events
    ]
    windows = process_session_windows(pd.DataFrame(rows), norm_factor, rate_norm)
    # One 30s span → one zero-anchored window row.
    return windows[0] if windows else None


def score_window_features(
    feature_row: dict, scaler, detectors: dict[str, object]
) -> dict[str, float]:
    """Feature-row dict → {detector_name: anomaly_score} (higher = more anomalous).

    NaN features (e.g. a window with no clicks/keys) are filled with 0.0, the
    same as the training pipeline.
    """
    x_vals = []
    for col in CHEAT_FEATURE_COLS:
        v = feature_row.get(col, 0.0)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0
        x_vals.append(float(v))
    x_scaled = scaler.transform(np.array(x_vals, dtype=np.float64).reshape(1, -1))
    return {
        name: float(-det.score_samples(x_scaled)[0]) for name, det in detectors.items()
    }


def score_chunk(
    events: list[dict],
    model,
    stats: dict,
    chunk_length: int,
    norm_factor: float,
    device: str,
) -> Optional[float]:
    """chunk_length events → LSTM-AE reconstruction error (or None if too short).

    torch + the sequence model are imported lazily so importing this module and
    running the classical-only path never require torch.
    """
    import torch

    from pipeline.models.lstm_ae import score_sequences
    from pipeline.sequences.preprocessing import (
        apply_normalizer,
        session_to_event_tensor,
    )

    # session_to_event_tensor recomputes norm_factor = sensitivity*dpi/800, so we
    # feed sensitivity=norm_factor (dpi=800) to reproduce this session's actual
    # norm_factor exactly (see SessionStreamState.configure_for_session). Hardcoding
    # 1.0/800 mis-scaled mouse kinematics on non-default hardware (sens≠1, dpi≠800).
    tensor = session_to_event_tensor(
        {"events": events, "sensitivity": norm_factor, "dpi": 800.0}
    )
    if len(tensor) < chunk_length:
        return None
    tensor = tensor[:chunk_length]
    normalized = apply_normalizer(tensor, stats)
    scores = score_sequences(
        model, torch.from_numpy(normalized[None]).float(), batch_size=1, device=device
    )
    return float(scores[0])


# ---------------------------------------------------------------------------
# Streaming engine
# ---------------------------------------------------------------------------


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
        rate_norm: float = 1.0,
        device: str = "auto",
    ) -> None:
        self.classical_detectors = classical_detectors
        self.feature_scaler = feature_scaler
        self.aggregator = aggregator
        self.lstm_ae_model = lstm_ae_model
        self.lstm_ae_stats = lstm_ae_stats
        self.chunk_length = chunk_length
        self.norm_factor = norm_factor
        # Polling-rate normalisation factor for the classical window features.
        # Defaults to 1.0 (no-op); the live recorder will report its own rate
        # in a future iteration so each connection can set this per-session.
        self.rate_norm = rate_norm
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
        self._finalized = False

    # -------------------------------------------------------------------
    # Per-session configuration
    # -------------------------------------------------------------------

    def configure_for_session(
        self,
        *,
        sensitivity: float | None = None,
        dpi: float | None = None,
        polling_rate: float | None = None,
    ) -> None:
        """Set per-session normalisation from recorder metadata.

        Call this *before* pushing a session's events. It scales the classical
        window features (``norm_factor``), the LSTM-AE chunk tensor (same
        ``norm_factor``) and the rate-proportional features (``rate_norm``) the
        same way the offline training pipeline did. ``build_stream_state`` only
        provides the no-op defaults (1.0); without this call a real
        mixed-hardware session (sens≠1, dpi≠800, rate≠1000 Hz) is mis-scaled and
        every window/chunk looks anomalous. Mock sessions (1.0/800/1000 Hz) are
        unaffected. Missing fields leave the current value untouched.
        """
        if sensitivity is not None and dpi is not None:
            self.norm_factor = max(float(sensitivity) * float(dpi) / 800.0, 1e-6)
        if polling_rate is not None:
            self.rate_norm = polling_rate_norm(polling_rate)

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
        """Force a score update at the end of a session (e.g. on WS disconnect).

        Flushes the trailing **partial window** — the events since the last 30s
        boundary — so the final sub-30s of play is scored, mirroring the offline
        pipeline (which scores the final short window by its actual duration).

        The trailing **partial chunk** is intentionally *not* scored: the LSTM-AE
        requires exactly ``chunk_length`` events, and padding/masking a short
        chunk would feed it an input distribution it never saw in training, so
        those leftover events are dropped. Idempotent — safe to call repeatedly.
        """
        if not self.events:
            return None
        t = (
            float(final_t)
            if final_t is not None
            else float(self.events[-1].get("t", 0.0))
        )
        # Flush the trailing partial window (no-op if it has <2 events or was
        # already flushed). The partial chunk is deliberately discarded above.
        if self.next_window_end_t is not None and not self._finalized:
            self._flush_window(
                self.next_window_end_t - WINDOW_MS, self.next_window_end_t
            )
            self._finalized = True
        return self._build_update(t, triggered_by="finalize")

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _flush_window(self, w_start: float, w_end: float) -> None:
        """Score the completed window: delegate feature extraction + scoring,
        then fold the result into running session state."""
        feat_row = compute_window_feature_row(
            self.events, w_start, w_end, self.norm_factor, self.rate_norm
        )
        if feat_row is None:
            return
        scores = score_window_features(
            feat_row, self.feature_scaler, self.classical_detectors
        )
        for name, score in scores.items():
            if score > self._classical_max[name]:
                self._classical_max[name] = score
        self.completed_windows.append({"start": w_start, "end": w_end, **scores})

    def _flush_chunk(self) -> None:
        """Score the next ``chunk_length`` events with the LSTM-AE and record it."""
        chunk_events = self.chunk_buffer[: self.chunk_length]
        del self.chunk_buffer[: self.chunk_length]
        if self.lstm_ae_model is None or self.lstm_ae_stats is None:
            return  # No LSTM — the chunk's events are dropped above.
        score = score_chunk(
            chunk_events,
            self.lstm_ae_model,
            self.lstm_ae_stats,
            self.chunk_length,
            self.norm_factor,
            self.device,
        )
        if score is not None:
            self.lstm_chunk_scores.append(score)

    def _build_update(self, t: float, triggered_by: str) -> ScoreUpdate:
        # Only include detectors that have actually produced a real score.
        # An unfired detector (no window/chunk yet) is "no evidence" — feeding
        # 0.0 to the calibrator would lie because 0.0 might mean "very
        # cheat-like" or "very legit" depending on its training distribution.
        # Skipping it lets the aggregator's prior do the right thing instead.
        per_detector_real: dict[str, float] = {}
        per_detector_display: dict[str, float] = {}
        for name, val in self._classical_max.items():
            display_val = val if val != -np.inf else 0.0
            per_detector_display[name] = display_val
            if val != -np.inf:
                per_detector_real[name] = float(val)
        if self.lstm_chunk_scores:
            lstm_val = float(np.percentile(self.lstm_chunk_scores, 95))
            per_detector_display["LSTMAutoencoder"] = lstm_val
            per_detector_real["LSTMAutoencoder"] = lstm_val

        explanation = self.aggregator.explain(per_detector_real)
        update = ScoreUpdate(
            t=t,
            n_events=len(self.events),
            n_windows=len(self.completed_windows),
            n_chunks=len(self.lstm_chunk_scores),
            per_detector=per_detector_display,
            session_risk=float(explanation["posterior_risk"]),
            detector_logits=dict(explanation["per_detector_logit"]),
            triggered_by=triggered_by,
            lstm_chunk_score=(
                float(self.lstm_chunk_scores[-1]) if self.lstm_chunk_scores else None
            ),
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
    X = feats[CHEAT_FEATURE_COLS].fillna(0.0).to_numpy()
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


# ---------------------------------------------------------------------------
# Serving bundle: persist the fitted detectors/scaler/calibrators so the API
# LOADS immutable, versioned artifacts instead of fitting from data/synthetic
# at every startup (slow, and impossible on a host without the research data).
# ---------------------------------------------------------------------------

SERVING_BUNDLE_SCHEMA_VERSION = 1
SERVING_BUNDLE_NAME = "serving_bundle.pkl"


def _default_bundle_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / SERVING_BUNDLE_NAME


def save_stream_bundle(
    state: SessionStreamState,
    path: Path | None = None,
    *,
    metadata: dict | None = None,
) -> Path:
    """Persist the fitted classical detectors + scaler + aggregator + schema.

    The bundle is hardware-agnostic (no ``norm_factor``/``rate_norm``) —
    per-session normalisation happens at serve time via
    ``configure_for_session``. The LSTM-AE is **not** bundled; it stays a
    separately DVC-tracked artifact (``models/lstm_ae.pt``) loaded alongside the
    bundle by ``load_stream_state``.
    """
    import pickle
    from datetime import datetime, timezone

    path = Path(path) if path is not None else _default_bundle_path()
    bundle = {
        "schema_version": SERVING_BUNDLE_SCHEMA_VERSION,
        "classical_detectors": state.classical_detectors,
        "feature_scaler": state.feature_scaler,
        "aggregator": state.aggregator,
        "chunk_length": state.chunk_length,
        "feature_cols": list(CHEAT_FEATURE_COLS),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    log.info(
        "Saved serving bundle (schema v%d) → %s", SERVING_BUNDLE_SCHEMA_VERSION, path
    )
    return path


def load_stream_state(
    path: Path | None = None,
    *,
    model_dir: Path | None = None,
    device: str = "auto",
) -> SessionStreamState:
    """Construct a ``SessionStreamState`` from a persisted bundle + the LSTM-AE.

    This is the **serving** path: it loads immutable, versioned artifacts and
    fits nothing (it never reads ``data/synthetic``). Raises if the bundle is
    missing or its schema version is incompatible.
    """
    import pickle

    path = Path(path) if path is not None else _default_bundle_path()
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    sv = bundle.get("schema_version")
    if sv != SERVING_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"serving bundle schema v{sv} != expected v{SERVING_BUNDLE_SCHEMA_VERSION} "
            f"({path}); rebuild with `python -m scripts.build_serving_bundle`"
        )

    chunk_length = int(bundle["chunk_length"])
    lstm_ae_model = None
    lstm_ae_stats = None
    if model_dir is None:
        model_dir = Path(__file__).resolve().parents[2] / "models"
    try:
        from pipeline.models.lstm_ae import LSTM_AE_WEIGHTS_NAME, load_lstm_ae

        if (Path(model_dir) / LSTM_AE_WEIGHTS_NAME).exists():
            lstm_ae_model, lstm_ae_stats, meta = load_lstm_ae(model_dir, device=device)
            chunk_length = int(
                (meta.get("config") or {}).get("chunk_length", chunk_length)
            )
        else:
            log.warning(
                "No LSTM-AE artifact at %s — serving without the chunk signal",
                model_dir,
            )
    except ImportError:
        log.warning("torch unavailable — serving without the LSTM-AE chunk signal")

    return SessionStreamState(
        classical_detectors=bundle["classical_detectors"],
        feature_scaler=bundle["feature_scaler"],
        aggregator=bundle["aggregator"],
        lstm_ae_model=lstm_ae_model,
        lstm_ae_stats=lstm_ae_stats,
        chunk_length=chunk_length,
        device=device,
    )


def load_or_build_stream_state(
    *,
    bundle_path: Path | None = None,
    device: str = "auto",
    **build_kwargs,
) -> SessionStreamState:
    """Prefer the persisted serving bundle; fall back to fitting from scratch.

    Production / clean-clone serving uses the versioned bundle (fast, no
    research data needed). When the bundle is absent (e.g. a dev box that hasn't
    run ``scripts.build_serving_bundle``), this falls back to
    ``build_stream_state`` — slower, and it requires ``data/synthetic``.
    """
    bundle_path = (
        Path(bundle_path) if bundle_path is not None else _default_bundle_path()
    )
    if bundle_path.exists():
        try:
            state = load_stream_state(bundle_path, device=device)
            log.info("Loaded versioned serving bundle → %s", bundle_path)
            return state
        except Exception as e:  # corrupt/incompatible → fall back loudly
            log.warning(
                "Serving bundle at %s unusable (%s) — falling back to fitting",
                bundle_path,
                e,
            )
    else:
        log.warning(
            "No serving bundle at %s — fitting from data/synthetic (dev fallback; "
            "slow, needs the synthetic dataset). Build one with "
            "`python -m scripts.build_serving_bundle`.",
            bundle_path,
        )
    return build_stream_state(device=device, **build_kwargs)
