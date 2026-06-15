"""
api/main.py
===========
BehaviorDNA FastAPI inference endpoint.

Loads models/model.pkl once at startup and exposes three endpoints:

  GET  /health          — model type, trained status, feature count, calibration flag
  POST /predict/player  — LightGBM player identification
  POST /predict/anomaly — IsolationForest anomaly scoring

Both prediction endpoints return HTTP 400 if the loaded model type does not
match the endpoint, and HTTP 503 if the model has not been trained yet.

Run from the project root:
    uvicorn api.main:app --reload
"""

import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from pipeline.features.run import FEATURE_COLS

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class FeatureVector(BaseModel):
    session_id: str
    # Mouse kinematics
    speed_mean: Optional[float] = None
    speed_std: Optional[float] = None
    speed_p50: Optional[float] = None
    speed_p90: Optional[float] = None
    speed_p99: Optional[float] = None
    accel_mean: Optional[float] = None
    accel_std: Optional[float] = None
    jitter: Optional[float] = None
    click_interval_mean: Optional[float] = None
    click_interval_std: Optional[float] = None
    # Mouse trajectory (Phase 1; fast_segment_straightness — Phase 1.5)
    mouse_curvature_mean: Optional[float] = None
    mouse_curvature_std: Optional[float] = None
    path_efficiency: Optional[float] = None
    fast_segment_straightness: Optional[float] = None
    direction_changes_per_sec: Optional[float] = None
    # Keyboard patterns
    hold_mean: Optional[float] = None
    hold_std: Optional[float] = None
    iki_mean: Optional[float] = None
    iki_std: Optional[float] = None
    burst_rate: Optional[float] = None
    wasd_rhythm: Optional[float] = None
    # Reaction timing (Phase 1; click_reaction_p5 — Phase 1.5)
    click_reaction_mean: Optional[float] = None
    click_reaction_p5: Optional[float] = None
    inter_click_movement: Optional[float] = None
    # Keystroke geometry (Phase 1)
    keystroke_periodicity: Optional[float] = None
    # Session aggregates
    event_rate: Optional[float] = None
    mouse_key_ratio: Optional[float] = None
    active_time_pct: Optional[float] = None
    scroll_count: Optional[float] = None
    scroll_direction_ratio: Optional[float] = None


class PlayerPrediction(BaseModel):
    session_id: str
    predicted_player: str
    # Raw LightGBM predict_proba — uncalibrated (no calibrator is persisted into
    # the serving artifact; see MODEL_CARD.md and /health's "calibrated" flag).
    probabilities: dict[str, float]


class AnomalyPrediction(BaseModel):
    session_id: str
    anomaly_score: float
    is_anomaly: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vec_to_frame(vec: FeatureVector, cols: list[str] = FEATURE_COLS) -> pd.DataFrame:
    """Convert Pydantic model to a 1-row named float64 frame over ``cols``.

    ``cols`` should be the loaded artifact's ``feature_cols`` — identification
    and anomaly models use different feature sets (see docs/SIGNALS.md).

    A named frame (not a bare array) keeps the persisted scaler + classifier —
    both fitted with feature names — from emitting the sklearn "X does not have
    valid feature names" warning at predict time. None → 0.0 (matches training).
    """
    vals = {col: float(getattr(vec, col) or 0.0) for col in cols}
    return pd.DataFrame([vals], columns=cols)


def _get_artifact(request: Request) -> dict:
    return request.app.state.artifact


def _require_model_type(artifact: dict, expected: str) -> None:
    if artifact["model_type"] != expected:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Loaded model type is '{artifact['model_type']}', "
                f"not '{expected}'. Use the matching endpoint."
            ),
        )


def _require_trained(artifact: dict) -> None:
    if not artifact.get("trained", False):
        raise HTTPException(
            status_code=503,
            detail="Model has not been trained yet. Collect more sessions and run 'dvc repro'.",
        )


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
        log.info(
            "Loaded model: type=%s  trained=%s  features=%d",
            artifact.get("model_type"),
            artifact.get("trained"),
            len(artifact.get("feature_cols", [])),
        )
    else:
        log.warning("models/model.pkl not found — serving health endpoint only.")
        artifact = {
            "model_type": "none",
            "trained": False,
            "feature_cols": FEATURE_COLS,
        }
    app.state.artifact = artifact

    # Streaming pre-load: build the classical detectors + LSTM-AE + aggregator
    # once so every /stream connection reuses them. Failure here is non-fatal —
    # the batch endpoints still work.
    try:
        from pipeline.inference.streaming import load_or_build_stream_state

        app.state.stream_template = load_or_build_stream_state()
        log.info("Streaming /stream endpoint ready (models + aggregator pre-loaded).")
    except Exception as e:
        log.warning("Could not initialise streaming components: %s", e)
        app.state.stream_template = None

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BehaviorDNA Inference API",
    description="Player behavioral biometrics — identification and anomaly detection.",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount the WebSocket /stream endpoint from api/streaming.py
from api.streaming import streaming_router  # noqa: E402

app.include_router(streaming_router)


@app.get("/health")
def health(request: Request) -> dict:
    """Returns model type, trained status, feature count, and whether served
    probabilities are calibrated (they are not — see MODEL_CARD.md)."""
    artifact = _get_artifact(request)
    return {
        "model_type": artifact.get("model_type"),
        "trained": artifact.get("trained", False),
        "feature_count": len(artifact.get("feature_cols", FEATURE_COLS)),
        "classes": artifact.get("classes"),
        # Player probabilities are raw predict_proba; no calibrator is persisted
        # into the serving artifact (the val fold is too small — see MODEL_CARD.md).
        # Defaults False; flips to True only if a calibrated artifact is ever served.
        "calibrated": bool(artifact.get("calibrated", False)),
    }


@app.post("/predict/player", response_model=PlayerPrediction)
def predict_player(vec: FeatureVector, request: Request) -> PlayerPrediction:
    """Identify the player from a 30s behavioral window using LightGBM."""
    artifact = _get_artifact(request)
    _require_model_type(artifact, "lightgbm")
    _require_trained(artifact)

    X = _vec_to_frame(vec, artifact.get("feature_cols", FEATURE_COLS))
    X_scaled = artifact["scaler"].transform(X)
    model = artifact["model"]
    le = artifact["label_encoder"]

    pred_idx = int(model.predict(X_scaled)[0])
    predicted_player = le.classes_[pred_idx]

    proba = model.predict_proba(X_scaled)[0]
    probabilities = {cls: float(p) for cls, p in zip(le.classes_, proba)}

    return PlayerPrediction(
        session_id=vec.session_id,
        predicted_player=predicted_player,
        probabilities=probabilities,
    )


@app.post("/predict/anomaly", response_model=AnomalyPrediction)
def predict_anomaly(vec: FeatureVector, request: Request) -> AnomalyPrediction:
    """Score a 30s behavioral window for automation/bot anomalies using IsolationForest."""
    artifact = _get_artifact(request)
    _require_model_type(artifact, "isolation_forest")
    _require_trained(artifact)

    # Anomaly scaler/model were fit on nameless numpy (see training/run.py).
    X = _vec_to_frame(vec, artifact.get("feature_cols", FEATURE_COLS)).to_numpy()
    X_scaled = artifact["scaler"].transform(X)
    model = artifact["model"]

    score = float(model.score_samples(X_scaled)[0])
    pred = int(model.predict(X_scaled)[0])
    is_anomaly = pred == -1

    return AnomalyPrediction(
        session_id=vec.session_id,
        anomaly_score=score,
        is_anomaly=is_anomaly,
    )
