"""
tests/test_api.py
=================
Unit tests for api/main.py
"""

import pandas as pd
from fastapi.testclient import TestClient

from api.main import FeatureVector, _vec_to_array, app
from pipeline.features.run import FEATURE_COLS
from pipeline.training.run import train_isolation_forest, train_lightgbm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_cfg() -> dict:
    return {
        "model": {"type": "lightgbm", "task": "identification"},
        "lightgbm": {
            "num_leaves": 4,
            "learning_rate": 0.1,
            "n_estimators": 10,
            "min_child_samples": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        "isolation_forest": {
            "n_estimators": 10,
            "contamination": 0.1,
            "max_features": 1.0,
        },
        "data": {
            "test_size": 0.15,
            "val_size": 0.15,
            "random_seed": 42,
            "min_sessions_per_player": 1,
        },
        "mlflow": {"experiment_name": "test", "tracking_uri": "http://localhost:5000"},
    }


def make_train_df(players=("alice", "bob"), n_windows=6) -> pd.DataFrame:
    rows = []
    for i, player in enumerate(players):
        for w in range(n_windows):
            row = {
                "session_id": f"s_{player}",
                "window_idx": w,
                "player": player,
                "game": "cs2",
                "sensitivity": 1.0,
                "dpi": 800,
                "recorded_at": pd.Timestamp("2026-01-01", tz="UTC"),
                "duration_ms": 90_000.0,
            }
            row.update({c: float(i + w * 0.1) for c in FEATURE_COLS})
            rows.append(row)
    return pd.DataFrame(rows)


def make_lgbm_artifact(trained=True) -> dict:
    if not trained:
        return {
            "model_type": "lightgbm",
            "task": "identification",
            "model": None,
            "scaler": None,
            "feature_cols": FEATURE_COLS,
            "label_encoder": None,
            "classes": None,
            "trained": False,
        }
    artifact, _ = train_lightgbm(make_train_df(), make_train_df().iloc[0:0], make_cfg())
    return artifact


def make_if_artifact(trained=True) -> dict:
    if not trained:
        return {
            "model_type": "isolation_forest",
            "task": "anomaly_detection",
            "model": None,
            "scaler": None,
            "feature_cols": FEATURE_COLS,
            "label_encoder": None,
            "classes": None,
            "trained": False,
        }
    artifact, _ = train_isolation_forest(make_train_df(), make_cfg())
    return artifact


def get_client(artifact: dict) -> TestClient:
    app.state.artifact = artifact
    return TestClient(app)


FEATURE_PAYLOAD = {c: 0.5 for c in FEATURE_COLS}


# ---------------------------------------------------------------------------
# TestVecToArray
# ---------------------------------------------------------------------------


class TestVecToArray:
    def test_none_values_become_zero(self):
        vec = FeatureVector(session_id="s1")
        arr = _vec_to_array(vec)
        assert (arr == 0.0).all()

    def test_array_shape_is_1_by_18(self):
        vec = FeatureVector(session_id="s1")
        arr = _vec_to_array(vec)
        assert arr.shape == (1, 18)


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_model_info(self):
        client = get_client(make_lgbm_artifact())
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["model_type"] == "lightgbm"
        assert body["trained"] is True
        assert body["feature_count"] == 18

    def test_untrained_model_reports_not_trained(self):
        client = get_client(make_lgbm_artifact(trained=False))
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["trained"] is False


# ---------------------------------------------------------------------------
# TestPredictPlayerEndpoint
# ---------------------------------------------------------------------------


class TestPredictPlayerEndpoint:
    def test_returns_200_with_prediction(self):
        client = get_client(make_lgbm_artifact())
        r = client.post("/predict/player", json={"session_id": "s1", **FEATURE_PAYLOAD})
        assert r.status_code == 200
        body = r.json()
        assert "predicted_player" in body
        assert "probabilities" in body

    def test_wrong_model_type_returns_400(self):
        client = get_client(make_if_artifact())
        r = client.post("/predict/player", json={"session_id": "s1", **FEATURE_PAYLOAD})
        assert r.status_code == 400

    def test_untrained_returns_503(self):
        client = get_client(make_lgbm_artifact(trained=False))
        r = client.post("/predict/player", json={"session_id": "s1", **FEATURE_PAYLOAD})
        assert r.status_code == 503

    def test_none_features_handled(self):
        client = get_client(make_lgbm_artifact())
        r = client.post("/predict/player", json={"session_id": "s1"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# TestPredictAnomalyEndpoint
# ---------------------------------------------------------------------------


class TestPredictAnomalyEndpoint:
    def test_returns_200_with_score(self):
        client = get_client(make_if_artifact())
        r = client.post(
            "/predict/anomaly", json={"session_id": "s1", **FEATURE_PAYLOAD}
        )
        assert r.status_code == 200
        body = r.json()
        assert "anomaly_score" in body
        assert isinstance(body["is_anomaly"], bool)

    def test_wrong_model_type_returns_400(self):
        client = get_client(make_lgbm_artifact())
        r = client.post(
            "/predict/anomaly", json={"session_id": "s1", **FEATURE_PAYLOAD}
        )
        assert r.status_code == 400

    def test_untrained_returns_503(self):
        client = get_client(make_if_artifact(trained=False))
        r = client.post(
            "/predict/anomaly", json={"session_id": "s1", **FEATURE_PAYLOAD}
        )
        assert r.status_code == 503
