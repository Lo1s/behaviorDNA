"""
tests/test_training.py
======================
Unit tests for pipeline/training/run.py
"""

import math
import tempfile
from pathlib import Path

import pandas as pd

from pipeline.features.run import FEATURE_COLS
from pipeline.training.run import (
    export_onnx,
    train_isolation_forest,
    train_lightgbm,
    train_lof,
    train_one_class_svm,
    train_random_forest,
    train_svc,
    train_xgboost,
)

ARTIFACT_KEYS = {
    "model_type",
    "task",
    "model",
    "scaler",
    "feature_cols",
    "label_encoder",
    "classes",
    "trained",
}


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
        "random_forest": {"n_estimators": 10, "class_weight": "balanced"},
        "xgboost": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
        "svc": {"kernel": "rbf", "class_weight": "balanced", "probability": True},
        "lof": {"n_neighbors": 5, "contamination": 0.1},
        "one_class_svm": {"kernel": "rbf", "nu": 0.1},
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


EMPTY_VAL = pd.DataFrame(columns=["player"] + FEATURE_COLS)


# ---------------------------------------------------------------------------
# TestTrainLightGBM
# ---------------------------------------------------------------------------


class TestTrainLightGBM:
    def test_trains_with_two_players(self):
        artifact, metrics = train_lightgbm(make_train_df(), EMPTY_VAL, make_cfg())
        assert artifact["trained"] is True
        assert artifact["model"] is not None

    def test_artifact_has_required_keys(self):
        artifact, _ = train_lightgbm(make_train_df(), EMPTY_VAL, make_cfg())
        assert ARTIFACT_KEYS == set(artifact.keys())

    def test_single_player_returns_untrained(self):
        df = make_train_df(players=("alice",))
        artifact, metrics = train_lightgbm(df, EMPTY_VAL, make_cfg())
        assert artifact["trained"] is False
        assert metrics["trained"] is False

    def test_scaler_is_fitted(self):
        artifact, _ = train_lightgbm(make_train_df(), EMPTY_VAL, make_cfg())
        assert hasattr(artifact["scaler"], "mean_")

    def test_val_acc_nan_when_val_empty(self):
        _, metrics = train_lightgbm(make_train_df(), EMPTY_VAL, make_cfg())
        assert math.isnan(metrics["val_accuracy"])


# ---------------------------------------------------------------------------
# TestTrainIsolationForest
# ---------------------------------------------------------------------------


class TestTrainIsolationForest:
    def test_trains_with_single_player(self):
        df = make_train_df(players=("alice",))
        artifact, metrics = train_isolation_forest(df, make_cfg())
        assert artifact["trained"] is True

    def test_metrics_has_required_keys(self):
        _, metrics = train_isolation_forest(make_train_df(), make_cfg())
        for k in ("mean_score_train", "pct_predicted_outlier", "n_train_windows"):
            assert k in metrics, f"missing key: {k}"

    def test_label_encoder_is_none(self):
        artifact, _ = train_isolation_forest(make_train_df(), make_cfg())
        assert artifact["label_encoder"] is None


# ---------------------------------------------------------------------------
# TestExportOnnx
# ---------------------------------------------------------------------------


class TestExportOnnx:
    def test_untrained_artifact_writes_empty_bytes(self):
        artifact = {
            "trained": False,
            "model_type": "lightgbm",
            "model": None,
            "scaler": None,
            "feature_cols": FEATURE_COLS,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model.onnx"
            export_onnx(artifact, out)
            assert out.read_bytes() == b""

    def test_isolation_forest_exports_onnx(self):
        df = make_train_df()
        artifact, _ = train_isolation_forest(df, make_cfg())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model.onnx"
            export_onnx(artifact, out)
            assert out.stat().st_size > 0

    def test_exception_writes_empty_bytes_gracefully(self):
        # Corrupt artifact triggers exception path inside export_onnx
        artifact = {
            "trained": True,
            "model_type": "isolation_forest",
            "model": object(),  # not a real sklearn model
            "scaler": object(),
            "feature_cols": FEATURE_COLS,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model.onnx"
            export_onnx(artifact, out)
            assert out.read_bytes() == b""


# ---------------------------------------------------------------------------
# TestTrainRandomForest
# ---------------------------------------------------------------------------


class TestTrainRandomForest:
    def test_trains_with_two_players(self):
        artifact, metrics = train_random_forest(make_train_df(), EMPTY_VAL, make_cfg())
        assert artifact["trained"] is True
        assert artifact["model"] is not None

    def test_artifact_has_required_keys(self):
        artifact, _ = train_random_forest(make_train_df(), EMPTY_VAL, make_cfg())
        assert ARTIFACT_KEYS == set(artifact.keys())

    def test_scaler_is_fitted(self):
        artifact, _ = train_random_forest(make_train_df(), EMPTY_VAL, make_cfg())
        assert hasattr(artifact["scaler"], "mean_")

    def test_single_player_returns_untrained(self):
        df = make_train_df(players=("alice",))
        artifact, metrics = train_random_forest(df, EMPTY_VAL, make_cfg())
        assert artifact["trained"] is False


# ---------------------------------------------------------------------------
# TestTrainXGBoost
# ---------------------------------------------------------------------------


class TestTrainXGBoost:
    def test_trains_with_two_players(self):
        artifact, metrics = train_xgboost(make_train_df(), EMPTY_VAL, make_cfg())
        assert artifact["trained"] is True
        assert artifact["model"] is not None

    def test_artifact_has_required_keys(self):
        artifact, _ = train_xgboost(make_train_df(), EMPTY_VAL, make_cfg())
        assert ARTIFACT_KEYS == set(artifact.keys())

    def test_scaler_is_fitted(self):
        artifact, _ = train_xgboost(make_train_df(), EMPTY_VAL, make_cfg())
        assert hasattr(artifact["scaler"], "mean_")

    def test_single_player_returns_untrained(self):
        df = make_train_df(players=("alice",))
        artifact, metrics = train_xgboost(df, EMPTY_VAL, make_cfg())
        assert artifact["trained"] is False


# ---------------------------------------------------------------------------
# TestTrainSVC
# ---------------------------------------------------------------------------


class TestTrainSVC:
    def test_trains_with_two_players(self):
        # Use small n_windows to keep SVC fit time short
        df = make_train_df(n_windows=4)
        artifact, metrics = train_svc(df, EMPTY_VAL, make_cfg())
        assert artifact["trained"] is True
        assert artifact["model"] is not None

    def test_artifact_has_required_keys(self):
        df = make_train_df(n_windows=4)
        artifact, _ = train_svc(df, EMPTY_VAL, make_cfg())
        assert ARTIFACT_KEYS == set(artifact.keys())

    def test_single_player_returns_untrained(self):
        df = make_train_df(players=("alice",), n_windows=4)
        artifact, metrics = train_svc(df, EMPTY_VAL, make_cfg())
        assert artifact["trained"] is False


# ---------------------------------------------------------------------------
# TestTrainLOF
# ---------------------------------------------------------------------------


class TestTrainLOF:
    def test_trains_with_single_player(self):
        df = make_train_df(players=("alice",))
        artifact, metrics = train_lof(df, make_cfg())
        assert artifact["trained"] is True

    def test_artifact_has_required_keys(self):
        artifact, _ = train_lof(make_train_df(), make_cfg())
        assert ARTIFACT_KEYS == set(artifact.keys())

    def test_label_encoder_is_none(self):
        artifact, _ = train_lof(make_train_df(), make_cfg())
        assert artifact["label_encoder"] is None

    def test_metrics_has_required_keys(self):
        _, metrics = train_lof(make_train_df(), make_cfg())
        for k in ("mean_score_train", "pct_predicted_outlier", "n_train_windows"):
            assert k in metrics, f"missing key: {k}"


# ---------------------------------------------------------------------------
# TestTrainOneClassSVM
# ---------------------------------------------------------------------------


class TestTrainOneClassSVM:
    def test_trains_with_single_player(self):
        df = make_train_df(players=("alice",))
        artifact, metrics = train_one_class_svm(df, make_cfg())
        assert artifact["trained"] is True

    def test_artifact_has_required_keys(self):
        artifact, _ = train_one_class_svm(make_train_df(), make_cfg())
        assert ARTIFACT_KEYS == set(artifact.keys())

    def test_label_encoder_is_none(self):
        artifact, _ = train_one_class_svm(make_train_df(), make_cfg())
        assert artifact["label_encoder"] is None

    def test_metrics_has_required_keys(self):
        _, metrics = train_one_class_svm(make_train_df(), make_cfg())
        for k in ("mean_score_train", "pct_predicted_outlier", "n_train_windows"):
            assert k in metrics, f"missing key: {k}"
