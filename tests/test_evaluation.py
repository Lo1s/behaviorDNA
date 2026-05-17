"""
tests/test_evaluation.py
========================
Unit tests for pipeline/evaluation/run.py
"""

import pandas as pd

from pipeline.evaluation.run import evaluate_isolation_forest, evaluate_lightgbm
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


# ---------------------------------------------------------------------------
# TestEvaluateLightGBM
# ---------------------------------------------------------------------------


class TestEvaluateLightGBM:
    def test_returns_correct_metric_keys(self):
        train_df = make_train_df()
        artifact, _ = train_lightgbm(train_df, train_df.iloc[0:0], make_cfg())
        metrics, _ = evaluate_lightgbm(artifact, train_df)
        for k in ("evaluated", "test_accuracy", "f1_weighted", "n_test_windows"):
            assert k in metrics, f"missing key: {k}"

    def test_confusion_matrix_has_player_labels(self):
        train_df = make_train_df()
        artifact, _ = train_lightgbm(train_df, train_df.iloc[0:0], make_cfg())
        _, cm_df = evaluate_lightgbm(artifact, train_df)
        assert list(cm_df.index) == artifact["classes"]

    def test_perfect_prediction_accuracy_is_1(self):
        # Train and test on the same perfectly separable data
        train_df = make_train_df()
        artifact, _ = train_lightgbm(train_df, train_df.iloc[0:0], make_cfg())
        metrics, _ = evaluate_lightgbm(artifact, train_df)
        assert abs(metrics["test_accuracy"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# TestEvaluateIsolationForest
# ---------------------------------------------------------------------------


class TestEvaluateIsolationForest:
    def test_returns_correct_metric_keys(self):
        train_df = make_train_df()
        artifact, _ = train_isolation_forest(train_df, make_cfg())
        metrics, _ = evaluate_isolation_forest(artifact, train_df)
        for k in ("mean_score", "pct_anomaly", "n_test_windows"):
            assert k in metrics, f"missing key: {k}"

    def test_confusion_matrix_is_empty(self):
        train_df = make_train_df()
        artifact, _ = train_isolation_forest(train_df, make_cfg())
        _, cm_df = evaluate_isolation_forest(artifact, train_df)
        assert cm_df.empty

    def test_pct_anomaly_between_0_and_1(self):
        train_df = make_train_df()
        artifact, _ = train_isolation_forest(train_df, make_cfg())
        metrics, _ = evaluate_isolation_forest(artifact, train_df)
        assert 0.0 <= metrics["pct_anomaly"] <= 1.0
