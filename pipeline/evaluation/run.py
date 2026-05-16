"""
pipeline/evaluation/run.py
==========================
Stage 5 — Evaluation: model artifact + test split → metrics + confusion matrix.

Loads models/model.pkl and data/splits/test.parquet. Applies the fitted scaler
(transform only — never refit) and computes metrics appropriate to the model type:

  lightgbm         — accuracy, precision, recall, F1, confusion matrix CSV
  isolation_forest — anomaly score stats, fraction predicted as outliers

Empty-data / untrained-model handling: writes placeholder outputs and exits
cleanly so 'dvc repro' stays green.

Outputs:
  reports/eval_metrics.json
  reports/confusion_matrix.csv

Run via DVC:
    dvc repro evaluate

Or directly:
    python -m pipeline.evaluation.run
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.features.run import FEATURE_COLS

log = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
MODEL_IN = ROOT / "models" / "model.pkl"
TEST_IN = ROOT / "data" / "splits" / "test.parquet"
EVAL_OUT = ROOT / "reports" / "eval_metrics.json"
CM_OUT = ROOT / "reports" / "confusion_matrix.csv"


def _prep_X(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS].fillna(0.0).values


def evaluate_lightgbm(
    artifact: dict,
    test_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Predict player classes and compute classification metrics.

    Returns (metrics_dict, confusion_matrix_df).
    Scaler is applied via .transform() only — it was fit on train data.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    scaler = artifact["scaler"]
    le = artifact["label_encoder"]
    model = artifact["model"]
    classes = artifact["classes"]

    X_test = scaler.transform(_prep_X(test_df))
    y_true = le.transform(test_df["player"])
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = "true \\ predicted"

    metrics = {
        "evaluated": True,
        "model_type": "lightgbm",
        "test_accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
        "n_test_windows": len(test_df),
        "n_classes": len(classes),
    }
    log.info(
        "LightGBM eval: acc=%.3f  precision=%.3f  recall=%.3f  f1=%.3f",
        acc,
        prec,
        rec,
        f1,
    )
    return metrics, cm_df


def evaluate_isolation_forest(
    artifact: dict,
    test_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Compute anomaly scores and outlier fraction on test data.

    Returns (metrics_dict, empty_df) — no confusion matrix for unsupervised models.
    """
    scaler = artifact["scaler"]
    model = artifact["model"]

    X_test = scaler.transform(_prep_X(test_df))
    scores = model.score_samples(X_test)
    preds = model.predict(X_test)
    pct_anomaly = float((preds == -1).mean())

    metrics = {
        "evaluated": True,
        "model_type": "isolation_forest",
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "pct_anomaly": pct_anomaly,
        "n_test_windows": len(test_df),
    }
    log.info(
        "IsolationForest eval: mean_score=%.4f  pct_anomaly=%.1f%%  n_test=%d",
        scores.mean(),
        pct_anomaly * 100,
        len(test_df),
    )
    # No confusion matrix for anomaly detection
    return metrics, pd.DataFrame()


def run() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_IN, "rb") as f:
        artifact = pickle.load(f)

    test_df = pd.read_parquet(TEST_IN)

    log.info(
        "Evaluation: model_type=%s  trained=%s  n_test=%d",
        artifact.get("model_type", "unknown"),
        artifact.get("trained", False),
        len(test_df),
    )

    # Guard: empty test data or untrained model
    if test_df.empty or not artifact.get("trained", False):
        reason = "empty_test_data" if test_df.empty else "model_not_trained"
        log.warning("Skipping evaluation: %s", reason)
        with open(EVAL_OUT, "w") as f:
            json.dump({"evaluated": False, "reason": reason}, f, indent=2)
        pd.DataFrame().to_csv(CM_OUT, index=False)
        return

    model_type = artifact["model_type"]

    if model_type == "lightgbm":
        metrics, cm_df = evaluate_lightgbm(artifact, test_df)
    elif model_type == "isolation_forest":
        metrics, cm_df = evaluate_isolation_forest(artifact, test_df)
    else:
        log.error("Unknown model_type '%s' in artifact.", model_type)
        sys.exit(1)

    with open(EVAL_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved eval metrics: %s", EVAL_OUT)

    cm_df.to_csv(CM_OUT)
    log.info("Saved confusion matrix: %s", CM_OUT)

    log.info("")
    log.info("=== Evaluation summary ===")
    for k, v in metrics.items():
        if isinstance(v, float) and not isinstance(v, bool):
            log.info("  %-25s %.4f", k, v)
        elif isinstance(v, int):
            log.info("  %-25s %d", k, v)


if __name__ == "__main__":
    run()
