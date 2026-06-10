"""
pipeline/evaluation/run.py
==========================
Stage 5 — Evaluation: model artifact + test split → metrics + confusion matrix.

Loads models/model.pkl and data/splits/test.parquet. Applies the fitted scaler
(transform only — never refit) and computes metrics appropriate to the model type:

  Identification (lightgbm, random_forest, xgboost, svc):
    accuracy, precision, recall, F1, confusion matrix CSV
  Anomaly detection (isolation_forest, lof, one_class_svm):
    anomaly score stats, fraction predicted as outliers

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

import pandas as pd

from pipeline.features.run import ID_FEATURE_COLS

log = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
MODEL_IN = ROOT / "models" / "model.pkl"
TEST_IN = ROOT / "data" / "splits" / "test.parquet"
EVAL_OUT = ROOT / "reports" / "eval_metrics.json"
CM_OUT = ROOT / "reports" / "confusion_matrix.csv"


def _prep_X(df: pd.DataFrame, cols: list[str] = ID_FEATURE_COLS) -> pd.DataFrame:
    # Named frame so the (set_output="pandas") scaler + classifier stay
    # feature-name-aware — no predict-time sklearn warning. See training/run.py.
    # The artifact's own feature_cols is authoritative (ID vs cheat sets differ).
    return df[cols].fillna(0.0)


_CLASSIFIER_TYPES = frozenset({"lightgbm", "random_forest", "xgboost", "svc"})
_ANOMALY_TYPES = frozenset({"isolation_forest", "lof", "one_class_svm"})


def _evaluate_classifier(
    artifact: dict,
    test_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Shared evaluation logic for all supervised identification models.

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
    model_type = artifact["model_type"]

    X_test = scaler.transform(_prep_X(test_df, artifact["feature_cols"]))
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
        "model_type": model_type,
        "test_accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
        "n_test_windows": len(test_df),
        "n_classes": len(classes),
    }
    log.info(
        "%s eval: acc=%.3f  precision=%.3f  recall=%.3f  f1=%.3f",
        model_type,
        acc,
        prec,
        rec,
        f1,
    )
    return metrics, cm_df


def _evaluate_anomaly_detector(
    artifact: dict,
    test_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Shared evaluation logic for all unsupervised anomaly detection models.

    Returns (metrics_dict, empty_df) — no confusion matrix for unsupervised models.
    """
    scaler = artifact["scaler"]
    model = artifact["model"]
    model_type = artifact["model_type"]

    # Anomaly scaler/model were fit on nameless numpy (see training/run.py) —
    # feed numpy so we don't trip the reverse feature-names warning.
    X_test = scaler.transform(_prep_X(test_df, artifact["feature_cols"]).to_numpy())
    scores = model.score_samples(X_test)
    preds = model.predict(X_test)
    pct_anomaly = float((preds == -1).mean())

    metrics = {
        "evaluated": True,
        "model_type": model_type,
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "pct_anomaly": pct_anomaly,
        "n_test_windows": len(test_df),
    }
    log.info(
        "%s eval: mean_score=%.4f  pct_anomaly=%.1f%%  n_test=%d",
        model_type,
        scores.mean(),
        pct_anomaly * 100,
        len(test_df),
    )
    return metrics, pd.DataFrame()


# Public wrappers — preserve names imported by tests and external callers.
def evaluate_lightgbm(
    artifact: dict,
    test_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate a LightGBM identification artifact on test_df."""
    return _evaluate_classifier(artifact, test_df)


def evaluate_isolation_forest(
    artifact: dict,
    test_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate an IsolationForest anomaly artifact on test_df."""
    return _evaluate_anomaly_detector(artifact, test_df)


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

    if model_type in _CLASSIFIER_TYPES:
        metrics, cm_df = _evaluate_classifier(artifact, test_df)
    elif model_type in _ANOMALY_TYPES:
        metrics, cm_df = _evaluate_anomaly_detector(artifact, test_df)
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
