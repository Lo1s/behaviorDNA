"""
pipeline/training/run.py
========================
Stage 4 — Training: split Parquets → fitted model artifact + ONNX + metrics.

Supports seven model types (configured via configs/training.yaml model.type):

  Identification (supervised multi-class):
    lightgbm         — LGBMClassifier
    random_forest    — RandomForestClassifier
    xgboost          — XGBClassifier
    svc              — SVC (RBF kernel, probability=True required)

  Anomaly detection (unsupervised):
    isolation_forest — IsolationForest
    lof              — LocalOutlierFactor (novelty=True)
    one_class_svm    — OneClassSVM

Both paths apply the same preprocessing:
  1. NaN-fill feature columns with 0.0 (e.g. wasd_rhythm=NaN → 0 = no WASD activity)
  2. StandardScaler fit on train features only (never val or test)

Outputs:
  models/model.pkl          — artifact dict (see ARTIFACT_KEYS below)
  models/model.onnx         — ONNX model (empty bytes if export fails)
  reports/train_metrics.json

Empty-data handling: if train.parquet is empty the stage writes a placeholder
artifact (trained=False) and exits cleanly so 'dvc repro' stays green while
more sessions are collected.

MLflow logging is optional — skipped silently if MLFLOW_TRACKING_USERNAME is
not set in the environment (standard in CI and non-interactive DVC runs).

Run via DVC:
    dvc repro train

Or directly:
    python -m pipeline.training.run
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, OneClassSVM

from pipeline.constants import IDENTIFIER_REGISTRY_NAME
from pipeline.features.run import FEATURE_COLS

log = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
TRAIN_IN = ROOT / "data" / "splits" / "train.parquet"
VAL_IN = ROOT / "data" / "splits" / "val.parquet"
CONFIG_IN = ROOT / "configs" / "training.yaml"
MODEL_OUT = ROOT / "models" / "model.pkl"
ONNX_OUT = ROOT / "models" / "model.onnx"
METRICS_OUT = ROOT / "reports" / "train_metrics.json"


def _prep_X(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix as a FEATURE_COLS-named frame.

    Returning a named frame (not a bare array) lets the scaler + classifier
    record real feature names (``feature_names_in_``), which keeps predict-time
    free of the sklearn "X does not have valid feature names" warning and makes
    SHAP/feature-importance outputs read as real names instead of Column_N.
    """
    return df[FEATURE_COLS].fillna(0.0)


def train_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict, dict]:
    """Fit LabelEncoder + StandardScaler + LGBMClassifier.

    Returns (artifact, metrics). artifact['trained'] is False if there are
    fewer than 2 unique players in the training fold.
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score

    n_classes = train_df["player"].nunique()
    if n_classes < 2:
        log.warning(
            "Only %d unique player(s) in train fold — LightGBM requires ≥ 2."
            " Writing untrained artifact.",
            n_classes,
        )
        return (
            {
                "model_type": "lightgbm",
                "task": "identification",
                "model": None,
                "scaler": StandardScaler(),
                "feature_cols": FEATURE_COLS,
                "label_encoder": None,
                "classes": None,
                "trained": False,
            },
            {
                "trained": False,
                "reason": "fewer_than_2_players",
                "n_classes": n_classes,
            },
        )

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["player"])

    scaler = StandardScaler().set_output(transform="pandas")
    X_train = scaler.fit_transform(_prep_X(train_df))

    lgbm_params = {k: v for k, v in cfg["lightgbm"].items() if k != "class_weight"}
    model = LGBMClassifier(**lgbm_params, class_weight="balanced", verbose=-1)
    model.fit(X_train, y_train)

    train_acc = float(accuracy_score(y_train, model.predict(X_train)))

    val_acc = float("nan")
    if not val_df.empty and val_df["player"].isin(le.classes_).all():
        X_val = scaler.transform(_prep_X(val_df))
        y_val = le.transform(val_df["player"])
        val_acc = float(accuracy_score(y_val, model.predict(X_val)))

    artifact = {
        "model_type": "lightgbm",
        "task": "identification",
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "label_encoder": le,
        "classes": list(le.classes_),
        "trained": True,
    }
    metrics = {
        "trained": True,
        "model_type": "lightgbm",
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "n_train_windows": len(train_df),
        "n_val_windows": len(val_df),
        "n_classes": n_classes,
    }
    log.info(
        "LightGBM: train_acc=%.3f  val_acc=%s  classes=%s",
        train_acc,
        f"{val_acc:.3f}" if not np.isnan(val_acc) else "N/A",
        list(le.classes_),
    )
    return artifact, metrics


def train_isolation_forest(
    train_df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict, dict]:
    """Fit StandardScaler + IsolationForest (unsupervised — no player labels needed)."""
    # Anomaly detectors stay on nameless numpy: they don't need feature names
    # (no SHAP-for-identification use), and naming them makes LOF's novelty mode
    # emit a spurious sklearn feature-names warning. .to_numpy() drops the names.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(_prep_X(train_df).to_numpy())

    seed = cfg["data"]["random_seed"]
    if_params = dict(cfg["isolation_forest"])
    model = IsolationForest(**if_params, random_state=seed)
    model.fit(X_train)

    scores = model.score_samples(X_train)
    preds = model.predict(X_train)
    pct_outlier = float((preds == -1).mean())

    artifact = {
        "model_type": "isolation_forest",
        "task": "anomaly_detection",
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "label_encoder": None,
        "classes": None,
        "trained": True,
    }
    metrics = {
        "trained": True,
        "model_type": "isolation_forest",
        "mean_score_train": float(scores.mean()),
        "std_score_train": float(scores.std()),
        "pct_predicted_outlier": pct_outlier,
        "n_train_windows": len(train_df),
    }
    log.info(
        "IsolationForest: mean_score=%.4f  pct_outlier=%.1f%%  n_train=%d",
        scores.mean(),
        pct_outlier * 100,
        len(train_df),
    )
    return artifact, metrics


def train_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict, dict]:
    """Fit LabelEncoder + StandardScaler + RandomForestClassifier."""
    from sklearn.metrics import accuracy_score

    n_classes = train_df["player"].nunique()
    if n_classes < 2:
        log.warning(
            "Only %d unique player(s) in train fold — RandomForest requires ≥ 2."
            " Writing untrained artifact.",
            n_classes,
        )
        return (
            {
                "model_type": "random_forest",
                "task": "identification",
                "model": None,
                "scaler": StandardScaler(),
                "feature_cols": FEATURE_COLS,
                "label_encoder": None,
                "classes": None,
                "trained": False,
            },
            {
                "trained": False,
                "reason": "fewer_than_2_players",
                "n_classes": n_classes,
            },
        )

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["player"])
    scaler = StandardScaler().set_output(transform="pandas")
    X_train = scaler.fit_transform(_prep_X(train_df))

    seed = cfg["data"]["random_seed"]
    rf_params = {k: v for k, v in cfg["random_forest"].items() if k != "class_weight"}
    model = RandomForestClassifier(
        **rf_params, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)

    train_acc = float(accuracy_score(y_train, model.predict(X_train)))
    val_acc = float("nan")
    if not val_df.empty and val_df["player"].isin(le.classes_).all():
        X_val = scaler.transform(_prep_X(val_df))
        y_val = le.transform(val_df["player"])
        val_acc = float(accuracy_score(y_val, model.predict(X_val)))

    artifact = {
        "model_type": "random_forest",
        "task": "identification",
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "label_encoder": le,
        "classes": list(le.classes_),
        "trained": True,
    }
    metrics = {
        "trained": True,
        "model_type": "random_forest",
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "n_train_windows": len(train_df),
        "n_val_windows": len(val_df),
        "n_classes": n_classes,
    }
    log.info(
        "RandomForest: train_acc=%.3f  val_acc=%s  classes=%s",
        train_acc,
        f"{val_acc:.3f}" if not np.isnan(val_acc) else "N/A",
        list(le.classes_),
    )
    return artifact, metrics


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict, dict]:
    """Fit LabelEncoder + StandardScaler + XGBClassifier."""
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    n_classes = train_df["player"].nunique()
    if n_classes < 2:
        log.warning(
            "Only %d unique player(s) in train fold — XGBoost requires ≥ 2."
            " Writing untrained artifact.",
            n_classes,
        )
        return (
            {
                "model_type": "xgboost",
                "task": "identification",
                "model": None,
                "scaler": StandardScaler(),
                "feature_cols": FEATURE_COLS,
                "label_encoder": None,
                "classes": None,
                "trained": False,
            },
            {
                "trained": False,
                "reason": "fewer_than_2_players",
                "n_classes": n_classes,
            },
        )

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["player"])
    scaler = StandardScaler().set_output(transform="pandas")
    X_train = scaler.fit_transform(_prep_X(train_df))

    seed = cfg["data"]["random_seed"]
    xgb_params = dict(cfg["xgboost"])
    model = XGBClassifier(
        **xgb_params, random_state=seed, verbosity=0, eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    train_acc = float(accuracy_score(y_train, model.predict(X_train)))
    val_acc = float("nan")
    if not val_df.empty and val_df["player"].isin(le.classes_).all():
        X_val = scaler.transform(_prep_X(val_df))
        y_val = le.transform(val_df["player"])
        val_acc = float(accuracy_score(y_val, model.predict(X_val)))

    artifact = {
        "model_type": "xgboost",
        "task": "identification",
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "label_encoder": le,
        "classes": list(le.classes_),
        "trained": True,
    }
    metrics = {
        "trained": True,
        "model_type": "xgboost",
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "n_train_windows": len(train_df),
        "n_val_windows": len(val_df),
        "n_classes": n_classes,
    }
    log.info(
        "XGBoost: train_acc=%.3f  val_acc=%s  classes=%s",
        train_acc,
        f"{val_acc:.3f}" if not np.isnan(val_acc) else "N/A",
        list(le.classes_),
    )
    return artifact, metrics


def train_svc(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict, dict]:
    """Fit LabelEncoder + StandardScaler + SVC (RBF kernel).

    Requires probability=True in config to enable predict_proba.
    """
    from sklearn.metrics import accuracy_score

    n_classes = train_df["player"].nunique()
    if n_classes < 2:
        log.warning(
            "Only %d unique player(s) in train fold — SVC requires ≥ 2."
            " Writing untrained artifact.",
            n_classes,
        )
        return (
            {
                "model_type": "svc",
                "task": "identification",
                "model": None,
                "scaler": StandardScaler(),
                "feature_cols": FEATURE_COLS,
                "label_encoder": None,
                "classes": None,
                "trained": False,
            },
            {
                "trained": False,
                "reason": "fewer_than_2_players",
                "n_classes": n_classes,
            },
        )

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["player"])
    scaler = StandardScaler().set_output(transform="pandas")
    X_train = scaler.fit_transform(_prep_X(train_df))

    seed = cfg["data"]["random_seed"]
    svc_params = {k: v for k, v in cfg["svc"].items() if k != "class_weight"}
    model = SVC(**svc_params, class_weight="balanced", random_state=seed)
    model.fit(X_train, y_train)

    train_acc = float(accuracy_score(y_train, model.predict(X_train)))
    val_acc = float("nan")
    if not val_df.empty and val_df["player"].isin(le.classes_).all():
        X_val = scaler.transform(_prep_X(val_df))
        y_val = le.transform(val_df["player"])
        val_acc = float(accuracy_score(y_val, model.predict(X_val)))

    artifact = {
        "model_type": "svc",
        "task": "identification",
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "label_encoder": le,
        "classes": list(le.classes_),
        "trained": True,
    }
    metrics = {
        "trained": True,
        "model_type": "svc",
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "n_train_windows": len(train_df),
        "n_val_windows": len(val_df),
        "n_classes": n_classes,
    }
    log.info(
        "SVC: train_acc=%.3f  val_acc=%s  classes=%s",
        train_acc,
        f"{val_acc:.3f}" if not np.isnan(val_acc) else "N/A",
        list(le.classes_),
    )
    return artifact, metrics


def train_lof(
    train_df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict, dict]:
    """Fit StandardScaler + LocalOutlierFactor (novelty=True)."""
    # Nameless numpy on purpose — see train_isolation_forest (LOF novelty mode
    # otherwise emits a spurious feature-names warning even on a named frame).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(_prep_X(train_df).to_numpy())

    lof_params = dict(cfg["lof"])
    model = LocalOutlierFactor(**lof_params, novelty=True)
    model.fit(X_train)

    scores = model.score_samples(X_train)
    preds = model.predict(X_train)
    pct_outlier = float((preds == -1).mean())

    artifact = {
        "model_type": "lof",
        "task": "anomaly_detection",
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "label_encoder": None,
        "classes": None,
        "trained": True,
    }
    metrics = {
        "trained": True,
        "model_type": "lof",
        "mean_score_train": float(scores.mean()),
        "std_score_train": float(scores.std()),
        "pct_predicted_outlier": pct_outlier,
        "n_train_windows": len(train_df),
    }
    log.info(
        "LOF: mean_score=%.4f  pct_outlier=%.1f%%  n_train=%d",
        scores.mean(),
        pct_outlier * 100,
        len(train_df),
    )
    return artifact, metrics


def train_one_class_svm(
    train_df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict, dict]:
    """Fit StandardScaler + OneClassSVM."""
    # Nameless numpy on purpose — anomaly detectors don't need feature names.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(_prep_X(train_df).to_numpy())

    ocsvm_params = dict(cfg["one_class_svm"])
    model = OneClassSVM(**ocsvm_params)
    model.fit(X_train)

    scores = model.score_samples(X_train)
    preds = model.predict(X_train)
    pct_outlier = float((preds == -1).mean())

    artifact = {
        "model_type": "one_class_svm",
        "task": "anomaly_detection",
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "label_encoder": None,
        "classes": None,
        "trained": True,
    }
    metrics = {
        "trained": True,
        "model_type": "one_class_svm",
        "mean_score_train": float(scores.mean()),
        "std_score_train": float(scores.std()),
        "pct_predicted_outlier": pct_outlier,
        "n_train_windows": len(train_df),
    }
    log.info(
        "OneClassSVM: mean_score=%.4f  pct_outlier=%.1f%%  n_train=%d",
        scores.mean(),
        pct_outlier * 100,
        len(train_df),
    )
    return artifact, metrics


def export_onnx(artifact: dict, out_path: Path) -> None:
    """Export the trained model to ONNX via skl2onnx.

    LightGBM requires onnxmltools to register its converter with skl2onnx.
    Writes empty bytes on ImportError or any conversion failure so the pipeline
    stays green regardless of dependency availability.
    """
    if not artifact.get("trained"):
        out_path.write_bytes(b"")
        return

    try:
        from skl2onnx import convert_sklearn, update_registered_converter
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
        )
        from sklearn.pipeline import Pipeline as SKPipeline

        model = artifact["model"]
        scaler = artifact["scaler"]
        initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
        pipe = SKPipeline([("scaler", scaler), ("model", model)])

        if artifact["model_type"] == "lightgbm":
            from lightgbm import LGBMClassifier
            from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
                convert_lightgbm,
            )

            update_registered_converter(
                LGBMClassifier,
                "LightGbmLGBMClassifier",
                calculate_linear_classifier_output_shapes,
                convert_lightgbm,
                options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
            )
            onnx_model = convert_sklearn(
                pipe,
                initial_types=initial_type,
                options={"zipmap": False},
                target_opset={"": 17, "ai.onnx.ml": 3},
            )
        else:
            onnx_model = convert_sklearn(
                pipe,
                initial_types=initial_type,
                target_opset={"": 17, "ai.onnx.ml": 3},
            )

        out_path.write_bytes(onnx_model.SerializeToString())
        log.info("ONNX model saved: %s  (%d bytes)", out_path, out_path.stat().st_size)
        _validate_onnx_fidelity(artifact, out_path)
    except ImportError as exc:
        log.warning("Missing ONNX dependency (%s) — writing empty model.onnx", exc)
        out_path.write_bytes(b"")
    except Exception as exc:
        log.warning(
            "ONNX export failed (%s: %s) — writing empty model.onnx",
            type(exc).__name__,
            exc,
        )
        out_path.write_bytes(b"")


def _validate_onnx_fidelity(artifact: dict, out_path: Path, tol: float = 1e-3) -> None:
    """Warn if the exported ONNX probabilities diverge from the sklearn model.

    Catches silent serving bugs (e.g. the onnxmltools/LightGBM multiclass
    converter mismatch documented in docs/FINDINGS.md) at export time rather
    than in production. Best-effort: skipped if onnxruntime is unavailable.
    """
    model = artifact.get("model")
    if model is None or not hasattr(model, "predict_proba"):
        return
    try:
        import numpy as _np
        import onnxruntime as _ort

        rng = _np.random.default_rng(0)
        X = rng.normal(size=(64, len(FEATURE_COLS))).astype(_np.float64)
        p_sk = model.predict_proba(artifact["scaler"].transform(_np.asarray(X)))
        sess = _ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        p_ox = _np.asarray(
            sess.run(["probabilities"], {name: X.astype(_np.float32)})[0]
        )
        mae = float(_np.abs(p_sk - p_ox).mean())
        if mae > tol:
            log.warning(
                "ONNX export fidelity check FAILED: probability MAE %.3f > %.0e — "
                "the serving graph disagrees with the sklearn model (see "
                "docs/FINDINGS.md). Do not use models/model.onnx for production "
                "scoring until resolved.",
                mae,
                tol,
            )
        else:
            log.info("ONNX export fidelity check passed (probability MAE %.2e).", mae)
    except ImportError:
        pass
    except Exception as exc:  # never fail training on a diagnostic
        log.warning("ONNX fidelity check skipped (%s).", exc)


def log_to_mlflow(artifact: dict, metrics: dict, cfg: dict) -> None:
    """Log run to MLflow/DagsHub. Skipped silently if credentials are absent."""
    if not os.environ.get("MLFLOW_TRACKING_USERNAME"):
        log.info("MLFLOW_TRACKING_USERNAME not set — skipping MLflow logging.")
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name=f"train_{artifact['model_type']}"):
            mlflow.log_params(
                {
                    "model_type": artifact["model_type"],
                    "n_features": len(FEATURE_COLS),
                    **{
                        f"{artifact['model_type']}.{k}": v
                        for k, v in cfg.get(artifact["model_type"], {}).items()
                    },
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            mlflow.log_artifact(str(MODEL_OUT), artifact_path="models")

            model = artifact.get("model")
            if artifact["model_type"] == "lightgbm" and model is not None:
                importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
                for feat, imp in importances.nlargest(10).items():
                    mlflow.log_metric(f"importance_{feat}", float(imp))

            # Log a registerable MLflow *model* (scaler + classifier as one
            # servable pipeline) and register a new version in the Model
            # Registry. scripts/promote_model.py later promotes the best version
            # to Production. Identification models only; skipped gracefully if the
            # backend has no registry.
            if (
                artifact.get("trained")
                and artifact.get("task") == "identification"
                and model is not None
            ):
                try:
                    import mlflow.sklearn
                    from sklearn.pipeline import Pipeline

                    pipe = Pipeline([("scaler", artifact["scaler"]), ("model", model)])
                    mlflow.sklearn.log_model(
                        pipe,
                        artifact_path="model",
                        registered_model_name=IDENTIFIER_REGISTRY_NAME,
                    )
                    log.info("Registered model version: %s", IDENTIFIER_REGISTRY_NAME)
                except Exception as reg_exc:
                    log.warning(
                        "log_model/registry skipped (%s) — run still logged.", reg_exc
                    )

        log.info("MLflow run logged to %s", cfg["mlflow"]["tracking_uri"])
    except Exception as exc:
        log.warning("MLflow logging failed (%s) — continuing.", exc)


def run() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    load_dotenv(ROOT / ".env")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(TRAIN_IN)
    val_df = pd.read_parquet(VAL_IN)

    with open(CONFIG_IN) as f:
        cfg = yaml.safe_load(f)

    model_type = cfg["model"]["type"]
    log.info(
        "Training: model_type=%s  n_train=%d  n_val=%d",
        model_type,
        len(train_df),
        len(val_df),
    )

    if train_df.empty:
        log.warning("train.parquet is empty — writing untrained artifact placeholder.")
        artifact = {
            "model_type": model_type,
            "task": cfg["model"]["task"],
            "model": None,
            "scaler": StandardScaler(),
            "feature_cols": FEATURE_COLS,
            "label_encoder": None,
            "classes": None,
            "trained": False,
        }
        with open(MODEL_OUT, "wb") as f:
            pickle.dump(artifact, f)
        ONNX_OUT.write_bytes(b"")
        with open(METRICS_OUT, "w") as f:
            json.dump({"trained": False, "reason": "insufficient_data"}, f, indent=2)
        log.info("Collect more sessions to enable training.")
        return

    if model_type == "lightgbm":
        artifact, metrics = train_lightgbm(train_df, val_df, cfg)
    elif model_type == "random_forest":
        artifact, metrics = train_random_forest(train_df, val_df, cfg)
    elif model_type == "xgboost":
        artifact, metrics = train_xgboost(train_df, val_df, cfg)
    elif model_type == "svc":
        artifact, metrics = train_svc(train_df, val_df, cfg)
    elif model_type == "isolation_forest":
        artifact, metrics = train_isolation_forest(train_df, cfg)
    elif model_type == "lof":
        artifact, metrics = train_lof(train_df, cfg)
    elif model_type == "one_class_svm":
        artifact, metrics = train_one_class_svm(train_df, cfg)
    else:
        log.error("Unknown model.type '%s' in config.", model_type)
        sys.exit(1)

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(artifact, f)
    log.info("Saved model artifact: %s", MODEL_OUT)

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved train metrics: %s", METRICS_OUT)

    export_onnx(artifact, ONNX_OUT)
    log_to_mlflow(artifact, metrics, cfg)

    log.info("")
    log.info("=== Training summary ===")
    log.info("  model_type : %s", artifact["model_type"])
    log.info("  trained    : %s", artifact["trained"])
    for k, v in metrics.items():
        if isinstance(v, float) and not isinstance(v, bool):
            log.info("  %-25s %.4f", k, v)
        elif isinstance(v, int):
            log.info("  %-25s %d", k, v)


if __name__ == "__main__":
    run()
