"""
pipeline/adversarial/benchmark.py
=================================
Score the BehaviorDNA detector zoo against the synthetic-cheat dataset.

Reads every JSON in ``data/synthetic/`` (produced by ``generate_dataset``),
runs each file through the ingestion + feature-engineering pipeline in
memory (no DVC, no parquet writes), trains the unsupervised detectors on
legit-only feature rows, then scores **all** rows and computes per-detector
× per-cheat-type metrics:

  * ROC AUC                              — overall ranking quality
  * PR AUC                               — quality when cheats are rare
  * Detection rate at fixed False Positive Rate (default 5%)
  * Mean anomaly score per cheat type    — interpretability

Run:
    python -m pipeline.adversarial.benchmark
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from pipeline.adversarial.bot_generator import CHEAT_LEGIT
from pipeline.features.run import FEATURE_COLS, process_session_windows
from pipeline.ingestion.run import parse_events, parse_session_metadata

log = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
RESULTS_DIR = ROOT / "reports" / "adversarial"


# ---------------------------------------------------------------------------
# Loading + feature extraction (in-memory mini-pipeline)
# ---------------------------------------------------------------------------


def _features_from_synthetic_file(path: Path) -> pd.DataFrame:
    """Run ingestion + feature engineering on one synthetic JSON, in memory.

    Returns one DataFrame row per 30-second window, with ``cheat_label`` and
    ``cheat_source_file`` columns added for grouping in the benchmark.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta = parse_session_metadata(data, path)
    events_df = parse_events(data)
    if events_df.empty:
        return pd.DataFrame()

    norm_factor = (float(meta["sensitivity"]) * float(meta["dpi"])) / 800.0
    sess_events = events_df.sort_values("t")
    windows = process_session_windows(sess_events, norm_factor)

    if not windows:
        return pd.DataFrame()

    df = pd.DataFrame(windows)
    df["session_id"] = meta["session_id"]
    df["player"] = meta["player"]
    df["game"] = meta["game"]
    df["cheat_label"] = data.get("cheat_label", CHEAT_LEGIT)
    df["cheat_source_file"] = path.name
    return df


def load_synthetic_features(synthetic_dir: Path = SYNTHETIC_DIR) -> pd.DataFrame:
    """Build a feature DataFrame for the whole synthetic dataset."""
    files = sorted(synthetic_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files in {synthetic_dir}. "
            "Run `python -m pipeline.adversarial.generate_dataset` first."
        )

    frames = []
    for path in files:
        try:
            frames.append(_features_from_synthetic_file(path))
        except Exception as e:
            log.warning("Failed to process %s: %s", path.name, e)

    if not frames:
        raise RuntimeError("No synthetic features could be extracted")

    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


# ---------------------------------------------------------------------------
# Detector zoo + benchmarking
# ---------------------------------------------------------------------------


def _build_detectors() -> dict[str, object]:
    return {
        "IsolationForest": IsolationForest(
            n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1
        ),
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1
        ),
        "OneClassSVM": OneClassSVM(kernel="rbf", nu=0.05),
    }


@dataclass
class BenchmarkResult:
    detector: str
    cheat_label: str
    n_legit: int
    n_cheat: int
    roc_auc: float
    pr_auc: float
    detection_rate_at_fpr: float
    fpr_threshold: float
    mean_score_legit: float
    mean_score_cheat: float


def _detection_rate_at_fpr(y_true, scores, fpr_threshold: float) -> float:
    """Recall (true-positive rate) achieved when FPR is pinned to a target.

    ``scores`` should be such that *higher = more anomalous*. We invert
    sklearn's ``score_samples`` output before passing it here.
    """
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    # Find the largest operating point with FPR <= threshold
    valid = fpr <= fpr_threshold
    if not valid.any():
        return 0.0
    return float(tpr[valid].max())


def _benchmark_one_cheat(
    name: str,
    detector,
    legit_X: np.ndarray,
    cheat_X: np.ndarray,
    cheat_label: str,
    fpr_threshold: float,
) -> BenchmarkResult:
    """Train detector on legit-only, score both, compute metrics."""
    detector.fit(legit_X)

    legit_scores = -detector.score_samples(legit_X)  # higher = more anomalous
    cheat_scores = -detector.score_samples(cheat_X)

    scores = np.concatenate([legit_scores, cheat_scores])
    y_true = np.concatenate([np.zeros(len(legit_X)), np.ones(len(cheat_X))])

    try:
        roc = roc_auc_score(y_true, scores)
    except ValueError:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, scores)
    except ValueError:
        pr = float("nan")

    return BenchmarkResult(
        detector=name,
        cheat_label=cheat_label,
        n_legit=len(legit_X),
        n_cheat=len(cheat_X),
        roc_auc=float(roc),
        pr_auc=float(pr),
        detection_rate_at_fpr=_detection_rate_at_fpr(y_true, scores, fpr_threshold),
        fpr_threshold=fpr_threshold,
        mean_score_legit=float(np.mean(legit_scores)),
        mean_score_cheat=float(np.mean(cheat_scores)),
    )


def run_benchmark(
    features_df: pd.DataFrame,
    fpr_threshold: float = 0.05,
    detectors: dict | None = None,
) -> pd.DataFrame:
    """Train each detector on legit-only rows, score per cheat type."""
    if detectors is None:
        detectors = _build_detectors()

    X = features_df[FEATURE_COLS].fillna(0.0).to_numpy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    legit_mask = features_df["cheat_label"].eq(CHEAT_LEGIT).to_numpy()
    legit_X = X_scaled[legit_mask]
    if legit_X.shape[0] < 5:
        raise RuntimeError(
            "Not enough legit windows to train detectors "
            f"({legit_X.shape[0]} < 5). Generate more sessions first."
        )

    cheat_labels = sorted(set(features_df["cheat_label"]) - {CHEAT_LEGIT})
    if not cheat_labels:
        raise RuntimeError("No cheat-labelled rows in the dataset.")

    rows: list[BenchmarkResult] = []
    for name, detector in detectors.items():
        for cheat_label in cheat_labels:
            cheat_X = X_scaled[features_df["cheat_label"].eq(cheat_label).to_numpy()]
            if cheat_X.shape[0] == 0:
                continue
            rows.append(
                _benchmark_one_cheat(
                    name, detector, legit_X, cheat_X, cheat_label, fpr_threshold
                )
            )

    out = pd.DataFrame([r.__dict__ for r in rows])
    return out


def compute_roc_curves(
    features_df: pd.DataFrame, detectors: dict | None = None
) -> dict[tuple[str, str], dict]:
    """Return raw ROC curves keyed by (detector, cheat_label).

    Each value is a dict with ``fpr``, ``tpr``, ``thresholds``, ``auc`` —
    suitable for plotting a curve grid in the notebook.
    """
    if detectors is None:
        detectors = _build_detectors()

    X = features_df[FEATURE_COLS].fillna(0.0).to_numpy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    legit_mask = features_df["cheat_label"].eq(CHEAT_LEGIT).to_numpy()
    legit_X = X_scaled[legit_mask]
    cheat_labels = sorted(set(features_df["cheat_label"]) - {CHEAT_LEGIT})

    out: dict[tuple[str, str], dict] = {}
    for name, detector in detectors.items():
        # Re-fit per detector (LOF cannot be re-used across runs)
        det = _build_detectors()[name]
        det.fit(legit_X)
        legit_scores = -det.score_samples(legit_X)
        for cheat_label in cheat_labels:
            cheat_X = X_scaled[features_df["cheat_label"].eq(cheat_label).to_numpy()]
            if cheat_X.shape[0] == 0:
                continue
            cheat_scores = -det.score_samples(cheat_X)
            scores = np.concatenate([legit_scores, cheat_scores])
            y_true = np.concatenate([np.zeros(len(legit_X)), np.ones(len(cheat_X))])
            fpr, tpr, thresholds = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            out[(name, cheat_label)] = dict(
                fpr=fpr, tpr=tpr, thresholds=thresholds, auc=float(auc)
            )
    return out


def compute_pr_curves(
    features_df: pd.DataFrame, detectors: dict | None = None
) -> dict[tuple[str, str], dict]:
    """Return raw precision-recall curves keyed by (detector, cheat_label)."""
    if detectors is None:
        detectors = _build_detectors()

    X = features_df[FEATURE_COLS].fillna(0.0).to_numpy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    legit_mask = features_df["cheat_label"].eq(CHEAT_LEGIT).to_numpy()
    legit_X = X_scaled[legit_mask]
    cheat_labels = sorted(set(features_df["cheat_label"]) - {CHEAT_LEGIT})

    out: dict[tuple[str, str], dict] = {}
    for name, _ in detectors.items():
        det = _build_detectors()[name]
        det.fit(legit_X)
        legit_scores = -det.score_samples(legit_X)
        for cheat_label in cheat_labels:
            cheat_X = X_scaled[features_df["cheat_label"].eq(cheat_label).to_numpy()]
            if cheat_X.shape[0] == 0:
                continue
            cheat_scores = -det.score_samples(cheat_X)
            scores = np.concatenate([legit_scores, cheat_scores])
            y_true = np.concatenate([np.zeros(len(legit_X)), np.ones(len(cheat_X))])
            precision, recall, thresholds = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            out[(name, cheat_label)] = dict(
                precision=precision, recall=recall, thresholds=thresholds, ap=float(ap)
            )
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _format_summary(results: pd.DataFrame) -> str:
    """Pretty-print a tabular benchmark summary."""
    pivot = results.pivot_table(
        index="detector",
        columns="cheat_label",
        values="roc_auc",
        aggfunc="first",
    )
    return pivot.round(3).to_string()


def run() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading synthetic features from %s", SYNTHETIC_DIR)
    df = load_synthetic_features()
    log.info(
        "  %d windows across %d files | cheat label counts: %s",
        len(df),
        df["cheat_source_file"].nunique(),
        df["cheat_label"].value_counts().to_dict(),
    )

    results = run_benchmark(df)
    out_path = RESULTS_DIR / "benchmark_results.csv"
    results.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)

    log.info("\nROC AUC by detector × cheat type:")
    log.info("\n%s", _format_summary(results))


if __name__ == "__main__":
    run()
