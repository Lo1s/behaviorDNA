"""
tests/test_benchmark.py
=======================
Unit tests for the classical (non-LSTM) paths of pipeline.adversarial.benchmark.

These exercise the detector-scoring + per-session aggregation + ROC/PR curve
machinery on a small in-memory synthetic feature frame — fast, deterministic,
no disk I/O, no torch. The LSTM-AE and CLI paths are covered by manual runs
and the smoke tests in test_lstm_ae / test_streaming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.adversarial.benchmark import (
    _build_detectors,
    _detection_rate_at_fpr,
    compute_pr_curves,
    compute_roc_curves,
    run_benchmark,
)
from pipeline.features.run import FEATURE_COLS


def _synthetic_feature_frame(seed: int = 0) -> pd.DataFrame:
    """Build a feature DataFrame: legit + 3 cheat types, multiple windows/session.

    Cheat rows are shifted on a couple of feature columns so the detectors have
    something to separate. Each session contributes several windows so the
    per-session (session_max) aggregation path is exercised.
    """
    rng = np.random.default_rng(seed)
    rows = []

    def add_session(fname: str, label: str, n_windows: int, shift: float):
        for w in range(n_windows):
            row = {col: float(rng.normal(0, 1)) for col in FEATURE_COLS}
            # Inject separation on a few columns for cheat sessions
            if shift:
                row["click_reaction_mean"] += shift
                row["keystroke_periodicity"] -= shift
            row["cheat_label"] = label
            row["cheat_source_file"] = fname
            row["window_idx"] = w
            rows.append(row)

    # 6 legit sessions, 4 windows each
    for i in range(6):
        add_session(f"legit_{i}.json", "legit", 4, shift=0.0)
    # cheat sessions with separation
    for i in range(4):
        add_session(f"aimbot_{i}.json", "aimbot", 4, shift=2.5)
    for i in range(4):
        add_session(f"macro_{i}.json", "macro", 4, shift=2.0)
    for i in range(4):
        add_session(f"triggerbot_{i}.json", "triggerbot", 4, shift=3.0)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _detection_rate_at_fpr
# ---------------------------------------------------------------------------


class TestDetectionRateAtFpr:
    def test_perfect_separation_full_recall(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.99])
        rate = _detection_rate_at_fpr(y, scores, fpr_threshold=0.05)
        assert rate == 1.0

    def test_all_one_class_returns_nan(self):
        y = np.zeros(5)
        scores = np.arange(5.0)
        assert np.isnan(_detection_rate_at_fpr(y, scores, 0.05))


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_window_aggregation_shape_and_columns(self):
        df = _synthetic_feature_frame()
        out = run_benchmark(df, aggregation="window")
        # 3 detectors × 3 cheat types = 9 rows
        assert len(out) == 9
        for col in ("detector", "cheat_label", "roc_auc", "pr_auc"):
            assert col in out.columns
        assert out["roc_auc"].between(0, 1).all()

    def test_session_max_aggregation_runs(self):
        df = _synthetic_feature_frame()
        out = run_benchmark(df, aggregation="session_max")
        assert len(out) == 9
        # With injected separation, at least one detector should beat chance
        assert (out["roc_auc"] > 0.6).any()

    def test_detectors_are_three(self):
        dets = _build_detectors()
        assert set(dets) == {"IsolationForest", "LocalOutlierFactor", "OneClassSVM"}


# ---------------------------------------------------------------------------
# ROC / PR curve helpers
# ---------------------------------------------------------------------------


class TestCurves:
    def test_roc_curves_keys_and_auc(self):
        df = _synthetic_feature_frame()
        curves = compute_roc_curves(df, aggregation="session_max")
        # one curve per (detector, cheat_label)
        assert len(curves) == 9
        for (det, label), c in curves.items():
            assert set(c) >= {"fpr", "tpr", "thresholds", "auc"}
            assert 0.0 <= c["auc"] <= 1.0

    def test_pr_curves_keys_and_ap(self):
        df = _synthetic_feature_frame()
        curves = compute_pr_curves(df, aggregation="session_max")
        assert len(curves) == 9
        for (det, label), c in curves.items():
            assert set(c) >= {"precision", "recall", "thresholds", "ap"}
            assert 0.0 <= c["ap"] <= 1.0

    def test_window_aggregation_curves_also_work(self):
        df = _synthetic_feature_frame()
        curves = compute_roc_curves(df, aggregation="window")
        assert len(curves) == 9
