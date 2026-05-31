"""
tests/test_calibration.py
=========================
Unit tests for pipeline/calibration.py (Phase 5b).
"""

from __future__ import annotations

import numpy as np

from pipeline.calibration import (
    expected_calibration_error,
    multiclass_brier,
    reliability_curve,
)


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_is_zero(self):
        # Confidence exactly matches accuracy within each bin:
        # 100 preds at conf 0.9 of which 90 correct → bin gap 0.
        conf = np.r_[np.full(100, 0.9), np.full(100, 0.6)]
        correct = np.r_[
            np.r_[np.ones(90), np.zeros(10)], np.r_[np.ones(60), np.zeros(40)]
        ]
        ece = expected_calibration_error(conf, correct, n_bins=10)
        assert ece < 1e-9

    def test_overconfident_has_positive_ece(self):
        # Always 99% confident but only 50% right → large gap.
        conf = np.full(100, 0.99)
        correct = np.r_[np.ones(50), np.zeros(50)]
        ece = expected_calibration_error(conf, correct, n_bins=10)
        assert ece > 0.4

    def test_empty_is_nan(self):
        assert np.isnan(expected_calibration_error(np.array([]), np.array([])))


class TestReliabilityCurve:
    def test_shapes_and_counts(self):
        conf = np.linspace(0, 1, 50)
        correct = (conf > 0.5).astype(float)
        centers, acc, conf_b, count = reliability_curve(conf, correct, n_bins=10)
        assert len(centers) == len(acc) == len(conf_b) == len(count) == 10
        assert count.sum() == 50

    def test_empty_bins_are_nan(self):
        # All confidences in one region → other bins empty (NaN acc).
        conf = np.full(20, 0.05)
        correct = np.ones(20)
        _, acc, _, count = reliability_curve(conf, correct, n_bins=10)
        assert count[0] == 20
        assert np.isnan(acc[5])

    def test_top_edge_lands_in_last_bin(self):
        _, _, _, count = reliability_curve([1.0], [1.0], n_bins=10)
        assert count[-1] == 1


class TestMulticlassBrier:
    def test_perfect_is_zero(self):
        proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert multiclass_brier([0, 1], proba) == 0.0

    def test_worst_case_is_two(self):
        # All mass on the wrong class → each sample contributes 1+1 = 2.
        proba = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert abs(multiclass_brier([0, 1], proba) - 2.0) < 1e-9

    def test_uniform_three_class(self):
        proba = np.full((4, 3), 1 / 3)
        # per sample: (1/3-1)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 = 0.6667
        assert abs(multiclass_brier([0, 1, 2, 0], proba) - 2 / 3) < 1e-9
