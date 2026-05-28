"""
tests/test_drift.py
===================
Unit tests for pipeline/monitoring/drift.py (KS test + PSI + report).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.monitoring.drift import (
    PSI_SIGNIFICANT,
    compute_drift_report,
    ks_drift,
    psi,
)

# ---------------------------------------------------------------------------
# KS test
# ---------------------------------------------------------------------------


class TestKsDrift:
    def test_identical_not_drifted(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        out = ks_drift(x, x.copy())
        assert out["drifted"] is False
        assert out["p_value"] > 0.05
        assert out["statistic"] == 0.0  # identical samples

    def test_shifted_is_drifted(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(3, 1, 500)  # mean shifted by 3σ
        out = ks_drift(ref, cur)
        assert out["drifted"] is True
        assert out["p_value"] < 0.05
        assert out["statistic"] > 0.5

    def test_empty_input_safe(self):
        out = ks_drift(np.array([]), np.array([1.0, 2.0]))
        assert out["drifted"] is False
        assert np.isnan(out["statistic"])

    def test_nan_dropped(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 200)
        cur = np.concatenate([rng.normal(0, 1, 200), [np.nan] * 50])
        out = ks_drift(ref, cur)
        # Same underlying distribution (NaNs ignored) → not drifted
        assert out["drifted"] is False


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


class TestPsi:
    def test_identical_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 2000)
        # Split the same distribution in half — PSI should be tiny
        value = psi(x[:1000], x[1000:])
        assert value < 0.1

    def test_large_shift_significant(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(3, 1, 2000)
        value = psi(ref, cur)
        assert value > PSI_SIGNIFICANT

    def test_worked_example_two_bins(self):
        # Reference uniformly split across [0,1) and [1,2); current shifts to bin 2.
        # With ref evenly in two bins and cur 30/70, hand-computed PSI ≈ 0.169.
        ref = np.concatenate([np.full(500, 0.5), np.full(500, 1.5)])
        cur = np.concatenate([np.full(300, 0.5), np.full(700, 1.5)])
        value = psi(ref, cur, bins=2)
        assert abs(value - 0.169) < 0.02

    def test_empty_returns_nan(self):
        assert np.isnan(psi(np.array([]), np.array([1.0, 2.0])))

    def test_constant_reference_returns_nan(self):
        # Reference all identical → no meaningful bins
        assert np.isnan(
            psi(np.full(100, 5.0), np.random.default_rng(0).normal(0, 1, 100))
        )

    def test_finite_when_current_bin_empty(self):
        # Current never lands in some reference bins → Laplace smoothing keeps it finite
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000) + 10  # entirely outside reference range
        value = psi(ref, cur)
        assert np.isfinite(value)
        assert value > PSI_SIGNIFICANT


# ---------------------------------------------------------------------------
# compute_drift_report
# ---------------------------------------------------------------------------


class TestDriftReport:
    def _frames(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame(
            {
                "stable": rng.normal(0, 1, 500),
                "drifted": rng.normal(0, 1, 500),
            }
        )
        cur = pd.DataFrame(
            {
                "stable": rng.normal(0, 1, 500),
                "drifted": rng.normal(4, 1, 500),  # big shift
            }
        )
        return ref, cur

    def test_report_columns_and_rows(self):
        ref, cur = self._frames()
        report = compute_drift_report(ref, cur, ["stable", "drifted"])
        assert list(report.columns) == [
            "feature",
            "ks_stat",
            "ks_pvalue",
            "ks_drifted",
            "psi",
            "psi_severity",
            "n_ref",
            "n_cur",
        ]
        assert len(report) == 2

    def test_sorted_by_psi_descending(self):
        ref, cur = self._frames()
        report = compute_drift_report(ref, cur, ["stable", "drifted"])
        # The drifted feature must be first (highest PSI)
        assert report.iloc[0]["feature"] == "drifted"
        assert report.iloc[0]["psi"] >= report.iloc[1]["psi"]
        assert report.iloc[0]["psi_severity"] == "significant"
        assert bool(report.iloc[0]["ks_drifted"]) is True
        # The stable feature should not be flagged as significant
        assert report.iloc[1]["psi_severity"] in ("none", "moderate")

    def test_missing_feature_skipped(self):
        ref, cur = self._frames()
        report = compute_drift_report(ref, cur, ["stable", "nonexistent"])
        assert set(report["feature"]) == {"stable"}
