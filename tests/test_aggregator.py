"""
tests/test_aggregator.py
========================
Unit tests for pipeline/inference/aggregator.py.

Validates the Naive-Bayes log-odds combination math and the isotonic
calibrator wrapper.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.inference.aggregator import (
    IsotonicCalibrator,
    RiskAggregator,
)

# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------


class TestIsotonicCalibrator:
    def test_untrained_returns_half(self):
        cal = IsotonicCalibrator()
        out = cal.predict_proba(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(out, np.full(3, 0.5))

    def test_monotonic_increasing(self):
        rng = np.random.default_rng(0)
        n = 200
        scores = np.concatenate([rng.normal(0, 1, n), rng.normal(3, 1, n)])
        labels = np.concatenate([np.zeros(n), np.ones(n)])
        cal = IsotonicCalibrator().fit(scores, labels)
        # Probabilities must be non-decreasing as score increases
        xs = np.linspace(-2, 5, 50)
        ps = cal.predict_proba(xs)
        assert np.all(np.diff(ps) >= -1e-9), "calibrator must be monotonic"

    def test_clamps_to_eps_range(self):
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        cal = IsotonicCalibrator(eps=0.01).fit(scores, labels)
        # Extreme score should clamp at 1 - eps
        very_high = cal.predict_proba(np.array([100.0]))[0]
        assert very_high <= 1.0 - 0.01 + 1e-9
        very_low = cal.predict_proba(np.array([-100.0]))[0]
        assert very_low >= 0.01 - 1e-9

    def test_fit_empty_raises(self):
        with pytest.raises(ValueError):
            IsotonicCalibrator().fit(np.array([]), np.array([]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            IsotonicCalibrator().fit(np.array([1, 2, 3]), np.array([0, 1]))


# ---------------------------------------------------------------------------
# RiskAggregator
# ---------------------------------------------------------------------------


def _make_simple_aggregator(prior: float = 0.05) -> RiskAggregator:
    rng = np.random.default_rng(0)
    n = 200
    legit_scores = rng.normal(0, 1, n)
    cheat_scores = rng.normal(3, 1, n)
    scores = np.concatenate([legit_scores, cheat_scores])
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    return RiskAggregator(prior_cheat_rate=prior).fit(
        {
            "det1": (scores, labels),
            "det2": (scores, labels),
            "det3": (scores, labels),
        }
    )


class TestRiskAggregator:
    def test_prior_logit_matches_formula(self):
        agg = RiskAggregator(prior_cheat_rate=0.05)
        expected = math.log(0.05 / 0.95)
        assert abs(agg.prior_logit - expected) < 1e-9

    def test_three_strong_signals_high_risk(self):
        agg = _make_simple_aggregator()
        risk = agg.aggregate({"det1": 5.0, "det2": 5.0, "det3": 5.0})
        assert risk > 0.95

    def test_three_weak_signals_low_risk(self):
        agg = _make_simple_aggregator()
        risk = agg.aggregate({"det1": -3.0, "det2": -3.0, "det3": -3.0})
        assert risk < 0.05

    def test_one_strong_two_silent_still_low(self):
        """Prior + two 'no evidence' detectors should require multiple strong signals."""
        agg = _make_simple_aggregator(prior=0.05)
        risk = agg.aggregate({"det1": 5.0, "det2": -3.0, "det3": -3.0})
        # One strong, two strongly-disagreeing → low risk
        assert risk < 0.2

    def test_unknown_detector_ignored(self):
        agg = _make_simple_aggregator()
        with_extra = agg.aggregate(
            {"det1": 5.0, "det2": 5.0, "det3": 5.0, "novel": 99.0}
        )
        without_extra = agg.aggregate({"det1": 5.0, "det2": 5.0, "det3": 5.0})
        assert abs(with_extra - without_extra) < 1e-9

    def test_nan_score_ignored(self):
        agg = _make_simple_aggregator()
        all_strong = agg.aggregate({"det1": 5.0, "det2": 5.0, "det3": 5.0})
        with_nan = agg.aggregate({"det1": 5.0, "det2": 5.0, "det3": float("nan")})
        # NaN drops one detector → posterior closer to "two-strong, one-silent"
        assert with_nan != all_strong
        # but still elevated (two strong signals dominate)
        assert with_nan > 0.5

    def test_output_in_unit_interval(self):
        agg = _make_simple_aggregator()
        for d1 in (-10.0, 0.0, 10.0):
            for d2 in (-10.0, 0.0, 10.0):
                risk = agg.aggregate({"det1": d1, "det2": d2, "det3": 0.0})
                assert 0.0 <= risk <= 1.0

    def test_explain_components_sum(self):
        agg = _make_simple_aggregator()
        scores = {"det1": 2.0, "det2": -1.0, "det3": 3.0}
        info = agg.explain(scores)
        expected_total = info["prior_logit"] + sum(info["per_detector_logit"].values())
        assert abs(info["posterior_logit"] - expected_total) < 1e-9
        # posterior_risk = sigmoid(posterior_logit)
        sig = 1.0 / (1.0 + math.exp(-info["posterior_logit"]))
        assert abs(info["posterior_risk"] - sig) < 1e-9

    def test_aggregate_many(self):
        agg = _make_simple_aggregator()
        out = agg.aggregate_many(
            {
                "session_a": {"det1": 5.0, "det2": 5.0, "det3": 5.0},
                "session_b": {"det1": -3.0, "det2": -3.0, "det3": -3.0},
            }
        )
        assert out["session_a"] > 0.95
        assert out["session_b"] < 0.05

    def test_fit_empty_raises(self):
        with pytest.raises(ValueError):
            RiskAggregator().fit({})
