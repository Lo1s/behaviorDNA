"""
tests/test_promote_model.py
===========================
Unit tests for the best-run selection logic in scripts/promote_model.py.

The live registry write needs MLflow credentials, so we test the pure selection
function with fake run objects (the same shape MlflowClient returns).
"""

from __future__ import annotations

import types

from scripts.promote_model import select_best_run


def _run(run_id: str, **metrics):
    return types.SimpleNamespace(
        info=types.SimpleNamespace(run_id=run_id),
        data=types.SimpleNamespace(metrics=metrics),
    )


class TestSelectBestRun:
    def test_picks_highest_metric(self):
        runs = [
            _run("a", val_accuracy=0.70),
            _run("b", val_accuracy=0.85),
            _run("c", val_accuracy=0.80),
        ]
        assert select_best_run(runs, "val_accuracy").info.run_id == "b"

    def test_skips_runs_missing_metric(self):
        runs = [
            _run("a", train_accuracy=1.0),  # no val_accuracy
            _run("b", val_accuracy=0.60),
        ]
        assert select_best_run(runs, "val_accuracy").info.run_id == "b"

    def test_none_when_no_run_has_metric(self):
        runs = [_run("a", train_accuracy=1.0), _run("b", brier=0.2)]
        assert select_best_run(runs, "val_accuracy") is None

    def test_empty_returns_none(self):
        assert select_best_run([], "val_accuracy") is None

    def test_first_seen_wins_on_tie(self):
        runs = [_run("a", val_accuracy=0.8), _run("b", val_accuracy=0.8)]
        # strictly-greater comparison → first stays
        assert select_best_run(runs, "val_accuracy").info.run_id == "a"
