"""
pipeline.inference
==================
Online-scoring + multi-detector aggregation utilities.

Phase 4 of the roadmap. Two submodules:

- ``aggregator``: combines per-detector session scores into one calibrated
  risk in [0, 1] via Naive-Bayes log-odds combination
  (per-detector isotonic calibration + cheat-rate prior).
- ``streaming``: a transport-independent ``SessionStreamState`` that ingests
  events one at a time and emits running per-detector and per-session
  scores, driven by either the WebSocket API or an offline replay client.
"""

from pipeline.inference.aggregator import (
    IsotonicCalibrator,
    RiskAggregator,
    fit_aggregator_from_synthetic,
)
from pipeline.inference.streaming import (
    ScoreUpdate,
    SessionStreamState,
    build_stream_state,
)

__all__ = [
    "IsotonicCalibrator",
    "RiskAggregator",
    "ScoreUpdate",
    "SessionStreamState",
    "build_stream_state",
    "fit_aggregator_from_synthetic",
]
