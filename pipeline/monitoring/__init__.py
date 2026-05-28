"""
pipeline.monitoring
===================
Production observability for the feature pipeline. Currently:

- ``drift``: per-feature data-drift detection (KS test + PSI) between a
  reference distribution and a current one. Used to quantify how far new
  recordings (real gameplay) drift from the training baseline (mock data).

See ``docs/MONITORING.md`` for the plain-English explanation of KS / PSI
and the Recording Arrival Runbook.
"""

from pipeline.monitoring.drift import (
    compute_drift_report,
    ks_drift,
    psi,
)

__all__ = [
    "compute_drift_report",
    "ks_drift",
    "psi",
]
