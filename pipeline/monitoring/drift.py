"""
pipeline/monitoring/drift.py
============================
Per-feature data-drift detection: KS test + Population Stability Index.

What "drift" means
------------------
Data drift is when the data a model sees in production stops looking like the
data it was trained on. In BehaviorDNA the immediate case is concrete: the
models were trained on mock "mouse-on-desktop" recordings, but real GTA
gameplay will be statistically different. Drift detection *measures* that
difference per feature, so you know which features moved and by how much
(and therefore whether the model needs retraining).

Two complementary metrics
-------------------------
**KS test (Kolmogorov–Smirnov, two-sample).** A statistical test for "are
these two samples drawn from the same distribution?". It finds the largest
vertical gap between the two empirical cumulative distribution functions
(the KS *statistic*, 0–1) and returns a p-value. A small p-value (< 0.05)
means the two distributions are significantly different — that feature has
drifted. Intuition: stack both histograms' running totals and measure the
widest gap between the curves.

**PSI (Population Stability Index).** An industry-standard *single number*
for "how much did this feature's distribution shift?". Procedure:

  1. Bin the reference sample into `bins` quantile buckets (deciles by default).
  2. Compute the fraction of reference points and current points in each bin.
  3. PSI = Σ_i (cur%_i − ref%_i) · ln(cur%_i / ref%_i)

Rule-of-thumb thresholds (used across credit-risk / fraud ML):
  - PSI < 0.10  → no significant shift
  - 0.10–0.25   → moderate shift (keep an eye on it)
  - PSI > 0.25  → significant shift (investigate / retrain)

Why both? KS gives a principled significance test; PSI gives an
interpretable magnitude with battle-tested thresholds. Reporting them
together is standard practice.

Worked PSI example (tiny, by hand)
----------------------------------
Reference falls evenly across 2 bins: ref% = [0.5, 0.5].
Current shifts toward the second bin:    cur% = [0.3, 0.7].

  PSI = (0.3 − 0.5)·ln(0.3/0.5) + (0.7 − 0.5)·ln(0.7/0.5)
      = (−0.2)·(−0.511)        + (0.2)·(0.336)
      = 0.102 + 0.067
      = 0.169   → "moderate shift"

A bigger move (cur% = [0.1, 0.9]) gives PSI ≈ 0.64 → "significant".

CLI
---
    python -m pipeline.monitoring.drift \
        --reference data/splits/train.parquet \
        --current   data/splits/test.parquet \
        --out       reports/drift_report.csv

When real recordings land, the headline use is to compare the mock feature
distribution against the real one:

    python -m pipeline.monitoring.drift \
        --reference <mock features.parquet> \
        --current  <real features.parquet>

See ``docs/MONITORING.md`` for the Recording Arrival Runbook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

log = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
SPLITS_DIR = ROOT / "data" / "splits"
REPORTS_DIR = ROOT / "reports"

# PSI severity thresholds (industry-standard)
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

# KS significance level
KS_ALPHA = 0.05


def ks_drift(reference: np.ndarray, current: np.ndarray) -> dict:
    """Two-sample KS test between reference and current samples.

    Returns ``{"statistic", "p_value", "drifted"}``. ``drifted`` is True when
    the p-value is below ``KS_ALPHA`` (distributions significantly differ).
    NaNs are dropped before testing; if either side is empty the result is
    NaN / not-drifted.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "drifted": False}

    result = ks_2samp(ref, cur)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "drifted": bool(result.pvalue < KS_ALPHA),
    }


def psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index between reference and current samples.

    Bin edges come from the reference quantiles (so each reference bin holds
    ~equal mass). ``epsilon`` is Laplace smoothing that keeps the log finite
    when a bin is empty on either side. Returns ``nan`` if either input is
    empty or the reference has no spread (all-constant).
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    # Quantile-based bin edges from the reference; dedupe to handle low-variance
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        # Reference is essentially constant — no meaningful binning
        return float("nan")
    # Stretch the outer edges so points outside the reference range are counted
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    # Laplace smoothing to avoid divide-by-zero / log(0)
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_severity(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value < PSI_MODERATE:
        return "none"
    if value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    """Per-feature drift report comparing two feature DataFrames.

    Returns one row per feature with columns:
      feature, ks_stat, ks_pvalue, ks_drifted, psi, psi_severity, n_ref, n_cur

    Sorted by PSI descending so the worst-drifting features surface first.
    Features missing from either DataFrame are skipped with a warning.
    """
    rows = []
    for feat in feature_cols:
        if feat not in reference_df.columns or feat not in current_df.columns:
            log.warning("Feature %s missing from one of the frames — skipping", feat)
            continue
        ref = reference_df[feat].to_numpy()
        cur = current_df[feat].to_numpy()
        ks = ks_drift(ref, cur)
        psi_val = psi(ref, cur, bins=bins)
        rows.append(
            {
                "feature": feat,
                "ks_stat": ks["statistic"],
                "ks_pvalue": ks["p_value"],
                "ks_drifted": ks["drifted"],
                "psi": psi_val,
                "psi_severity": _psi_severity(psi_val),
                "n_ref": int(np.sum(~np.isnan(ref))),
                "n_cur": int(np.sum(~np.isnan(cur))),
            }
        )

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(
            "psi", ascending=False, na_position="last"
        ).reset_index(drop=True)
    return report


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Per-feature data-drift report (KS + PSI)"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=SPLITS_DIR / "train.parquet",
        help="Reference (baseline) feature parquet",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=SPLITS_DIR / "test.parquet",
        help="Current feature parquet to compare against the reference",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPORTS_DIR / "drift_report.csv",
    )
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    # Imported here so the module's functions can be used without the full
    # feature pipeline being importable in minimal environments.
    from pipeline.features.run import FEATURE_COLS

    if not args.reference.exists():
        log.error("Reference parquet not found: %s", args.reference)
        sys.exit(1)
    if not args.current.exists():
        log.error("Current parquet not found: %s", args.current)
        sys.exit(1)

    ref_df = pd.read_parquet(args.reference)
    cur_df = pd.read_parquet(args.current)
    log.info(
        "Reference: %d rows (%s) | Current: %d rows (%s)",
        len(ref_df),
        args.reference.name,
        len(cur_df),
        args.current.name,
    )

    report = compute_drift_report(ref_df, cur_df, FEATURE_COLS, bins=args.bins)
    if report.empty:
        log.warning("No comparable features — empty report")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    log.info("Wrote %s", args.out)

    n_sig = int((report["psi_severity"] == "significant").sum())
    n_mod = int((report["psi_severity"] == "moderate").sum())
    log.info(
        "Drift summary: %d significant, %d moderate, %d stable (of %d features)",
        n_sig,
        n_mod,
        len(report) - n_sig - n_mod,
        len(report),
    )
    log.info(
        "\nTop drifting features by PSI:\n%s", report.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    run()
