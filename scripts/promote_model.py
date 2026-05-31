"""
scripts/promote_model.py
========================
Promote the best logged identification model to the MLflow Model Registry's
**Production** stage, selected by validation accuracy.

Each `dvc repro` (training) logs a run to the DagsHub-hosted MLflow with a
`val_accuracy` metric and the model artifact. This script closes the MLOps loop:
find the best run, register its model as a new version, and mark that version
Production (archiving the previous Production version). Promotion is a
deliberate, auditable step — separate from training — which is how you'd gate a
model into serving.

Degrades gracefully: with no MLflow credentials in `.env` it logs and exits 0
(same contract as the training-stage MLflow logging), so it never breaks an
offline/CI run. Registry support varies by backend; if the hosted registry
rejects the operation the error is reported and the script exits non-zero
without crashing the caller.

Usage:
    python -m scripts.promote_model
    python -m scripts.promote_model --metric val_accuracy --stage Production
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from pipeline.constants import IDENTIFIER_REGISTRY_NAME

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "training.yaml"
DEFAULT_MODEL_NAME = IDENTIFIER_REGISTRY_NAME
DEFAULT_METRIC = "val_accuracy"


def select_best_run(runs, metric: str = DEFAULT_METRIC):
    """Return the run with the highest ``metric`` (ties → first seen).

    Pure and backend-agnostic: ``runs`` is any iterable of objects exposing
    ``.info.run_id`` and ``.data.metrics`` (a dict). Runs missing the metric are
    skipped. Returns ``None`` when no run carries the metric.
    """
    best, best_val = None, float("-inf")
    for r in runs:
        val = r.data.metrics.get(metric)
        if val is None:
            continue
        if val > best_val:
            best, best_val = r, val
    return best


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote best model to Production")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--stage", default="Production")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    load_dotenv(ROOT / ".env")

    if not os.environ.get("MLFLOW_TRACKING_USERNAME"):
        log.warning("MLFLOW_TRACKING_USERNAME not set — skipping promotion.")
        return 0

    import mlflow
    from mlflow.tracking import MlflowClient

    cfg = yaml.safe_load(open(CONFIG))
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = MlflowClient()

    # Training (log_to_mlflow) registers a model version per run. Pick the
    # version whose run has the best metric, then promote *that* version.
    versions = list(client.search_model_versions(f"name='{args.model_name}'"))
    if not versions:
        log.error(
            "No registered versions for '%s' — run training with MLflow "
            "credentials first (it logs+registers a version per run).",
            args.model_name,
        )
        return 1

    runs, version_by_run = [], {}
    for v in versions:
        if not v.run_id:
            continue
        try:
            runs.append(client.get_run(v.run_id))
            version_by_run[v.run_id] = v
        except Exception:  # run deleted / inaccessible
            continue

    best = select_best_run(runs, args.metric)
    if best is None:
        log.error("No registered version's run carries metric '%s'.", args.metric)
        return 1
    best_version = version_by_run[best.info.run_id]
    log.info(
        "Best: version %s (run %s, %s=%.4f)",
        best_version.version,
        best.info.run_id,
        args.metric,
        best.data.metrics[args.metric],
    )

    try:
        client.transition_model_version_stage(
            args.model_name,
            best_version.version,
            args.stage,
            archive_existing_versions=True,
        )
        log.info(
            "Promoted %s v%s → %s", args.model_name, best_version.version, args.stage
        )
    except Exception as stage_err:
        # MLflow ≥ 2.9 deprecates stages in favour of aliases; some backends
        # only support one. Fall back to a 'production' alias.
        log.warning("Stage transition unsupported (%s) — setting alias.", stage_err)
        try:
            client.set_registered_model_alias(
                args.model_name, "production", best_version.version
            )
            log.info(
                "Set alias 'production' → %s v%s",
                args.model_name,
                best_version.version,
            )
        except Exception as alias_err:
            log.error("Registry promotion failed (%s).", alias_err)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(run())
