"""
scripts/build_serving_bundle.py
===============================
Build the versioned serving bundle the API loads at startup.

Fits the classical detectors + feature scaler + isotonic calibrators once (from
the synthetic-cheat dataset) and persists them to ``models/serving_bundle.pkl``,
so serving **loads** immutable artifacts instead of fitting at startup — which
would be slow and would require ``data/synthetic`` on the serving host.

Run AFTER (re)generating the synthetic dataset, then version the bundle::

    python -m pipeline.adversarial.generate_dataset   # deterministic, seed 42
    python -m scripts.build_serving_bundle
    dvc add models/serving_bundle.pkl && dvc push

The LSTM-AE is not bundled — it stays a separate DVC-tracked artifact
(``models/lstm_ae.pt``) loaded alongside the bundle at serve time.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from pipeline.inference.streaming import (
    SERVING_BUNDLE_SCHEMA_VERSION,
    build_stream_state,
    save_stream_bundle,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    n_files = len(list(SYNTHETIC_DIR.glob("*.json"))) if SYNTHETIC_DIR.exists() else 0
    if n_files == 0:
        log.error(
            "data/synthetic is empty — run `python -m pipeline.adversarial."
            "generate_dataset` first (deterministic, seed 42)."
        )
        sys.exit(1)

    log.info("Fitting serving components from %d synthetic sessions…", n_files)
    state = build_stream_state()

    metadata = {
        "n_synthetic_files": n_files,
        "generator_seed": 42,
        "source": "data/synthetic (regenerated deterministically from data/raw)",
        "git_sha": _git_sha(),
    }
    path = save_stream_bundle(state, metadata=metadata)
    log.info(
        "Done. Bundle (schema v%d, %d bytes) → %s\nNext: `dvc add %s && dvc push`",
        SERVING_BUNDLE_SCHEMA_VERSION,
        path.stat().st_size,
        path,
        path.relative_to(ROOT),
    )


if __name__ == "__main__":
    run()
