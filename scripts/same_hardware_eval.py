"""
scripts/same_hardware_eval.py
=============================
Recompute the same-hardware identification result — **hydra vs dninix**, recorded
on the *same PC with identical game settings*, so telling them apart cannot lean
on any hardware artefact (DPI / polling / OS). This is the cleanest behavioural-
identity number in the project; the cross-hardware score is optimistic by
comparison and should not be the headline.

Mirrors ``notebooks/12_explainability.ipynb`` Part 3 exactly: filter the window
features to the same-hardware pair, apply the pipeline's player-stratified
session-level split, train the pipeline LightGBM, and evaluate test accuracy vs
the majority-class baseline with a window bootstrap CI.

Writes a machine-readable, provenance-stamped ``reports/same_hardware.json`` that
``scripts/generate_results.py`` consumes (replacing hand-entered constants).
Re-run whenever the pipeline regenerates features::

    dvc repro && python -m scripts.same_hardware_eval
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score

from pipeline.evaluation.run import bootstrap_ci
from pipeline.features.split import split
from pipeline.training.run import train_lightgbm

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PAIR = ["dninix", "hydra"]  # same physical rig, identical in-game settings
FEATURES = ROOT / "data" / "processed" / "features.parquet"
OUT = ROOT / "reports" / "same_hardware.json"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    cfg = yaml.safe_load(open(ROOT / "configs" / "training.yaml"))
    feats = pd.read_parquet(FEATURES)
    pair = feats[feats["player"].isin(PAIR)].copy()
    if pair["player"].nunique() < 2:
        log.error(
            "Same-hardware pair %s not both present in features — got %s",
            PAIR,
            sorted(pair["player"].unique()),
        )
        sys.exit(1)

    tr, va, te = split(
        pair,
        cfg["data"]["test_size"],
        cfg["data"]["val_size"],
        cfg["data"]["random_seed"],
        cfg["data"]["min_sessions_per_player"],
    )
    art, _ = train_lightgbm(tr, va, cfg)
    if not art.get("trained"):
        log.error("LightGBM did not train (need >=2 players in the train fold).")
        sys.exit(1)

    cols = art["feature_cols"]
    le = art["label_encoder"]
    y_true = le.transform(te["player"])
    x_te = art["scaler"].transform(te[cols].fillna(0.0))
    y_pred = art["model"].predict(x_te)

    acc = float(accuracy_score(y_true, y_pred))
    counts = np.bincount(y_true)
    baseline = float(counts.max() / counts.sum())
    acc_ci = bootstrap_ci(y_true, y_pred, accuracy_score)

    report = {
        "experiment": "same_hardware_identification",
        "description": (
            "hydra vs dninix on identical hardware (same PC + in-game settings) "
            "— pure behavioural identity, no DPI/polling/OS confound."
        ),
        "pair": PAIR,
        "model": cfg["model"]["type"],
        "feature_set": f"ID_FEATURE_COLS ({len(cols)} features)",
        "split": "player-stratified session-level holdout (pipeline/features/split.py)",
        "accuracy": round(acc, 4),
        "majority_baseline": round(baseline, 4),
        "accuracy_ci95": [round(float(acc_ci[0]), 4), round(float(acc_ci[1]), 4)],
        "ci_method": (
            "window bootstrap; windows within a session are correlated, so this "
            "interval is optimistic (same caveat as the main eval)."
        ),
        "n_train_windows": int(len(tr)),
        "n_val_windows": int(len(va)),
        "n_test_windows": int(len(te)),
        "n_test_sessions": int(te["session_id"].nunique()),
        "source_notebook": "notebooks/12_explainability.ipynb (Part 3)",
        "features_md5": _md5(FEATURES),
        "git_sha": _git_sha(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    log.info(
        "Same-hardware: acc=%.3f vs baseline=%.3f (CI95 %.2f-%.2f) over %d test "
        "windows / %d sessions -> %s",
        acc,
        baseline,
        acc_ci[0],
        acc_ci[1],
        len(te),
        report["n_test_sessions"],
        OUT,
    )


if __name__ == "__main__":
    run()
