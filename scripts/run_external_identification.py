"""
scripts/run_external_identification.py
======================================
Phase 6 experiment runner: the *exact* windowed-feature pipeline on public
mouse-dynamics corpora, at 10–120 users (docs/ROADMAP.md Phase 6).

Two corpora, three questions:

  SapiMouse (120 users)  — closed-set users-curve (3→120): does the windowed
                           identifier survive beyond 3 friends? Plus window-
                           level verification EER and an open-set split
                           (60 enrolled / 60 unknown).
                           Protocol = the SapiMouse paper's own: train on each
                           user's 3-minute session, test on their 1-minute one
                           (session-held-out by construction).
  Balabit (10 users)     — closed-set accuracy (session-held-out) and the
                           challenge's real verification task: 816 labelled
                           test sessions (genuine vs impostor) → session EER.

Features: MOUSE_ID_FEATURE_COLS (keyboard features excluded — these corpora
have no keyboard channel). Sessions are segmented at idle gaps before
windowing (desktop captures are not continuous gameplay; see
``pipeline/external/base.py:split_on_idle``).

Output: reports/external_identification.json (consumed by
``scripts/generate_results.py`` for the README results block).

Usage:
    python -m scripts.run_external_identification                # both corpora
    python -m scripts.run_external_identification --corpus sapimouse
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.evaluation.run import bootstrap_ci
from pipeline.external.balabit import BalabitAdapter
from pipeline.external.base import split_on_idle
from pipeline.external.sapimouse import SapiMouseAdapter
from pipeline.features.run import MOUSE_ID_FEATURE_COLS, process_session_windows
from pipeline.ingestion.run import parse_events
from pipeline.verification import eer, far_at_frr, verification_scores

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
BALABIT_DIR = ROOT / "data" / "external" / "balabit"
SAPIMOUSE_DIR = ROOT / "data" / "external" / "sapimouse"
OUT_JSON = ROOT / "reports" / "external_identification.json"

SEED = 42
LGBM_PARAMS = dict(
    num_leaves=31,
    n_estimators=200,
    learning_rate=0.1,
    min_child_samples=3,  # 120-class runs have ~6 train windows per class
    subsample=1.0,
    colsample_bytree=1.0,
)


# --- feature extraction ------------------------------------------------------


def session_windows(session: dict) -> pd.DataFrame:
    """One recorder-schema session → window-feature rows (mouse-only path).

    Splits at idle gaps first; norm_factor / rate_norm are 1.0 (the corpora
    carry no sens/DPI/polling metadata; adapter defaults make the factor 1.0).
    """
    events = parse_events(session)
    rows = []
    for seg_idx, seg in enumerate(split_on_idle(events)):
        for w in process_session_windows(seg, norm_factor=1.0, rate_norm=1.0):
            w["player"] = session["player"]
            w["session_id"] = session["session_id"]
            w["segment_idx"] = seg_idx
            rows.append(w)
    return pd.DataFrame(rows)


def windows_frame(sessions) -> pd.DataFrame:
    frames = [session_windows(s) for s in sessions]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- model -------------------------------------------------------------------


def fit_classifier(train_df: pd.DataFrame):
    """LabelEncoder + StandardScaler + LGBMClassifier on the mouse feature set."""
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y = le.fit_transform(train_df["player"])
    scaler = StandardScaler().set_output(transform="pandas")
    X = scaler.fit_transform(train_df[MOUSE_ID_FEATURE_COLS].fillna(0.0))
    model = LGBMClassifier(**LGBM_PARAMS, class_weight="balanced", verbose=-1)
    model.fit(X, y)
    return model, scaler, le


def predict_proba(model, scaler, df: pd.DataFrame) -> np.ndarray:
    X = scaler.transform(df[MOUSE_ID_FEATURE_COLS].fillna(0.0))
    return model.predict_proba(X)


# --- SapiMouse ---------------------------------------------------------------


def run_sapimouse(src: Path) -> dict:
    log.info("SapiMouse: extracting window features (240 sessions) ...")
    feats = windows_frame(SapiMouseAdapter(src).iter_sessions())
    feats["protocol"] = np.where(
        feats["session_id"].str.endswith("1min"), "1min", "3min"
    )
    train_all = feats[feats["protocol"] == "3min"]
    test_all = feats[feats["protocol"] == "1min"]
    users = sorted(feats["player"].unique())
    log.info(
        "SapiMouse: %d users, %d train / %d test windows",
        len(users),
        len(train_all),
        len(test_all),
    )

    rng = np.random.default_rng(SEED)
    curve = []
    for n_users in (3, 10, 30, 60, len(users)):
        n_users = min(n_users, len(users))
        n_draws = 5 if n_users < len(users) else 1
        accs = []
        for _ in range(n_draws):
            chosen = rng.choice(users, size=n_users, replace=False)
            tr = train_all[train_all["player"].isin(chosen)]
            te = test_all[test_all["player"].isin(chosen)]
            model, scaler, le = fit_classifier(tr)
            te = te[te["player"].isin(le.classes_)]
            proba = predict_proba(model, scaler, te)
            y_true = le.transform(te["player"])
            accs.append(float((proba.argmax(axis=1) == y_true).mean()))
        entry = {
            "n_users": int(n_users),
            "n_draws": n_draws,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_min": float(np.min(accs)),
            "accuracy_max": float(np.max(accs)),
            "chance": 1.0 / n_users,
        }
        if n_draws == 1:
            y_pred = proba.argmax(axis=1)
            entry["accuracy_ci95"] = list(
                bootstrap_ci(y_true, y_pred, lambda t, p: float((t == p).mean()))
            )
            entry["n_test_windows"] = int(len(te))
            # verification at full scale: every (window, user) pair is a trial
            genuine, impostor = verification_scores(proba, y_true)
            window_eer, _ = eer(genuine, impostor)
            entry["window_eer"] = float(window_eer)
        log.info("SapiMouse n_users=%d: %s", n_users, entry)
        curve.append(entry)

    # open-set: enrol half the users, the other half are unknown impostors
    enrolled = rng.choice(users, size=len(users) // 2, replace=False)
    unknown = [u for u in users if u not in set(enrolled)]
    model, scaler, le = fit_classifier(train_all[train_all["player"].isin(enrolled)])
    te_known = test_all[test_all["player"].isin(enrolled)]
    te_unknown = test_all[test_all["player"].isin(unknown)]
    known_scores = predict_proba(model, scaler, te_known).max(axis=1)
    unknown_scores = predict_proba(model, scaler, te_unknown).max(axis=1)
    openset_eer, _ = eer(known_scores, unknown_scores)
    open_set = {
        "n_enrolled": int(len(enrolled)),
        "n_unknown": int(len(unknown)),
        "eer": float(openset_eer),
        "far_at_frr05": float(far_at_frr(known_scores, unknown_scores, 0.05)),
        "n_known_windows": int(len(te_known)),
        "n_unknown_windows": int(len(te_unknown)),
    }
    log.info("SapiMouse open-set: %s", open_set)

    return {
        "n_users": len(users),
        "protocol": "train on 3-min session, test on 1-min session (paper protocol)",
        "users_curve": curve,
        "open_set": open_set,
    }


# --- Balabit -----------------------------------------------------------------


def run_balabit(src: Path) -> dict:
    adapter = BalabitAdapter(src)
    log.info("Balabit: extracting training window features ...")
    feats = windows_frame(adapter.iter_sessions())
    users = sorted(feats["player"].unique())
    log.info("Balabit: %d users, %d train windows", len(users), len(feats))

    # closed-set: hold out the last 20% of each user's sessions (≥1)
    train_parts, test_parts = [], []
    for _, user_df in feats.groupby("player"):
        sessions = sorted(user_df["session_id"].unique())
        n_hold = max(1, round(0.2 * len(sessions)))
        held = set(sessions[-n_hold:])
        test_parts.append(user_df[user_df["session_id"].isin(held)])
        train_parts.append(user_df[~user_df["session_id"].isin(held)])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    model, scaler, le = fit_classifier(train_df)
    proba = predict_proba(model, scaler, test_df)
    y_true = le.transform(test_df["player"])
    y_pred = proba.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    closed_set = {
        "n_users": len(users),
        "accuracy": acc,
        "accuracy_ci95": list(
            bootstrap_ci(y_true, y_pred, lambda t, p: float((t == p).mean()))
        ),
        "chance": 1.0 / len(users),
        "n_train_windows": int(len(train_df)),
        "n_test_windows": int(len(test_df)),
    }
    log.info("Balabit closed-set: %s", closed_set)

    # verification: the challenge's labelled test sessions. Retrain on ALL
    # training windows (no holdout needed — test files are separate), score
    # each test session as mean P(claimed user | window), EER over sessions.
    model, scaler, le = fit_classifier(feats)
    genuine, impostor = [], []
    n_skipped = 0
    log.info("Balabit: scoring labelled test sessions ...")
    for session, claimed, is_impostor in adapter.iter_test_sessions():
        wins = session_windows(session)
        if wins.empty or claimed not in set(le.classes_):
            n_skipped += 1
            continue
        proba = predict_proba(model, scaler, wins)
        claimed_idx = int(np.where(le.classes_ == claimed)[0][0])
        score = float(proba[:, claimed_idx].mean())
        (impostor if is_impostor else genuine).append(score)
    session_eer, _ = eer(np.array(genuine), np.array(impostor))
    verification = {
        "task": "challenge protocol: is this test session really the claimed user?",
        "session_eer": float(session_eer),
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "n_skipped_no_windows": n_skipped,
    }
    log.info("Balabit verification: %s", verification)

    return {
        "n_users": len(users),
        "closed_set": closed_set,
        "verification": verification,
    }


# --- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", choices=["both", "balabit", "sapimouse"], default="both"
    )
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    result: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_set": f"MOUSE_ID_FEATURE_COLS ({len(MOUSE_ID_FEATURE_COLS)} mouse-only features)",
        "model": f"LightGBM {LGBM_PARAMS}",
        "seed": SEED,
    }
    if args.corpus in ("both", "sapimouse"):
        result["sapimouse"] = run_sapimouse(SAPIMOUSE_DIR)
    if args.corpus in ("both", "balabit"):
        result["balabit"] = run_balabit(BALABIT_DIR)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
