"""
pipeline/features/split.py
===========================
Stage 3 — Split: features.parquet → stratified train / val / test parquets.

Splits are **player-stratified at the session level**: every retained player's
sessions are partitioned into train/val/test independently, so all windows from
one session stay in one fold (no leakage) AND every player is represented in
every non-empty fold. This matters most with few sessions — a purely random
session split can drop a whole identity out of the test set, making
identification metrics meaningless. Players with fewer than
min_sessions_per_player sessions are excluded before splitting.

With very few sessions (e.g. current single-session data) all players may fall below
the threshold — in that case empty but schema-valid parquets are written and a
warning is logged. This keeps the DVC pipeline green while more data is collected.

Output:
  data/splits/train.parquet
  data/splits/val.parquet
  data/splits/test.parquet

Run via DVC:
    dvc repro split

Or directly:
    python -m pipeline.features.split
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
FEATURES_IN = ROOT / "data" / "processed" / "features.parquet"
CONFIG_IN = ROOT / "configs" / "training.yaml"
SPLITS_DIR = ROOT / "data" / "splits"


def split(
    features_df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_seed: int,
    min_sessions_per_player: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) with no session_id crossing splits.

    All windows from a given session_id stay in a single fold, and each retained
    player's sessions are split independently so every player appears in every
    non-empty fold. Players with fewer than min_sessions_per_player sessions are
    excluded first. Cheat sessions (``is_cheat_session``) are excluded entirely —
    identification fingerprints players from legit play, and cheating partially
    erases the biometric (see notebooks/17).
    """
    # Drop cheat sessions from the identification set (legit play only).
    # Guarded for backward compatibility with feature frames lacking the column.
    if "is_cheat_session" in features_df.columns:
        cheat_mask = features_df["is_cheat_session"].fillna(False).astype(bool)
        if cheat_mask.any():
            log.info(
                "Excluding %d cheat session(s) / %d window(s) from identification",
                features_df.loc[cheat_mask, "session_id"].nunique(),
                int(cheat_mask.sum()),
            )
            features_df = features_df[~cheat_mask]

    # Filter under-represented players
    session_player = features_df.drop_duplicates("session_id").set_index("session_id")[
        "player"
    ]
    session_counts = session_player.value_counts()
    valid_players = session_counts[session_counts >= min_sessions_per_player].index

    dropped = set(session_counts.index) - set(valid_players)
    if dropped:
        log.warning(
            "Dropping %d player(s) with < %d sessions: %s",
            len(dropped),
            min_sessions_per_player,
            sorted(dropped),
        )

    df = features_df[features_df["player"].isin(valid_players)].copy()

    empty = features_df.iloc[0:0].copy()

    if df.empty:
        log.warning(
            "No players meet min_sessions_per_player=%d — writing empty splits.",
            min_sessions_per_player,
        )
        return empty, empty, empty

    # Per-player whole-session holdout: for each player, shuffle their sessions
    # deterministically and peel off test, then val, leaving the rest for train.
    # This guarantees every retained player is present in each non-empty fold.
    rng = np.random.default_rng(random_seed)
    test_sids: list = []
    val_sids: list = []
    train_sids: list = []

    for player in sorted(valid_players):
        sids = df.loc[df["player"] == player, "session_id"].unique().tolist()
        rng.shuffle(sids)
        n = len(sids)

        # At least one session per fold once enough sessions exist; a retained
        # 1- or 2-session player (only when min_sessions is lowered) degrades
        # gracefully to train-only / train+test rather than erroring.
        n_test = max(1, round(test_size * n)) if n >= 2 else 0
        n_val = max(1, round(val_size * n)) if n >= 3 else 0
        # Never starve train: keep at least one session out of test+val.
        while n_test + n_val >= n and (n_test + n_val) > 0:
            if n_val > 0:
                n_val -= 1
            else:
                n_test -= 1

        test_sids.extend(sids[:n_test])
        val_sids.extend(sids[n_test : n_test + n_val])
        train_sids.extend(sids[n_test + n_val :])

    df_train = df[df["session_id"].isin(train_sids)].reset_index(drop=True)
    df_val = df[df["session_id"].isin(val_sids)].reset_index(drop=True)
    df_test = df[df["session_id"].isin(test_sids)].reset_index(drop=True)

    # Verify no session crosses split boundaries
    assert not (
        set(df_train["session_id"]) & set(df_test["session_id"])
    ), "Session leakage: train ∩ test non-empty"
    assert not (
        set(df_val["session_id"]) & set(df_test["session_id"])
    ), "Session leakage: val ∩ test non-empty"
    assert not (
        set(df_train["session_id"]) & set(df_val["session_id"])
    ), "Session leakage: train ∩ val non-empty"

    return df_train, df_val, df_test


def run() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_parquet(FEATURES_IN)
    log.info(
        "Loaded features: %d rows, %d sessions, %d players",
        len(features_df),
        features_df["session_id"].nunique(),
        features_df["player"].nunique(),
    )

    with open(CONFIG_IN) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    train_df, val_df, test_df = split(
        features_df,
        test_size=data_cfg["test_size"],
        val_size=data_cfg["val_size"],
        random_seed=data_cfg["random_seed"],
        min_sessions_per_player=data_cfg["min_sessions_per_player"],
    )

    train_df.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    val_df.to_parquet(SPLITS_DIR / "val.parquet", index=False)
    test_df.to_parquet(SPLITS_DIR / "test.parquet", index=False)

    log.info(
        "Split complete: train=%d  val=%d  test=%d  (windows)",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    log.info(
        "  Players in train: %s",
        sorted(train_df["player"].unique()) if len(train_df) else [],
    )


if __name__ == "__main__":
    run()
