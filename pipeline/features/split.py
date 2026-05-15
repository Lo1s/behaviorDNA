"""
pipeline/features/split.py
===========================
Stage 3 — Split: features.parquet → stratified train / val / test parquets.

Groups windows by session_id so all windows from one session stay in one fold,
preventing leakage. Players with fewer than min_sessions_per_player sessions are
excluded before splitting.

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

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

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

    All windows from a given session_id stay in a single fold. Players with
    fewer than min_sessions_per_player sessions are excluded first.
    """
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

    groups = df["session_id"].values

    # Stage 1: carve out test set
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    if df_train_val.empty:
        log.warning(
            "Not enough data to produce a val split — all non-test goes to train."
        )
        return df_train_val, empty, df_test

    # Stage 2: carve val from train_val (adjust fraction for reduced pool)
    val_fraction = val_size / (1.0 - test_size)
    groups_tv = df_train_val["session_id"].values
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=random_seed
    )
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=groups_tv))
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # Verify no session crosses split boundaries
    assert not (
        set(df_train["session_id"]) & set(df_test["session_id"])
    ), "Session leakage: train ∩ test non-empty"
    assert not (
        set(df_val["session_id"]) & set(df_test["session_id"])
    ), "Session leakage: val ∩ test non-empty"

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
