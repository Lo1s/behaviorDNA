"""
tests/test_split.py
===================
Unit tests for pipeline/features/split.py
"""

import pandas as pd

from pipeline.features.run import FEATURE_COLS
from pipeline.features.split import split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_features_df(n_players=4, n_sessions_per_player=3, n_windows=3) -> pd.DataFrame:
    rows = []
    for p in range(n_players):
        for s in range(n_sessions_per_player):
            sid = f"p{p}_s{s}"
            for w in range(n_windows):
                row = {
                    "session_id": sid,
                    "window_idx": w,
                    "player": f"player{p}",
                    "game": "cs2",
                    "sensitivity": 1.0,
                    "dpi": 800,
                    "recorded_at": pd.Timestamp("2026-01-01", tz="UTC"),
                    "duration_ms": 90_000.0,
                }
                row.update({c: float(p + w) for c in FEATURE_COLS})
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestSplit
# ---------------------------------------------------------------------------


class TestSplit:
    def test_no_session_crosses_splits(self):
        df = make_features_df()
        train, val, test = split(
            df, test_size=0.15, val_size=0.15, random_seed=42, min_sessions_per_player=1
        )
        train_sids = set(train["session_id"])
        val_sids = set(val["session_id"])
        test_sids = set(test["session_id"])
        assert train_sids & test_sids == set()
        assert val_sids & test_sids == set()
        assert train_sids & val_sids == set()

    def test_empty_df_returns_three_empty(self):
        df = make_features_df()
        empty = df.iloc[0:0]
        train, val, test = split(
            empty,
            test_size=0.15,
            val_size=0.15,
            random_seed=42,
            min_sessions_per_player=1,
        )
        assert train.empty
        assert val.empty
        assert test.empty

    def test_player_filtered_when_below_min_sessions(self):
        # 4 players: 3 with 3 sessions, 1 with only 1 session
        df = make_features_df(n_players=3, n_sessions_per_player=3)
        lone = make_features_df(n_players=1, n_sessions_per_player=1)
        lone["player"] = "lone_player"
        lone["session_id"] = "lone_s0"
        combined = pd.concat([df, lone], ignore_index=True)

        train, val, test = split(
            combined,
            test_size=0.15,
            val_size=0.15,
            random_seed=42,
            min_sessions_per_player=3,
        )
        all_players = set(train["player"]) | set(val["player"]) | set(test["player"])
        assert "lone_player" not in all_players

    def test_all_windows_from_session_stay_together(self):
        df = make_features_df()
        train, val, test = split(
            df, test_size=0.15, val_size=0.15, random_seed=42, min_sessions_per_player=1
        )
        for sid in df["session_id"].unique():
            in_train = sid in set(train["session_id"])
            in_val = sid in set(val["session_id"])
            in_test = sid in set(test["session_id"])
            assert (
                sum([in_train, in_val, in_test]) == 1
            ), f"session {sid} appears in multiple splits"

    def test_output_sizes_approximately_correct(self):
        # 4 players × 3 sessions × 3 windows = 36 total windows
        df = make_features_df(n_players=4, n_sessions_per_player=3, n_windows=3)
        train, val, test = split(
            df, test_size=0.15, val_size=0.15, random_seed=42, min_sessions_per_player=1
        )
        total = len(train) + len(val) + len(test)
        assert total == len(df)
        # train should be the largest fold
        assert len(train) > len(val)
        assert len(train) > len(test)
