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
    def test_cheat_sessions_excluded_from_identification(self):
        # 3 players × 4 legit sessions, plus one extra cheat session for player0.
        df = make_features_df(n_players=3, n_sessions_per_player=4)
        cheat = make_features_df(n_players=1, n_sessions_per_player=1).assign(
            session_id="p0_cheat", player="player0"
        )
        df["is_cheat_session"] = False
        cheat["is_cheat_session"] = True
        combined = pd.concat([df, cheat], ignore_index=True)

        train, val, test = split(
            combined,
            test_size=0.25,
            val_size=0.25,
            random_seed=42,
            min_sessions_per_player=1,
        )
        all_sids = (
            set(train["session_id"]) | set(val["session_id"]) | set(test["session_id"])
        )
        assert "p0_cheat" not in all_sids
        for fold in (train, val, test):
            assert not fold["is_cheat_session"].any()

    def test_missing_cheat_column_is_backward_compatible(self):
        # Frames without the column (old features.parquet) split unchanged.
        df = make_features_df()  # no is_cheat_session column
        assert "is_cheat_session" not in df.columns
        train, val, test = split(
            df, test_size=0.15, val_size=0.15, random_seed=42, min_sessions_per_player=1
        )
        assert len(train) + len(val) + len(test) == len(df)

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
        # 4 players × 5 sessions × 3 windows = 60 total windows.
        # Per-player holdout at 5 sessions → 3 train / 1 val / 1 test each,
        # so train is genuinely the largest fold.
        df = make_features_df(n_players=4, n_sessions_per_player=5, n_windows=3)
        train, val, test = split(
            df, test_size=0.15, val_size=0.15, random_seed=42, min_sessions_per_player=1
        )
        total = len(train) + len(val) + len(test)
        assert total == len(df)
        # train should be the largest fold
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_every_player_in_every_fold(self):
        # With ≥3 sessions per player, every retained player must appear in
        # all three folds — the property the player-stratified split buys us.
        df = make_features_df(n_players=3, n_sessions_per_player=5, n_windows=3)
        train, val, test = split(
            df, test_size=0.15, val_size=0.15, random_seed=42, min_sessions_per_player=3
        )
        retained = set(df["player"].unique())
        assert set(train["player"]) == retained
        assert set(val["player"]) == retained
        assert set(test["player"]) == retained
