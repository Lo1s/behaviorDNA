"""
tests/test_benchmark_cs2cd.py
=============================
Unit tests for the CS2CD stream-recovery adapter (the novel bit of
scripts/benchmark_cs2cd_ae.py): the balanced file interleaves each player's
cheat- and clean-match by tick, so contiguous same-label streams are recovered
by grouping on (steamid, cheater_present) and splitting on tick gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.benchmark_cs2cd_ae import CHUNK, GAP, _streams_from_df

COLS = [
    "steamid",
    "tick",
    "cheater_present",
    "usercmd_mouse_dx",
    "usercmd_mouse_dy",
    "FIRE",
    "RIGHTCLICK",
]


def _df(rows):
    return pd.DataFrame(rows, columns=COLS)


def _player(sid, ticks, label):
    return [[sid, t, label, float(t % 5), float(t % 3), t % 2, 0] for t in ticks]


class TestStreamsFromDf:
    def test_recovers_one_stream_per_label_despite_tick_interleave(self):
        # Same player, legit + cheat share the tick range (interleaved by tick).
        rows = _player("P", range(100), 0) + _player("P", range(100), 1)
        streams = _streams_from_df(_df(rows))
        assert len(streams) == 2
        labels = sorted(lab for lab, _ in streams)
        assert labels == [0, 1]
        for _lab, arr in streams:
            assert arr.shape == (100, len(COLS) - 3)  # F features
            assert arr.dtype == np.float32

    def test_runs_shorter_than_chunk_are_dropped(self):
        rows = _player("P", range(CHUNK - 1), 0)  # one short legit run
        assert _streams_from_df(_df(rows)) == []

    def test_tick_gap_splits_into_separate_runs(self):
        # two contiguous blocks separated by a gap > GAP → two streams
        block_a = list(range(70))
        block_b = list(range(70 + GAP + 50, 70 + GAP + 50 + 70))
        rows = _player("P", block_a + block_b, 0)
        streams = _streams_from_df(_df(rows))
        assert len(streams) == 2
        assert all(lab == 0 and len(arr) == 70 for lab, arr in streams)

    def test_duplicate_ticks_are_dropped(self):
        rows = _player("P", list(range(100)) + list(range(50)), 0)  # 50 dup ticks
        streams = _streams_from_df(_df(rows))
        assert len(streams) == 1
        assert len(streams[0][1]) == 100  # dups collapsed
