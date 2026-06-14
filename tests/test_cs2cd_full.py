"""
tests/test_cs2cd_full.py
========================
Unit gate for the Phase 8.1 full-release CS2CD ingest (CPU-fast, no network).

Everything is exercised on tiny synthetic match parquets shaped like the real
full release — the 6 projection columns (``tick, steamid, usercmd_mouse_dx,
usercmd_mouse_dy, FIRE, RIGHTCLICK``) plus a decoy column, and crucially **no**
``cheater_present`` column (the label is the subdir). Confirms the encode→shard
round-trip, gap-splitting, the lazy LRU dataset's contract (matches
``MaskedDenoisingDataset``), match-disjoint manifest splits, and the S2 dt hook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from pipeline.models.lstm_ae import LSTMAutoencoder
from pipeline.pretraining.corpora import CS2_MS_PER_TICK
from pipeline.pretraining.cs2cd_full import (
    MIN_TICKS,
    PROJECT_COLS,
    CS2CDShardChunkDataset,
    ShardGroupedSampler,
    _apply_dt_override,
    _load_shard,
    _read_run_lengths,
    build_manifest,
    build_shard_cache,
    encode_match_to_shard,
    fit_shard_normalizer,
)
from pipeline.pretraining.pretrain import pretrain_masked_ae
from pipeline.sequences.preprocessing import (
    COL_DT,
    COL_IS_MOUSE_CLICK_PRESS,
    COL_IS_MOUSE_MOVE,
    EVENT_FEATURE_DIM,
)


def _write_match(path, *, tick_spans, seed=0, decoy_col=True):
    """Write a synthetic full-release-style match parquet (no cheater_present)."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid, ticks in tick_spans.items():
        for tk in ticks:
            rows.append(
                {
                    "tick": int(tk),
                    "steamid": sid,
                    "usercmd_mouse_dx": float(rng.normal()),
                    "usercmd_mouse_dy": float(rng.normal()),
                    "FIRE": bool(rng.random() < 0.15),
                    "RIGHTCLICK": bool(rng.random() < 0.05),
                }
            )
    df = pd.DataFrame(rows)
    if decoy_col:
        df["kills_total"] = np.arange(len(df))  # extra col → projection must ignore it
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _legit_match(tmp_path, mid, **kw):
    p = tmp_path / "raw" / "no_cheater_present" / f"{mid}.parquet"
    _write_match(p, **kw)
    return p


# ---------------------------------------------------------------------------
# projection + encode round-trip
# ---------------------------------------------------------------------------
def test_projection_columns_present_in_fixture(tmp_path):
    df = _write_match(tmp_path / "m.parquet", tick_spans={"Player_1": range(80)})
    assert set(PROJECT_COLS) <= set(df.columns)
    assert "cheater_present" not in df.columns  # the full-release reality


def test_encode_round_trip(tmp_path):
    p = _legit_match(
        tmp_path, 0, tick_spans={"Player_1": range(200), "Player_2": range(200)}
    )
    cache = tmp_path / "cache"
    shard_path = encode_match_to_shard(p, label=0, cache_dir=cache, source="s1")

    assert shard_path.exists()
    assert shard_path.with_suffix("").with_suffix(".idx.json").exists()
    assert shard_path.parts[-3:] == ("s1", "no_cheater_present", "0.pt")

    shard = _load_shard(shard_path)
    assert shard["label"] == 0 and shard["source"] == "s1"
    assert len(shard["runs"]) == 2
    for run in shard["runs"]:
        t = run["tensor"].numpy()
        assert t.shape == (200, EVENT_FEATURE_DIM)
        assert np.allclose(t[:, COL_IS_MOUSE_MOVE], 1.0)  # sampled-move channel
        assert np.allclose(t[1:, COL_DT], np.log1p(CS2_MS_PER_TICK))  # native S1 dt
        assert t[0, COL_DT] == 0.0
        assert set(np.unique(t[:, COL_IS_MOUSE_CLICK_PRESS])) <= {0.0, 1.0}

    assert _read_run_lengths(shard_path) == [200, 200]


def test_encode_splits_on_tick_gaps(tmp_path):
    # Player_1 has a >2-tick gap → two runs; Player_2 is contiguous → one run.
    p = _legit_match(
        tmp_path,
        7,
        tick_spans={
            "Player_1": list(range(100)) + list(range(300, 400)),
            "Player_2": list(range(200)),
        },
    )
    shard = _load_shard(
        encode_match_to_shard(p, label=0, cache_dir=tmp_path / "c", source="s1")
    )
    lengths = sorted(t["tensor"].shape[0] for t in shard["runs"])
    assert lengths == [100, 100, 200]


def test_encode_drops_runs_below_min_ticks(tmp_path):
    p = _legit_match(tmp_path, 1, tick_spans={"Player_1": range(MIN_TICKS - 1)})
    shard = _load_shard(
        encode_match_to_shard(p, label=0, cache_dir=tmp_path / "c", source="s1")
    )
    assert shard["runs"] == []


def test_encode_idempotent(tmp_path):
    p = _legit_match(tmp_path, 2, tick_spans={"Player_1": range(80)})
    cache = tmp_path / "c"
    a = encode_match_to_shard(p, label=0, cache_dir=cache, source="s1")
    mtime = a.stat().st_mtime_ns
    b = encode_match_to_shard(p, label=0, cache_dir=cache, source="s1")  # no overwrite
    assert a == b and b.stat().st_mtime_ns == mtime


# ---------------------------------------------------------------------------
# lazy dataset
# ---------------------------------------------------------------------------
def _build_two_shard_ds(tmp_path, **ds_kw):
    cache = tmp_path / "c"
    shards = build_shard_cache(
        [
            _legit_match(
                tmp_path, 0, tick_spans={"Player_1": range(200), "Player_2": range(200)}
            ),
            _legit_match(
                tmp_path, 1, tick_spans={"Player_1": range(200), "Player_2": range(200)}
            ),
        ],
        label=0,
        cache_dir=cache,
        log_every=0,
    )
    # fit stats with the same dt transform the dataset uses (matches the real pipeline)
    stats = fit_shard_normalizer(shards, dt_override_ms=ds_kw.get("dt_override_ms"))
    return CS2CDShardChunkDataset(shards, stats=stats, **ds_kw), shards


def test_dataset_len_matches_chunk_count(tmp_path):
    ds, _ = _build_two_shard_ds(tmp_path, chunk_length=64, stride=32)
    # _chunk_indices(200, 64, 32) → [0,32,64,96,128] = 5 chunks; 4 runs → 20.
    assert len(ds) == 20


def test_dataset_pair_shapes_and_determinism(tmp_path):
    ds, _ = _build_two_shard_ds(
        tmp_path, chunk_length=64, stride=32, mask_frac=0.15, seed=3
    )
    masked, clean = ds[0]
    assert masked.shape == (64, EVENT_FEATURE_DIM)
    assert clean.shape == (64, EVENT_FEATURE_DIM)
    # masking is deterministic per index, and the masked input differs from clean.
    masked2, clean2 = ds[0]
    assert np.array_equal(masked.numpy(), masked2.numpy())
    assert np.array_equal(clean.numpy(), clean2.numpy())
    assert not np.array_equal(masked.numpy(), clean.numpy())
    assert np.isfinite(clean.numpy()).all()


def test_dataset_lru_bound(tmp_path):
    cache = tmp_path / "c"
    shards = build_shard_cache(
        [
            _legit_match(tmp_path, i, tick_spans={"Player_1": range(120)})
            for i in range(4)
        ],
        label=0,
        cache_dir=cache,
        log_every=0,
    )
    ds = CS2CDShardChunkDataset(
        shards,
        stats=fit_shard_normalizer(shards),
        chunk_length=64,
        stride=32,
        lru_shards=2,
    )
    for i in range(len(ds)):  # touch every shard
        _ = ds[i]
    assert len(ds._lru) <= 2


# ---------------------------------------------------------------------------
# manifest (PLAYER_THIN: match-disjoint)
# ---------------------------------------------------------------------------
def test_manifest_match_disjoint_and_nested_subsets(tmp_path):
    raw = tmp_path / "raw"
    for i in range(10):
        (raw / "no_cheater_present").mkdir(parents=True, exist_ok=True)
        (raw / "no_cheater_present" / f"{i}.parquet").touch()
    for i in range(3):
        (raw / "with_cheater_present").mkdir(parents=True, exist_ok=True)
        (raw / "with_cheater_present" / f"{i}.parquet").touch()

    m = build_manifest(
        raw_dir=raw, diversity_points=(2, 5), out=tmp_path / "manifest.json"
    )
    assert m["branch"] == "THIN" and m["split_unit"] == "match"
    assert m["n_legit_matches"] == 10 and m["n_cheat_matches"] == 3
    tr, va, ho = (
        set(m["pretrain_matches"]),
        set(m["val_matches"]),
        set(m["heldout_matches"]),
    )
    assert tr.isdisjoint(va) and tr.isdisjoint(ho) and va.isdisjoint(ho)
    assert tr | va | ho == {str(i) for i in range(10)}
    # nested volume subsets, all drawn from the pretrain pool
    s2, s5 = set(m["diversity_subsets"]["2"]), set(m["diversity_subsets"]["5"])
    assert len(s2) == 2 and len(s5) == 5 and s2 <= s5 <= tr


# ---------------------------------------------------------------------------
# S2 dt hook
# ---------------------------------------------------------------------------
def test_apply_dt_override_sets_constant_dt():
    t = np.zeros((10, EVENT_FEATURE_DIM), dtype=np.float32)
    t[1:, COL_DT] = np.log1p(CS2_MS_PER_TICK)
    out = _apply_dt_override(t, 8.0)
    assert np.allclose(out[1:, COL_DT], np.log1p(8.0))
    assert out[0, COL_DT] == 0.0
    assert not np.shares_memory(out, t)  # copy, not in-place


# ---------------------------------------------------------------------------
# shard-grouped sampler + end-to-end pretrain integration
# ---------------------------------------------------------------------------
def test_shard_grouped_sampler_is_permutation(tmp_path):
    ds, _ = _build_two_shard_ds(tmp_path, chunk_length=64, stride=32)
    sampler = ShardGroupedSampler(ds, shuffle=True, seed=0)
    order = list(sampler)
    assert sorted(order) == list(range(len(ds)))  # a true permutation
    assert len(sampler) == len(ds)
    # consecutive epochs reshuffle
    assert list(ShardGroupedSampler(ds, shuffle=True, seed=0)) != list(range(len(ds)))


def test_shard_dataset_pretrains_one_epoch(tmp_path):
    ds, _ = _build_two_shard_ds(
        tmp_path, chunk_length=64, stride=32, mask_frac=0.15, seed=1
    )
    loader = DataLoader(ds, batch_size=16, sampler=ShardGroupedSampler(ds, seed=1))
    model, history = pretrain_masked_ae(
        loader,
        loader,
        hidden_dim=16,
        bottleneck_dim=8,
        num_layers=1,
        epochs=1,
        device="cpu",
        log_every=1,
    )
    assert np.isfinite(history.best_val_loss)
    # transfer contract: pretrained weights load into a fresh same-shape AE.
    fresh = LSTMAutoencoder(hidden_dim=16, bottleneck_dim=8, num_layers=1)
    fresh.load_state_dict(model.state_dict())


def test_dt_neutralized_source_zeros_dt_channel(tmp_path):
    # s2 (dt_override_ms=0.0) → the dt channel is identically zero pre-normalisation.
    ds, shards = _build_two_shard_ds(
        tmp_path, chunk_length=64, stride=32, dt_override_ms=0.0
    )
    runs = ds._decoded(shards[0])  # normalised runs
    # dt was constant (native) → after neutralise+zscore it collapses to ~0 variance.
    assert np.allclose(np.asarray([r[:, COL_DT] for r in runs]), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# real-release smoke (only if the download is present)
# ---------------------------------------------------------------------------
def test_real_release_smoke():
    from pipeline.pretraining.cs2cd_full import local_matches

    legit = local_matches("no_cheater_present")
    if not legit:
        pytest.skip("full CS2CD release not downloaded")
    shard = _load_shard(encode_match_to_shard(legit[0], label=0))
    assert shard["runs"], "expected at least one stream in a real match"
    assert shard["runs"][0]["tensor"].shape[1] == EVENT_FEATURE_DIM
