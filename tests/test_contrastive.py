"""
tests/test_contrastive.py
=========================
Unit gate for Phase 8.2 contrastive pretraining (CPU-fast).

Covers the new machinery on tiny synthetic fixtures so the suite runs without the
big external corpora or a GPU: the augmentations, the NT-Xent loss, the projection
head, the two-view datasets (in-memory + over a synthetic CS2CD shard), the
contrastive train loop + weight-transfer contract, and the frozen-embedding
evaluators. Also pins the behaviour-preserving ``_clean_window`` refactor in
``cs2cd_full``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from pipeline.models.lstm_ae import LSTMAutoencoder, load_lstm_ae, save_lstm_ae
from pipeline.pretraining.augment import (
    Augmenter,
    crop_resize,
    jitter,
    scale,
    time_mask,
)
from pipeline.pretraining.contrastive import (
    ContrastiveSequenceDataset,
    CS2CDContrastiveShardDataset,
    ProjectionHead,
    nt_xent_loss,
    pretrain_contrastive,
)
from pipeline.pretraining.cs2cd_full import (
    CS2CDShardChunkDataset,
    ShardGroupedSampler,
    encode_match_to_shard,
    fit_shard_normalizer,
)
from pipeline.pretraining.embed_eval import (
    embed_chunks,
    knn_scores,
    linear_probe_auc,
    mahalanobis_scores,
    ocsvm_scores,
    oneclass_auc,
)
from pipeline.sequences.preprocessing import (
    COL_DT,
    COL_DX,
    COL_DY,
    COL_IS_MOUSE_MOVE,
    EVENT_FEATURE_DIM,
)

L = 64


def _motion_chunk(length=L, seed=0):
    rng = np.random.default_rng(seed)
    c = np.zeros((length, EVENT_FEATURE_DIM), dtype=np.float32)
    c[:, COL_DT] = 0.5
    c[:, COL_DX] = rng.normal(size=length).astype(np.float32) + 1.0
    c[:, COL_DY] = rng.normal(size=length).astype(np.float32) + 2.0
    c[:, COL_IS_MOUSE_MOVE] = 1.0
    return c


def _rand_tensors(n=4, length=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = np.zeros((length, EVENT_FEATURE_DIM), dtype=np.float32)
        t[:, COL_DX] = rng.normal(size=length)
        t[:, COL_DY] = rng.normal(size=length)
        t[:, COL_IS_MOUSE_MOVE] = 1.0
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------
class TestAugment:
    def test_jitter_only_touches_motion_channels(self):
        c = _motion_chunk()
        out = jitter(c, np.random.default_rng(0), sigma=0.5)
        assert out.shape == c.shape
        assert not np.allclose(out[:, COL_DX], c[:, COL_DX])
        assert not np.allclose(out[:, COL_DY], c[:, COL_DY])
        # dt + one-hot channels untouched
        np.testing.assert_array_equal(out[:, COL_DT], c[:, COL_DT])
        np.testing.assert_array_equal(
            out[:, COL_IS_MOUSE_MOVE], c[:, COL_IS_MOUSE_MOVE]
        )

    def test_scale_is_single_factor_on_motion(self):
        c = _motion_chunk()
        out = scale(c, np.random.default_rng(1), low=2.0, high=2.0)  # deterministic ×2
        np.testing.assert_allclose(out[:, COL_DX], c[:, COL_DX] * 2.0, rtol=1e-5)
        np.testing.assert_allclose(out[:, COL_DY], c[:, COL_DY] * 2.0, rtol=1e-5)
        np.testing.assert_array_equal(out[:, COL_DT], c[:, COL_DT])

    def test_time_mask_zeros_at_least_one_step(self):
        c = _motion_chunk()
        out = time_mask(c, np.random.default_rng(0), frac=0.2)
        assert out.shape == c.shape
        assert int((out.sum(axis=1) == 0).sum()) >= 1

    def test_crop_resize_preserves_shape_and_constant_signal(self):
        c = np.ones((L, EVENT_FEATURE_DIM), dtype=np.float32) * 3.0
        out = crop_resize(c, np.random.default_rng(2), min_frac=0.5)
        assert out.shape == (L, EVENT_FEATURE_DIM)
        np.testing.assert_allclose(
            out, c, atol=1e-5
        )  # interp of a constant is constant

    def test_crop_resize_noop_when_too_short(self):
        c = _motion_chunk(length=3)
        out = crop_resize(c, np.random.default_rng(0))
        np.testing.assert_array_equal(out, c)

    def test_augmenter_shape_determinism_and_two_views_differ(self):
        c = _motion_chunk()
        aug = Augmenter()
        a = aug(c, np.random.default_rng(7))
        b = aug(c, np.random.default_rng(7))
        assert a.shape == (L, EVENT_FEATURE_DIM)
        np.testing.assert_array_equal(a, b)  # same rng seed → same view
        # one rng advanced across two calls → two different views
        rng = np.random.default_rng(7)
        v1, v2 = aug(c, rng), aug(c, rng)
        assert not np.array_equal(v1, v2)

    def test_augmenter_rejects_wrong_feature_dim(self):
        with pytest.raises(ValueError):
            Augmenter()(np.zeros((L, 4), np.float32), np.random.default_rng(0))


# ---------------------------------------------------------------------------
# NT-Xent loss + projection head
# ---------------------------------------------------------------------------
class TestNTXent:
    def test_finite_scalar(self):
        z1 = torch.randn(8, 16)
        z2 = torch.randn(8, 16)
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        assert loss.dim() == 0 and torch.isfinite(loss)

    def test_aligned_pairs_beat_misaligned(self):
        torch.manual_seed(0)
        z = torch.randn(16, 16)
        aligned = nt_xent_loss(z, z.clone(), temperature=0.5)
        mis = nt_xent_loss(z, z[torch.randperm(16)], temperature=0.5)
        assert aligned < mis

    def test_gradient_flows(self):
        z1 = torch.randn(8, 16, requires_grad=True)
        z2 = torch.randn(8, 16, requires_grad=True)
        nt_xent_loss(z1, z2).backward()
        assert z1.grad is not None and torch.isfinite(z1.grad).all()

    def test_shape_guard(self):
        with pytest.raises(ValueError):
            nt_xent_loss(torch.randn(4, 16), torch.randn(5, 16))

    def test_projection_head_shape(self):
        head = ProjectionHead(16, 64, 32)
        assert head(torch.randn(10, 16)).shape == (10, 32)


# ---------------------------------------------------------------------------
# Two-view datasets
# ---------------------------------------------------------------------------
class TestContrastiveDatasets:
    def test_in_memory_two_views_differ_and_deterministic(self):
        ds = ContrastiveSequenceDataset(
            _rand_tensors(), chunk_length=L, stride=L, seed=5
        )
        assert len(ds) > 0
        v1, v2 = ds[0]
        assert v1.shape == (L, EVENT_FEATURE_DIM) and v2.shape == (L, EVENT_FEATURE_DIM)
        assert not torch.equal(v1, v2)
        other = ContrastiveSequenceDataset(
            _rand_tensors(), chunk_length=L, stride=L, seed=5
        )
        assert torch.equal(
            ds[3][0], other[3][0]
        )  # deterministic per (seed, epoch, idx)

    def test_set_epoch_changes_views(self):
        ds = ContrastiveSequenceDataset(
            _rand_tensors(), chunk_length=L, stride=L, seed=5
        )
        before = ds[0][0].clone()
        ds.set_epoch(1)
        assert not torch.equal(before, ds[0][0])

    def test_rejects_wrong_feature_dim(self):
        with pytest.raises(ValueError):
            ContrastiveSequenceDataset([np.zeros((100, 4), np.float32)])


# ---------------------------------------------------------------------------
# CS2CD shard fixtures: contrastive dataset + _clean_window refactor
# ---------------------------------------------------------------------------
def _write_cs2cd_full_fixture(path, n=200, sids=("p1", "p2")):
    """Full-release-style parquet (NO cheater_present column — label is the subdir)."""
    rng = np.random.default_rng(0)
    rows = []
    for sid in sids:
        for tick in range(n):
            rows.append(
                {
                    "tick": tick,
                    "steamid": sid,
                    "usercmd_mouse_dx": float(rng.normal()),
                    "usercmd_mouse_dy": float(rng.normal()),
                    "FIRE": float(tick % 10 == 0),
                    "RIGHTCLICK": 0.0,
                }
            )
    pd.DataFrame(rows).to_parquet(path)


def _build_shard(tmp_path):
    match_dir = tmp_path / "no_cheater_present"
    match_dir.mkdir()
    pq = match_dir / "0.parquet"
    _write_cs2cd_full_fixture(pq)
    return encode_match_to_shard(
        pq, label=0, cache_dir=tmp_path / "_cache", source="s1"
    )


class TestShardContrastive:
    def test_clean_window_matches_masked_dataset_target(self, tmp_path):
        shard = _build_shard(tmp_path)
        stats = fit_shard_normalizer([shard])
        ds = CS2CDShardChunkDataset([shard], stats=stats, chunk_length=L, stride=L)
        # __getitem__ returns (masked, clean); _clean_window must equal that clean.
        for i in (0, len(ds) - 1):
            _masked, clean = ds[i]
            np.testing.assert_allclose(ds._clean_window(i), clean.numpy(), atol=1e-6)

    def test_contrastive_shard_two_views(self, tmp_path):
        shard = _build_shard(tmp_path)
        stats = fit_shard_normalizer([shard])
        ds = CS2CDContrastiveShardDataset(
            [shard], stats=stats, chunk_length=L, stride=L, augment=Augmenter(), seed=3
        )
        assert len(ds) > 0
        v1, v2 = ds[0]
        assert v1.shape == (L, EVENT_FEATURE_DIM)
        assert not torch.equal(v1, v2)

    def test_works_with_shard_grouped_sampler(self, tmp_path):
        shard = _build_shard(tmp_path)
        stats = fit_shard_normalizer([shard])
        ds = CS2CDContrastiveShardDataset(
            [shard], stats=stats, chunk_length=L, stride=L, augment=Augmenter()
        )
        sampler = ShardGroupedSampler(ds, shuffle=True, seed=0)
        loader = DataLoader(ds, batch_size=2, sampler=sampler)
        v1, v2 = next(iter(loader))
        assert v1.shape[1:] == (L, EVENT_FEATURE_DIM)
        assert v1.shape == v2.shape


# ---------------------------------------------------------------------------
# Train loop + transfer contract
# ---------------------------------------------------------------------------
class TestPretrainContrastive:
    def test_one_cpu_epoch_finite_loss(self):
        ds = ContrastiveSequenceDataset(_rand_tensors(), chunk_length=L, stride=32)
        loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
        backbone, head, history = pretrain_contrastive(
            loader,
            None,
            epochs=1,
            device="cpu",
            **{"hidden_dim": 64, "bottleneck_dim": 16, "num_layers": 2},
        )
        assert len(history.train_loss) == 1 and np.isfinite(history.train_loss[0])
        assert isinstance(head, ProjectionHead)

    def test_backbone_transfers_and_reloads(self, tmp_path):
        ds = ContrastiveSequenceDataset(_rand_tensors(), chunk_length=L, stride=32)
        loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
        backbone, _head, _h = pretrain_contrastive(
            loader,
            None,
            epochs=1,
            device="cpu",
            hidden_dim=64,
            bottleneck_dim=16,
            num_layers=2,
        )
        # transfer contract: state_dict drops into a fresh LSTMAutoencoder
        fresh = LSTMAutoencoder(hidden_dim=64, bottleneck_dim=16, num_layers=2)
        fresh.load_state_dict(backbone.state_dict())
        # and round-trips through the standard artifact format
        stats = {"mean": np.zeros(8, np.float32), "std": np.ones(8, np.float32)}
        save_lstm_ae(backbone, stats, tmp_path)
        model, _stats, _meta = load_lstm_ae(tmp_path, device="cpu")
        z = model.encode(torch.zeros(2, L, EVENT_FEATURE_DIM))
        assert z.shape == (2, 16)


# ---------------------------------------------------------------------------
# Frozen-embedding evaluation
# ---------------------------------------------------------------------------
class TestEmbedEval:
    def test_embed_chunks_shape(self):
        model = LSTMAutoencoder(hidden_dim=64, bottleneck_dim=16, num_layers=2)
        chunks = np.zeros((7, L, EVENT_FEATURE_DIM), dtype=np.float32)
        emb = embed_chunks(model, chunks, device="cpu")
        assert emb.shape == (7, 16)

    def test_oneclass_scorers_flag_outliers(self):
        rng = np.random.default_rng(0)
        train = rng.normal(size=(200, 16))
        inliers = rng.normal(size=(50, 16))
        outliers = rng.normal(loc=6.0, size=(50, 16))
        for scorer in (mahalanobis_scores, ocsvm_scores, knn_scores):
            s_in = scorer(train, inliers)
            s_out = scorer(train, outliers)
            assert s_out.mean() > s_in.mean()

    def test_oneclass_auc_separable(self):
        rng = np.random.default_rng(1)
        train = rng.normal(size=(200, 16))
        legit = rng.normal(size=(60, 16))
        cheat = rng.normal(loc=6.0, size=(60, 16))
        auc = oneclass_auc(train, legit, cheat, mahalanobis_scores)
        assert auc > 0.9

    def test_oneclass_auc_nan_on_empty(self):
        train = np.zeros((10, 16))
        assert np.isnan(
            oneclass_auc(
                train, np.zeros((0, 16)), np.zeros((3, 16)), mahalanobis_scores
            )
        )

    def test_linear_probe_separable_vs_random(self):
        rng = np.random.default_rng(2)
        legit = rng.normal(size=(80, 16))
        cheat = rng.normal(loc=4.0, size=(80, 16))
        emb = np.r_[legit, cheat]
        y = np.r_[np.zeros(80), np.ones(80)]
        assert linear_probe_auc(emb, y, seed=0) > 0.9
        # shuffled labels → no signal
        y_rand = rng.permutation(y)
        assert 0.3 < linear_probe_auc(emb, y_rand, seed=0) < 0.7
