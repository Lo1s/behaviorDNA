"""
tests/test_pretraining.py
=========================
Unit gate for Phase 8 self-supervised pretraining (CPU-fast).

Core logic (masking, the sampled→8-D encoding, the masked-denoising loop, the
weight-transfer contract) is exercised on tiny synthetic fixtures so the suite
runs without the big external corpora. The real-corpus adapters are smoke-
tested with a synthetic CS2CD parquet fixture, plus a skip-if-present check on
the real CaptchaSolve30k file when it happens to be cached locally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from pipeline.models.lstm_ae import LSTMAutoencoder
from pipeline.pretraining.corpora import (
    CAPTCHA_PARQUET,
    _sampled_stream_to_tensor,
    captcha_to_tensors,
    channel_summary_frame,
    cs2cd_to_tensors_8d,
)
from pipeline.pretraining.masking import MaskedDenoisingDataset, mask_chunk
from pipeline.pretraining.pretrain import pretrain_masked_ae
from pipeline.sequences.preprocessing import (
    COL_DT,
    COL_DX,
    COL_DY,
    COL_IS_MOUSE_CLICK_PRESS,
    COL_IS_MOUSE_MOVE,
    EVENT_FEATURE_DIM,
)


def _rand_tensors(n_sessions=4, length=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_sessions):
        t = np.zeros((length, EVENT_FEATURE_DIM), dtype=np.float32)
        t[:, COL_DX] = rng.normal(size=length)
        t[:, COL_DY] = rng.normal(size=length)
        t[:, COL_IS_MOUSE_MOVE] = 1.0
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class TestMasking:
    def test_mask_chunk_zeros_fraction_and_keeps_shape(self):
        chunk = np.ones((100, EVENT_FEATURE_DIM), dtype=np.float32)
        masked = mask_chunk(chunk, frac=0.2, rng=np.random.default_rng(0))
        zeroed = int((masked.sum(axis=1) == 0).sum())
        assert masked.shape == chunk.shape
        assert zeroed == 20
        # original is untouched (we masked a copy)
        assert chunk.sum() == 100 * EVENT_FEATURE_DIM

    def test_mask_chunk_masks_at_least_one(self):
        chunk = np.ones((3, EVENT_FEATURE_DIM), dtype=np.float32)
        masked = mask_chunk(chunk, frac=0.01, rng=np.random.default_rng(0))
        assert int((masked.sum(axis=1) == 0).sum()) >= 1

    def test_mask_frac_zero_is_identity(self):
        chunk = np.ones((10, EVENT_FEATURE_DIM), dtype=np.float32)
        masked = mask_chunk(chunk, frac=0.0, rng=np.random.default_rng(0))
        assert np.array_equal(masked, chunk)

    def test_dataset_returns_clean_target_and_masked_input(self):
        ds = MaskedDenoisingDataset(
            _rand_tensors(), chunk_length=64, stride=32, mask_frac=0.15
        )
        assert len(ds) > 0
        masked, clean = ds[0]
        assert masked.shape == (64, EVENT_FEATURE_DIM)
        assert clean.shape == (64, EVENT_FEATURE_DIM)
        # the masked input must differ from (and be "smaller" than) the clean target
        assert not torch.equal(masked, clean)
        assert int((masked.abs().sum(dim=1) == 0).sum()) >= 1

    def test_dataset_masking_is_deterministic(self):
        tensors = _rand_tensors()
        a = MaskedDenoisingDataset(tensors, mask_frac=0.15, seed=7)[3][0]
        b = MaskedDenoisingDataset(tensors, mask_frac=0.15, seed=7)[3][0]
        assert torch.equal(a, b)

    def test_dataset_rejects_wrong_feature_dim(self):
        bad = [np.zeros((100, 4), dtype=np.float32)]
        with pytest.raises(ValueError):
            MaskedDenoisingDataset(bad)


# ---------------------------------------------------------------------------
# Sampled-stream → 8-D encoding
# ---------------------------------------------------------------------------


class TestSampledEncoding:
    def test_deltas_dt_move_and_click_rising_edge(self):
        x = np.array([0.0, 1.0, 3.0, 3.0], dtype=np.float32)
        y = np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32)
        down = np.array([False, True, True, False])  # rising edge at index 1 only
        t = _sampled_stream_to_tensor(x, y, down, ms_per_tick=10.0)

        assert t.shape == (4, EVENT_FEATURE_DIM)
        # dx = first difference, dx[0] = 0
        np.testing.assert_allclose(t[:, COL_DX], [0.0, 1.0, 2.0, 0.0], atol=1e-6)
        # dt: 0 on first row, constant log1p(10) after
        assert t[0, COL_DT] == 0.0
        np.testing.assert_allclose(t[1:, COL_DT], np.log1p(10.0), atol=1e-6)
        # every tick is a movement sample
        assert np.all(t[:, COL_IS_MOUSE_MOVE] == 1.0)
        # click fires only on the rising edge (index 1), not while held (index 2)
        np.testing.assert_array_equal(
            t[:, COL_IS_MOUSE_CLICK_PRESS], [0.0, 1.0, 0.0, 0.0]
        )

    def test_starts_pressed_counts_as_click(self):
        t = _sampled_stream_to_tensor(
            np.zeros(3, np.float32),
            np.zeros(3, np.float32),
            np.array([True, True, False]),
            ms_per_tick=5.0,
        )
        assert t[0, COL_IS_MOUSE_CLICK_PRESS] == 1.0

    def test_channel_summary_frame_columns(self):
        sf = channel_summary_frame(_rand_tensors(n_sessions=2, length=50))
        assert len(sf) == 100
        assert "dt" in sf.columns and "is_mouse_move" in sf.columns


# ---------------------------------------------------------------------------
# CS2CD 8-D adapter (synthetic parquet fixture)
# ---------------------------------------------------------------------------


def _write_cs2cd_fixture(path, n=120):
    rng = np.random.default_rng(0)
    rows = []
    for sid, label in [("111", 0), ("222", 1)]:
        for tick in range(n):
            rows.append(
                {
                    "tick": tick,
                    "steamid": sid,
                    "cheater_present": label,
                    "usercmd_mouse_dx": float(rng.normal()),
                    "usercmd_mouse_dy": float(rng.normal()),
                    "FIRE": float(tick % 10 == 0),  # periodic fire
                    "RIGHTCLICK": 0.0,
                }
            )
    pd.DataFrame(rows).to_parquet(path)


class TestCS2CDAdapter:
    def test_recovers_labelled_8d_streams(self, tmp_path):
        path = tmp_path / "cs2cd.parquet"
        _write_cs2cd_fixture(path)
        streams = cs2cd_to_tensors_8d(path)
        assert len(streams) == 2
        labels = sorted(lab for lab, _ in streams)
        assert labels == [0, 1]
        for _, t in streams:
            assert t.shape[1] == EVENT_FEATURE_DIM
            assert t.shape[0] == 120
            assert np.all(t[:, COL_IS_MOUSE_MOVE] == 1.0)
            # FIRE rising edges → at least one click press
            assert t[:, COL_IS_MOUSE_CLICK_PRESS].sum() >= 1


# ---------------------------------------------------------------------------
# Pretraining loop + transfer contract
# ---------------------------------------------------------------------------


class TestPretrainLoop:
    def test_one_cpu_step_finite_loss(self):
        ds = MaskedDenoisingDataset(_rand_tensors(), chunk_length=64, stride=32)
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        model, history = pretrain_masked_ae(loader, None, epochs=1, device="cpu")
        assert len(history.train_loss) == 1
        assert np.isfinite(history.train_loss[0])

    def test_pretrained_weights_load_into_fresh_autoencoder(self):
        ds = MaskedDenoisingDataset(_rand_tensors(), chunk_length=64, stride=32)
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        model, _ = pretrain_masked_ae(
            loader,
            None,
            epochs=1,
            device="cpu",
            hidden_dim=64,
            bottleneck_dim=16,
            num_layers=2,
        )
        fresh = LSTMAutoencoder(hidden_dim=64, bottleneck_dim=16, num_layers=2)
        # the transfer contract the whole phase rests on: state_dict drops in
        fresh.load_state_dict(model.state_dict())


# ---------------------------------------------------------------------------
# Real CaptchaSolve30k adapter — smoke when the cached parquet is present
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not CAPTCHA_PARQUET.exists(), reason="CaptchaSolve30k parquet not cached locally"
)
def test_captcha_adapter_smoke():
    tensors = captcha_to_tensors(max_sessions=5, seed=42)
    assert tensors
    for t in tensors:
        assert t.shape[1] == EVENT_FEATURE_DIM
        assert np.all(t[:, COL_IS_MOUSE_MOVE] == 1.0)
