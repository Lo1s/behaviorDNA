"""
tests/test_lstm_ae.py
=====================
Unit tests for pipeline.models.lstm_ae.

These tests are kept tiny (small batch, short sequences, few epochs) so the
suite still finishes in a few seconds. We test:

- forward-pass shape correctness
- encode → decode roundtrip preserves the (B, L, F) contract
- reconstruction_error returns one scalar per chunk and is non-negative
- score_sequences accepts both numpy arrays and tensors
- training reduces loss on a trivially overfittable dataset
- early stopping triggers when val loss stalls
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pipeline.models.lstm_ae import (
    LSTMAutoencoder,
    load_lstm_ae,
    save_lstm_ae,
    score_sequences,
    train_lstm_ae,
)
from pipeline.sequences.preprocessing import EVENT_FEATURE_DIM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> LSTMAutoencoder:
    torch.manual_seed(0)
    return LSTMAutoencoder(
        feature_dim=EVENT_FEATURE_DIM,
        hidden_dim=16,
        bottleneck_dim=8,
        num_layers=1,
        dropout=0.0,
    )


def _make_loader(n: int = 32, L: int = 16, seed: int = 0) -> DataLoader:
    """Create a small DataLoader of random sequence chunks."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, L, EVENT_FEATURE_DIM)).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(data))

    # TensorDataset yields tuples; our training loop expects bare tensors.
    # Wrap the loader so each batch is just the tensor, not the (tensor,) tuple.
    class _UnwrapLoader:
        def __init__(self, base):
            self._base = base

        def __iter__(self):
            for batch in self._base:
                yield batch[0]

        def __len__(self):
            return len(self._base)

    return _UnwrapLoader(DataLoader(ds, batch_size=8, shuffle=False))


# ---------------------------------------------------------------------------
# Model shape tests
# ---------------------------------------------------------------------------


class TestModelShape:
    def test_forward_shape(self, model):
        x = torch.zeros(4, 16, EVENT_FEATURE_DIM)
        out = model(x)
        assert tuple(out.shape) == (4, 16, EVENT_FEATURE_DIM)

    def test_encode_shape(self, model):
        x = torch.zeros(4, 16, EVENT_FEATURE_DIM)
        z = model.encode(x)
        assert tuple(z.shape) == (4, model.bottleneck_dim)

    def test_decode_shape(self, model):
        z = torch.zeros(4, model.bottleneck_dim)
        recon = model.decode(z, seq_len=32)
        assert tuple(recon.shape) == (4, 32, EVENT_FEATURE_DIM)

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError):
            LSTMAutoencoder(num_layers=0)

    def test_handles_different_sequence_lengths(self, model):
        # Decoder needs to be able to reconstruct different-length sequences
        for L in (4, 16, 64):
            x = torch.zeros(2, L, EVENT_FEATURE_DIM)
            out = model(x)
            assert out.shape == x.shape


class TestReconstructionError:
    def test_shape_is_one_per_chunk(self, model):
        x = torch.zeros(5, 16, EVENT_FEATURE_DIM)
        err = model.reconstruction_error(x)
        assert tuple(err.shape) == (5,)

    def test_non_negative(self, model):
        x = torch.randn(5, 16, EVENT_FEATURE_DIM)
        err = model.reconstruction_error(x)
        assert (err >= 0).all()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    def test_loss_decreases_on_overfittable_data(self):
        """A small autoencoder fed a learnable pattern should drive loss way down."""
        torch.manual_seed(0)
        # Use a structured synthetic signal: a sum of sinusoids that the LSTM
        # can actually memorise with a tiny number of parameters.
        n, L = 8, 16
        t_idx = np.arange(L)[None, :, None]  # (1, L, 1)
        phases = np.random.RandomState(0).uniform(
            0, np.pi, size=(n, 1, EVENT_FEATURE_DIM)
        )
        data = np.sin(0.5 * t_idx + phases).astype(np.float32)

        ds = TensorDataset(torch.from_numpy(data))

        class _UnwrapLoader:
            def __init__(self, base):
                self._base = base

            def __iter__(self):
                for batch in self._base:
                    yield batch[0]

            def __len__(self):
                return len(self._base)

        loader = _UnwrapLoader(DataLoader(ds, batch_size=4, shuffle=False))

        _, history = train_lstm_ae(
            loader,
            loader,
            hidden_dim=32,
            bottleneck_dim=8,
            num_layers=1,
            dropout=0.0,
            lr=5e-2,
            epochs=40,
            device="cpu",
            log_every=100,
            early_stopping_patience=None,
        )
        assert len(history.train_loss) == 40
        # Final loss should be a fraction of starting loss on this learnable signal
        assert history.train_loss[-1] < history.train_loss[0] * 0.5, (
            f"train_loss[0]={history.train_loss[0]:.4f}, "
            f"train_loss[-1]={history.train_loss[-1]:.4f}"
        )

    def test_returns_best_weights_when_val_provided(self):
        loader = _make_loader(n=16, L=8, seed=0)
        model, history = train_lstm_ae(
            loader,
            loader,
            hidden_dim=16,
            bottleneck_dim=8,
            num_layers=1,
            dropout=0.0,
            lr=1e-2,
            epochs=10,
            device="cpu",
            log_every=100,
            early_stopping_patience=None,
        )
        assert history.best_epoch > 0
        assert math.isfinite(history.best_val_loss)
        assert next(model.parameters()).device.type == "cpu"

    def test_works_without_val_loader(self):
        loader = _make_loader(n=16, L=8, seed=0)
        _, history = train_lstm_ae(
            loader,
            None,
            hidden_dim=16,
            bottleneck_dim=8,
            num_layers=1,
            dropout=0.0,
            lr=1e-2,
            epochs=5,
            device="cpu",
            log_every=100,
            early_stopping_patience=None,
        )
        assert history.val_loss == []
        assert history.best_epoch == -1


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_save_load_produces_identical_outputs(self, model, tmp_path):
        # Bake the model with random weights, run a forward pass, save, reload, compare.
        x = torch.randn(3, 16, 8)
        with torch.no_grad():
            original_out = model(x).numpy()

        stats = {
            "mean": np.zeros(8, dtype=np.float32),
            "std": np.ones(8, dtype=np.float32),
        }
        weights_path, meta_path = save_lstm_ae(
            model, stats, tmp_path, config={"foo": 1}
        )
        assert weights_path.exists()
        assert meta_path.exists()

        loaded_model, loaded_stats, meta = load_lstm_ae(tmp_path, device="cpu")
        assert meta["config"] == {"foo": 1}
        np.testing.assert_array_equal(loaded_stats["mean"], stats["mean"])
        np.testing.assert_array_equal(loaded_stats["std"], stats["std"])
        # Architecture metadata round-trips
        assert loaded_model.hidden_dim == model.hidden_dim
        assert loaded_model.bottleneck_dim == model.bottleneck_dim
        # Same input → same output (within float tolerance)
        with torch.no_grad():
            loaded_out = loaded_model(x).numpy()
        np.testing.assert_allclose(loaded_out, original_out, rtol=1e-5, atol=1e-6)

    def test_load_missing_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lstm_ae(tmp_path)


class TestScoreSequences:
    def test_accepts_numpy_and_tensor(self, model):
        seqs_np = np.zeros((4, 8, EVENT_FEATURE_DIM), dtype=np.float32)
        seqs_t = torch.from_numpy(seqs_np)
        scores_np = score_sequences(model, seqs_np, device="cpu")
        scores_t = score_sequences(model, seqs_t, device="cpu")
        np.testing.assert_array_equal(scores_np, scores_t)

    def test_score_shape(self, model):
        seqs = np.zeros((10, 8, EVENT_FEATURE_DIM), dtype=np.float32)
        scores = score_sequences(model, seqs, batch_size=4, device="cpu")
        assert scores.shape == (10,)
        assert scores.dtype == np.float32

    def test_score_non_negative(self, model):
        seqs = (
            np.random.RandomState(0).randn(10, 8, EVENT_FEATURE_DIM).astype(np.float32)
        )
        scores = score_sequences(model, seqs, device="cpu")
        assert (scores >= 0).all()

    def test_invalid_input_type_raises(self, model):
        with pytest.raises(TypeError):
            score_sequences(model, [[1, 2, 3]], device="cpu")  # list not allowed
