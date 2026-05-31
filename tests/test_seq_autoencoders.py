"""
tests/test_seq_autoencoders.py
==============================
Shape/interface tests for the alternative sequence autoencoders (TCN,
Transformer) used in the architecture comparison. They must match the
LSTM-AE interface so score_sequences works on all three.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.models.tcn_ae import TCNAutoencoder
from pipeline.models.transformer_ae import TransformerAutoencoder

L, F = 32, 8


def _models():
    return [
        TCNAutoencoder(seq_len=L, hidden_dim=16, bottleneck_dim=8),
        TransformerAutoencoder(
            seq_len=L, d_model=16, nhead=2, num_layers=1, bottleneck_dim=8
        ),
    ]


@pytest.mark.parametrize("model", _models())
class TestAutoencoderInterface:
    def test_forward_shape(self, model):
        x = torch.randn(5, L, F)
        assert model(x).shape == (5, L, F)

    def test_reconstruction_error_shape_and_nonneg(self, model):
        x = torch.randn(5, L, F)
        err = model.reconstruction_error(x)
        assert err.shape == (5,)
        assert torch.all(err >= 0) and torch.isfinite(err).all()

    def test_score_sequences_compat(self, model):
        from pipeline.models.lstm_ae import score_sequences

        x = np.random.default_rng(0).normal(size=(6, L, F)).astype(np.float32)
        scores = score_sequences(model, x, batch_size=4, device="cpu")
        assert scores.shape == (6,)

    def test_has_trainable_params(self, model):
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0


def test_one_train_step_reduces_loss():
    # A few gradient steps on a fixed batch should lower reconstruction loss.
    torch.manual_seed(0)
    model = TCNAutoencoder(seq_len=L, hidden_dim=16, bottleneck_dim=8)
    x = torch.randn(16, L, F)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = loss_fn(model(x), x)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_transformer_rejects_bad_head_dim():
    with pytest.raises(ValueError):
        TransformerAutoencoder(d_model=64, nhead=5)  # 64 % 5 != 0
