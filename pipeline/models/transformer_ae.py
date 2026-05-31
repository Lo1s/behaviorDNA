"""
pipeline/models/transformer_ae.py
==================================
Small Transformer autoencoder — a self-attention alternative to the LSTM-AE on
the same chunk-level reconstruction task.

A linear embedding + learned positional encoding feed a stack of
``TransformerEncoder`` layers; a flatten → narrow bottleneck → unflatten forces
compression (so attention can't learn the identity), and a per-position linear
head reconstructs the 8 channels.

Included mainly for the architecture comparison: at this dataset scale
(18 sessions) a transformer is more data-hungry than the LSTM/TCN and the
sequence is short (L=64), so it is *not* expected to win — the value is the
honest head-to-head. Same interface as ``LSTMAutoencoder`` so it drops into
``score_sequences`` unchanged. Fixed sequence length.
"""

from __future__ import annotations

import torch
from torch import nn

from pipeline.sequences.preprocessing import EVENT_FEATURE_DIM


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = EVENT_FEATURE_DIM,
        seq_len: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        bottleneck_dim: int = 16,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        self.seq_len = seq_len
        self.d_model = d_model

        self.embed = nn.Linear(feature_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        flat = seq_len * d_model
        self.to_bottleneck = nn.Linear(flat, bottleneck_dim)
        self.from_bottleneck = nn.Linear(bottleneck_dim, flat)
        self.head = nn.Linear(d_model, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        h = self.embed(x) + self.pos  # (B, L, d_model)
        h = self.encoder(h)
        z = self.to_bottleneck(h.reshape(b, -1))  # (B, bottleneck)
        h2 = self.from_bottleneck(z).reshape(b, self.seq_len, self.d_model)
        return self.head(h2)  # (B, L, F)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=(1, 2))
