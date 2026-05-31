"""
pipeline/models/tcn_ae.py
=========================
Temporal Convolutional Network autoencoder — a conv-based alternative to the
LSTM-AE for the same chunk-level reconstruction task.

Dilated 1-D convolutions give each output position a wide receptive field
without recurrence, so a TCN often matches an LSTM on sequence tasks while
training faster and with fewer parameters (and less overfitting on small data).
A true bottleneck (flatten → linear → narrow code → linear → unflatten) forces
compression so the model can't learn the identity.

Same interface as ``LSTMAutoencoder`` (``forward`` → ``(B, L, F)``;
``reconstruction_error`` → ``(B,)``), so it drops into ``score_sequences`` and the
architecture-comparison harness unchanged. Fixed sequence length (chunks are a
constant 64 events).
"""

from __future__ import annotations

import torch
from torch import nn

from pipeline.sequences.preprocessing import EVENT_FEATURE_DIM


class TCNAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = EVENT_FEATURE_DIM,
        seq_len: int = 64,
        hidden_dim: int = 32,
        bottleneck_dim: int = 16,
        dilations: tuple[int, ...] = (1, 2, 4),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        def conv_block(in_c: int, out_c: int, dilation: int) -> nn.Sequential:
            pad = (kernel_size - 1) // 2 * dilation  # 'same' length
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size, padding=pad, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        enc = []
        in_c = feature_dim
        for d in dilations:
            enc.append(conv_block(in_c, hidden_dim, d))
            in_c = hidden_dim
        self.encoder = nn.Sequential(*enc)

        flat = hidden_dim * seq_len
        self.to_bottleneck = nn.Linear(flat, bottleneck_dim)
        self.from_bottleneck = nn.Linear(bottleneck_dim, flat)
        self.hidden_dim = hidden_dim

        self.decoder = nn.Sequential(
            conv_block(hidden_dim, hidden_dim, 1),
            nn.Conv1d(
                hidden_dim, feature_dim, kernel_size, padding=(kernel_size - 1) // 2
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) → conv wants (B, F, L)
        b = x.size(0)
        h = self.encoder(x.transpose(1, 2))  # (B, hidden, L)
        z = self.to_bottleneck(h.reshape(b, -1))  # (B, bottleneck)
        h2 = self.from_bottleneck(z).reshape(b, self.hidden_dim, self.seq_len)
        recon = self.decoder(h2)  # (B, F, L)
        return recon.transpose(1, 2)  # (B, L, F)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=(1, 2))
