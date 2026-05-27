"""
pipeline.models
===============
Custom PyTorch models that don't fit the scikit-learn shape used by
``pipeline.training.run``. Currently:

- ``lstm_ae``: bidirectional LSTM autoencoder for unsupervised behavioural
  anomaly detection on raw event sequences (Phase 2 of the roadmap).
"""

from pipeline.models.lstm_ae import (
    LSTMAutoencoder,
    score_sequences,
    train_lstm_ae,
)

__all__ = [
    "LSTMAutoencoder",
    "score_sequences",
    "train_lstm_ae",
]
