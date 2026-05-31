"""
tests/test_explainability.py
============================
Unit tests for pipeline/explainability.py (Phase 5a).

Covers the two helpers: TreeExplainer SHAP summarisation for the supervised
classifier, and per-channel reconstruction-error attribution for the LSTM-AE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.explainability import (
    CHANNEL_NAMES,
    mean_abs_shap,
    per_channel_reconstruction_error,
    tree_shap,
)

# ---------------------------------------------------------------------------
# Classifier SHAP
# ---------------------------------------------------------------------------


def _tiny_classifier(n=120, n_feat=6, n_classes=3, seed=0):
    from lightgbm import LGBMClassifier

    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_feat)]
    X = pd.DataFrame(rng.normal(size=(n, n_feat)), columns=cols)
    # Label is a pure function of f0 (tertiles) so f0 is unambiguously the
    # dominant feature in SHAP — the other columns are pure noise.
    edges = np.quantile(X["f0"], [1 / 3, 2 / 3])
    y = np.digitize(X["f0"], edges)
    model = LGBMClassifier(n_estimators=30, num_leaves=7, verbose=-1).fit(X, y)
    return model, X, cols, sorted(np.unique(y).tolist())


class TestTreeShap:
    def test_shape_and_named_features(self):
        model, X, cols, classes = _tiny_classifier()
        expl = tree_shap(model, X)
        # multiclass → (n_samples, n_features, n_classes)
        assert expl.values.shape == (len(X), len(cols), len(classes))

    def test_mean_abs_shap_table(self):
        model, X, cols, classes = _tiny_classifier()
        expl = tree_shap(model, X)
        imp = mean_abs_shap(expl.values, cols, classes)
        assert list(imp.index) == cols
        assert list(imp.columns) == classes
        assert (imp.to_numpy() >= 0).all()  # mean |SHAP| is non-negative
        # f0 carries the signal → should not be the least-important feature
        assert imp.sum(axis=1).idxmax() == "f0"

    def test_mean_abs_shap_handles_2d_values(self):
        # Single-output (binary/regression) shape (n, feat) → one column
        vals = np.array([[1.0, -2.0], [3.0, 0.0]])
        out = mean_abs_shap(vals, ["a", "b"], ["pos"])
        assert out.shape == (2, 1)
        assert out.loc["b", "pos"] == 1.0  # mean(|-2|, |0|)


# ---------------------------------------------------------------------------
# LSTM-AE per-channel reconstruction attribution
# ---------------------------------------------------------------------------


def _tiny_ae():
    from pipeline.models.lstm_ae import LSTMAutoencoder

    return LSTMAutoencoder(hidden_dim=8, bottleneck_dim=4, num_layers=1, dropout=0.0)


class TestPerChannelReconstruction:
    def test_shape_and_nonnegative(self):
        model = _tiny_ae()
        chunks = np.random.default_rng(0).normal(size=(5, 16, 8)).astype(np.float32)
        err = per_channel_reconstruction_error(model, chunks)
        assert err.shape == (5, 8)
        assert (err >= 0).all() and np.isfinite(err).all()

    def test_single_chunk_gets_batch_dim(self):
        model = _tiny_ae()
        one = np.random.default_rng(1).normal(size=(16, 8)).astype(np.float32)
        err = per_channel_reconstruction_error(model, one)
        assert err.shape == (1, 8)

    def test_channel_mean_matches_scalar_score(self):
        # Averaging the per-channel row over channels must equal the scalar
        # reconstruction_error the inference path uses.
        import torch

        from pipeline.models.lstm_ae import score_sequences

        model = _tiny_ae()
        chunks = np.random.default_rng(2).normal(size=(4, 16, 8)).astype(np.float32)
        per_ch = per_channel_reconstruction_error(model, chunks)
        scalar = score_sequences(
            model, torch.from_numpy(chunks).float(), batch_size=4, device="cpu"
        )
        assert np.allclose(per_ch.mean(axis=1), scalar, atol=1e-5)

    def test_channel_names_length(self):
        assert len(CHANNEL_NAMES) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
