"""
pipeline/explainability.py
==========================
Phase 5a — model explainability helpers.

Two complementary tools, one per model family:

1. **SHAP for the supervised identification model** (LightGBM / RF / XGB).
   ``TreeExplainer`` gives *exact* per-prediction Shapley values; we summarise
   them into a per-player feature-importance table and feed the raw values to
   beeswarm / waterfall plots in the notebook.

2. **Per-channel reconstruction-error attribution for the LSTM-AE.** The
   autoencoder flags a chunk by reconstructing it poorly; the scalar score is
   the reconstruction MSE averaged over timesteps *and* the 8 input channels.
   Keeping the per-channel breakdown answers *which* channel drove the flag
   (e.g. the mouse dx/dy channels for an aimbot) — robust and exact, without
   the fragility of running SHAP through an LSTM.

These power ``notebooks/12_explainability.ipynb`` and are unit-tested in
``tests/test_explainability.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Channel order produced by
# pipeline.sequences.preprocessing.session_to_event_tensor (COL_* constants).
CHANNEL_NAMES = [
    "dt",  # log1p inter-event time
    "dx",  # mouse dx (sens/DPI-normalised)
    "dy",  # mouse dy (sens/DPI-normalised)
    "is_mouse_move",
    "is_mouse_click_press",
    "is_mouse_scroll",
    "is_key_press",
    "is_key_release",
]


def tree_shap(model, X):
    """Exact SHAP ``Explanation`` for a tree classifier (LightGBM/RF/XGB).

    ``X`` should be the scaled feature frame the model was trained on (named
    columns → SHAP carries real feature names). For a multiclass classifier the
    returned ``Explanation.values`` has shape ``(n_samples, n_features,
    n_classes)``.
    """
    import shap

    return shap.TreeExplainer(model)(X)


def mean_abs_shap(values, feature_cols, classes) -> pd.DataFrame:
    """Mean |SHAP| per feature per class → DataFrame (features × classes).

    ``values`` is the array from a TreeExplainer: ``(n_samples, n_features,
    n_classes)`` for multiclass, or ``(n_samples, n_features)`` for a single
    output. Each cell is the mean absolute SHAP value — a global importance.

    Binary classifiers are the single-output case: SHAP emits one array because
    one class's contribution is exactly the negative of the other's, so mean
    |SHAP| is identical for both. We therefore label that single column for the
    *pair* it separates (``"a vs b"``) rather than mislabelling it with one
    class name — it is each class's importance equally.
    """
    values = np.asarray(values)
    if values.ndim == 2:
        imp = np.abs(values).mean(axis=0)  # (n_features,)
        if classes is not None and len(classes) == 2:
            col = f"{classes[0]} vs {classes[1]}"
        elif classes is not None and len(classes):
            col = classes[0]
        else:
            col = "importance"
        return pd.DataFrame({col: imp}, index=list(feature_cols))
    imp = np.abs(values).mean(axis=0)  # (n_features, n_classes)
    return pd.DataFrame(imp, index=list(feature_cols), columns=list(classes))


def per_channel_reconstruction_error(model, normalized_chunks) -> np.ndarray:
    """Per-channel reconstruction MSE for a stack of normalised ``(L, 8)`` chunks.

    Returns ``(n_chunks, 8)``: for each chunk, the mean over timesteps of
    ``(reconstruction − input)²`` for each of the 8 input channels. A high value
    in a channel means that channel reconstructed poorly — it is what drove the
    anomaly. Averaging a row over its 8 channels recovers the scalar
    ``score_sequences`` reconstruction error.
    """
    import torch

    arr = np.asarray(normalized_chunks, dtype=np.float32)
    if arr.ndim == 2:  # single (L, 8) chunk → add batch dim
        arr = arr[None]
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(arr).float().to(device)
        recon = model(x)
        per_channel = ((recon - x) ** 2).mean(dim=1)  # mean over timesteps → (B, 8)
    return per_channel.cpu().numpy()
