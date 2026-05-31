"""
tests/test_benchmark_inference.py
=================================
Tests for the sklearn/ONNX prediction helpers in scripts/benchmark_inference.py.

We export a *sklearn-native* pipeline (StandardScaler + LogisticRegression) to
ONNX — skl2onnx converts these faithfully — and assert the two prediction paths
agree to high precision. This both exercises the helpers and documents the
contrast with the LightGBM→ONNX export, which is *not* faithful with the current
converter versions (see docs/FINDINGS.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark_inference import (
    predict_labels_onnx,
    predict_labels_sklearn,
    predict_proba_onnx,
)


@pytest.fixture(scope="module")
def faithful_pipeline():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    skl2onnx = pytest.importorskip("skl2onnx")
    ort = pytest.importorskip("onnxruntime")
    from skl2onnx.common.data_types import FloatTensorType

    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(6)]
    X = pd.DataFrame(rng.normal(size=(200, 6)), columns=cols)
    y = (X["f0"] + 0.5 * X["f1"] > 0).astype(int)

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(X), y)
    artifact = {"scaler": scaler, "model": model}

    onnx = skl2onnx.convert_sklearn(
        __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
            [("scaler", scaler), ("model", model)]
        ),
        initial_types=[("float_input", FloatTensorType([None, 6]))],
        options={"zipmap": False},
    )
    sess = ort.InferenceSession(
        onnx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return artifact, sess, X


def test_labels_agree(faithful_pipeline):
    artifact, sess, X = faithful_pipeline
    y_sk = predict_labels_sklearn(artifact, X)
    y_ox = predict_labels_onnx(sess, X.to_numpy(np.float32))
    assert (y_sk == y_ox).mean() == 1.0  # faithful export → exact label match


def test_probabilities_match_to_high_precision(faithful_pipeline):
    artifact, sess, X = faithful_pipeline
    p_sk = artifact["model"].predict_proba(artifact["scaler"].transform(X))
    p_ox = predict_proba_onnx(sess, X.to_numpy(np.float32))
    assert np.abs(p_sk - p_ox).mean() < 1e-5  # contrast: LightGBM export ≈ 0.13


def test_predict_helpers_shapes(faithful_pipeline):
    artifact, sess, X = faithful_pipeline
    assert predict_labels_sklearn(artifact, X).shape == (len(X),)
    assert predict_proba_onnx(sess, X.to_numpy(np.float32)).shape == (len(X), 2)
