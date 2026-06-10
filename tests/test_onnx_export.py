"""
tests/test_onnx_export.py
=========================
Regression gate for the ONNX serving-fidelity bug (docs/FINDINGS.md #7).

The float32 export path silently flipped predictions (probability MAE ~0.27
on a probe); the float64 composed export in pipeline/onnx_export.py must be
bit-faithful. These tests fail CI if the export ever regresses.
"""

import numpy as np
import pandas as pd
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxmltools")

from pipeline.features.run import ID_FEATURE_COLS  # noqa: E402
from pipeline.onnx_export import (  # noqa: E402
    OnnxExportError,
    _reconstruct_double_tensors,
    convert_lightgbm_pipeline_double,
    parity_report,
)
from pipeline.training.run import export_onnx, train_lightgbm  # noqa: E402


def make_cfg() -> dict:
    return {
        "model": {"type": "lightgbm", "task": "identification"},
        "lightgbm": {
            "num_leaves": 15,
            "learning_rate": 0.1,
            "n_estimators": 40,
            "min_child_samples": 2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        "data": {"random_seed": 42},
        "mlflow": {"experiment_name": "test", "tracking_uri": "http://localhost"},
    }


def make_train_df(players=("alice", "bob", "carol"), n_windows=40) -> pd.DataFrame:
    """Random per-player-shifted features — deep enough trees for real splits."""
    rng = np.random.default_rng(7)
    rows = []
    for i, player in enumerate(players):
        for w in range(n_windows):
            row = {
                "session_id": f"s_{player}",
                "window_idx": w,
                "player": player,
                "game": "gta",
                "sensitivity": 1.0,
                "dpi": 800,
                "recorded_at": pd.Timestamp("2026-01-01", tz="UTC"),
                "duration_ms": 90_000.0,
            }
            row.update(
                {c: float(rng.normal(loc=0.3 * i, scale=1.0)) for c in ID_FEATURE_COLS}
            )
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def artifact() -> dict:
    art, _ = train_lightgbm(make_train_df(), make_train_df().iloc[0:0], make_cfg())
    assert art["trained"]
    return art


@pytest.fixture(scope="module")
def session(artifact, tmp_path_factory):
    out = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_onnx(artifact, out)
    assert out.stat().st_size > 0, "export wrote empty bytes — conversion failed"
    return ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])


class TestDoubleExportParity:
    def test_input_is_float64(self, session):
        inp = session.get_inputs()[0]
        assert "double" in inp.type

    def test_parity_on_training_features(self, artifact, session):
        X = make_train_df()[ID_FEATURE_COLS].to_numpy(dtype=np.float64)
        report = parity_report(artifact, session, X)
        assert report["label_agreement"] == 1.0
        assert report["probability_mae"] < 1e-6

    def test_parity_on_random_probe(self, artifact, session):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2048, len(ID_FEATURE_COLS)))
        report = parity_report(artifact, session, X)
        assert report["label_agreement"] == 1.0
        assert report["probability_mae"] < 1e-6
        assert report["probability_max_abs"] < 1e-5

    def test_parity_under_tiny_perturbation(self, artifact, session):
        """The bug class this gate exists for: inputs nudged by ~1e-7 must
        still agree because the whole graph runs in float64 (the old float32
        graph flipped labels under exactly this perturbation)."""
        rng = np.random.default_rng(1)
        base = make_train_df()[ID_FEATURE_COLS].to_numpy(dtype=np.float64)
        X = base + rng.normal(scale=1e-7, size=base.shape)
        report = parity_report(artifact, session, X)
        assert report["label_agreement"] == 1.0


class TestReconstruction:
    def test_checksum_matches_converter_float32(self, artifact):
        from onnx import helper
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        model = artifact["model"]
        n = len(artifact["feature_cols"])
        float_model = convert_lightgbm(
            model,
            initial_types=[("scaled", FloatTensorType([None, n]))],
            zipmap=False,
            target_opset=15,
        )
        tec = next(
            nd
            for nd in float_model.graph.node
            if nd.op_type == "TreeEnsembleClassifier"
        )
        attrs = {a.name: helper.get_attribute_value(a) for a in tec.attribute}
        values64, weights64 = _reconstruct_double_tensors(model.booster_)
        assert np.array_equal(
            values64.astype(np.float32),
            np.asarray(attrs["nodes_values"], dtype=np.float32),
        )
        assert np.array_equal(
            weights64.astype(np.float32),
            np.asarray(attrs["class_weights"], dtype=np.float32),
        )

    def test_double_thresholds_carry_more_precision_than_float32(self, artifact):
        values64, _ = _reconstruct_double_tensors(artifact["model"].booster_)
        branch = values64[values64 != 0.0]
        # at least one threshold should not be float32-representable —
        # otherwise this model wouldn't exercise the double path at all
        assert (branch.astype(np.float32).astype(np.float64) != branch).any()

    def test_rejects_unsupported_split_types(self, artifact):
        class FakeBooster:
            def dump_model(self):
                return {
                    "tree_info": [
                        {
                            "tree_structure": {
                                "decision_type": "==",
                                "threshold": 1.0,
                                "left_child": {"leaf_value": 0.1},
                                "right_child": {"leaf_value": 0.2},
                            }
                        }
                    ]
                }

        with pytest.raises(OnnxExportError):
            _reconstruct_double_tensors(FakeBooster())


class TestExportOnnxDispatch:
    def test_untrained_artifact_writes_empty_bytes(self, tmp_path):
        out = tmp_path / "model.onnx"
        export_onnx({"trained": False}, out)
        assert out.read_bytes() == b""

    def test_convert_returns_checked_model(self, artifact):
        model = convert_lightgbm_pipeline_double(
            artifact["model"], artifact["scaler"], len(artifact["feature_cols"])
        )
        onnx.checker.check_model(model)
        out_names = [o.name for o in model.graph.output]
        assert out_names == ["label", "probabilities"]
