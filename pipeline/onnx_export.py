"""
pipeline/onnx_export.py
=======================
Bit-faithful ONNX export for the LightGBM identification pipeline.

Why this module exists (the serving-fidelity bug, docs/FINDINGS.md #7)
----------------------------------------------------------------------
The original export converted ``Pipeline(scaler, LGBMClassifier)`` through
skl2onnx with the onnxmltools LightGBM converter and produced probabilities
that disagreed with sklearn by MAE ~0.27 (38% of labels flipped on a random
probe). Investigation showed the converter itself is *faithful* — the
divergence has two precision causes, neither of them a converter bug:

1. **The ai.onnx.ml Scaler op computes in float32.** Standardising
   ``(x - mean) * (1/std)`` in single precision perturbs the scaled features
   by up to ~2e-4 (the worst features have tiny stds, so the error is
   amplified by 1/std).
2. **The model is razor-margin sensitive.** This LightGBM model (trained to
   100% train accuracy on 187 windows) memorises via hairline splits whose
   thresholds sit within float32 epsilon of real feature values. Feeding
   *sklearn itself* the float32-scaled features reproduces the exact same
   0.27 MAE — so any float32 stage anywhere in the graph breaks parity.

The fix is to keep the entire graph in float64:

* the scaler becomes plain ``Sub`` + ``Div`` nodes on double tensors
  (bit-identical to numpy's ``(X - mean_) / scale_``),
* the trees run as an ``ai.onnx.ml v3`` ``TreeEnsembleClassifier`` with
  ``nodes_values_as_tensor`` / ``class_weights_as_tensor`` holding the
  booster's *original float64* thresholds and leaf weights (the classic
  float-list attributes are single precision by ONNX spec).

onnxmltools doesn't emit the double variant, so we convert normally to get
the verified tree *structure*, then re-derive the exact float64 values from
``booster_.dump_model()`` by replicating the converter's node-id assignment.
A float32 checksum guarantees the re-derivation matches the converter
node-for-node; any mismatch raises and the caller falls back to the float
export.

Result: probability MAE ~1e-8 (pure float32 rounding of the final softmax)
and 100% label agreement on train/val/test and random probes. Gated by
``tests/test_onnx_export.py`` in CI and re-checked at every export by
``pipeline/training/run.py:_validate_onnx_fidelity``.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Opset ceiling of the installed onnxmltools converter (it rejects higher).
_ONNXMLTOOLS_MAX_OPSET = 15
_AI_ONNX_ML_DOUBLE_OPSET = 3  # *_as_tensor attributes need ai.onnx.ml >= 3


class OnnxExportError(RuntimeError):
    """Raised when the double-precision export cannot be built faithfully."""


def _reconstruct_double_tensors(booster) -> tuple[np.ndarray, np.ndarray]:
    """Exact float64 node thresholds + leaf class weights from the booster dump.

    Replicates onnxmltools' node-id assignment (``_parse_tree_structure``):
    ids are handed out in discovery order where each branch node assigns ids
    to both children before descending left then right; leaf class weights
    are emitted in visit order. The converter later sorts ``nodes_*`` by
    (tree, node id), so per-tree threshold arrays are returned id-ascending.

    Returns (nodes_values64, class_weights64) matching the converter's
    ``nodes_values`` / ``class_weights`` attribute layouts.
    """
    dump = booster.dump_model()
    nodes_values: list[float] = []
    class_weights: list[float] = []

    for tree in dump["tree_info"]:
        by_id: dict[int, float] = {}
        counter = iter(range(1 << 62))

        def visit(node: dict, nid: int) -> None:
            if "left_child" in node and "right_child" in node:
                if node["decision_type"] != "<=":
                    raise OnnxExportError(
                        "Unsupported decision_type "
                        f"{node['decision_type']!r} (only numerical '<=' "
                        "splits are handled)."
                    )
                left_id, right_id = next(counter), next(counter)
                by_id[nid] = float(node["threshold"])
                visit(node["left_child"], left_id)
                visit(node["right_child"], right_id)
            else:
                by_id[nid] = 0.0  # converter stores 0.0 for leaves
                class_weights.append(float(node["leaf_value"]))

        visit(tree["tree_structure"], next(counter))
        nodes_values.extend(by_id[i] for i in range(len(by_id)))

    return (
        np.asarray(nodes_values, dtype=np.float64),
        np.asarray(class_weights, dtype=np.float64),
    )


def convert_lightgbm_pipeline_double(model, scaler, n_features: int):
    """StandardScaler + LGBMClassifier → float64-faithful ONNX ModelProto.

    Graph contract:
      input   ``input``          float64 [None, n_features]  (raw features)
      output  ``label``          int64   [None]
      output  ``probabilities``  float32 [None, n_classes]

    Raises OnnxExportError if the reconstruction checksum fails (converter
    behaviour changed) or the model uses unsupported split types.
    """
    import onnx
    from onnx import TensorProto, compose, helper, numpy_helper
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    # 1. Trusted tree *structure* from the stock converter.
    float_model = convert_lightgbm(
        model,
        initial_types=[("scaled", FloatTensorType([None, n_features]))],
        zipmap=False,
        target_opset=_ONNXMLTOOLS_MAX_OPSET,
    )
    tec = next(
        (n for n in float_model.graph.node if n.op_type == "TreeEnsembleClassifier"),
        None,
    )
    if tec is None:
        raise OnnxExportError("Converter produced no TreeEnsembleClassifier node.")
    attrs = {a.name: helper.get_attribute_value(a) for a in tec.attribute}

    # 2. Exact float64 values, checksummed against the converter's float32.
    values64, weights64 = _reconstruct_double_tensors(model.booster_)
    attr_values = np.asarray(attrs["nodes_values"], dtype=np.float32)
    attr_weights = np.asarray(attrs["class_weights"], dtype=np.float32)
    if len(values64) != len(attr_values) or not np.array_equal(
        values64.astype(np.float32), attr_values
    ):
        raise OnnxExportError("nodes_values reconstruction checksum failed.")
    if len(weights64) != len(attr_weights) or not np.array_equal(
        weights64.astype(np.float32), attr_weights
    ):
        raise OnnxExportError("class_weights reconstruction checksum failed.")

    # 3. Double-precision tree node (drop float32 lists + optional hitrates).
    keep = {
        name: val
        for name, val in attrs.items()
        if name not in ("nodes_values", "class_weights", "nodes_hitrates")
    }
    keep["nodes_modes"] = [m.decode() for m in keep["nodes_modes"]]
    keep["post_transform"] = keep["post_transform"].decode()
    tree_node = helper.make_node(
        "TreeEnsembleClassifier",
        ["scaled"],
        ["label", "probabilities"],
        domain="ai.onnx.ml",
        nodes_values_as_tensor=numpy_helper.from_array(values64, name=""),
        class_weights_as_tensor=numpy_helper.from_array(weights64, name=""),
        **keep,
    )
    n_classes = len(keep["classlabels_int64s"])

    # 4. Double-precision scaler: Sub + Div is bit-identical to sklearn's
    #    (X - mean_) / scale_ in numpy float64.
    scaler_graph = helper.make_graph(
        [
            helper.make_node("Sub", ["input", "scaler_mean"], ["centered"]),
            helper.make_node("Div", ["centered", "scaler_scale"], ["scaled"]),
        ],
        "scaler_float64",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.DOUBLE, [None, n_features]
            )
        ],
        [
            helper.make_tensor_value_info(
                "scaled", TensorProto.DOUBLE, [None, n_features]
            )
        ],
        initializer=[
            numpy_helper.from_array(
                np.asarray(scaler.mean_, dtype=np.float64), name="scaler_mean"
            ),
            numpy_helper.from_array(
                np.asarray(scaler.scale_, dtype=np.float64), name="scaler_scale"
            ),
        ],
    )
    tree_graph = helper.make_graph(
        [tree_node],
        "lightgbm_float64",
        [
            helper.make_tensor_value_info(
                "scaled", TensorProto.DOUBLE, [None, n_features]
            )
        ],
        [
            helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
            # ai.onnx.ml TreeEnsembleClassifier scores are spec'd float32;
            # rounding the final softmax output is the only precision loss.
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, n_classes]
            ),
        ],
    )

    opsets = [
        helper.make_opsetid("", _ONNXMLTOOLS_MAX_OPSET),
        helper.make_opsetid("ai.onnx.ml", _AI_ONNX_ML_DOUBLE_OPSET),
    ]
    scaler_model = helper.make_model(scaler_graph, opset_imports=opsets)
    tree_model = helper.make_model(tree_graph, opset_imports=opsets)
    scaler_model.ir_version = tree_model.ir_version  # compose requires equality

    merged = compose.merge_models(
        scaler_model, tree_model, io_map=[("scaled", "scaled")]
    )
    onnx.checker.check_model(merged)
    return merged


def parity_report(artifact: dict, session, X: np.ndarray) -> dict:
    """Compare sklearn vs an ONNX session on float64 features X.

    Returns {"probability_mae", "probability_max_abs", "label_agreement"}.
    The session input dtype is honoured (float64 for the double export,
    float32 for legacy float graphs).
    """
    import pandas as pd

    model, scaler = artifact["model"], artifact["scaler"]
    # Named frame: the identification scaler was fit with feature names.
    X_df = pd.DataFrame(
        np.asarray(X, dtype=np.float64), columns=artifact["feature_cols"]
    )
    p_sk = model.predict_proba(scaler.transform(X_df))

    inp = session.get_inputs()[0]
    dtype = np.float64 if "double" in inp.type else np.float32
    feed = {inp.name: np.ascontiguousarray(X, dtype=dtype)}
    label, p_ox = session.run(["label", "probabilities"], feed)
    p_ox = np.asarray(p_ox, dtype=np.float64)

    return {
        "probability_mae": float(np.abs(p_sk - p_ox).mean()),
        "probability_max_abs": float(np.abs(p_sk - p_ox).max()),
        "label_agreement": float((p_sk.argmax(axis=1) == np.asarray(label)).mean()),
    }
