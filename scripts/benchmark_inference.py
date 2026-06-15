"""
scripts/benchmark_inference.py
==============================
Inference latency + throughput benchmark: scikit-learn vs ONNX Runtime.

Anti-cheat runs at population scale — millions of concurrent players — so "what
does one inference cost?" is a first-class question. We export an ONNX model
(`models/model.onnx`) from training; this script actually *uses* it and measures:

* **per-window latency** (single sample — the real-time path): p50 / p95 / mean,
* **throughput** (batched): windows scored per second,
* **parity**: ONNX and sklearn agree on the predicted labels.

Outputs `reports/inference_benchmark.json` + `reports/figures/inference_latency.png`.

Usage:
    python -m scripts.benchmark_inference            # uses the test split if present
    python -m scripts.benchmark_inference --n 5000 --batch 512
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PKL = ROOT / "models" / "model.pkl"
MODEL_ONNX = ROOT / "models" / "model.onnx"
TEST_PARQUET = ROOT / "data" / "splits" / "test.parquet"
OUT_JSON = ROOT / "reports" / "inference_benchmark.json"
OUT_FIG = ROOT / "reports" / "figures" / "inference_latency.png"

# Max acceptable probability MAE between sklearn and ONNX for the export to be
# considered production-trustworthy. The float64 composed export
# (pipeline/onnx_export.py) achieves ~1e-8; anything above this means the
# serving-fidelity regression of docs/FINDINGS.md #7 is back.
FIDELITY_THRESHOLD = 1e-6


# --- prediction paths (small + testable) ----------------------------------


def predict_labels_sklearn(artifact: dict, X_df) -> np.ndarray:
    """Production sklearn path: scaler.transform → classifier.predict (encoded labels)."""
    return np.asarray(artifact["model"].predict(artifact["scaler"].transform(X_df)))


def _onnx_input(session) -> tuple[str, type]:
    """(input name, numpy dtype) — the float64 export takes double tensors."""
    inp = session.get_inputs()[0]
    return inp.name, (np.float64 if "double" in inp.type else np.float32)


def predict_labels_onnx(session, X_np: np.ndarray) -> np.ndarray:
    """ONNX Runtime path on raw features (scaler is baked into the graph)."""
    name, dtype = _onnx_input(session)
    out = session.run(["label"], {name: np.ascontiguousarray(X_np, dtype=dtype)})
    return np.asarray(out[0]).ravel()


def predict_proba_onnx(session, X_np: np.ndarray) -> np.ndarray:
    """ONNX class probabilities — the high-fidelity parity signal vs predict_proba."""
    name, dtype = _onnx_input(session)
    out = session.run(
        ["probabilities"], {name: np.ascontiguousarray(X_np, dtype=dtype)}
    )
    return np.asarray(out[0])


# --- timing helpers --------------------------------------------------------


def _latency_ms(call, rows, n_iter: int) -> dict:
    lat = np.empty(n_iter)
    for i in range(n_iter):
        r = rows[i % len(rows)]
        t0 = time.perf_counter()
        call(r)
        lat[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "mean_ms": float(lat.mean()),
    }


def _throughput(call, batch, reps: int) -> float:
    call(batch)  # warm up
    t0 = time.perf_counter()
    for _ in range(reps):
        call(batch)
    dt = time.perf_counter() - t0
    return float(reps * len(batch) / dt)


def _load_X(feature_cols: list[str], n_min: int = 256) -> np.ndarray:
    """Feature matrix from the test split, tiled up to n_min rows; else random."""
    import pandas as pd

    if TEST_PARQUET.exists():
        df = pd.read_parquet(TEST_PARQUET)
        if len(df):
            X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
            reps = int(np.ceil(n_min / len(X)))
            return np.tile(X, (reps, 1))[:n_min]
    log.warning("No test split — using random features (timings still valid).")
    return np.random.default_rng(0).normal(size=(n_min, len(feature_cols)))


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="sklearn vs ONNX inference benchmark")
    parser.add_argument("--n", type=int, default=3000, help="single-sample iterations")
    parser.add_argument("--batch", type=int, default=256, help="throughput batch size")
    parser.add_argument("--reps", type=int, default=200, help="throughput repetitions")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if ONNX fidelity fails (use as a release gate)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    if not MODEL_PKL.exists() or not MODEL_ONNX.exists():
        log.error("Need models/model.pkl + models/model.onnx (run `dvc repro train`).")
        return 1

    import pandas as pd

    artifact = pickle.load(open(MODEL_PKL, "rb"))
    if not artifact.get("trained"):
        log.error("Model artifact is untrained.")
        return 1

    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL_ONNX), providers=["CPUExecutionProvider"])

    feature_cols = artifact["feature_cols"]
    X = _load_X(feature_cols, max(args.batch, 256))
    X_df = pd.DataFrame(X, columns=feature_cols)  # named → no sklearn warning
    X_onnx = X  # float64 contract; predict_*_onnx cast to the session dtype

    # Parity. Probabilities are the true fidelity measure; label flips happen only
    # at argmax near-ties (this 3-class model is intentionally low-confidence at
    # N=18), so we report both + how many disagreements are within a 0.05 top-2 gap.
    y_sk = predict_labels_sklearn(artifact, X_df)
    y_ox = predict_labels_onnx(session, X_onnx)
    agreement = float((y_sk == y_ox).mean())
    p_sk = artifact["model"].predict_proba(artifact["scaler"].transform(X_df))
    p_ox = predict_proba_onnx(session, X_onnx)
    prob_mae = float(np.abs(p_sk - p_ox).mean())
    prob_max = float(np.abs(p_sk - p_ox).max())
    disagree = y_sk != y_ox
    top2_gap = np.sort(p_sk, axis=1)[:, -1] - np.sort(p_sk, axis=1)[:, -2]
    flips_at_ties = float((top2_gap[disagree] < 0.05).mean()) if disagree.any() else 1.0

    # Single-window latency (the real-time path)
    sk_rows = [X_df.iloc[[i]] for i in range(len(X_df))]
    ox_rows = [X_onnx[i : i + 1] for i in range(len(X_onnx))]
    sk_lat = _latency_ms(lambda r: predict_labels_sklearn(artifact, r), sk_rows, args.n)
    ox_lat = _latency_ms(lambda r: predict_labels_onnx(session, r), ox_rows, args.n)

    # Batched throughput (windows/sec)
    sk_tp = _throughput(
        lambda b: predict_labels_sklearn(artifact, b),
        X_df.iloc[: args.batch],
        args.reps,
    )
    ox_tp = _throughput(
        lambda b: predict_labels_onnx(session, b), X_onnx[: args.batch], args.reps
    )

    fidelity_ok = prob_mae < FIDELITY_THRESHOLD
    result = {
        "onnx_fidelity_ok": fidelity_ok,
        "label_agreement": agreement,
        "probability_mae": prob_mae,
        "probability_max_abs": prob_max,
        "fraction_of_label_flips_at_near_ties": flips_at_ties,
        "n_features": len(feature_cols),
        "single_window_latency": {"sklearn": sk_lat, "onnx": ox_lat},
        "throughput_windows_per_s": {"sklearn": sk_tp, "onnx": ox_tp},
        "onnx_speedup_p50": (
            sk_lat["p50_ms"] / ox_lat["p50_ms"] if ox_lat["p50_ms"] else float("nan")
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    log.info(
        "parity: prob MAE %.2e (max %.2e); label agreement %.3f; %.0f%% of flips are argmax near-ties",
        prob_mae,
        prob_max,
        agreement,
        flips_at_ties * 100,
    )
    log.info(
        "single-window p50/p95 ms — sklearn %.3f/%.3f | onnx %.3f/%.3f (%.1fx faster p50)",
        sk_lat["p50_ms"],
        sk_lat["p95_ms"],
        ox_lat["p50_ms"],
        ox_lat["p95_ms"],
        result["onnx_speedup_p50"],
    )
    log.info("throughput windows/s — sklearn %.0f | onnx %.0f", sk_tp, ox_tp)

    if not fidelity_ok:
        log.warning(
            "ONNX export is NOT production-trustworthy: probability MAE %.2e > %.0e. "
            "The float64 composed export (pipeline/onnx_export.py) should be "
            "bit-faithful — this is the docs/FINDINGS.md #7 regression. ONNX "
            "latencies are reference-only until resolved.",
            prob_mae,
            FIDELITY_THRESHOLD,
        )

    _render_figure(result)
    log.info("Wrote %s and %s", OUT_JSON, OUT_FIG)
    if args.strict and not fidelity_ok:
        log.error("--strict: failing because ONNX fidelity check did not pass.")
        return 1
    return 0


def _render_figure(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sk, ox = (
        result["single_window_latency"]["sklearn"],
        result["single_window_latency"]["onnx"],
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(2)
    w = 0.35
    ax1.bar(x - w / 2, [sk["p50_ms"], ox["p50_ms"]], w, label="p50", color="#4c78a8")
    ax1.bar(x + w / 2, [sk["p95_ms"], ox["p95_ms"]], w, label="p95", color="#e94560")
    faithful = result.get("onnx_fidelity_ok", True)
    onnx_tag = "ONNX Runtime" if faithful else "ONNX Runtime*"
    ax1.set_xticks(x)
    ax1.set_xticklabels(["scikit-learn", onnx_tag])
    ax1.set_ylabel("per-window latency (ms)")
    ax1.set_title("Single-window latency (lower is better)")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    tp = result["throughput_windows_per_s"]
    ax2.bar(
        ["scikit-learn", onnx_tag],
        [tp["sklearn"], tp["onnx"]],
        color=["#4c78a8", "#54a24b"],
    )
    ax2.set_ylabel("throughput (windows / sec, batched)")
    ax2.set_title("Batched throughput")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Identification inference latency — scikit-learn is the production path "
        f"(p50 {sk['p50_ms']:.2f} ms · {tp['sklearn']:,.0f} windows/s)",
        fontsize=12,
    )
    if not faithful:
        fig.text(
            0.5,
            0.005,
            f"* ONNX export currently UNFAITHFUL (probability MAE {result['probability_mae']:.2f}) — "
            "latency shown for reference only; see docs/FINDINGS.md",
            ha="center",
            color="#c0392b",
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(run())
