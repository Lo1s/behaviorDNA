"""
pipeline/adversarial/benchmark.py
=================================
Score the BehaviorDNA detector zoo against the synthetic-cheat dataset.

Reads every JSON in ``data/synthetic/`` (produced by ``generate_dataset``),
runs each file through the ingestion + feature-engineering pipeline in
memory (no DVC, no parquet writes), trains the unsupervised detectors on
legit-only feature rows, then scores **all** rows and computes per-detector
× per-cheat-type metrics:

  * ROC AUC                              — overall ranking quality
  * PR AUC                               — quality when cheats are rare
  * Detection rate at fixed False Positive Rate (default 5%)
  * Mean anomaly score per cheat type    — interpretability

Run:
    python -m pipeline.adversarial.benchmark
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from pipeline.adversarial.bot_generator import CHEAT_LEGIT
from pipeline.features.run import FEATURE_COLS, process_session_windows
from pipeline.ingestion.run import parse_events, parse_session_metadata

log = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
RESULTS_DIR = ROOT / "reports" / "adversarial"


# ---------------------------------------------------------------------------
# Loading + feature extraction (in-memory mini-pipeline)
# ---------------------------------------------------------------------------


def _features_from_synthetic_file(path: Path) -> pd.DataFrame:
    """Run ingestion + feature engineering on one synthetic JSON, in memory.

    Returns one DataFrame row per 30-second window, with ``cheat_label`` and
    ``cheat_source_file`` columns added for grouping in the benchmark.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta = parse_session_metadata(data, path)
    events_df = parse_events(data)
    if events_df.empty:
        return pd.DataFrame()

    norm_factor = (float(meta["sensitivity"]) * float(meta["dpi"])) / 800.0
    sess_events = events_df.sort_values("t")
    windows = process_session_windows(sess_events, norm_factor)

    if not windows:
        return pd.DataFrame()

    df = pd.DataFrame(windows)
    df["session_id"] = meta["session_id"]
    df["player"] = meta["player"]
    df["game"] = meta["game"]
    df["cheat_label"] = data.get("cheat_label", CHEAT_LEGIT)
    df["cheat_source_file"] = path.name
    return df


def load_synthetic_features(synthetic_dir: Path = SYNTHETIC_DIR) -> pd.DataFrame:
    """Build a feature DataFrame for the whole synthetic dataset."""
    files = sorted(synthetic_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files in {synthetic_dir}. "
            "Run `python -m pipeline.adversarial.generate_dataset` first."
        )

    frames = []
    for path in files:
        try:
            frames.append(_features_from_synthetic_file(path))
        except Exception as e:
            log.warning("Failed to process %s: %s", path.name, e)

    if not frames:
        raise RuntimeError("No synthetic features could be extracted")

    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


# ---------------------------------------------------------------------------
# Detector zoo + benchmarking
# ---------------------------------------------------------------------------


def _build_detectors() -> dict[str, object]:
    return {
        "IsolationForest": IsolationForest(
            n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1
        ),
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1
        ),
        "OneClassSVM": OneClassSVM(kernel="rbf", nu=0.05),
    }


@dataclass
class BenchmarkResult:
    detector: str
    cheat_label: str
    n_legit: int
    n_cheat: int
    roc_auc: float
    pr_auc: float
    detection_rate_at_fpr: float
    fpr_threshold: float
    mean_score_legit: float
    mean_score_cheat: float


def _detection_rate_at_fpr(y_true, scores, fpr_threshold: float) -> float:
    """Recall (true-positive rate) achieved when FPR is pinned to a target.

    ``scores`` should be such that *higher = more anomalous*. We invert
    sklearn's ``score_samples`` output before passing it here.
    """
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    # Find the largest operating point with FPR <= threshold
    valid = fpr <= fpr_threshold
    if not valid.any():
        return 0.0
    return float(tpr[valid].max())


def _benchmark_one_cheat(
    name: str,
    detector,
    legit_X: np.ndarray,
    cheat_X: np.ndarray,
    cheat_label: str,
    fpr_threshold: float,
) -> BenchmarkResult:
    """Train detector on legit-only, score both, compute metrics."""
    detector.fit(legit_X)

    legit_scores = -detector.score_samples(legit_X)  # higher = more anomalous
    cheat_scores = -detector.score_samples(cheat_X)

    scores = np.concatenate([legit_scores, cheat_scores])
    y_true = np.concatenate([np.zeros(len(legit_X)), np.ones(len(cheat_X))])

    try:
        roc = roc_auc_score(y_true, scores)
    except ValueError:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, scores)
    except ValueError:
        pr = float("nan")

    return BenchmarkResult(
        detector=name,
        cheat_label=cheat_label,
        n_legit=len(legit_X),
        n_cheat=len(cheat_X),
        roc_auc=float(roc),
        pr_auc=float(pr),
        detection_rate_at_fpr=_detection_rate_at_fpr(y_true, scores, fpr_threshold),
        fpr_threshold=fpr_threshold,
        mean_score_legit=float(np.mean(legit_scores)),
        mean_score_cheat=float(np.mean(cheat_scores)),
    )


def _session_scores(
    detector,
    legit_X: np.ndarray,
    cheat_X: np.ndarray,
    legit_sessions: np.ndarray,
    cheat_sessions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-window anomaly scores into per-session max scores.

    Production anti-cheat flags whole sessions, not individual windows: cheats
    are typically present in only a fraction of a session's windows, so the
    max anomaly score across a session is what matters. Per-window evaluation
    dilutes the signal with the many legit-looking windows inside cheat files.
    """
    legit_w = -detector.score_samples(legit_X)
    cheat_w = -detector.score_samples(cheat_X)

    # Aggregate: max score per session
    legit_df = pd.DataFrame({"session": legit_sessions, "score": legit_w})
    cheat_df = pd.DataFrame({"session": cheat_sessions, "score": cheat_w})
    legit_max = legit_df.groupby("session")["score"].max().to_numpy()
    cheat_max = cheat_df.groupby("session")["score"].max().to_numpy()

    scores = np.concatenate([legit_max, cheat_max])
    y_true = np.concatenate([np.zeros(len(legit_max)), np.ones(len(cheat_max))])
    return y_true, scores


def _benchmark_one_cheat_per_session(
    name: str,
    detector,
    legit_X: np.ndarray,
    cheat_X: np.ndarray,
    legit_sessions: np.ndarray,
    cheat_sessions: np.ndarray,
    cheat_label: str,
    fpr_threshold: float,
) -> BenchmarkResult:
    """Train detector on legit-only, score per-session, compute metrics."""
    detector.fit(legit_X)
    y_true, scores = _session_scores(
        detector, legit_X, cheat_X, legit_sessions, cheat_sessions
    )

    n_legit = int((y_true == 0).sum())
    n_cheat = int((y_true == 1).sum())

    try:
        roc = roc_auc_score(y_true, scores)
    except ValueError:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, scores)
    except ValueError:
        pr = float("nan")

    legit_mask = y_true == 0
    return BenchmarkResult(
        detector=name,
        cheat_label=cheat_label,
        n_legit=n_legit,
        n_cheat=n_cheat,
        roc_auc=float(roc),
        pr_auc=float(pr),
        detection_rate_at_fpr=_detection_rate_at_fpr(y_true, scores, fpr_threshold),
        fpr_threshold=fpr_threshold,
        mean_score_legit=float(np.mean(scores[legit_mask])),
        mean_score_cheat=float(np.mean(scores[~legit_mask])),
    )


def run_benchmark(
    features_df: pd.DataFrame,
    fpr_threshold: float = 0.05,
    detectors: dict | None = None,
    aggregation: str = "session_max",
) -> pd.DataFrame:
    """Train each detector on legit-only rows, score per cheat type.

    ``aggregation`` controls evaluation granularity:
      - ``"session_max"`` (default): collapse a session's per-window scores to
        their max, then evaluate per session. Matches production anti-cheat,
        where whole sessions get flagged.
      - ``"window"``: evaluate every window independently (the original
        Phase-3 behaviour — useful for diagnostics but dilutes cheat signal
        because most windows in a cheat-labelled file contain no cheat events).
    """
    if detectors is None:
        detectors = _build_detectors()

    X = features_df[FEATURE_COLS].fillna(0.0).to_numpy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    legit_mask = features_df["cheat_label"].eq(CHEAT_LEGIT).to_numpy()
    legit_X = X_scaled[legit_mask]
    legit_sessions = features_df.loc[legit_mask, "cheat_source_file"].to_numpy()
    if legit_X.shape[0] < 5:
        raise RuntimeError(
            "Not enough legit windows to train detectors "
            f"({legit_X.shape[0]} < 5). Generate more sessions first."
        )

    cheat_labels = sorted(set(features_df["cheat_label"]) - {CHEAT_LEGIT})
    if not cheat_labels:
        raise RuntimeError("No cheat-labelled rows in the dataset.")

    rows: list[BenchmarkResult] = []
    for name, detector in detectors.items():
        for cheat_label in cheat_labels:
            cheat_mask = features_df["cheat_label"].eq(cheat_label).to_numpy()
            cheat_X = X_scaled[cheat_mask]
            cheat_sessions = features_df.loc[cheat_mask, "cheat_source_file"].to_numpy()
            if cheat_X.shape[0] == 0:
                continue
            if aggregation == "session_max":
                rows.append(
                    _benchmark_one_cheat_per_session(
                        name,
                        detector,
                        legit_X,
                        cheat_X,
                        legit_sessions,
                        cheat_sessions,
                        cheat_label,
                        fpr_threshold,
                    )
                )
            else:
                rows.append(
                    _benchmark_one_cheat(
                        name, detector, legit_X, cheat_X, cheat_label, fpr_threshold
                    )
                )

    return pd.DataFrame([r.__dict__ for r in rows])


def compute_roc_curves(
    features_df: pd.DataFrame,
    detectors: dict | None = None,
    aggregation: str = "session_max",
) -> dict[tuple[str, str], dict]:
    """Return raw ROC curves keyed by (detector, cheat_label).

    Each value is a dict with ``fpr``, ``tpr``, ``thresholds``, ``auc`` —
    suitable for plotting a curve grid in the notebook. See ``run_benchmark``
    for the ``aggregation`` parameter.
    """
    if detectors is None:
        detectors = _build_detectors()

    X = features_df[FEATURE_COLS].fillna(0.0).to_numpy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    legit_mask = features_df["cheat_label"].eq(CHEAT_LEGIT).to_numpy()
    legit_X = X_scaled[legit_mask]
    legit_sessions = features_df.loc[legit_mask, "cheat_source_file"].to_numpy()
    cheat_labels = sorted(set(features_df["cheat_label"]) - {CHEAT_LEGIT})

    out: dict[tuple[str, str], dict] = {}
    for name, _ in detectors.items():
        det = _build_detectors()[name]
        det.fit(legit_X)
        for cheat_label in cheat_labels:
            cheat_mask = features_df["cheat_label"].eq(cheat_label).to_numpy()
            cheat_X = X_scaled[cheat_mask]
            cheat_sessions = features_df.loc[cheat_mask, "cheat_source_file"].to_numpy()
            if cheat_X.shape[0] == 0:
                continue
            if aggregation == "session_max":
                y_true, scores = _session_scores(
                    det, legit_X, cheat_X, legit_sessions, cheat_sessions
                )
            else:
                legit_scores = -det.score_samples(legit_X)
                cheat_scores = -det.score_samples(cheat_X)
                scores = np.concatenate([legit_scores, cheat_scores])
                y_true = np.concatenate([np.zeros(len(legit_X)), np.ones(len(cheat_X))])
            fpr, tpr, thresholds = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            out[(name, cheat_label)] = dict(
                fpr=fpr, tpr=tpr, thresholds=thresholds, auc=float(auc)
            )
    return out


def compute_pr_curves(
    features_df: pd.DataFrame,
    detectors: dict | None = None,
    aggregation: str = "session_max",
) -> dict[tuple[str, str], dict]:
    """Return raw precision-recall curves keyed by (detector, cheat_label).

    See ``run_benchmark`` for the ``aggregation`` parameter.
    """
    if detectors is None:
        detectors = _build_detectors()

    X = features_df[FEATURE_COLS].fillna(0.0).to_numpy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    legit_mask = features_df["cheat_label"].eq(CHEAT_LEGIT).to_numpy()
    legit_X = X_scaled[legit_mask]
    legit_sessions = features_df.loc[legit_mask, "cheat_source_file"].to_numpy()
    cheat_labels = sorted(set(features_df["cheat_label"]) - {CHEAT_LEGIT})

    out: dict[tuple[str, str], dict] = {}
    for name, _ in detectors.items():
        det = _build_detectors()[name]
        det.fit(legit_X)
        for cheat_label in cheat_labels:
            cheat_mask = features_df["cheat_label"].eq(cheat_label).to_numpy()
            cheat_X = X_scaled[cheat_mask]
            cheat_sessions = features_df.loc[cheat_mask, "cheat_source_file"].to_numpy()
            if cheat_X.shape[0] == 0:
                continue
            if aggregation == "session_max":
                y_true, scores = _session_scores(
                    det, legit_X, cheat_X, legit_sessions, cheat_sessions
                )
            else:
                legit_scores = -det.score_samples(legit_X)
                cheat_scores = -det.score_samples(cheat_X)
                scores = np.concatenate([legit_scores, cheat_scores])
                y_true = np.concatenate([np.zeros(len(legit_X)), np.ones(len(cheat_X))])
            precision, recall, thresholds = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            out[(name, cheat_label)] = dict(
                precision=precision, recall=recall, thresholds=thresholds, ap=float(ap)
            )
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _format_summary(results: pd.DataFrame) -> str:
    """Pretty-print a tabular benchmark summary."""
    pivot = results.pivot_table(
        index="detector",
        columns="cheat_label",
        values="roc_auc",
        aggfunc="first",
    )
    return pivot.round(3).to_string()


# ---------------------------------------------------------------------------
# LSTM Autoencoder benchmark (Phase 2)
# ---------------------------------------------------------------------------
#
# The classical detectors above consume per-window aggregated features. The
# LSTM-AE operates natively on the raw event stream, so it bypasses the
# feature pipeline entirely. We give it its own benchmark function and merge
# the results into the same BenchmarkResult format so the existing CSV /
# heatmap rendering keeps working.


def _load_session_tensors(synthetic_dir: Path = SYNTHETIC_DIR) -> dict[str, list]:
    """Load every synthetic session JSON into ``(label, file_name, tensor)`` tuples.

    Returns a dict keyed by cheat label, value = list of ``(file_name, tensor)``.
    Sessions with no events are dropped.
    """
    # Imported lazily so the classical benchmark can run without torch.
    from pipeline.sequences.preprocessing import session_to_event_tensor

    out: dict[str, list] = {}
    for path in sorted(synthetic_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tensor = session_to_event_tensor(data)
        if len(tensor) == 0:
            continue
        label = data.get("cheat_label", "legit")
        out.setdefault(label, []).append((path.name, tensor))
    return out


def run_lstm_ae_benchmark(
    synthetic_dir: Path = SYNTHETIC_DIR,
    *,
    chunk_length: int = 64,
    stride: int = 32,
    hidden_dim: int = 64,
    bottleneck_dim: int = 16,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 256,
    score_percentile: float = 95.0,
    fpr_threshold: float = 0.05,
    device: str = "auto",
    val_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Train an LSTM-AE on legit sessions, score every session per cheat type.

    Returns ``BenchmarkResult`` rows in the same format as :func:`run_benchmark`,
    one per cheat type **and** evaluation granularity:

    - ``detector="LSTMAutoencoder/chunk"`` — chunk-level AUC, using the
      ``cheat_segments`` field of each synthetic session to label individual
      chunks. This is the headline metric: it shows what the model actually
      learnt to flag.
    - ``detector="LSTMAutoencoder/session"`` — session-level AUC via
      ``score_percentile`` aggregation. Comparable to the classical
      detectors but typically much weaker — cheats affect a small minority
      of chunks in any session, and the legit-baseline's natural variance
      tail overlaps with the cheat signal at the session level. This gap
      motivates the multi-detector Bayesian aggregation planned in Phase 4.
    """
    # Lazy imports — torch + the LSTM module only needed for this path.
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from pipeline.models.lstm_ae import score_sequences, train_lstm_ae
    from pipeline.sequences.dataset import EventSequenceDataset
    from pipeline.sequences.preprocessing import (
        apply_normalizer,
        fit_normalizer,
        session_to_event_tensor,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    log.info("Loading synthetic session tensors from %s", synthetic_dir)
    by_label = _load_session_tensors(synthetic_dir)
    if "legit" not in by_label or not by_label["legit"]:
        raise RuntimeError("No legit sessions found — cannot train LSTM-AE")

    legit_items = by_label["legit"]
    log.info(
        "  %d legit / %s cheat sessions",
        len(legit_items),
        {k: len(v) for k, v in by_label.items() if k != "legit"},
    )

    # Train / val split on legit only
    n_val = max(1, int(round(len(legit_items) * val_fraction)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(legit_items))
    val_idx = set(perm[:n_val].tolist())
    train_legit = [legit_items[i] for i in range(len(legit_items)) if i not in val_idx]
    val_legit = [legit_items[i] for i in range(len(legit_items)) if i in val_idx]

    # Fit normalizer on training-fold tensors only
    stats = fit_normalizer([t for _, t in train_legit])

    def make_loader(items, shuffle):
        tensors = [apply_normalizer(t, stats) for _, t in items]
        sids = [name for name, _ in items]
        ds = EventSequenceDataset(
            tensors, chunk_length=chunk_length, stride=stride, session_ids=sids
        )
        if len(ds) == 0:
            return None
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True
        )

    # Train loader uses overlapping chunks; val uses non-overlapping so the
    # held-out signal isn't measured on near-duplicates.
    train_loader = make_loader(train_legit, shuffle=True)
    val_loader = make_loader(val_legit, shuffle=False)

    if train_loader is None:
        raise RuntimeError(
            "Train loader empty — every legit session is shorter than chunk_length"
        )

    log.info(
        "Training LSTM-AE (chunk_length=%d, stride=%d, hidden=%d, bottleneck=%d)",
        chunk_length,
        stride,
        hidden_dim,
        bottleneck_dim,
    )
    model, history = train_lstm_ae(
        train_loader,
        val_loader,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        device=device,
        log_every=5,
    )
    log.info(
        "Training complete — best val_loss=%.5f at epoch %d",
        history.best_val_loss,
        history.best_epoch,
    )

    # --- Chunk-level scoring: load every session, score every non-overlapping
    # chunk, and (for cheat files) flag chunks that overlap a cheat_segment.
    def chunk_data(path: Path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tensor = session_to_event_tensor(data)
        if len(tensor) < chunk_length:
            return None
        normalized = apply_normalizer(tensor, stats)
        n_chunks = len(normalized) // chunk_length
        chunks = np.stack(
            [
                normalized[i * chunk_length : (i + 1) * chunk_length]
                for i in range(n_chunks)
            ]
        )

        # Flag chunks overlapping any cheat_segment
        segs = data.get("cheat_segments") or []
        contains_cheat = np.zeros(n_chunks, dtype=bool)
        if segs and data.get("events"):
            times = np.array([ev.get("t", 0.0) for ev in data["events"]])
            in_seg = np.zeros(len(times), dtype=bool)
            for s_start, s_end in segs:
                in_seg |= (times >= s_start) & (times <= s_end)
            for i in range(n_chunks):
                contains_cheat[i] = in_seg[
                    i * chunk_length : (i + 1) * chunk_length
                ].any()

        chunk_scores = score_sequences(
            model,
            torch.from_numpy(chunks).float(),
            batch_size=batch_size,
            device=device,
        )
        return chunk_scores, contains_cheat

    # Aggregate chunk-level + session-level metrics per cheat type
    rows: list[BenchmarkResult] = []
    paths_by_label = {label: [] for label in by_label}
    for path in sorted(synthetic_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        paths_by_label.setdefault(data.get("cheat_label", "legit"), []).append(path)

    # Pre-score legit sessions once — used by both granularities
    legit_chunk_scores: list[np.ndarray] = []
    legit_session_scores: list[float] = []
    for path in paths_by_label.get("legit", []):
        out = chunk_data(path)
        if out is None:
            continue
        scores, _ = out
        legit_chunk_scores.append(scores)
        legit_session_scores.append(float(np.percentile(scores, score_percentile)))

    if not legit_chunk_scores:
        raise RuntimeError(
            "No legit session produced any chunks at the configured chunk_length"
        )

    legit_chunks_flat = np.concatenate(legit_chunk_scores)
    legit_session_arr = np.array(legit_session_scores, dtype=np.float64)

    for label in sorted(set(paths_by_label) - {"legit"}):
        cheat_chunk_scores_pos: list[float] = []  # chunks overlapping a cheat_segment
        cheat_chunk_scores_neg: list[float] = []  # chunks in the cheat file but clean
        cheat_session_scores: list[float] = []
        for path in paths_by_label[label]:
            out = chunk_data(path)
            if out is None:
                continue
            scores, contains_cheat = out
            cheat_chunk_scores_pos.extend(scores[contains_cheat])
            cheat_chunk_scores_neg.extend(scores[~contains_cheat])
            cheat_session_scores.append(float(np.percentile(scores, score_percentile)))

        # --- Chunk-level AUC: cheat-bearing chunks vs all legit chunks ---
        if cheat_chunk_scores_pos:
            chunk_y = np.concatenate(
                [
                    np.zeros(len(legit_chunks_flat)),
                    np.ones(len(cheat_chunk_scores_pos)),
                ]
            )
            chunk_s = np.concatenate(
                [legit_chunks_flat, np.array(cheat_chunk_scores_pos)]
            )
            try:
                chunk_roc = roc_auc_score(chunk_y, chunk_s)
                chunk_pr = average_precision_score(chunk_y, chunk_s)
                chunk_det = _detection_rate_at_fpr(chunk_y, chunk_s, fpr_threshold)
            except ValueError:
                chunk_roc = chunk_pr = chunk_det = float("nan")
            rows.append(
                BenchmarkResult(
                    detector="LSTMAutoencoder/chunk",
                    cheat_label=label,
                    n_legit=int(len(legit_chunks_flat)),
                    n_cheat=int(len(cheat_chunk_scores_pos)),
                    roc_auc=float(chunk_roc),
                    pr_auc=float(chunk_pr),
                    detection_rate_at_fpr=float(chunk_det),
                    fpr_threshold=fpr_threshold,
                    mean_score_legit=float(np.mean(legit_chunks_flat)),
                    mean_score_cheat=float(np.mean(cheat_chunk_scores_pos)),
                )
            )

        # --- Session-level AUC: per-session p95 ---
        if cheat_session_scores:
            sess_y = np.concatenate(
                [np.zeros(len(legit_session_arr)), np.ones(len(cheat_session_scores))]
            )
            sess_s = np.concatenate([legit_session_arr, np.array(cheat_session_scores)])
            try:
                sess_roc = roc_auc_score(sess_y, sess_s)
                sess_pr = average_precision_score(sess_y, sess_s)
                sess_det = _detection_rate_at_fpr(sess_y, sess_s, fpr_threshold)
            except ValueError:
                sess_roc = sess_pr = sess_det = float("nan")
            rows.append(
                BenchmarkResult(
                    detector="LSTMAutoencoder/session",
                    cheat_label=label,
                    n_legit=int(len(legit_session_arr)),
                    n_cheat=int(len(cheat_session_scores)),
                    roc_auc=float(sess_roc),
                    pr_auc=float(sess_pr),
                    detection_rate_at_fpr=float(sess_det),
                    fpr_threshold=fpr_threshold,
                    mean_score_legit=float(np.mean(legit_session_arr)),
                    mean_score_cheat=float(np.mean(cheat_session_scores)),
                )
            )

    return pd.DataFrame([r.__dict__ for r in rows])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run(include_lstm_ae: bool = True) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading synthetic features from %s", SYNTHETIC_DIR)
    df = load_synthetic_features()
    log.info(
        "  %d windows across %d files | cheat label counts: %s",
        len(df),
        df["cheat_source_file"].nunique(),
        df["cheat_label"].value_counts().to_dict(),
    )

    results = run_benchmark(df)

    if include_lstm_ae:
        log.info("\n--- LSTM autoencoder benchmark ---")
        try:
            lstm_results = run_lstm_ae_benchmark()
            results = pd.concat([results, lstm_results], ignore_index=True)
        except Exception as e:
            log.warning(
                "LSTM-AE benchmark failed: %s — continuing with classical only", e
            )

    out_path = RESULTS_DIR / "benchmark_results.csv"
    results.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)

    log.info("\nROC AUC by detector × cheat type:")
    log.info("\n%s", _format_summary(results))


if __name__ == "__main__":
    run()
