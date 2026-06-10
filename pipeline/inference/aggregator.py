"""
pipeline/inference/aggregator.py
================================
Bayesian multi-detector risk aggregator.

Why this module exists
----------------------
After Phases 1–3, every cheat type is detectable by some detector, but no
single detector wins everything at the session level. Triggerbot belongs to
``OneClassSVM`` (session AUC 0.87); aimbot belongs to the LSTM-AE at the
chunk level (chunk AUC 0.78, session 0.50 alone); macro is somewhere in
between. The honest production move is to combine them — and that is what
``RiskAggregator`` does.

The math, in plain English
--------------------------
1. **Calibrate each detector**: every detector emits scores on its own
   scale (negative log-likelihoods, anomaly distances, MSEs, …). We fit a
   monotonic *isotonic regression* per detector that maps "raw score" →
   "probability this is a cheat" using a held-out labelled sample.
   ``IsotonicCalibrator`` is just a thin wrapper around
   ``sklearn.isotonic.IsotonicRegression``.

2. **Convert each probability to log-odds**: the "logit" function is
   ``logit(p) = log(p / (1 − p))``. A probability of 0.5 becomes 0; 0.9
   becomes ~2.2; 0.99 becomes ~4.6. Log-odds is the natural additive scale
   for combining independent evidence.

3. **Combine via a Naive-Bayes sum**: assuming detectors are conditionally
   independent given "session is a cheat", the posterior log-odds equal
   the sum of per-detector log-odds **plus a prior log-odds term**::

        logit(P(cheat | scores)) = Σ_i logit(p_i)  +  prior_logit

   where ``prior_logit = log(prior_cheat_rate / (1 − prior_cheat_rate))``.

4. **Why the prior matters**: without one, the formula assumes a 50%
   base-rate of cheating, which is absurd. With a 5%-cheat assumption
   the prior is ``log(0.05 / 0.95) ≈ -2.94``. The prior is a head-start
   *subtraction*: detectors then have to provide enough evidence to lift
   the posterior back above 0 (= 50% probability). This is what keeps a
   weak signal from one detector from flagging a session by itself.

5. **Sigmoid back to a probability**: apply ``sigmoid`` to the combined
   log-odds to get the final session risk ∈ [0, 1].

Worked example
--------------
Three detectors, all calibrated, prior = 5%::

    detector       raw_score   p_i = calibrated_prob   logit(p_i)
    --------       ---------   ---------------------   ----------
    IsolationFor.  0.42        0.10                    -2.20
    OneClassSVM    -0.05       0.30                    -0.85
    LSTM-AE        2.10        0.85                    +1.73

    Σ logit(p_i)        =  -2.20 - 0.85 + 1.73  =  -1.32
    prior_logit         =  log(0.05 / 0.95)     ≈  -2.94
    posterior_logit     =  -1.32 + (-2.94)      =  -4.26
    posterior_risk      =  sigmoid(-4.26)        ≈  0.014   ← low risk

Same example but every detector says 0.9::

    Σ logit(p_i)        =  3 × 2.20             =  +6.59
    posterior_logit     =  6.59 + (-2.94)        =  +3.65
    posterior_risk      =  sigmoid(3.65)         ≈  0.975   ← clearly flag

Three independent strong signals override the conservative prior; one
medium signal does not. That's the point of multi-detector aggregation.

Public API
----------
- ``IsotonicCalibrator``  — fit / predict_proba over one detector
- ``RiskAggregator``      — fits a calibrator per detector, then
                             ``aggregate(scores_dict) -> float`` returns
                             ``P(cheat | scores)`` in [0, 1].
- ``fit_aggregator_from_synthetic(synthetic_dir, …)`` — convenience that
  pulls per-session scores from every detector for every synthetic
  session, then fits the aggregator on them. Used by the benchmark and
  by the streaming API's startup hook.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IsotonicCalibrator — per-detector score → P(cheat)
# ---------------------------------------------------------------------------


class IsotonicCalibrator:
    """Maps an unconstrained anomaly score to a calibrated cheat probability.

    Wraps ``sklearn.isotonic.IsotonicRegression`` with two tweaks:

    1. **Clipping**: probabilities are clamped to ``[eps, 1 - eps]`` so that
       ``logit()`` stays finite when we feed them to the aggregator.
    2. **Untrained fallback**: ``predict_proba`` returns 0.5 for any input
       if ``fit`` hasn't been called. Lets callers treat the calibrator as
       "no evidence yet" before training data is available.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        self.eps = float(eps)
        self._iso: IsotonicRegression | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        if scores.shape != labels.shape:
            raise ValueError(
                f"scores and labels must match shape, got {scores.shape} vs {labels.shape}"
            )
        if scores.size == 0:
            raise ValueError("Cannot fit IsotonicCalibrator on empty input")
        self._iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True
        )
        self._iso.fit(scores, labels)
        return self

    def predict_proba(self, scores: np.ndarray | float) -> np.ndarray:
        if self._iso is None:
            arr = np.atleast_1d(np.asarray(scores, dtype=np.float64))
            return np.full(arr.shape, 0.5, dtype=np.float64)
        arr = np.atleast_1d(np.asarray(scores, dtype=np.float64).ravel())
        p = self._iso.predict(arr)
        return np.clip(p, self.eps, 1.0 - self.eps)


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    """Log-odds — see module docstring."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# RiskAggregator — Naive-Bayes log-odds combination
# ---------------------------------------------------------------------------


@dataclass
class RiskAggregator:
    """Combine per-detector session scores into one calibrated risk.

    Typical use:

    >>> agg = RiskAggregator(prior_cheat_rate=0.05)
    >>> agg.fit({
    ...     "ifor": (np.array([...]), np.array([0, 0, 1, 1, ...])),
    ...     "ocsvm": (np.array([...]), np.array([0, 0, 1, 1, ...])),
    ...     "lstm_chunk": (np.array([...]), np.array([0, 0, 1, 1, ...])),
    ... })
    >>> risk = agg.aggregate({"ifor": 0.42, "ocsvm": -0.05, "lstm_chunk": 2.10})
    >>> # risk ∈ [0, 1]

    Detector names are arbitrary strings; whichever names appear in ``fit``
    are the ones ``aggregate`` knows about. Missing detectors at scoring
    time are treated as "no evidence" (skipped).
    """

    prior_cheat_rate: float = 0.05
    calibrators: dict[str, IsotonicCalibrator] = field(default_factory=dict)

    @property
    def prior_logit(self) -> float:
        p = float(np.clip(self.prior_cheat_rate, 1e-6, 1 - 1e-6))
        return math.log(p / (1.0 - p))

    def fit(
        self, training_data: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> "RiskAggregator":
        """Fit one isotonic calibrator per detector.

        ``training_data`` maps ``detector_name -> (scores, labels)``.
        Labels: 0 = legit, 1 = cheat. Same labels across detectors are fine
        (typically the same set of sessions).
        """
        if not training_data:
            raise ValueError("training_data must contain at least one detector")
        self.calibrators = {}
        for name, (scores, labels) in training_data.items():
            cal = IsotonicCalibrator().fit(scores, labels)
            self.calibrators[name] = cal
        return self

    def aggregate(self, scores: dict[str, float]) -> float:
        """Combine per-detector session scores → one risk in [0, 1].

        Detectors not in ``self.calibrators`` are ignored. NaN scores are
        ignored (treated as missing evidence).
        """
        log_odds = self.prior_logit
        for name, raw in scores.items():
            if name not in self.calibrators:
                continue
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                continue
            p = float(self.calibrators[name].predict_proba(np.asarray([raw]))[0])
            log_odds += float(_logit(p))
        return float(_sigmoid(log_odds))

    def aggregate_many(
        self, scores_by_session: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Convenience: aggregate a batch of sessions in one call."""
        return {
            sid: self.aggregate(scores) for sid, scores in scores_by_session.items()
        }

    def explain(self, scores: dict[str, float]) -> dict:
        """Return per-detector log-odds contributions for explainability.

        Useful for the dashboard's stacked-area chart: shows which detectors
        pushed the risk up or down for a given session.
        """
        contribs: dict[str, float] = {}
        for name, raw in scores.items():
            if name not in self.calibrators or raw is None:
                continue
            if isinstance(raw, float) and math.isnan(raw):
                continue
            p = float(self.calibrators[name].predict_proba(np.asarray([raw]))[0])
            contribs[name] = float(_logit(p))
        total = self.prior_logit + sum(contribs.values())
        return {
            "prior_logit": self.prior_logit,
            "per_detector_logit": contribs,
            "posterior_logit": total,
            "posterior_risk": float(_sigmoid(total)),
        }


# ---------------------------------------------------------------------------
# Convenience: fit an aggregator from the synthetic-cheat dataset
# ---------------------------------------------------------------------------


def fit_aggregator_from_synthetic(
    synthetic_dir: Path,
    *,
    prior_cheat_rate: float = 0.05,
    detector_names: list[str] | None = None,
    seed: int = 42,
) -> tuple[RiskAggregator, dict]:
    """Fit a ``RiskAggregator`` using per-session scores from every detector.

    Runs the existing classical + LSTM-AE benchmarks against the synthetic
    dataset to collect (session, detector → score, label) records, then
    fits the calibrators.

    Returns
    -------
    (aggregator, training_data)
        - ``aggregator`` — fitted ``RiskAggregator`` ready to use.
        - ``training_data`` — the raw per-detector ``(scores, labels)`` pairs
          used for fitting, so callers can inspect / log them.
    """
    # Imported lazily to keep this module light when only the aggregator math
    # is needed.
    from sklearn.preprocessing import StandardScaler

    from pipeline.adversarial.benchmark import (
        _build_detectors,
        _load_session_tensors,
        _session_scores,
        load_synthetic_features,
    )
    from pipeline.features.run import CHEAT_FEATURE_COLS

    # --- Classical detector session scores from the 25-D feature pipeline ---
    feats = load_synthetic_features(synthetic_dir)
    X = feats[CHEAT_FEATURE_COLS].fillna(0.0).to_numpy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    legit_mask = feats["cheat_label"].eq("legit").to_numpy()
    legit_X = X_scaled[legit_mask]
    legit_sessions = feats.loc[legit_mask, "cheat_source_file"].to_numpy()

    cheat_labels = sorted(set(feats["cheat_label"]) - {"legit"})

    detectors = _build_detectors()
    if detector_names is not None:
        detectors = {k: v for k, v in detectors.items() if k in detector_names}

    training_data: dict[str, tuple[list[float], list[int]]] = {
        name: ([], []) for name in detectors
    }

    # For each classical detector: get per-session-max score for legit + each cheat type
    for det_name, detector in detectors.items():
        detector.fit(legit_X)
        for cheat_label in cheat_labels:
            cheat_mask = feats["cheat_label"].eq(cheat_label).to_numpy()
            cheat_X = X_scaled[cheat_mask]
            cheat_sessions = feats.loc[cheat_mask, "cheat_source_file"].to_numpy()
            if cheat_X.shape[0] == 0:
                continue
            y_true, scores = _session_scores(
                detector, legit_X, cheat_X, legit_sessions, cheat_sessions
            )
            # _session_scores already aggregated to per-session max
            for score, label in zip(scores, y_true):
                training_data[det_name][0].append(float(score))
                training_data[det_name][1].append(int(label))

    # --- LSTM-AE per-session p95 score ---
    # Run the LSTM-AE benchmark in scoring-only mode, then read the per-session
    # scores from the BenchmarkResult rows it produced + recompute the underlying
    # per-session scores. Simpler: replicate the per-session p95 path directly.
    try:
        import torch

        from pipeline.models.lstm_ae import (
            LSTM_AE_WEIGHTS_NAME,
            load_lstm_ae,
            score_sequences,
        )
        from pipeline.sequences.preprocessing import apply_normalizer

        model_dir = Path(synthetic_dir).resolve().parent.parent / "models"
        if (model_dir / LSTM_AE_WEIGHTS_NAME).exists():
            model, stats, meta = load_lstm_ae(model_dir, device="auto")
            chunk_length = int((meta.get("config") or {}).get("chunk_length", 64))
            tensors_by_label = _load_session_tensors(synthetic_dir)
            lstm_scores: list[float] = []
            lstm_labels: list[int] = []
            for label, items in tensors_by_label.items():
                is_cheat = 0 if label == "legit" else 1
                for _name, tensor in items:
                    normalized = apply_normalizer(tensor, stats)
                    if len(normalized) < chunk_length:
                        continue
                    n_chunks = len(normalized) // chunk_length
                    chunks = np.stack(
                        [
                            normalized[i * chunk_length : (i + 1) * chunk_length]
                            for i in range(n_chunks)
                        ]
                    )
                    scores = score_sequences(
                        model,
                        torch.from_numpy(chunks).float(),
                        batch_size=256,
                        device="auto",
                    )
                    lstm_scores.append(float(np.percentile(scores, 95)))
                    lstm_labels.append(is_cheat)
            if lstm_scores:
                training_data["LSTMAutoencoder"] = (lstm_scores, lstm_labels)
        else:
            log.warning(
                "LSTM-AE artifact not found at %s — aggregator will only use classical detectors",
                model_dir,
            )
    except ImportError:
        log.warning(
            "torch not available — aggregator will only use classical detectors"
        )

    training_data_np = {
        name: (np.asarray(s, dtype=np.float64), np.asarray(labs, dtype=np.int64))
        for name, (s, labs) in training_data.items()
        if s
    }

    aggregator = RiskAggregator(prior_cheat_rate=prior_cheat_rate).fit(training_data_np)
    return aggregator, training_data_np
