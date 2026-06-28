# Model Card — BehaviorDNA

Two models ship in this project. This card documents both, their intended use,
measured performance on **real** GTA5 data, and — most importantly for an
anti-cheat context — their **limitations and the cost of being wrong**.

> Status: research / portfolio. Trained on a small real dataset (22 sessions, 4
> players) plus synthetic cheats. Numbers are **directional**, not production
> guarantees. See [docs/FINDINGS.md](docs/FINDINGS.md) for the honest narrative.

---

## 1. Player identification (behavioural biometric)

| | |
|---|---|
| **Type** | `LGBMClassifier` (multiclass) on the 25-feature `ID_FEATURE_COLS` set (decoupled from the cheat detectors' 30 — [docs/SIGNALS.md](docs/SIGNALS.md)) |
| **Input** | one 30 s window → 25 features (mouse kinematics/trajectory, click dynamics, keyboard, session aggregates), sens/DPI- + polling-normalised |
| **Output** | player label + raw class probabilities (`predict_proba`; **uncalibrated** at serving — see Calibration below) |
| **Code** | `pipeline/training/run.py`, `pipeline/features/run.py` |

**Metrics (real data, held-out test):** accuracy **0.72** (95% CI 0.60–0.83),
weighted F1 **0.73** (95% CI 0.62–0.85) — 4 players, 53 test windows,
2000-resample window bootstrap. The intervals are wide because the test set is
small, and they are quoted for exactly that reason; `reports/eval_metrics.json`
is the source of truth (the README results block regenerates from it). On the
**same-hardware pair** (hydra vs dninix — identical PC/settings, only the human
differs) accuracy is **0.75** — the honest behavioural-biometric number, since
the multi-class figure is partly aided by players on distinct hardware (shotik
especially; the 4th player ropyk is *not* trivially separable, evidence the
normalisation works — [FINDINGS](docs/FINDINGS.md)).

**Calibration:** ECE/Brier are *measured* (isotonic improves Brier 0.275→0.224;
Platt does not — small-N fragility); see `notebooks/13_calibration.ipynb`. This
is a **diagnostic only**: the served artifact (`models/model.pkl`) and the API
return raw `predict_proba`, **not** calibrated probabilities. No calibrator is
persisted into the serving path because the validation fold (62 windows) is too
small to fit a trustworthy one — treat the API probabilities as uncalibrated
scores, not thresholdable likelihoods.

## 2. Cheat detection (anomaly)

| | |
|---|---|
| **Type** | Bidirectional **LSTM autoencoder** on raw 8-channel event sequences (64-event chunks) |
| **Signal** | reconstruction error — legit play reconstructs well, cheat chunks poorly |
| **Code** | `pipeline/models/lstm_ae.py`, `pipeline/adversarial/benchmark.py` |

**Metrics (real legit data + synthetic cheats), chunk-level ROC AUC:** triggerbot
**0.93**, aimbot **0.79**, macro **0.60**; classical hand-crafted window features
sit at **~0.50 (chance)** for aimbot — the motivation for the sequence model.
**Session-level detection is ~0.50** (does not separate) on the current
synthetic data — a documented ceiling, not a tuning gap ([FINDINGS](docs/FINDINGS.md)).

---

## Intended use

- **In scope:** research into input-based behavioural biometrics and automation
  detection; a portfolio demonstration of an end-to-end ML/MLOps system.
- **Out of scope:** issuing real bans or moderation actions; any
  production/safety-critical use; generalising beyond the 4 enrolled players or
  beyond GTA5 mouse/keyboard play.
- **Users:** the project author / reviewers, offline.

## The cost of being wrong (why thresholds matter)

In anti-cheat the asymmetry is everything: **a false positive bans an innocent
player.** A detector at 95% TPR / 5% FPR wrongly flags 1 in 20 legitimate
players — unacceptable at population scale. Production systems tune to FPR
≤ 0.1% and treat a model score as *evidence*, not a verdict. This project
therefore (a) *measures* probability calibration (ECE/Brier) so the gap to a
thresholdable probability is quantified — though the served API returns raw,
uncalibrated probabilities (§1), (b) keeps promotion to "Production" a deliberate, audited
step (`scripts/promote_model.py`), and (c) treats the live risk score as
advisory — it is never wired to an automated action.

## Inference cost

scikit-learn single-window latency p50 **1.40 ms** / p95 **1.90 ms**,
**~89k windows/s** batched (CPU) — comfortably real-time (one window per 30 s of
play). The ONNX-Runtime path is faster still and — after an earlier float32
fidelity bug was found and fixed — is now a **bit-faithful float64 export**
(probability MAE ~1e-8, 100% label agreement), guarded by a CI parity regression
test (`pipeline/onnx_export.py`, `tests/test_onnx_export.py`, [FINDINGS #7](docs/FINDINGS.md)).

## Limitations & caveats

- **Tiny dataset:** 22 sessions / 4 players. Metrics have wide confidence
  intervals; the 25-feature model is **over-parameterised** at this N (ablation:
  dropping a feature family can *raise* validation accuracy).
- **Hardware confound:** cross-hardware identification is optimistic; only the
  same-hardware number is a clean behavioural result.
- **Sensing-layer blind spot:** this is **input-based** detection. A
  *memory-only* aimbot that never moves the OS cursor is invisible to it — which
  is why production anti-cheat also does memory/integrity scanning. Input
  biometrics is one layer, not the whole stack. Full observable-vs-evasion map:
  [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
- **Synthetic cheats:** the headline detection metrics use synthetic cheat
  injection on real legit play. Three real continuous-cheat recordings
  (F8/F9/F10-labelled) now exist; the per-type real-cheat benchmark on them is
  the next step (`docs/CHEAT_DATA_COLLECTION.md`).

## Ethics & data

- Recordings are from a small set of consenting participants for this research;
  raw data is DVC-tracked (not in git) and not redistributed.
- Cheat *signatures* are generated offline (single-player) with a controllable
  harness that has **no target acquisition, no memory reads, no networking**
  (`collector/cheat_sim.py`) — for training a **detector**, never to gain an
  advantage over a human. See [docs/ETHICS.md](docs/ETHICS.md).

## How to reproduce

`dvc repro` (pipeline) · `python -m pipeline.adversarial.benchmark` (detection) ·
`python -m scripts.benchmark_inference` (latency) · notebooks 12–15
(explainability / calibration / drift / ablation).
