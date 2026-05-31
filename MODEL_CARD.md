# Model Card — BehaviorDNA

Two models ship in this project. This card documents both, their intended use,
measured performance on **real** GTA5 data, and — most importantly for an
anti-cheat context — their **limitations and the cost of being wrong**.

> Status: research / portfolio. Trained on a small real dataset (18 sessions, 3
> players) plus synthetic cheats. Numbers are **directional**, not production
> guarantees. See [docs/FINDINGS.md](docs/FINDINGS.md) for the honest narrative.

---

## 1. Player identification (behavioural biometric)

| | |
|---|---|
| **Type** | `LGBMClassifier` (multiclass) on 25 windowed behavioural features |
| **Input** | one 30 s window → 25 features (mouse kinematics/trajectory, click dynamics, keyboard, session aggregates), sens/DPI- + polling-normalised |
| **Output** | player label + calibrated class probabilities |
| **Code** | `pipeline/training/run.py`, `pipeline/features/run.py` |

**Metrics (real data, held-out test):** accuracy **0.853**, weighted F1 **0.862**
(3 players, 34 test windows). On the **same-hardware pair** (hydra vs dninix —
identical PC/settings, only the human differs) accuracy is **0.75** — the honest
behavioural-biometric number, since the 3-class figure is partly inflated by a
third player on different hardware ([FINDINGS](docs/FINDINGS.md)).

**Calibration:** ECE/Brier measured; isotonic improves Brier (0.275→0.224),
Platt does not (small-N fragility). See `notebooks/13_calibration.ipynb`.

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
  production/safety-critical use; generalising beyond the 3 enrolled players or
  beyond GTA5 mouse/keyboard play.
- **Users:** the project author / reviewers, offline.

## The cost of being wrong (why thresholds matter)

In anti-cheat the asymmetry is everything: **a false positive bans an innocent
player.** A detector at 95% TPR / 5% FPR wrongly flags 1 in 20 legitimate
players — unacceptable at population scale. Production systems tune to FPR
≤ 0.1% and treat a model score as *evidence*, not a verdict. This project
therefore (a) reports calibrated probabilities and ECE/Brier (so a threshold
*means* something), (b) keeps promotion to "Production" a deliberate, audited
step (`scripts/promote_model.py`), and (c) treats the live risk score as
advisory — it is never wired to an automated action.

## Inference cost

scikit-learn single-window latency p50 **1.40 ms** / p95 **1.90 ms**,
**~89k windows/s** batched (CPU) — comfortably real-time (one window per 30 s of
play). An ONNX-Runtime path is ~90× faster but the current LightGBM→ONNX export
is **numerically unfaithful** (probability MAE 0.13) and is gated/flagged, not
used for scoring (`scripts/benchmark_inference.py`, [FINDINGS](docs/FINDINGS.md)).

## Limitations & caveats

- **Tiny dataset:** 18 sessions / 3 players. Metrics have wide confidence
  intervals; the 25-feature model is **over-parameterised** at this N (ablation:
  dropping a feature family can *raise* validation accuracy).
- **Hardware confound:** cross-hardware identification is optimistic; only the
  same-hardware number is a clean behavioural result.
- **Sensing-layer blind spot:** this is **input-based** detection. A
  *memory-only* aimbot that never moves the OS cursor is invisible to it — which
  is why production anti-cheat also does memory/integrity scanning. Input
  biometrics is one layer, not the whole stack.
- **Synthetic cheats:** detection is validated against synthetic injection;
  real continuous-cheat recordings are pending (`docs/CHEAT_DATA_COLLECTION.md`).
- **ONNX export** is not production-trustworthy for the LightGBM model (above).

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
