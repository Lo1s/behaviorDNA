# BehaviorDNA — Portfolio Roadmap

Targeted enhancements aimed at ML/AI roles at anti-cheat companies (Anybrain, Irdeto, BattlEye R&D, Riot's anti-cheat team, etc.).

The repo already has a solid classical-ML foundation: 18 features, 7 model types, batch API, Streamlit dashboard, DVC pipeline, CI, MLflow tracking, external dataset analysis (CS2CD + CaptchaSolve30k).

This roadmap adds the four things hiring managers at AI-focused anti-cheat companies look for:

1. **Deep learning on raw input sequences** (LSTM autoencoder)
2. **Adversarial / domain knowledge** (synthetic cheat generation + detection benchmarks)
3. **Production MLOps maturity** (streaming inference, drift detection, model registry)
4. **Research rigor** (calibration, ablation, explainability)

---

## Status at a glance

| Phase | Goal | Status |
|---|---|---|
| 1. [Trajectory & temporal features](#phase-1--trajectory--temporal-features) | 7 new anti-cheat-relevant features | ⬜ Not started |
| 2. [LSTM autoencoder](#phase-2--lstm-autoencoder-for-anomaly-detection) | Deep-learning sequence model | ⬜ Not started |
| 3. [Adversarial bots](#phase-3--adversarial-bot-generation--detection-benchmark) | Synthetic cheat generator + detection benchmark | ✅ Done |
| 4. [Streaming + risk aggregation](#phase-4--session-level-risk-aggregation--streaming-api) | Live inference dashboard | ⬜ Not started |
| 5. [Statistical rigor & MLOps](#phase-5--statistical-rigor--mlops-polish) | SHAP, calibration, drift, registry | ⬜ Not started |

Legend: ⬜ Not started · 🚧 In progress · ✅ Done

---

## Phase 1 — Trajectory & Temporal Features

**Why:** Current 18 features are per-window aggregates of counts and timing. Anti-cheat detection in practice leans on **geometric trajectory features** (curvature, turn angles, flick patterns) and **reaction-time features**. These distinguish aimbots from humans.

**New features (7):**
- `mouse_curvature_mean`, `mouse_curvature_std` — average turn angle between consecutive 3-point mouse segments
- `flick_count` — fast-acceleration mouse bursts (>3σ above session mean)
- `flick_precision` — dispersion of post-flick positions (aimbots snap perfectly)
- `direction_changes_per_sec` — velocity-vector sign flips per second
- `path_efficiency` — Euclidean / total-path ratio (smoother = more bot-like)
- `inter_click_movement` — mouse-movement distance between consecutive clicks
- `keystroke_overlap_ratio` — fraction of time multiple keys held simultaneously

**Deliverables:**
- [ ] 7 features added to `pipeline/features/run.py` and `FEATURE_COLS`
- [ ] Tests in `tests/test_features.py`
- [ ] `notebooks/08_trajectory_features.ipynb` — derivation, visualization, discrimination analysis
- [ ] `docs/FEATURES.md` — explanation of each feature's anti-cheat relevance

---

## Phase 2 — LSTM Autoencoder for Anomaly Detection

**Why:** Deep sequence models on raw event streams (not aggregated features) are how modern anti-cheat tackles behavioral biometrics. Unsupervised autoencoder fits perfectly: train on legit sessions, flag anything that reconstructs poorly.

**Approach:**
- Input: variable-length sequence of `(dt, dx, dy, event_type_onehot)` from raw events
- Architecture: bidirectional LSTM encoder → bottleneck → LSTM decoder
- Loss: per-step MSE reconstruction
- Anomaly score: mean reconstruction error over the session

**Notebook 09 is a thorough tutorial** with diagrams and visualizations covering: what is an autoencoder, why LSTM, building the sequence dataset, PyTorch dataloader & padding, model architecture, training loop, watching it learn, latent-space exploration, anomaly scoring, threshold selection, ablation.

**Deliverables:**
- [ ] `pipeline/sequences/dataset.py` + `preprocessing.py`
- [ ] `pipeline/models/lstm_ae.py`
- [ ] Integration into `pipeline/training/run.py` (selectable via `configs/training.yaml`)
- [ ] `notebooks/09_lstm_autoencoder.ipynb` — 11-step tutorial
- [ ] `docs/LSTM_AE.md` — architecture diagram + write-up
- [ ] MLflow logging of reconstruction-error curves

---

## Phase 3 — Adversarial Bot Generation & Detection Benchmark

**Why:** The killer feature for an anti-cheat portfolio. Generating synthetic cheating trajectories and benchmarking detectors against them demonstrates you understand the adversarial nature of the problem.

**Approach:**
- **Aimbot**: perfect-aim snapping with configurable smoothing
- **Triggerbot**: instant click when crosshair crosses target (zero reaction latency)
- **Macro**: repeated keystroke patterns at perfectly regular intervals
- Overlay bot actions on baseline legit sessions to produce labeled hybrid sessions
- Benchmark IsolationForest, LOF, OneClassSVM (and later LSTM-AE) per cheat type

**Notebook 10 is a thorough tutorial** with math and visualizations covering: what is a cheat mathematically, aimbot simulator design + visualization, triggerbot design, macro design + FFT signature, hybrid session construction, benchmark setup, ROC grid, per-cheat analysis, failure analysis, production discussion.

**No real recordings needed** — fully synthetic. Can proceed in parallel with data collection.

**Deliverables:**
- [x] `pipeline/adversarial/__init__.py`
- [x] `pipeline/adversarial/bot_generator.py` — three cheat-type generators
- [x] `pipeline/adversarial/generate_dataset.py` — builds labelled dataset from `data/raw/`
- [x] `pipeline/adversarial/benchmark.py` — runs all detectors against synthetic sessions
- [x] `data/synthetic/` — 90 generated labelled JSON sessions (15 legit + 45 aimbot + 15 triggerbot + 15 macro)
- [x] `notebooks/10_adversarial_bots.ipynb` — 11-step tutorial with ROC grid + event-level analysis
- [x] `docs/ADVERSARIAL.md` — methodology, results, lessons learned
- [x] `tests/test_adversarial.py` — 11 unit tests covering all three generators

**Key finding:** the current 18 window-level features cannot discriminate any of the three cheat types from legit play (AUC ≈ 0.5). Event-level signals (curvature, click reaction time, FFT) discriminate cleanly. This motivates Phases 1 and 2.

---

## Phase 4 — Session-Level Risk Aggregation + Streaming API

**Why:** Real anti-cheat systems aggregate evidence across a session into a Bayesian-style confidence score, and they score events as they arrive (streaming) rather than at end-of-session.

**What this means concretely:** real-time scoring during actual gameplay. Three modes:
1. **Live mode**: recorder pushes events via WebSocket as you play → dashboard updates live
2. **Replay mode**: stream a recorded session JSON at original timestamps → dashboard updates as if live
3. **Synthetic-cheat replay**: inject Phase 3 aimbot into a legit session, watch risk score spike

**Approach:**
- Bayesian log-likelihood accumulation across windows for player-identity confidence + cheat-risk score
- WebSocket endpoint in FastAPI streams per-window probabilities + accumulated risk
- Recorder gets a `--stream-to ws://...` flag (additive, JSON saving still happens)

**Deliverables:**
- [ ] `pipeline/inference/aggregator.py` — Bayesian risk aggregator
- [ ] `api/streaming.py` — WebSocket endpoint
- [ ] `collector/streamer.py` — async WebSocket client used by the recorder
- [ ] `scripts/replay_session.py` — replays a recorded JSON through the API
- [ ] Dashboard "Live session" tab
- [ ] `notebooks/11_session_risk.ipynb` — aggregator derivation + offline analysis
- [ ] Tests in `tests/test_streaming.py`
- [ ] Demo video / GIF embedded in README

---

## Phase 5 — Statistical Rigor & MLOps Polish

**Why:** Tie everything together with scientific rigor and production-ready observability. Reads as "ready to ship" rather than "promising prototype."

**Deliverables:**

**5a. SHAP explainability** — `notebooks/12_explainability.ipynb`
- [ ] Per-prediction SHAP values for LightGBM + LSTM-AE
- [ ] Population-level feature importance heatmaps per player
- [ ] "Why was this session flagged?" walkthroughs

**5b. Calibration** — `notebooks/13_calibration.ipynb`
- [ ] Reliability diagrams for each classifier
- [ ] Brier score + ECE
- [ ] Isotonic / Platt scaling applied, before/after comparison

**5c. Drift detection** — `pipeline/monitoring/drift.py` + `notebooks/14_drift.ipynb`
- [ ] KS test on feature distributions (training set vs latest week)
- [ ] PSI (Population Stability Index)
- [ ] CLI: `python -m pipeline.monitoring.drift`

**5d. Ablation study** — `notebooks/15_ablation.ipynb`
- [ ] Remove each feature group, re-evaluate, heatmap marginal contribution

**5e. Model registry hardening**
- [ ] MLflow Model Registry stages (Staging → Production)
- [ ] `scripts/promote_model.py` — promotes best run based on test accuracy
- [ ] `docs/MLOPS.md` — documents drift monitoring + registry promotion

---

## Cross-cutting

**README rewrite at end of Phase 5** — new sections for anti-cheat features, deep learning, production architecture, model rigor; updated architecture diagram; portfolio video embed.

**`docs/RESEARCH_LOG.md`** (updated throughout) — research questions answered with measured results:
- "How few sessions per player do we need for reliable identification?"
- "Can fingerprints transfer across games?"
- "What's the latency-accuracy tradeoff at different window sizes?"
- "Does the LSTM autoencoder catch synthetic aimbots better than IsolationForest?"
- "How does session-level Bayesian aggregation compare to single-window confidence?"

---

## Data dependency map

| Phase | Needs real recordings? | Can start before recordings arrive? |
|---|---|---|
| 1. Trajectory features | helpful but not blocking | ✅ yes — works with existing data |
| 2. LSTM autoencoder | helpful for training | ✅ architecture + tutorial can start; final training after recordings |
| 3. Adversarial bots | no | ✅ fully synthetic |
| 4. Streaming + aggregator | no | ✅ uses any session |
| 5. Statistical rigor | yes, ideally | partial — works better with more data |

**Order if recordings are delayed:** 3 → 4 → 1 → 2 → 5.

---

## Demo path (end of roadmap)

**60-second elevator demo:** live dashboard with replayed session + synthetic aimbot injected at minute 3, cheat-risk line spikes visibly.

**15-minute deep-dive demo (technical interview):**
*Foundation:* README → notebook 01 (raw data) → notebook 02 + `pipeline/features/run.py` → notebook 05 (external datasets) → notebook 06 (model comparison) → notebook 07 (behavioral differentiation).
*Anti-cheat-specific:* notebook 10 (adversarial bots) → notebook 09 (LSTM tutorial) → live dashboard (Phase 4) → notebook 12 (SHAP).
*Production:* `docs/MLOPS.md` → CI badge + workflow.

**Portfolio video (3 min, OBS-recorded):** what the project is → live demo playing GTA → inject synthetic aimbot, risk spikes → notebook 09 LSTM walkthrough → `docs/MLOPS.md` scroll. Embed in README, link from LinkedIn / CV.
