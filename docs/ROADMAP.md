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
| 1. [Trajectory & temporal features](#phase-1--trajectory--temporal-features) | 7 new anti-cheat-relevant features | ✅ Done |
| 1.5. [Feature expansion (optional)](#phase-15--feature-expansion-optional) | Backlog of further feature ideas, revisited after Phase 5 | 📝 Backlog |
| 2. [LSTM autoencoder](#phase-2--lstm-autoencoder-for-anomaly-detection) | Deep-learning sequence model | ✅ Done |
| 3. [Adversarial bots](#phase-3--adversarial-bot-generation--detection-benchmark) | Synthetic cheat generator + detection benchmark | ✅ Done |
| 4. [Streaming + risk aggregation](#phase-4--session-level-risk-aggregation--streaming-api) | Bayesian multi-detector aggregator + WebSocket API + live dashboard | ✅ Done |
| 4.1. [Live recorder + multi-user backlog](#phase-41--live-recorder--multi-user-backlog) | Phase 4 follow-ups (live recorder, WS auth, MLflow logging) | 📝 Backlog |
| 5. [Statistical rigor & MLOps](#phase-5--statistical-rigor--mlops-polish) | SHAP, calibration, drift, registry | 🚧 5c drift done; rest not started |

Legend: ⬜ Not started · 🚧 In progress · ✅ Done · 📝 Backlog

> **Pre-recording readiness (done):** ahead of the real GTA recordings, shipped data-independent infra — drift detection (5c), a recording QC gate (`scripts/validate_recordings.py`), polling-rate normalization, and dependency fixes. See the [Pre-recording readiness](#pre-recording-readiness-done-while-waiting-for-real-recordings) section and the Recording Arrival Runbook in [docs/MONITORING.md](MONITORING.md).

---

## Phase 1 — Trajectory & Temporal Features

**Why:** Current 18 features are per-window aggregates of counts and timing. Anti-cheat detection in practice leans on **geometric trajectory features** (curvature, turn angles, flick patterns) and **reaction-time features**. These distinguish aimbots from humans.

**New features (7):**
- `mouse_curvature_mean`, `mouse_curvature_std` — turn-angle distribution between consecutive 3-point mouse segments
- `path_efficiency` — Euclidean displacement / total-path ratio (smoother = more bot-like)
- `direction_changes_per_sec` — velocity-vector sign flips per second
- `click_reaction_mean` — mean gap (ms) between each click and the prior `mouse_move` (triggerbot signature)
- `inter_click_movement` — mean distance moved between consecutive clicks
- `keystroke_periodicity` — coefficient of variation of inter-key-press intervals (macro signature)

> The draft list also included `flick_count`, `flick_precision`, `keystroke_overlap_ratio`. After the Phase-3 benchmark showed which event-level signals actually separate cheats from legit, those were swapped for the targeted set above.

**Deliverables:**
- [x] 7 features added to `pipeline/features/run.py` and `FEATURE_COLS` (now 25 total)
- [x] Unit tests in `tests/test_features.py` (17 new tests, all passing)
- [x] Per-session aggregation added to `pipeline/adversarial/benchmark.py` (production-realistic evaluation)
- [x] `notebooks/08_trajectory_features.ipynb` — derivation, discrimination plots, before/after AUC comparison
- [x] `docs/FEATURES.md` — full feature catalogue with anti-cheat relevance
- [x] Re-ran Phase 3 benchmark with new features + per-session aggregation

**Key results:** triggerbot AUC **0.50 → 0.87** (OneClassSVM), macro AUC **0.55 → 0.68**. Aimbot remained at AUC 0.53 — the 150 ms snap signal is still buried in 30 s of mouse data, motivating the Phase 2 LSTM autoencoder operating directly on raw event sequences.

---

## Phase 1.5 — Feature expansion (optional)

**Backlog of further window-feature ideas**, revisited after Phase 5 calibration/SHAP analysis identifies gaps. Not scheduled — promote individual entries if a real signal gap motivates them.

- **Flick detection** — `flick_count`, `flick_precision` (post-flick dispersion). On the Phase 1 draft list; dropped because the signal partially lives in `mouse_curvature_std` + `path_efficiency`. Worth adding if aimbot variants emerge that snap-and-hold differently.
- **Keystroke overlap** — fraction of window with > 1 key held. Useful for crouch-jump / strafe macros common in FPS cheats.
- **Click pattern features** — left/right ratio, click hold duration, double-click rate. Useful for triggerbot variants that fire bursts.
- **Per-event-type kinematics** — split mouse_move statistics by activity context (aiming vs strafing). Needs game-state metadata.
- **Frequency-domain summaries** — windowed FFT energy in macro-relevant bands (5–10 Hz). The time-domain `keystroke_periodicity` already captures the signal cheaply; this is the thorough version.
- **Latency-distribution percentiles** instead of means — `click_reaction_p5`, `mouse_curvature_p10`. Lift the percentile-aggregation idea Phase 2 uses at the sequence level back into window features.

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
- [x] `pipeline/sequences/dataset.py` + `preprocessing.py` — 8-D event tensors, chunking, train-fold normaliser
- [x] `pipeline/models/lstm_ae.py` — bidirectional encoder + bottleneck + decoder; ~196k params; GPU auto-select
- [x] `pipeline/adversarial/benchmark.py` — LSTM-AE benchmark with **chunk-level + session-level AUC**
- [x] `notebooks/09_lstm_autoencoder.ipynb` — 11-step tutorial with diagrams, training curves, latent UMAP, score histograms
- [x] `docs/LSTM_AE.md` — architecture + benchmark write-up + WSL+CUDA notes
- [x] 38 new unit tests (`tests/test_sequences.py` + `tests/test_lstm_ae.py`)
- [ ] Integration into `pipeline/training/run.py` (deferred — current benchmark path proves the model; full DVC integration is Phase 2.1)
- [ ] MLflow logging of reconstruction-error curves (deferred — same reason)

**Key results vs Phase 1 baseline:**

| Cheat | Phase 1 best AUC | LSTM-AE chunk AUC | LSTM-AE session AUC |
|---|---|---|---|
| Aimbot | 0.53 | **0.78** | ~0.50 |
| Macro | 0.68 | 0.70 | ~0.58 |
| Triggerbot | 0.87 | **0.96** | ~0.51 |

Chunk-level AUC ≥ 0.75 success criterion met for aimbot (0.78). Session-level AUC is intentionally reported alongside chunk-level to expose the **single-detector aggregation gap** — most synthetic-file chunks are clean even when the file is cheat-labelled, so single-percentile session aggregation underperforms multi-detector aggregation. This motivates [Phase 4 (Bayesian multi-detector aggregator)](#phase-4--session-level-risk-aggregation--streaming-api).

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
- [x] `pipeline/inference/aggregator.py` — Bayesian risk aggregator (isotonic calibration + Naive-Bayes log-odds combination + configurable cheat-rate prior)
- [x] `pipeline/inference/streaming.py` — transport-independent `SessionStreamState` engine
- [x] `api/streaming.py` — `/stream` WebSocket endpoint mounted on the existing API
- [x] `scripts/train_lstm_ae.py` — persists `models/lstm_ae.pt` so the engine + benchmark can reuse weights
- [x] `scripts/replay_session.py` — WebSocket + offline replay client with optional synthetic-cheat injection
- [x] `scripts/build_phase4_demo.py` — programmatic PNG + GIF generator (matplotlib FuncAnimation + PillowWriter, no manual capture)
- [x] Dashboard "📡 Live Session" tab with live chart + per-detector contribution panel
- [x] `tests/test_aggregator.py` + `tests/test_streaming.py` + `tests/test_replay_session.py` (33 new tests)
- [x] `docs/STREAMING.md` — architecture + plain-English aggregator math + worked example
- [x] Demo GIF + PNG embedded in `docs/STREAMING.md` and the README hero

**Key results (mock data):** The aggregator math is correct (all 15 unit tests covering monotonicity, NaN handling, log-odds combination, explain components pass) and the streaming pipeline is end-to-end (event in → `ScoreUpdate` out with per-detector contributions). **However**, with the current mock-data legit baseline, classical detectors over-fire on any active session, so the combined session-level AUC does not exceed the best individual detector. The proof point at this stage is the working infrastructure + visible chunk-level signal from the LSTM-AE in the dashboard panel. Re-running against real GTA recordings (pending) should tighten the absolute numbers without code changes.

---

## Phase 4.1 — Live recorder + multi-user backlog

Follow-ups to Phase 4 that were intentionally **not** in scope this session. Move them up to a real phase if and when they unlock a demo or unblock a use case.

- **Live recorder integration** — modify `collector/recorder_gui.py` / `collector/record_session.py` to add a `--stream-to ws://localhost:8000/stream` flag that pushes events to the API in real time **as the user plays GTA**. Requires Windows-host testing because the recorder runs on Windows; the replay client already demonstrates the streaming pipeline.
- **WebSocket authentication** — token-based auth on `/stream` for multi-user / production deployment. Single-player demo doesn't need it.
- **Recorder ↔ API networking from Windows** — when the recorder runs on the Windows host and the API runs in WSL, the WS URL `ws://localhost:8000/stream` may or may not resolve depending on `uvicorn --host` flags. Documented in `docs/STREAMING.md`.
- **MLflow logging of streaming sessions** — log each replayed session's score timeline as an MLflow run for later analysis. Defer to Phase 5 (statistical rigor & MLOps).
- **Persistent session storage** — saving recent live sessions for replay/audit. Production concern, deferred.
- **Aggregator retraining on real GTA recordings** — once real gameplay data lands, re-train the isotonic calibrators on a held-out cheat-vs-legit split. This is the main path to a meaningful "combined > best individual" session-level result.

---

## Pre-recording readiness (done while waiting for real recordings)

Data-independent work shipped ahead of the real GTA recordings so their arrival is immediately productive:

- [x] **Drift detection** (Phase 5c above) — `pipeline/monitoring/drift.py`. Doubles as the tool to quantify the mock→real shift.
- [x] **Recording QC gate** — `scripts/validate_recordings.py`. Validates incoming JSONs (schema, event_count integrity, activity labels, polling-rate consistency, per-player counts) before `dvc repro`. Exit 1 on FAIL so it can gate ingestion.
- [x] **Polling-rate normalization** — `event_rate`, `mouse_key_ratio`, `direction_changes_per_sec` scaled to a 1000 Hz reference so mixed-hardware recordings are comparable (see `docs/FEATURES.md`).
- [x] **Dependency fixes** — `websockets` (replay WS client) + `shap` (staged for 5a) added to `requirements.txt`.
- [x] **Recording Arrival Runbook** — step-by-step for when data lands, in `docs/MONITORING.md`.

## Tooling backlog

- **CI pre-ingestion hook** — wire `scripts/validate_recordings.py` into a gate: either a `dvc repro` dependency or a GitHub Action that fails the build when a recording batch has FAILs. Keeps bad data out of the pipeline automatically. (Surfaced during the QC-script work; not yet built.)

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

**5c. Drift detection** — `pipeline/monitoring/drift.py` ✅ done (built early as pre-recording readiness)
- [x] KS test on feature distributions (`ks_drift`)
- [x] PSI (Population Stability Index) (`psi`) with standard 0.1/0.25 thresholds
- [x] `compute_drift_report` per-feature KS+PSI table, sorted by PSI
- [x] CLI: `python -m pipeline.monitoring.drift`
- [x] `docs/MONITORING.md` — plain-English KS/PSI explanation + worked example + Recording Arrival Runbook
- [x] `tests/test_drift.py` (13 tests)
- [ ] `notebooks/14_drift.ipynb` — visual mock-vs-real drift walkthrough (deferred until real recordings land)

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
