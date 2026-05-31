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
| 4. [Streaming + risk aggregation](#phase-4--session-level-risk-aggregation--streaming-api) | Bayesian multi-detector aggregator + WebSocket API + live dashboard | ✅ Infra done; combined risk saturates on real data → 4.1 |
| 4.1. [Live recorder + aggregator redesign](#phase-41--live-recorder--multi-user-backlog) | Aggregator redesign (real data), live recorder, WS auth | 📝 Backlog |
| 5. [Statistical rigor & MLOps](#phase-5--statistical-rigor--mlops-polish) | SHAP, calibration, drift, registry | 🚧 5a + 5b + 5c + 5d done; 5e registry to go |

Legend: ⬜ Not started · 🚧 In progress · ✅ Done · 📝 Backlog

### Recommended next order (post real-data)

Agreed sequencing for the remaining work, now that real recordings are in. Rigor/interpretability first (now meaningful on real data), cheap narrative-closers next, hard/data-limited problems last. **This order is a guide, not a contract — revisit it if implementing one phase changes what the next should be.**

1. ~~**5a — SHAP explainability.**~~ ✅ **done** — `notebooks/12_explainability.ipynb` + `pipeline/explainability.py`; included the same-hardware hydra-vs-dninix deep-dive and LSTM-AE per-channel attribution.
2. ~~**5c notebook 14 — mock→real drift walkthrough.**~~ ✅ **done** — `notebooks/14_drift.ipynb`.
3. ~~**5b — calibration** (reliability diagrams, Brier/ECE).~~ ✅ **done** — `notebooks/13_calibration.ipynb`; isotonic improved Brier, Platt hurt (small-N fragility, same root cause as the aggregator saturation).
4. ~~**5d — ablation study.**~~ ✅ **done** — revealed over-parameterisation at N=18; redundant fingerprint across families.
5. ~~**1.5 — feature expansion.**~~ ✅ **gate resolved: SKIP/defer** — 5d says the model already has too many features for the data (see Phase 1.5 note). Revisit only with much more data.
6. **4.1 — aggregator redesign.** ← **currently next.** Split it: feeding the chunk-level LSTM signal into the live score is doable now; the "combined > best-individual" recalibration is data-starved at 18 sessions — defer until more recordings land.
7. **5e + CI pre-ingestion hook.** Production-maturity polish; low dependency; good closing task.

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

> **Gate decision (2026-05-31): DEFERRED — do not add features now.** The Phase 5d ablation (`notebooks/15_ablation.ipynb`) showed the 25-feature model is already **over-parameterised at 18 sessions** — dropping whole feature families often *raises* validation accuracy, and single families classify well alone (redundant fingerprint). Adding more features would worsen overfitting, not help. **Revisit only once a much larger dataset makes the full feature set non-overfitting**; at that point re-run 5d and promote entries against a measured gap. (The nearer-term lever at this scale is *more recordings* or *feature reduction*, not feature addition.)

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

**Key results (real data, 2026-05-30):** The aggregator math is correct (15 unit tests) and the streaming pipeline is end-to-end. On 18 real GTA sessions the **chunk-level LSTM-AE detector works** (aimbot AUC 0.79, triggerbot 0.93 — see `docs/ADVERSARIAL.md`). **However**, the *session-level combined risk* still does not beat the best individual detector: its session-level inputs are near-chance (≈ 0.50) and the isotonic calibrators are fit on only 18 legit sessions, so the combination saturates. Real data also exposed a **normalisation bug** (the streaming engine never applied per-session sens/DPI + polling-rate norm) — **fixed this round** via `SessionStreamState.configure_for_session`. Recalibrating / redesigning the aggregator is **Phase 4.1** (below).

---

## Phase 4.1 — Live recorder + multi-user backlog

Follow-ups to Phase 4 that were intentionally **not** in scope this session. Move them up to a real phase if and when they unlock a demo or unblock a use case.

- **Live recorder integration** — modify `collector/recorder_gui.py` / `collector/record_session.py` to add a `--stream-to ws://localhost:8000/stream` flag that pushes events to the API in real time **as the user plays GTA**. Requires Windows-host testing because the recorder runs on Windows; the replay client already demonstrates the streaming pipeline.
- **WebSocket authentication** — token-based auth on `/stream` for multi-user / production deployment. Single-player demo doesn't need it.
- **Recorder ↔ API networking from Windows** — when the recorder runs on the Windows host and the API runs in WSL, the WS URL `ws://localhost:8000/stream` may or may not resolve depending on `uvicorn --host` flags. Documented in `docs/STREAMING.md`.
- **MLflow logging of streaming sessions** — log each replayed session's score timeline as an MLflow run for later analysis. Defer to Phase 5 (statistical rigor & MLOps).
- **Persistent session storage** — saving recent live sessions for replay/audit. Production concern, deferred.
- **Aggregator redesign (real data showed it saturates)** — on 18 real sessions the session-level combined risk saturates: its inputs (`LSTMAutoencoder/session`, classical session detectors) are ≈ 0.50 AUC and the isotonic calibrators are fit on only 18 legit sessions, so combining near-chance signals over a tiny set pushes even legit sessions to high risk. The discriminative power is at the **chunk level** (AUC 0.79–0.93). Fix options: (a) aggregate the **chunk-level** LSTM signal directly into the live score rather than per-session detector maxima; (b) recalibrate once there are far more sessions; (c) a per-chunk-labelled within-session eval so the live score can localise *when* cheating starts (the current detector is between-session only). The streaming **normalisation bug is already fixed** (`configure_for_session`).

---

## Pre-recording readiness (done while waiting for real recordings)

Data-independent work shipped ahead of the real GTA recordings so their arrival is immediately productive:

- [x] **Drift detection** (Phase 5c above) — `pipeline/monitoring/drift.py`. Doubles as the tool to quantify the mock→real shift.
- [x] **Recording QC gate** — `scripts/validate_recordings.py`. Validates incoming JSONs (schema, event_count integrity, activity labels, polling-rate consistency, per-player counts) before `dvc repro`. Exit 1 on FAIL so it can gate ingestion.
- [x] **Polling-rate normalization** — `event_rate`, `mouse_key_ratio`, `direction_changes_per_sec` scaled to a 1000 Hz reference so mixed-hardware recordings are comparable (see `docs/FEATURES.md`).
- [x] **Dependency fixes** — `websockets` (replay WS client) + `shap` (staged for 5a) added to `requirements.txt`.
- [x] **Recording Arrival Runbook** — step-by-step for when data lands, in `docs/MONITORING.md`.

## First real recordings — full runbook done (2026-05-30)

The first real GTA batch landed: **18 sessions, 3 players** (shotik 5, dninix 8, hydra 5), all 1000 Hz, mixed DPI (800 / 1600). Both the first pass and the full Recording Arrival Runbook are complete:

- [x] **QC gate** — all 18 PASS (`python -m scripts.validate_recordings --dir data/raw`).
- [x] **Player-stratified split** — `pipeline/features/split.py` rewritten from `GroupShuffleSplit` to a per-player whole-session holdout, so every identity appears in train/val/test (essential at N=18; a random split could drop a player from test). Real-data identification: **test acc 0.853** (f1 0.862); a same-hardware-only check (hydra vs dninix) lands at 0.750, exposing a cross-hardware confound from shotik's different rig.
- [x] **Mock→real drift quantified** — `reports/drift_mock_vs_real.csv`: **20 / 25 features significant**, led by `wasd_rhythm` (PSI 9.4), `speed/accel_*` (6.6–8.0), `event_rate` (5.1). Empirical proof the mock baseline was unrepresentative.
- [x] **LSTM-AE retrained on real legit data** (`scripts.train_lstm_ae`) — 1.35M events; persisted to `models/lstm_ae.pt`.
- [x] **Adversarial benchmark re-run on real data** — chunk-level LSTM-AE: aimbot 0.79 / triggerbot 0.93 / macro 0.60; classical detectors at chance for aimbot. Full table in `docs/ADVERSARIAL.md`.
- [x] **Phase-4 demo regenerated** — pivoted from the (saturated) live risk-timeline to an honest **chunk-detection distribution** figure (`reports/figures/phase4_chunk_detection.png`); the README hero now shows it.
- [x] **Streaming normalisation bug fixed** — `SessionStreamState.configure_for_session` (sens/DPI + polling-rate); previously the engine mis-scaled real-hardware sessions.
- [x] **Mock-data caveats updated** across `docs/ADVERSARIAL.md`, `docs/STREAMING.md`, `docs/LSTM_AE.md`, `README.md`, `CLAUDE.md`.

**Open follow-up:** the session-level live-risk aggregator saturates on real data → tracked in [Phase 4.1](#phase-41--live-recorder--multi-user-backlog).

## Tooling backlog

- **CI pre-ingestion hook** — wire `scripts/validate_recordings.py` into a gate: either a `dvc repro` dependency or a GitHub Action that fails the build when a recording batch has FAILs. Keeps bad data out of the pipeline automatically. (Surfaced during the QC-script work; not yet built.)

---

## Phase 5 — Statistical Rigor & MLOps Polish

**Why:** Tie everything together with scientific rigor and production-ready observability. Reads as "ready to ship" rather than "promising prototype."

**Deliverables:**

**5a. Explainability** ✅ done — `pipeline/explainability.py` + `notebooks/12_explainability.ipynb`
- [x] Exact SHAP (TreeExplainer) for the LightGBM identification model — per-player mean-|SHAP| heatmap + beeswarm
- [x] Per-prediction "why this window → player X?" waterfall walkthroughs
- [x] **Same-hardware deep-dive** (hydra vs dninix) — SHAP shows timing/rhythm features (`click_interval_std`, `keystroke_periodicity`, `burst_rate`) separate two players on identical hardware → a real behavioural biometric, not a hardware tell
- [x] LSTM-AE explained via **per-channel reconstruction attribution** (not SHAP-through-an-LSTM): triggerbot flags driven ~16× by the `is_mouse_click_press` channel. `tests/test_explainability.py` (7 tests); figures `reports/figures/phase5a_*.png`

**5b. Calibration** ✅ done — `pipeline/calibration.py` + `notebooks/13_calibration.ipynb`
- [x] Reliability diagram (top-label confidence) for the LightGBM identifier
- [x] ECE + multiclass Brier (`pipeline/calibration.py`, 9 tests — sklearn has no multiclass versions)
- [x] Isotonic / Platt scaling via `FrozenEstimator` (sklearn 1.8), fit on val / evaluated on test, before/after table
- **Finding (honest, N=34 test / 46 cal):** isotonic improved Brier 0.275 → 0.224 keeping accuracy 0.853; Platt *hurt* (acc → 0.824). The same small-data calibration fragility that makes the Phase-4 aggregator saturate (4.1). Figures `reports/figures/phase5b_*.png`

**5c. Drift detection** — `pipeline/monitoring/drift.py` ✅ done (built early as pre-recording readiness)
- [x] KS test on feature distributions (`ks_drift`)
- [x] PSI (Population Stability Index) (`psi`) with standard 0.1/0.25 thresholds
- [x] `compute_drift_report` per-feature KS+PSI table, sorted by PSI
- [x] CLI: `python -m pipeline.monitoring.drift`
- [x] `docs/MONITORING.md` — plain-English KS/PSI explanation + worked example + Recording Arrival Runbook
- [x] `tests/test_drift.py` (13 tests)
- [x] `notebooks/14_drift.ipynb` — visual mock-vs-real drift walkthrough (PSI bars, KS ECDF overlay, top-feature distribution overlays); figures `reports/figures/phase5c_*.png`

**5d. Ablation study** ✅ done — `notebooks/15_ablation.ipynb`
- [x] Leave-one-group-out + only-group, 5 feature families, 8-seed-averaged val accuracy
- **Finding:** the player fingerprint is **redundant across families** (mouse-kinematics and keyboard each ≈ 0.75–0.79 *alone* vs 0.33 chance), and the full 25-feature model is **over-parameterised at N=18 sessions** — dropping a whole family often *raises* val accuracy above the full model (0.72). → directly gates Phase 1.5 (see below). Figure `reports/figures/phase5d_ablation.png`

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
