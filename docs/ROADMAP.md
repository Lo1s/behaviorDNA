# BehaviorDNA — Portfolio Roadmap

Targeted enhancements aimed at ML/AI roles at anti-cheat companies (Anybrain, Irdeto, BattlEye R&D, Riot's anti-cheat team, etc.).

The repo already has a solid classical-ML foundation: 30 windowed features (decoupled into a 25-feature identifier slice + 30-feature cheat-detector slice), 7 model types, batch API, Streamlit dashboard, DVC pipeline, CI, MLflow tracking, external dataset analysis (CS2CD + CaptchaSolve30k).

This roadmap adds the four things hiring managers at AI-focused anti-cheat companies look for:

1. **Deep learning on raw input sequences** (LSTM autoencoder)
2. **Adversarial / domain knowledge** (synthetic cheat generation + detection benchmarks)
3. **Production MLOps maturity** (streaming inference, drift detection, model registry)
4. **Research rigor** (calibration, ablation, explainability)

---

## Recruiter-handover checklist (do last)

One-off actions to run at the **very end** of portfolio work, immediately before
sharing the repo with recruiters — not earlier, so they don't interfere with
ongoing iteration:

- [ ] **Enable public/anonymous DVC read on DagsHub** (repo Settings → make the
      DVC storage public) so a recruiter can `dvc pull` the versioned artifacts
      (LSTM-AE model, recordings) and reproduce the headline result **without a
      DagsHub account/token**. Anonymous pull is currently disabled → reproduction
      needs a token (see [docs/ADVERSARIAL.md](ADVERSARIAL.md) → "Reproducing").
      DagsHub-UI only; verified 2026-06-13 that token-free `dvc pull` fails.
- [ ] **Then delete the hosted Streamlit demo's DagsHub secrets.** The live demo
      (`behaviordna.streamlit.app` → Manage app → Settings → Secrets) carries
      `DAGSHUB_USER` / `DAGSHUB_TOKEN` only so it can `dvc pull` while the repo is
      private. Once DVC read is public the targeted pull works **anonymously**, so
      remove both secrets (and rotate the token if it's reused anywhere). No code
      change needed — `_ensure_artifact_or_stop()` falls back to an anonymous pull.

---

## Status at a glance

| Phase | Goal | Status |
|---|---|---|
| 1. [Trajectory & temporal features](#phase-1--trajectory--temporal-features) | 7 new anti-cheat-relevant features | ✅ Done |
| 1.5. [Feature expansion (optional)](#phase-15--feature-expansion-optional) | CS2-validated feature promotion + ID/cheat feature-set split | 🚧 Partial — 5 features promoted ([nb 18](../notebooks/18_signal_importance_cs2.ipynb)); ID/cheat sets decoupled; rest data-gated |
| 2. [LSTM autoencoder](#phase-2--lstm-autoencoder-for-anomaly-detection) | Deep-learning sequence model | ✅ Done |
| 3. [Adversarial bots](#phase-3--adversarial-bot-generation--detection-benchmark) | Synthetic cheat generator + detection benchmark | ✅ Done |
| 4. [Streaming + risk aggregation](#phase-4--session-level-risk-aggregation--streaming-api) | Bayesian multi-detector aggregator + WebSocket API + live dashboard | ✅ Infra done; combined risk saturates on real data → 4.1 |
| 4.1. [Live recorder + aggregator redesign](#phase-41--live-recorder--multi-user-backlog) | Aggregator redesign (real data), live recorder, WS auth | 📝 Backlog |
| 5. [Statistical rigor & MLOps](#phase-5--statistical-rigor--mlops-polish) | SHAP, calibration, drift, registry | ✅ Done (5a–5e); MLOps in [docs/MLOPS.md](docs/MLOPS.md) |
| 6. [Public-corpus identification + verification](#phase-6--public-corpus-identification--verification) | Run the pipeline at 10–120 users (Balabit/SapiMouse); reframe ID as verification/open-set (EER, smurf detection) | ✅ Done (2026-06-11) — Balabit EER 0.144 @ 10 users; REPORT.md §6 pending |
| 7. [Detection-vs-evasion frontier](#phase-7--detection-vs-evasion-frontier) | Parameterised cheat "humanizer"; detector-AUC-vs-evasion curve + equilibrium | ✅ Done (2026-06-13) — **defender-favoured frontier**: no λ is both undetectable and worth running (aimbot humanising *raises* AUC; triggerbot stays 0.76 at zero utility; macro reaches chance only at zero utility). [docs/ADVERSARIAL.md](ADVERSARIAL.md#the-arms-race--detection-vs-evasion-phase-7) |
| 8. [Self-supervised pretraining](#phase-8--self-supervised-pretraining) | Pretrain the sequence encoder on CaptchaSolve30k; data-efficiency curve | ✅ Done (2026-06-13) — **rigorous null**: no transfer benefit (CS2CD Δ≈0.000; GTA Δ≈−0.005, within ±std); the captcha→game domain gap dominates. [docs/PRETRAINING.md](PRETRAINING.md) |
| 8.1. [In-domain pretraining](#phase-81--in-domain-pretraining-does-closing-the-domain-gap-rescue-the-null) | Pretrain in-domain on full CS2CD (795 matches) + frozen-encoder arm; does closing the domain gap rescue the Phase 8 null? | ✅ Done (2026-06-15) — **null holds, deeper than the domain gap**: in-domain ≤ scratch, `s2`(dt-neutralised)≈`s1`, volume flat; in-domain CS2 isn't even closer to GTA. [docs/PRETRAINING.md](PRETRAINING.md#phase-81--in-domain-pretraining-does-closing-the-domain-gap-rescue-the-null) |
| 8.2. [Contrastive pretraining](#phase-82--contrastive-pretraining-does-the-objective-matter) | Swap reconstruction for a contrastive (NT-Xent) objective; eval on the **frozen** embedding (kNN / one-class / linear-probe) | ✅ Done (2026-06-15) — **first non-null**: in-domain contrastive beats random-init **and** the 8.1 reconstruction encoder on every probe (modest, in-domain-specific, volume-flat). The objective was the lever. [docs/PRETRAINING.md](PRETRAINING.md#phase-82--contrastive-pretraining-does-the-objective-matter) |
| 9. [Outcome-labelled telemetry](#phase-9--outcome-labelled-telemetry) | CS2 demo parsing → kills/damage/accuracy per window; supervised detection + aggregator re-attempt | 🚧 **Spike done (2026-06-14)** — `demoparser2` extracts kills/damage/shots/per-tick view-angles (validated on a real public demo); marker-free motion **clock-sync** recovers injected offsets to <1 sample & self-rejects mismatches. Supervised detector + 4.1 re-attempt await **dual-capture** data. |

Legend: ⬜ Not started · 🚧 In progress · ✅ Done · 📝 Backlog

### Recommended next order (post real-data)

Agreed sequencing for the remaining work, now that real recordings are in. Rigor/interpretability first (now meaningful on real data), cheap narrative-closers next, hard/data-limited problems last. **This order is a guide, not a contract — revisit it if implementing one phase changes what the next should be.**

1. ~~**5a — SHAP explainability.**~~ ✅ **done** — `notebooks/12_explainability.ipynb` + `pipeline/explainability.py`; included the same-hardware hydra-vs-dninix deep-dive and LSTM-AE per-channel attribution.
2. ~~**5c notebook 14 — mock→real drift walkthrough.**~~ ✅ **done** — `notebooks/14_drift.ipynb`.
3. ~~**5b — calibration** (reliability diagrams, Brier/ECE).~~ ✅ **done** — `notebooks/13_calibration.ipynb`; isotonic improved Brier, Platt hurt (small-N fragility, same root cause as the aggregator saturation).
4. ~~**5d — ablation study.**~~ ✅ **done** — revealed over-parameterisation at N=18; redundant fingerprint across families.
5. ~~**1.5 — feature expansion.**~~ ✅ **done (partial promotion, 2026-06-08→10)** — rather than add to the over-parameterised GTA set (the 5d gate still holds), validated a windowed signal bank on external **CS2CD** ([nb 18](../notebooks/18_signal_importance_cs2.ipynb)) where N is large, **promoted 5 cheat features**, and **decoupled the ID vs cheat feature sets** (`ID_FEATURE_COLS` 25 / `CHEAT_FEATURE_COLS` 30) so they don't trade off at small N. Remaining (outcome/system) signals are data-gated → [docs/SIGNALS.md](SIGNALS.md).
6. ~~**4.1 — aggregator redesign.**~~ ⏸️ **verified blocked & deferred** — prototyped feeding the chunk signal into the live score; all session aggregations ≈ 0.50 (legit natural-variance tail + synthetic sparse injection). Unblock = **real continuous cheat data** ([docs/CHEAT_DATA_COLLECTION.md](CHEAT_DATA_COLLECTION.md)), not more code.
7. ~~**5e + CI pre-ingestion hook.**~~ ✅ **done** — `scripts/promote_model.py` (registry promotion, verified live on DagsHub) + CI validation gate + `docs/MLOPS.md`. **Original (Phase 1–5) roadmap complete** — the extension (Phases 6–9 + tech report) is below; Phase 6 is done. Remaining classical work is data-gated (real cheat recordings → 4.1) or scales with more sessions.

> **Pre-recording readiness (done):** ahead of the real GTA recordings, shipped data-independent infra — drift detection (5c), a recording QC gate (`scripts/validate_recordings.py`), polling-rate normalization, and dependency fixes. See the [Pre-recording readiness](#pre-recording-readiness-done-while-waiting-for-real-recordings) section and the Recording Arrival Runbook in [docs/MONITORING.md](MONITORING.md).

### Extension sequencing (Phases 6–9 + the tech report)

Phases 1–5 closed the original portfolio scope (end-to-end pipeline, deep model, adversarial benchmark, statistical rigor) — but on **3 players / 1 cheat recorder**, which is the credibility ceiling. Phases 6–9 attack that ceiling, ordered by impact-per-evening **and** by what's unblocked now:

1. ~~**Phase 6 — public-corpus ID + verification (A + D merged).**~~ ✅ **done (2026-06-11)** — Balabit (10 users) closed-set 0.59 / impostor EER 0.144; SapiMouse (120 users) signal survives but data-starved + open-set rejection at chance. The honest two-sided scale answer ([docs/VERIFICATION.md](VERIFICATION.md), [nb 19](../notebooks/19_identification_at_scale_public.ipynb)). Only remaining bit: expand REPORT.md §6 from draft.
2. **Phase 9 — outcome telemetry, *spike now / execute later*.** ✅ **spike done (2026-06-14)** — `demoparser2` *does* pull per-tick view-angles + kills/damage/shots (validated on a real public `de_mirage` demo), and the demo↔recorder **clock-sync** is solved marker-free by cross-correlating recorder mouse-motion against demo view-angle-motion (recovers injected offsets to <1 sample, `peak_corr` self-validates and rejects mismatched pairs). Code: `pipeline/outcome/cs2_demo.py` + `scripts/parse_cs2_demo.py`; verdict in [docs/CHEAT_DATA_COLLECTION.md](CHEAT_DATA_COLLECTION.md#phase-9-feasibility-spike--cs2-outcome-telemetry--clock-sync-verdict--feasible). Remaining = let **dual-capture** sessions accumulate, then build the supervised detector + 4.1 re-attempt.
3. ~~**Phase 7 — detection-vs-evasion frontier (C).**~~ ✅ **done (2026-06-13)** — a humanisation knob λ ([`pipeline/adversarial/humanizer.py`](../pipeline/adversarial/humanizer.py)) turns each cheat from obvious bot → humanised toward the player's own play; [`scripts/evasion_frontier.py`](../scripts/evasion_frontier.py) sweeps λ on the 18 real GTA sessions and plots detection AUC(λ) vs cheat utility(λ). **Defender-favoured equilibrium:** no λ is both undetectable and worth running (humanising the aimbot snap *raises* AUC 0.79→0.84; triggerbot stays 0.76 at zero utility; macro reaches chance only once its cadence — its whole value — is gone). [docs/ADVERSARIAL.md](ADVERSARIAL.md#the-arms-race--detection-vs-evasion-phase-7), [nb 20](../notebooks/20_evasion_frontier.ipynb).
4. ~~**Phase 8 — self-supervised pretraining (B).**~~ ✅ **done (2026-06-13)** — masked-denoising pretraining of the LSTM-AE on CaptchaSolve30k (≈17.7k mouse sessions), then a pretrained-vs-scratch data-efficiency curve on **both** CS2CD (real cheats) and GTA. **Result is a rigorous null:** no transfer benefit at this scale, and the measured domain gap explains why (the `dt` channel — sampled fixed-tick captcha vs event-driven GTA — is PSI ≈ 10–12 mismatched). A publishable negative result: a generic human-mouse corpus is not a drop-in foundation for game-input biometrics. [docs/PRETRAINING.md](PRETRAINING.md), [nb 21](../notebooks/21_pretraining.ipynb).
5. **Tech report (F) — grown, not written.** ✅ **full draft (2026-06-13)** ([docs/REPORT.md](REPORT.md)) — §§1–10 + abstract & appendix drafted from the per-topic docs; **§7 (evasion) now run and reported** (Phase 7 done). The remaining action is the **manual arXiv submission** + a blog-post condensation. The structure falls out of [docs/FINDINGS.md](FINDINGS.md).

> Same contract as before: **a guide, not a promise** — revisit if implementing one phase changes what the next should be. The honest-positioning rule still holds: a null result (mouse-only ID drops; pretraining doesn't transfer; the domain gap dominates) is a publishable finding, not a failure.

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
>
> **Update (2026-06-08): partial promotion via external CS2CD.** Instead of validating on the over-parameterised GTA set, [`notebooks/18_signal_importance_cs2.ipynb`](../notebooks/18_signal_importance_cs2.ipynb) ranks a windowed signal bank on **CS2CD (10 players, 50k ticks, real cheats)** where we don't overfit. **Promoted 5 features** (`speed_p50/p90/p99`, `fast_segment_straightness`, `click_reaction_p5` → `FEATURE_COLS` now 30): new-only cheat AUC **0.74 > 0.71** existing-analog (player-held-out), neutral-to-better on GTA ID (0.600 → 0.625, small-N noise). Two findings in [docs/SIGNALS.md](SIGNALS.md): (1) the **strongest cheat signals are non-behavioural** (outcome/performance, system/process) and need telemetry we don't yet collect → a prioritised collection roadmap; (2) `FEATURE_COLS` is **shared by the identifier and the cheat detectors** → they should be **decoupled** so cheat features don't trade against identification at small N — **✅ implemented 2026-06-10**: `ID_FEATURE_COLS` (25, identifier) vs `CHEAT_FEATURE_COLS` (30, cheat detectors); artifacts carry their own `feature_cols` ([docs/SIGNALS.md](SIGNALS.md)). Data-hygiene flag: cheat sessions sat inside the identification set — **fixed** (excluded at split time; see SIGNALS.md).

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
- **Aggregator redesign (real data showed it saturates) — VERIFIED BLOCKED on current data (2026-05-31).** On 18 real sessions the session-level combined risk saturates. The obvious fix — "aggregate the chunk-level signal (AUC 0.79–0.93) directly into the live score" — was **prototyped and does not work**: every chunk→session aggregation (max, p95, mean-top-k, fraction-above-threshold of calibrated chunk-risk) scores **≈ 0.50** legit-vs-cheat at the session level. Root cause: legit gameplay has its own natural high-reconstruction-error chunks (rare fast flicks/scrolls) that are indistinguishable from the few injected cheat chunks once aggregated, *and* synthetic cheat files share almost all chunks with their legit twin (sparse injection). This is a hard ceiling of the **synthetic** data, not a tuning problem.
  - **Unblock:** record **real continuous cheat data** (a real aimbot/triggerbot user cheats throughout → most chunks elevated → session separates). Full methodology in [docs/CHEAT_DATA_COLLECTION.md](CHEAT_DATA_COLLECTION.md) — including the key point that only *input-level* aimbots are visible to input biometrics, and logging the cheat-toggle key for per-chunk ground truth. **4.1 is deferred until that data exists.**
  - The streaming **normalisation bug is already fixed** (`configure_for_session`); that part shipped.

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

## Cheat-data collection — live signature harness (in progress)

The Phase 4.1 verification showed synthetic *sparse* cheat injection can't separate at the session level. The unblock is **real continuous cheat telemetry**, captured live during offline GTA5. Rather than download a (malware-prone, unlabelled) real aimbot, we generate the cheat *input signature* under full control — methodology in [docs/CHEAT_DATA_COLLECTION.md](CHEAT_DATA_COLLECTION.md).

- [x] **`pipeline/adversarial/live_cheat.py`** — pure, unit-tested planning layer (14 tests): difficulty presets (shared with `bot_generator`), rng-seeded planners for aimbot micro-correction snaps (easing + overshoot/jitter for the soft/evasive case), sub-human triggerbot bursts, and recoil/rapid-fire macros, emitting abstract `InputAction`s; plus `toggle_log_to_segments`.
- [x] **`collector/cheat_sim.py`** — Windows-side actuator (`SendInput`) + `pynput` toggle-hotkey loop running alongside the recorder. **Safe by construction: no target acquisition, no memory reads, no online** — produces the input signature, not a functional competitive cheat. Offline-only guard + provenance log.
- [x] **`scripts/label_cheat_segments.py`** — derives `cheat_label` + `cheat_segments` from the toggle keys captured *in-band* by the recorder (no cross-process clock sync) and strips the control keys; drop-in for the pipeline (5 tests).
- [x] **Architecture comparison (unsupervised, done):** LSTM-AE vs TCN-AE vs Transformer-AE on the chunk task — all competitive at N=18 (Transformer marginally best, TCN cheapest; capacity isn't the bottleneck). `scripts/compare_architectures.py`, [docs/ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md).
- [ ] **Awaiting recordings:** capture real continuous-cheat sessions, then (a) re-run the adversarial benchmark on **real** cheats, (b) **re-attempt Phase 4.1** (continuous cheating → session separates), and (c) re-run the architecture comparison **supervised** (classifier head per backbone — labels should dominate; ranking may change).

## Tooling backlog

- [x] **CI pre-ingestion hook** ✅ done — `scripts/validate_recordings.py` now runs as a step in the CI `dvc-repro` job (after `dvc pull`, before `dvc repro`), failing the build on any malformed recording before it can poison the pipeline. See `.github/workflows/ci.yml` + [docs/MLOPS.md](MLOPS.md).
- [x] **Regenerate LSTM-AE weights + demo GIF on the GPU desktop** ✅ done (2026-06-12, RTX 3070). Canonical CUDA retrain on **18 legit sessions** (best val_loss **0.575**); the hero figure reproduces the documented detection AUCs (aimbot 0.805, triggerbot 0.940, macro 0.614). Re-rendered `reports/figures/phase4_chunk_flags.gif` (triggerbot session, 80 frames) + `phase4_chunk_detection.png`; full `pytest -q` passes in one process on Linux (371). **Also fixed a latent leak:** `scripts/train_lstm_ae.py:_load_legit_tensors` was loading *every* `data/raw` JSON despite its name — now skips cheat sessions via `_is_cheat_session` (same rationale as the identification-split exclusion), so the local cheat recordings no longer contaminate the legit-manifold AE. Guard test in `tests/test_train_lstm_ae.py`.
- [x] **DVC-version the LSTM-AE artifact + recipe-reproducible synthetic data** ✅ done (2026-06-13, review #4) — `models/lstm_ae.pt` is now `dvc add`-tracked (`models/lstm_ae.pt.dvc` pointer, `dvc push`ed); the 523 MB synthetic-cheat set is left unstored and regenerated deterministically (seed 42) from the versioned legit recordings (`docs/ADVERSARIAL.md` "Reproducing"). The headline cheat-detection result now reproduces from a fresh clone.
- [x] **Single source-of-truth repo metadata** ✅ done (2026-06-14) — `scripts/generate_metadata.py` derives the structural facts from the git-tracked tree (test count, dashboard tabs, notebook/doc counts) into `reports/repo_metadata.json` and rewrites the *owned spans* in README/CLAUDE (regex anchors that raise if the prose changes shape, so the generator can't silently go stale). CI-gated via `--check` next to the results-block gate; guarded by `tests/test_metadata.py`. Caught the live drift: README "340+" / CLAUDE "~372" → **418** test functions. Model status + artifact fidelity already come from `reports/*.json` via `generate_results.py`; **dataset/session counts stay manual** (data is DVC-gitignored, so not computable in CI without a `dvc pull`).

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

**5e. Model registry hardening** ✅ done — `scripts/promote_model.py` + `docs/MLOPS.md`
- [x] MLflow Model Registry stages (Staging → Production) — training logs+registers a `scaler+classifier` pipeline version per run; **verified live on DagsHub** (v1 → Production). Alias fallback for MLflow's stage→alias deprecation.
- [x] `scripts/promote_model.py` — selects the best registered version by `val_accuracy` and promotes it to Production; `select_best_run` unit-tested (`tests/test_promote_model.py`), live write degrades gracefully without creds
- [x] `docs/MLOPS.md` — drift monitoring + registry promotion + CI gates in one place

---

## Phase 6 — Public-corpus identification + verification

**Why:** The single most damaging question a reviewer can ask is *"does this survive beyond 3 friends?"* — and right now the honest answer is "unmeasured." The fix is **not** recruiting 15 people; it's running the *exact* windowed-feature + model pipeline on a public mouse-dynamics corpus at 10–120 users, the way notebook 17 already did for CS2CD. Two payloads ship from one dataset-adapter effort: (A) the **scale** claim, and (D) the **reframe** — closed-set "which of 3 players" is not the industry problem; *"is this account being played by its usual owner?"* (account-sharing / smurf / boost detection) is. Same data, harder and more product-relevant problem, and it opens the non-gaming story (continuous authentication, fraud/bot detection) that matters to Irdeto-style companies.

**Datasets:**
- **Balabit Mouse Dynamics Challenge** (10 users) — the classic benchmark, so EER is literature-comparable.
- **SapiMouse** (120 users) — the scale claim; short sessions, large N.
- (BeCAPTCHA-Mouse deferred — two corpora is enough to make the point.)
- Download locations + layout: [`data/external/README.md`](../data/external/README.md) (dirs pre-created; gitignored except the README).

**Approach:**
- **Adapter** maps each corpus's raw mouse stream into the existing `events.parquet` schema (same pattern as the CS2CD ingestion in notebook 17). These corpora are **mouse-only** → define a `MOUSE_ID_FEATURE_COLS` slice (trivial now that ID/cheat feature sets are [decoupled](SIGNALS.md)); handle missing keyboard features by *exclusion*, not zero-fill into the model.
- **Closed-set:** accuracy vs number-of-users curve (3 → 10 → 50 → 120), session-held-out splits, bootstrap CIs — wired into `scripts/generate_results.py` so the README row stays pipeline-backed.
- **Verification (the D reframe):** per-user one-vs-rest scores → **ROC / EER + DET curve**; **open-set:** enrol K users, hold the rest out as impostors, report false-accept at a fixed rejection rate. Frame as smurf / account-sharing detection in the docs, with a paragraph on the continuous-auth generalisation.

**Decision (2026-06-11):** build the adapter abstraction for **both Balabit + SapiMouse together** (shared base, one users-curve 3→120 in notebook 19).

**Deliverables:**
- [x] `MOUSE_ID_FEATURE_COLS` slice + tests (`pipeline/features/run.py`, `tests/test_external.py`)
- [x] `pipeline/external/` adapters — Balabit + SapiMouse parsing implemented & fixture-tested; incl. idle-gap segmentation (`split_on_idle` — desktop captures aren't continuous gameplay) and the (65535, 65535) sentinel-glitch guard found in Balabit
- [x] Corpora downloaded into `data/external/` (Balabit 241 MB from the archived GitHub repo; SapiMouse 8 MB zip from ms.sapientia.ro) — re-downloadable, see `data/external/README.md`
- [x] `pipeline/verification.py` — EER + DET + FAR@FRR + closed-set→verification conversion (unit-tested)
- [x] Users-curve + verification run — `scripts/run_external_identification.py` → `reports/external_identification.json` (seed 42)
- [x] README results rows via `generate_results.py` (auto-generated, CI-gated)
- [x] `docs/VERIFICATION.md` — results + the product reframe
- [x] `notebooks/19_identification_at_scale_public.ipynb` — tutorial walkthrough, executed end-to-end (users-curve + DET/EER recomputed live from the corpora; segmentation demo: 4.3 h session → 3 windows without `split_on_idle`, 245 with)
- [ ] REPORT.md §6 — expand the draft into the full section

**Key results (2026-06-11):** Balabit (10 users, hours/user): closed-set **0.59** acc (chance 0.10), **impostor-detection EER 0.144** over 784 labelled sessions — the literature-comparable headline, from the unmodified GTA pipeline. SapiMouse (120 users, *minutes*/user): accuracy stays **10–20× chance** all the way to 120 users (0.11 @ chance 0.008) but absolute performance is data-starved at ~6 train windows/user, and **open-set rejection is chance-level** — softmax confidence is not an identity score. The scale answer is honest and two-sided: *the signal survives; the data budget is the binding constraint* → directly motivates Phase 8 (pretraining) and embedding-based verification. Full numbers: [docs/VERIFICATION.md](VERIFICATION.md).

**Honest-outcome note (confirmed):** GTA features are partly keyboard-driven; mouse-only transfer behaved as predicted — strong where data is plentiful (Balabit), starved where it isn't (SapiMouse).

---

## Phase 7 — Detection-vs-evasion frontier

**Why:** The most anti-cheat-native thing a portfolio can show: you generate cheats — now make them **evade**, and characterise the equilibrium. Red-teaming your own detector and plotting where it breaks is literally the day job at Anybrain / BattlEye / Riot, and almost nobody has it in a portfolio.

**Approach:**
- **Humanizer with a strength knob λ ∈ [0, 1]** (`pipeline/adversarial/humanizer.py`):
  - reaction-delay injection sampled from a human RT distribution,
  - Bézier / minimum-jerk smoothing of aimbot snaps,
  - kinematic noise **matched to the target player's own** speed/curvature distribution (sample from that player's legit windows — we have them).
  - This is an *extension of what exists*: `pipeline/adversarial/live_cheat.py` already has easing, overshoot, and jitter planners; λ parameterises their strength.
- **The two curves that make the figure:**
  - **detector AUC(λ)** — chunk-level LSTM-AE + the window detectors,
  - **cheat utility(λ)** — residual advantage (e.g. reaction-time edge vs the player's legit baseline).
  - Headline plot: **detection vs utility**, with the equilibrium region marked — *"humanized enough to evade ≈ no longer worth running."*

**Deliverables:**
- [x] `pipeline/adversarial/humanizer.py` (λ knob + per-player baseline + closed-form utility) + `tests/test_humanizer.py` (28 tests)
- [x] `scripts/evasion_frontier.py` — λ-sweep runner (reuses the chunk-LSTM-AE + OneClassSVM window detector), writes the JSON + figure
- [x] `notebooks/20_evasion_frontier.ipynb` — λ-sweep, both curves, equilibrium (executed end-to-end)
- [x] "Arms race" section in `docs/ADVERSARIAL.md` + frontier figure in README + REPORT.md §7
- [x] `reports/evasion_frontier.json` + `reports/figures/phase7_evasion_frontier.png` (AUC + utility per λ)

**Key results (2026-06-13):** the frontier favours the defender — **no λ is both undetectable and worth running**. Humanising the **aimbot** snap *raises* detection (chunk AUC 0.79 → 0.84, window 0.59 → 0.72: player-matched jitter/overshoot is more anomalous than a clean robotic snap) while its speed edge vanishes; the **triggerbot** stays at AUC 0.76 even at full human reaction (utility 0); the **macro** reaches ~chance (0.40) only once its perfect cadence — its entire value — is jittered away. Honest caveats: closed-world (humanise toward the player's own logged distribution, fixed detector), the macro utility proxy is the weakest axis, N = 18/3.

---

## Phase 8 — Self-supervised pretraining

**Why:** Highest research-credential ROI, and it attacks the project's *actual* limiting factor. The [architecture comparison](ARCHITECTURE_COMPARISON.md) already showed capacity isn't the bottleneck — **data is** — so a transfer/pretraining result targets the real constraint and is the most "modern ML" headline available: *"a small foundation model for human input motion."* **Needs the GPU desktop** and is the heaviest lift, hence last.

**Approach:**
- **Objective:** start with **masked-step reconstruction** on the existing 8-D event tensors (reuses `pipeline/sequences/dataset.py` + the AE encoder nearly unchanged). Contrastive is the stretch goal, not the first move.
- **Corpus:** **CaptchaSolve30k** (~20k human mouse sessions, already cached — see notebook 05). First **measure the captcha→game domain gap** with the existing drift tooling (`pipeline/monitoring/drift.py`) — a finding either way, and it de-risks the transfer claim.
- **Headline experiment:** **data-efficiency curve** — chunk AUC vs number of fine-tuning sessions (2, 5, 10, 18), pretrained vs from-scratch, on **both** GTA and CS2CD. Pretraining-wins-at-low-N = the foundation-model line; pretraining-doesn't-help = domain gap dominates, also a real result.
- The canonical LSTM-AE retrain that used to be staged for "the first desktop session" is already [done](#tooling-backlog) (2026-06-12).

**Deliverables:**
- [x] `pipeline/pretraining/` — masked-denoising objective + masking dataset + corpus→8-D adapters (`corpora.py`, `masking.py`, `pretrain.py`); `tests/test_pretraining.py` (13 tests)
- [x] `scripts/pretrain_encoder.py` (CUDA) — persists `models/pretrained_encoder.pt` (+ `_meta.json`); DVC-tracked
- [x] domain-gap report (captcha vs CS2CD vs GTA) via the existing KS/PSI drift tooling — `scripts/domain_gap_report.py` → `reports/pretraining_domain_gap.json`
- [x] `scripts/data_efficiency.py --domain {cs2cd,gta}` + `notebooks/21_pretraining.ipynb` — data-efficiency curves, pretrained vs scratch (5 seeds)
- [x] `docs/PRETRAINING.md` — objective, shared-8-D-schema decision, domain gap, the honest null

**Key results (2026-06-13):** pretrained-init vs from-scratch chunk-AUC is **statistically indistinguishable** — CS2CD Δ ≈ 0.000 at every budget (and the task is near-separable at random init, AUC ≈ 0.70); GTA Δ = −0.001…−0.005, within ±std. The captcha→game domain gap (PSI: `dt` ≈ 10–12 both targets; GTA `dx` ≈ 0.37) dominates. **A generic human-mouse corpus does not transfer as a foundation for game-input cheat detection at this scale** — honest negative result; next levers are a matched temporal encoding, an in-domain pretraining corpus, or a contrastive objective.

---

## Phase 8.1 — In-domain pretraining (does closing the domain gap rescue the null?)

> ✅ **Done (2026-06-15) — the null holds, and is *deeper than the domain gap*.** In-domain CS2CD pretraining (6 encoders: native-`dt` `s1` / `dt`-neutralised `s2` × 50/200/382 matches) transfers to GTA cheat detection **at or below from-scratch** (scratch 0.562; best in-domain config 0.559; captcha 0.557). **`s2` ≈ `s1`** (the `dt` mismatch wasn't the cause), **volume 50→200→382 is flat**, and **frozen ≤ fine-tune ≤ scratch**. The CS2CD-reference domain-gap re-run shows *why*: in-domain CS2 isn't even closer to GTA (`dx` PSI 0.88 — worse than captcha's 0.37; `dt` KS 0.95 persists). Phase 8 named two fixes — match the temporal encoding, use an in-domain corpus — and **8.1 ran both; neither moved the needle**. The binding constraint is the task/data regime (real-cheat chunk signal ~0.56, N≈18/3), not the corpus. Full write-up: [docs/PRETRAINING.md](PRETRAINING.md#phase-81--in-domain-pretraining-does-closing-the-domain-gap-rescue-the-null).

**Why:** Phase 8's null is attributed to the **captcha→game domain gap** (the `dt` channel, PSI ≈ 10–12). That diagnosis is testable: remove the suspected cause by pretraining *in-domain* on the **full public CS2CD release** (795 matches — far more match/player diversity than the 10-player slice Phase 8's data-efficiency curve ever saw) instead of out-of-domain captcha. This is the single variant that could flip the null, and it directly answers the reviewer's P2 "true transfer study" the way Phase 8 didn't (in-domain source + the frozen arm). It also addresses the honest scope gaps left by Phase 8: no frozen-encoder condition, and the diversity axis was capped at 10 players.

**Hypothesis:** in-domain pretraining shrinks the spatial-channel gap and *may* lift low-budget transfer; the `dt` mismatch (CS2 fixed ~15.6 ms tick vs GTA event-driven) will cap GTA transfer unless the temporal encoding is matched.

**Design:**
- **Conditions (the frozen arm Phase 8 skipped goes here):** (A) scratch / trainable; (B) CS2CD-pretrained / **frozen** encoder; (C) CS2CD-pretrained / fine-tuned.
- **Two source variants to isolate the `dt` term:** **S1** = pretrain on full CS2CD as-is (native tick); **S2** = same but resampled/encoded to the GTA temporal grid (the [PRETRAINING.md](PRETRAINING.md) "match temporal encoding" fix). The S1↔S2 diff *is* the measured `dt`-gap contribution.
- **Targets / splits:** GTA cheat-detection chunk-AUC (the real transfer test, **player-disjoint**); CS2CD held-out as a weak sanity check (near-separable at random init, **match/player-disjoint**). Build a match/player manifest + fixed splits *before* sampling any window.
- **New lever = pretraining diversity:** number of CS2CD matches/players (e.g. 50 / 200 / 795), the axis Phase 8 never scaled past 10. GTA fine-tune budget stays `[2, 5, 10, 15]` sessions.
- **Rigor:** 3 seeds + CIs on final configs only; re-run the domain-gap report with CS2CD-as-reference *before* interpreting any null.

**The real engineering bottleneck (not disk):** models are 0.08–0.21M params, so a 3070 is never compute-bound — with models this small the run is **dataloader-bound**, the GPU starving while the CPU re-encodes per-tick streams. Disk is a one-time ~52 GB download; column-projecting to `{dx, dy, FIRE, RIGHTCLICK, match_id, player_id, tick, label}` collapses the working set to single-digit GB (the "100–150 GB" figure assumes a naive pipeline). The 16 GB system RAM is the hard ceiling that mandates streaming. **Mitigation:** encode tensors **once** to cached per-match `.pt`/memmap shards, then sample chunks lazily per epoch (near-zero per-epoch CPU). Reuses `iter_batches` streaming (already in `corpora.py`), `train_ae`, `score_sequences`, `_chunk_cheat_labels` unchanged.

**Deliverables:**
- [x] **Step 0 — diversity axis verified ⇒ PLAYER_THIN.** `scripts/cs2cd_diversity_probe.py` → `reports/cs2cd_diversity_probe.json`: the full 795-match release is **player-anonymised** (`Player_1..10` *per match*, not linkable across matches), so there is no cross-match identity. The "diversity" lever is therefore a *stream-volume* axis (50/200/382 matches), **NOT** player diversity, and CS2CD splits are match-disjoint only. (The roadmap's match-diversity ⇒ player-diversity assumption was false — exactly the risk this step guarded.)
- [x] full-CS2CD column-projected ingest + cached per-match tensor shards (`pipeline/pretraining/cs2cd_full.py`; 478 legit matches → 16 GB shard cache; lazy LRU dataset + `ShardGroupedSampler`; 16 GB-RAM-safe). Match label = subdir (the full release has **no** `cheater_present` column, unlike the balanced sample).
- [x] `scripts/indomain_transfer.py` — arms A/B/C × sources s1/s2 × volume {50,200,382} (+ captcha comparison source) → `reports/phase8_1_indomain_transfer.json` (192 runs) + `reports/figures/phase8_1_indomain_transfer_gta.png`. **S2 = `dt`-neutralised** (CS2's `dt` is constant → "resample to the GTA grid" is a no-op post-normalisation; zeroing the channel is the clean causal test). Target = the **Phase-8 non-disjoint GTA pool** (directly comparable; a player-disjoint variant was tried but floored every arm at chance — cross-player shift dominates at 2 training players). CS2CD sanity arm **dropped** (no recoverable per-player cheat label in the full release).
- [x] re-run `domain_gap_report.py --reference cs2cd` → `reports/pretraining_domain_gap_cs2cd_ref.json` + figure
- [x] update `docs/PRETRAINING.md` with the verdict

**Pre-registered outcomes (all publishable):** (a) B/C beat A at low budget → in-domain source matters, the foundation-model line holds; (b) still flat → the null is deeper than domain (the task's signal is in obvious snaps, not learnable priors); (c) S2 ≫ S1 → the gap was specifically temporal. → **Outcome (b) fired:** still flat (in-domain ≤ scratch, `s2`≈`s1`, volume flat); (a) and (c) are ruled out. The one remaining untested Phase-8 lever — a **contrastive objective** — is now run in **Phase 8.2** (below).

---

## Phase 8.2 — Contrastive pretraining (does the *objective* matter?)

> ✅ **Done (2026-06-15) — the first non-null pretraining result.** Swapping reconstruction for a
> **contrastive** objective (NT-Xent over two augmented views), scored on the *frozen* 16-D embedding
> (one-class + linear-probe, not reconstruction-error AUC), beats **both** random-init and the Phase-8.1
> reconstruction encoder on every probe: in-domain `contrastive cs2cd@382` → Maha **0.550** / OCSVM 0.540 /
> kNN **0.585** / linear-probe **0.662**, vs random 0.481/0.477/0.486/0.547 and recon 0.511/0.528/0.493/0.603
> (seed±std ≈ 0.01–0.04). Modest (absolute ceiling ~0.55–0.66, near the weak ~0.56 real-cheat signal) and
> **in-domain-specific** (out-of-domain captcha contrastive is random-level on the one-class metrics), volume
> flat (saturates by ~50 matches) — but **directional and real**: the *objective* (magnitude-invariant
> contrastive vs magnitude-dominated reconstruction) was the lever, not corpus / capacity / `dt`. Full
> write-up: [docs/PRETRAINING.md](PRETRAINING.md#phase-82--contrastive-pretraining-does-the-objective-matter).

**Why:** Phase 8/8.1's reconstruction MSE is magnitude-dominated (the CS2CD "near-separable at random init"
caveat). A contrastive prior is magnitude-invariant by construction and is evaluated on the embedding
*directly* (kNN / one-class / linear-probe), not by reconstruction-error AUC — the last item on Phase 8's
"what would change the verdict" list, and the only swing left after 8/8.1 closed the rest negative.

**Deliverables:**
- [x] `pipeline/pretraining/augment.py` (jitter / scale / time-mask / crop-resize) + `contrastive.py`
  (NT-Xent, projection head, two-view datasets reusing the 8.1 shard pipeline, `pretrain_contrastive`) +
  `embed_eval.py` (frozen-embedding Mahalanobis / OCSVM / kNN / linear-probe); `tests/test_contrastive.py`
  (25 tests). Behaviour-preserving `_clean_window` extraction in `cs2cd_full.py`.
- [x] `scripts/contrastive_transfer.py --phase {pretrain,eval}` → 4 encoders
  (`models/pretrained_contrastive_{cs2cd_50,cs2cd_200,cs2cd_382,captcha}.pt`, DVC-tracked) +
  `reports/contrastive_transfer.json` (72 runs) + `reports/figures/phase8_2_contrastive_transfer_gta.png`.
- [x] study notebook [22](../notebooks/22_contrastive_pretraining.ipynb); `docs/PRETRAINING.md` §8.2.

**Verdict:** the pretraining arc resolves — reconstruction doesn't transfer (8/8.1 null), contrastive does,
modestly and in-domain. The binding constraint remains the small-N task/data regime (the absolute ceiling
didn't move past ~0.56), so this is a **representation-quality** win, not a deployable detector on its own.

---

## Phase 9 — Outcome-labelled telemetry

**Why:** [docs/SIGNALS.md](SIGNALS.md) says the *causally strongest* cheat signals are outcome/performance stats (headshot ratio, damage/shot, accuracy) that **cannot be collected from GTA** — the CS2CD sample is action-sparse (`damage_total = 0`). An instrumentable game with demo/log parsing gives kills/damage/accuracy **per window with ground truth**, unlocking the supervised detection notebook 16's D2 lever is waiting on, and a *real* re-attempt of the Phase 4.1 session-level aggregator (which saturated on synthetic sparse injection).

**Approach — spike early, execute late:**
- **One-evening feasibility spike (run during Phase 6):** can `demoparser2` pull per-tick view-angles + damage/kill events from a demo of *your own* CS2 match, and **clock-sync** it against your recorder running simultaneously? *That sync is the whole risk* — settle it before committing to capture sessions.
- **If yes:** schedule **dual-capture** sessions (recorder + demo). Offline / `-insecure` only for anything `cheat_sim`-related, per [docs/ETHICS.md](ETHICS.md).
- **Unlocks:** SIGNALS.md collection-roadmap item 1 (outcome features) → supervised detection (notebook 16 D2) → honest Phase 4.1 aggregator re-attempt with data that can support it.

**Deliverables:**
- [x] `pipeline/outcome/cs2_demo.py` + `scripts/parse_cs2_demo.py` — demo → per-window outcome features + marker-free motion clock-sync to recorder; `tests/test_outcome.py` (14 tests). `demoparser2` added to `requirements.txt`.
- [x] feasibility-spike note in `docs/CHEAT_DATA_COLLECTION.md` (sync-strategy **verdict: feasible**) — validated end-to-end on a real public `de_mirage` SourceTV demo (60 MB, 10 players); cross-correlation sync recovers injected offsets to <1 grid sample (`peak_corr ≈ 0.99`) and self-rejects an unrelated recorder session (`peak_corr ≈ 0.04`).
- [x] outcome-feature columns + `OUTCOME_FEATURE_COLS` slice (13 features) in `pipeline/features/run.py` — a *separate, additive* slice (not part of `FEATURE_COLS`; data-gated), aligned onto the `WINDOW_MS` grid.
- [ ] supervised detection benchmark (classifier head) + Phase 4.1 aggregator re-attempt — **awaits dual-capture data** (recorder + demo recorded simultaneously).

**Key results (2026-06-14, spike):** the feasibility question is answered **yes**. `demoparser2` extracts kills (with `headshot`/hitgroup/distance), `player_hurt` damage, `weapon_fire` shots and **per-tick view-angles**; aggregating the most-active player gave realistic combat stats (0.13 headshot ratio, 0.185 accuracy over 31 windows) + per-window aim dynamics. The clock-sync — the part the roadmap flagged as "the whole risk" — is solved without a manual marker by cross-correlating recorder mouse-motion vs demo view-angle-motion, and it **self-reports its own confidence** (`peak_corr`), so a bad sync is detectable rather than silent. The only number that still needs a genuine simultaneous capture is how high `peak_corr` climbs on a real recorder↔demo pair (sensitivity scaling + aim-punch noise).

---

## Cross-cutting

**Tech report (F) — grown, not written.** Condense the 14 docs into one ~10-page arXiv-style report — *"Input-level behavioural biometrics for cheat detection: what works at small N"* — plus a blog-post condensation. Submitting (even arXiv-only) converts "portfolio repo" into "research output," the currency at R&D-flavoured teams (Irdeto, Anybrain). The structure falls out of [docs/FINDINGS.md](FINDINGS.md): problem → data → windowed vs sequence → small-N rigor (ablation / CIs / calibration / serving-fidelity) → scale-up (Phase 6) → verification reframe (Phase 6/D) → evasion frontier (Phase 7) → pretraining (Phase 8).
- [x] `docs/REPORT.md` — **full draft (2026-06-13)**: §§1–10 + abstract & appendix drafted from the per-topic docs; §7 (evasion) now run and reported (Phase 7 done).
- [ ] **arXiv submission** + blog post — the remaining (manual) action.

**README rewrite at end of Phase 5** — ✅ effectively done (self-updating results block via `scripts/generate_results.py`, GIF hero, funnel). Keep the results block pipeline-backed as Phases 6–9 add rows.

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
| 6. Public-corpus ID + verification | no — uses **public** corpora (Balabit/SapiMouse) | ✅ start now; CPU-only |
| 7. Evasion frontier | no — uses existing legit windows + LSTM-AE | ✅ start now; CPU-friendly |
| 8. Pretraining | no — CaptchaSolve30k already cached | ⚠️ needs **GPU desktop** |
| 8.1. In-domain pretraining | no — full CS2CD release is public/re-downloadable | ⚠️ needs **GPU desktop** + one-time ~52 GB download |
| 8.2. Contrastive pretraining | no — reuses the 8.1 CS2CD shard cache | ⚠️ needs **GPU desktop** (no new download) |
| 9. Outcome telemetry | **yes** — new CS2 dual-capture sessions | ✅ spike done (parser + clock-sync validated on a public demo); execution capture lead-time bound |

**Extension order:** 6 → (9 spike) → 7 → 8, with the tech report (F) growing alongside. See [Extension sequencing](#extension-sequencing-phases-69--the-tech-report).

---

## Demo path (end of roadmap)

**60-second elevator demo:** live dashboard with replayed session + synthetic aimbot injected at minute 3, cheat-risk line spikes visibly.

**15-minute deep-dive demo (technical interview):**
*Foundation:* README → notebook 01 (raw data) → notebook 02 + `pipeline/features/run.py` → notebook 05 (external datasets) → notebook 06 (model comparison) → notebook 07 (behavioral differentiation).
*Anti-cheat-specific:* notebook 10 (adversarial bots) → notebook 09 (LSTM tutorial) → live dashboard (Phase 4) → notebook 12 (SHAP).
*Production:* `docs/MLOPS.md` → CI badge + workflow.

**Portfolio video (3 min, OBS-recorded):** what the project is → live demo playing GTA → inject synthetic aimbot, risk spikes → notebook 09 LSTM walkthrough → `docs/MLOPS.md` scroll. Embed in README, link from LinkedIn / CV.
