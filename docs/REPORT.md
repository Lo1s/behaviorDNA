# Input-level behavioural biometrics for cheat detection: what works at small N

> **Status: FULL DRAFT (post-Phase-8).** This is the grow-as-you-go tech report
> (deliverable **F** in [docs/ROADMAP.md](ROADMAP.md)). §§1–6, 8, 9, 10 + the
> abstract and appendix are drafted from the per-topic docs; **§7 (evasion
> frontier) is Phase 7 and not yet run — it is flagged as planned, not
> reported.** The remaining action is the manual arXiv submission + a blog-post
> condensation. This report *condenses and frames* the evidence already in
> [docs/FINDINGS.md](FINDINGS.md) and the per-topic docs; it does not re-derive.

**Thesis.** Player identification and automation detection from raw mouse/keyboard
telemetry is feasible, but the honest story at small data scale is one of
*measured limits and checked claims* as much as headline metrics. We report what
works, what doesn't, and how we verified each.

---

## Abstract

We build an end-to-end system for player identification and automation (cheat)
detection from raw mouse/keyboard telemetry, and — more than the headline metrics
— we report an honest account of *what survives scrutiny at small data scale*. On
real gameplay (18 sessions, 3 players) a windowed-feature identifier reaches 0.85
3-class accuracy, but we isolate a hardware confound and report the narrower,
honest claim: **~0.75 on a same-hardware pair where only the human differs**. A
chunk-level LSTM autoencoder on raw event sequences detects synthetic aimbot at
**0.79 AUC** and triggerbot at **0.93**, where 30-second window features sit at
chance — but we show the *session-level* aggregation saturates at ~0.50 and prove
why before shipping. Run unmodified on public corpora, the pipeline reaches
**EER 0.144** on the Balabit mouse-dynamics challenge (10 users) and survives to
120 users on SapiMouse while exposing the per-user **data budget** as the binding
constraint. A self-supervised pretraining experiment (masked-denoising on ~17.7k
unlabelled human-mouse sessions) returns a **rigorous null** — no transfer benefit
— with a measured domain gap that explains it. Throughout, every headline number
is paired with the check that validates it: confound isolation, drift
quantification, ablation, calibration, bootstrap CIs, and a bit-faithful serving
export. The transferable asset is the discipline, not just the metrics.

## 1. Introduction & problem framing

Modern anti-cheat increasingly leans on **behavioural** signals at the **input
layer** — the millisecond-resolution stream of mouse moves, clicks, and keystrokes
— because that layer is observable without kernel/memory access and is hard to
spoof convincingly: an automation tool must reproduce not just *what* a human does
but *how* (the kinematics, timing, and micro-variability). We treat two tasks:

- **Identification → verification.** Closed-set "which of N players" is not the
  industry problem; *"is this account being played by its usual owner?"* is — the
  account-sharing / smurf / boost-detection framing, and the continuous-auth /
  fraud framing outside gaming. We report both, and argue (with data, §6) the
  verification framing is the product-relevant one.
- **Automation detection.** Aimbot (superhuman aim correction), triggerbot
  (zero-latency fire), and macro (perfectly periodic keystrokes) — each a distinct
  input signature.

The governing constraint is **cost asymmetry**: a false ban loses a paying player
and reputation, so precision, calibration, and *checked claims* matter more than a
headline accuracy point. That asymmetry is why this report's centre of gravity is
§5 (rigor): a system that *measures and verifies* is more deployable than one that
merely scores well once.

→ [README.md](../README.md), [docs/ETHICS.md](ETHICS.md), [docs/THREAT_MODEL.md](THREAT_MODEL.md)

## 2. Data

- **Real GTA V recordings — 18 sessions, 3 players** (shotik 5, dninix 8, hydra 5),
  all 1000 Hz, mixed DPI (800/1600). Crucially, **two players (hydra, dninix) share
  one PC and settings** — the controlled pair where only the human varies (§5.1).
  This is the credibility ceiling (§9), and we treat all real-data numbers as
  directional, not production guarantees.
- **Synthetic-cheat harness.** Aimbot/triggerbot/macro signatures injected into
  legit recordings while preserving the recorder JSON schema (deterministic, seed
  42), so the full pipeline accepts them unchanged. A separate, deliberately
  **safe** live actuator (`cheat_sim`) produces the cheat *input signature* only —
  no target acquisition, no memory reads, offline-only — for a controllable
  ground-truth capture path.
- **External corpora.** CS2CD (Counter-Strike 2, real third-party cheats, 10
  players, game-state telemetry); Balabit Mouse-Dynamics Challenge (10 users, the
  classic EER benchmark); SapiMouse (120 users, minutes each — the scale stress
  test); CaptchaSolve30k (~17.7k unlabelled human-mouse sessions, the pretraining
  corpus). Per-corpus audits, including the **hardware-confound** and
  domain-shift caveats, are in [docs/DATASET_CARDS.md](DATASET_CARDS.md).

→ [docs/ETHICS.md](ETHICS.md), [docs/DATA_LAYOUT.md](DATA_LAYOUT.md), [docs/CHEAT_DATA_COLLECTION.md](CHEAT_DATA_COLLECTION.md), [FINDINGS](FINDINGS.md) #1

## 3. Features: windowed aggregates vs raw sequences

The classical path computes **30 features over 30-second non-overlapping windows**.
Two normalisations make mixed hardware comparable: mouse kinematics are divided by
`norm_factor = sensitivity·dpi/800`, and the three polling-rate-proportional
features are scaled to a 1000 Hz reference (`rate_norm = 1000/polling_rate`). No
z-scaling happens in the feature stage — `StandardScaler` is applied **only on the
training fold** inside training, to prevent train/test leakage.

The headline limitation is structural: **a 150 ms aimbot snap is averaged away by a
30 s window** → window features detect it at ~0.50 AUC (chance). This directly
motivates the raw-sequence model (§4). We also **decouple the feature sets by
task**: `ID_FEATURE_COLS` (25, the identifier) vs `CHEAT_FEATURE_COLS` (30, the
detectors); each trained artifact carries its own `feature_cols`. At N=18 this
stops cheat-oriented features from trading against identification (§5.3).

→ [docs/FEATURES.md](FEATURES.md), [docs/SIGNALS.md](SIGNALS.md)

## 4. Models

- **Identifier.** A config-driven classifier (LightGBM default; 7 types available)
  trained on **player-stratified, session-level** splits — every player appears in
  train/val/test, whole sessions stay in one fold, and cheat sessions are excluded
  (you fingerprint players from legit play).
- **Sequence anomaly model.** A bidirectional **LSTM autoencoder** on raw 8-D event
  tensors `(dt, dx, dy, + 5 event-type one-hots)`, trained on legit play only;
  cheat chunks reconstruct poorly. An architecture comparison (LSTM vs TCN vs
  Transformer autoencoders, same loop, same eval) finds all three **competitive at
  N=18** (Transformer marginally best, TCN cheapest) — i.e. **capacity is not the
  bottleneck, data is**, the thesis §6 and §8 then test directly.
- **Session aggregator.** A Naive-Bayes log-odds combiner with per-detector
  isotonic calibration and a configurable cheat-rate prior, fed by a
  transport-independent streaming engine (WebSocket / replay / offline).

→ [docs/LSTM_AE.md](LSTM_AE.md), [docs/ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md), [docs/STREAMING.md](STREAMING.md)

## 5. Rigor at small N (the core contribution)

This is the section that distinguishes the report: each claim is paired with the
check that validates (or bounds) it. → [docs/FINDINGS.md](FINDINGS.md)

- **5.1 Hardware confound isolated.** 3-class ID scores **0.85** (95% CI 0.74–0.97;
  the interval is wide because the test set is 34 windows, and we say so). But one
  player sits on different hardware and is trivially separable, inflating the
  number. Evaluating *just the same-hardware pair* gives **0.75** (vs 0.65
  majority) — and SHAP attributes the separation to **timing/rhythm** features
  (`click_interval_std`, `keystroke_periodicity`, `burst_rate`), i.e. a behavioural
  fingerprint, not a hardware tell. The honest claim is the narrower one.
- **5.2 Drift measured, not assumed.** Replacing the original mock data with real
  recordings, a per-feature KS + PSI report showed **20 of 25 features drifted
  significantly** (`wasd_rhythm` PSI 9.4; `speed`/`accel` 6.6–8.0). That measured
  shift — not a hunch — triggered retraining on real data.
- **5.3 Ablation → over-parameterisation at N=18.** Splitting 25 features into 5
  families (8-seed-averaged), single families already classify well alone
  (mouse-kinematics, keyboard each ≈0.75–0.79) and **dropping a whole family often
  *raises* validation accuracy.** So the planned feature-expansion phase was
  **deferred on evidence** — the lever at this scale is more data or feature
  *reduction*, not more features.
- **5.4 Calibration helps but not blindly.** On measured ECE + multiclass Brier,
  **isotonic** improved Brier (0.275 → 0.224) holding accuracy; **Platt made it
  worse** — the small-data fragility you'd predict at 46 calibration windows.
  Reported as found.
- **5.5 Bootstrap CIs on every headline number** (e.g. the 0.74–0.97 above), so the
  reader sees the uncertainty the small N implies rather than a bare point estimate.
- **5.6 Serving-fidelity bug — found → root-caused → fixed → CI-gated.** The shipped
  ONNX model was numerically unfaithful (probability MAE **0.27**, ~38% labels
  flipped). The cause was *not* a converter bug: float32 standardisation perturbs
  razor-margin splits on a model overfit to 187 windows (feeding sklearn the same
  float32 inputs reproduces the error). The fix is a composed **float64** graph
  (MAE **~1e-8**, 100% label agreement), regression-gated in CI. (This is §5.3's
  over-parameterisation wearing a different hat.)

## 6. Scaling identification + verification *(Phase 6)*
<!-- Source: notebooks/19, docs/VERIFICATION.md, reports/external_identification.json. -->

**Draft (2026-06-11).** We run the unmodified windowed-feature pipeline
(mouse-only slice, 17 features) on two public corpora. On **Balabit** (10
users, hours of activity each) the pipeline reaches 0.59 closed-set accuracy
(chance 0.10) and — on the challenge's own labelled impostor task — **EER
0.144** over 784 test sessions, in the credible range for challenge-era
dedicated methods. On **SapiMouse** (120 users, *minutes* each, the paper's
3-min-train / 1-min-test protocol) accuracy stays 10–20× chance at every
enrolment size up to 120, but absolute accuracy is low (0.11) and **open-set
rejection is chance-level**: with ~6 training windows per user, 30-second
aggregate features are data-starved and closed-set softmax confidence is not
an identity score. The two corpora bracket the claim precisely: the
behavioural signal survives scale; the per-user data budget — not model
capacity (§4) — is the binding constraint, motivating §8's pretraining and
embedding-based verification over classifier confidence.

→ [docs/VERIFICATION.md](VERIFICATION.md)

## 7. The detection-vs-evasion frontier *(Phase 7 — planned, not yet run)*

> **Not reported.** Phase 7 is not yet executed; this section states the *planned*
> method only, and contains no results. It will be filled before any submission
> that claims an evasion frontier.

The most anti-cheat-native experiment: parameterise the synthetic cheats with a
**humanizer strength knob λ ∈ [0,1]** (reaction-delay injection, minimum-jerk
smoothing of aimbot snaps, kinematic noise matched to the target player's own
legit windows — extending the easing/overshoot/jitter planners that already exist
in `pipeline/adversarial/live_cheat.py`), then plot two curves against λ: **detector
AUC(λ)** (the §4 chunk model) and **cheat utility(λ)** (residual reaction-time
advantage). The headline would be the equilibrium region — *"humanized enough to
evade ≈ no longer worth running."*

→ [docs/ADVERSARIAL.md](ADVERSARIAL.md) (arms-race section, planned)

## 8. Self-supervised pretraining & data efficiency *(Phase 8)*
<!-- Source: notebooks/21, PRETRAINING.md, reports/{pretraining_domain_gap,data_efficiency_*}.json -->

We tested the "data is the bottleneck" thesis directly: masked-denoising-pretrain the LSTM-AE on
CaptchaSolve30k (≈17.7k unlabelled human-mouse sessions), transfer the full weights, and measure the
**data-efficiency curve** — pretrained-init vs from-scratch chunk-AUC as a function of fine-tuning
budget — on both CS2CD (real cheats, 10 players) and GTA (synthetic, N=18). The precondition for a
fair test is a single 8-D event-tensor schema shared across all three corpora (the sampled per-tick
captcha/CS2 streams re-encoded into the GTA event schema).

**Result: a rigorous null.** Pretrained ≈ scratch at every budget (CS2CD Δ ≈ 0.000; GTA Δ =
−0.001…−0.005, within ±std). A domain-gap report (the project's own KS/PSI drift tooling, reference =
captcha) explains it mechanistically: the **temporal channel `dt` is PSI ≈ 10–12 mismatched** against
*both* games (fixed-tick captcha vs CS2's ~15.6 ms tick and GTA's event-driven stream with idle gaps),
and GTA's mouse-delta geometry differs (`dx` PSI 0.37) — while, tellingly, captcha→CS2 movement
geometry transfers well (`dx/dy` PSI < 0.1). Two honesty caveats sharpen it: CS2CD is near-separable at
random init (a magnitude artifact → a weak discriminator of the transfer question), and GTA
fine-tuning *itself* helps while pretraining specifically does not. The publishable takeaway: **a
generic human-mouse corpus is not a drop-in foundation for game-input cheat detection at this scale** —
the next levers are a matched temporal encoding, an in-domain pretraining corpus, or a contrastive
objective. This is the data-not-capacity thesis confirmed from the other direction: out-of-domain
*data* didn't substitute for in-domain data.

→ [docs/PRETRAINING.md](PRETRAINING.md)

## 9. Limitations & ethics

- **Scale.** 3 players / 1 cheat recorder is the credibility ceiling; real-data CIs
  are wide (§5.1) and all such numbers are directional. The public-corpus work (§6)
  is the answer to *"does it survive beyond 3 friends?"* — and its honest finding is
  that the **per-user data budget**, not capacity, binds.
- **Sensor scope.** Input-level only. The causally strongest cheat signals —
  outcome/performance stats (headshot ratio, damage/shot), and system/process
  signals (injected modules, frame-time anomalies) — are **not collected**; the
  collection roadmap is in [SIGNALS.md](SIGNALS.md), and outcome-labelled telemetry
  is the (unstarted) Phase 9.
- **Session-level aggregation is unsolved on available data.** The chunk detector is
  strong (§5.4 / FINDINGS #4) but every session aggregation saturates at ~0.50 on
  *synthetic sparse* injection; the unblock is real *continuous* cheat data, not more
  modelling. We proved this before shipping a saturated score.
- **The cheat actuator is safe by construction** — an offline-only input-signature
  generator, not a functional competitive cheat. Out of scope: real bans /
  enforcement actions. → [ETHICS.md](ETHICS.md), [THREAT_MODEL.md](THREAT_MODEL.md)

## 10. Conclusion

What survived scrutiny: a **same-hardware behavioural fingerprint** (~0.75, driven
by timing/rhythm), **chunk-level deep cheat detection** (aimbot 0.79 / triggerbot
0.93 where window features are at chance), **literature-comparable verification at
scale** (Balabit EER 0.144), and — the throughline — a **methodology that measures
and verifies**: confound isolation, drift quantification, ablation, calibration,
bootstrap CIs, and a bit-faithful serving export. What is data-gated: session-level
risk aggregation, outcome-based detection, and (newly) out-of-domain pretraining
transfer (the §8 null). The one-line takeaway for an anti-cheat R&D reader: **the
input-biometric signal is real but data-bound; the transferable asset is the
discipline of checking each claim — exactly what matters where a wrong "ban" is the
expensive failure.**

---

### Appendix — reproducibility

The system is a 5-stage **DVC pipeline** (ingestion → features → split → train →
evaluate); model/data artifacts are DVC-tracked (`lstm_ae.pt`, `serving_bundle.pkl`,
`pretrained_encoder.pt`) and the synthetic-cheat set is regenerated deterministically
(seed 42). **CI gates**: full pytest, README-results staleness (`generate_results
--check`), ONNX float64 parity (MAE < 1e-6), and a pre-ingestion recording-QC gate.
Optional **MLflow** registry (Staging → Production) logs each run; missing
credentials degrade silently. All experiments seed 42.

→ [docs/MLOPS.md](MLOPS.md), [docs/MONITORING.md](MONITORING.md)
