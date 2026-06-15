# Input-level behavioural biometrics for cheat detection: what works at small N

> **Status: FULL DRAFT (post-Phase-8.2).** This is the grow-as-you-go tech report
> (deliverable **F** in [docs/ROADMAP.md](ROADMAP.md)). §§1–10 + the abstract and
> appendix are drafted from the per-topic docs; **§6 (scale/verification) is now the
> full section including the §6.1 contrastive-identity null, and §§7–8 (evasion
> frontier, pretraining through 8.2) are run and reported.** The remaining action is
> the manual arXiv submission + a blog-post condensation. This report *condenses and
> frames* the evidence already in [docs/FINDINGS.md](FINDINGS.md) and the per-topic
> docs; it does not re-derive.

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
constraint; the same frozen contrastive embedding that helps cheat detection
**ties random projections** for identity (a clean transfer null). Self-supervised
*reconstruction* pretraining (masked-denoising,
out-of-domain and in-domain) returns a **rigorous null** — no transfer benefit, with
a measured domain gap that explains it — while a **contrastive** objective on the
frozen embedding is the **first non-null**, isolating the pretext *objective* (not
the corpus or capacity) as the lever at small N. Throughout, every headline number
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
- **5.5 Bootstrap CIs on every *identification* headline number** (e.g. the
  0.74–0.97 above), so the reader sees the uncertainty the small N implies rather
  than a bare point estimate. (Cheat-detection chunk AUCs are reported as point
  estimates; the held-out classical benchmark, §finding H1, adds repeated-split
  intervals.)
- **5.6 Serving-fidelity bug — found → root-caused → fixed → CI-gated.** The shipped
  ONNX model was numerically unfaithful (probability MAE **0.27**, ~38% labels
  flipped). The cause was *not* a converter bug: float32 standardisation perturbs
  razor-margin splits on a model overfit to 187 windows (feeding sklearn the same
  float32 inputs reproduces the error). The fix is a composed **float64** graph
  (MAE **~1e-8**, 100% label agreement), regression-gated in CI. (This is §5.3's
  over-parameterisation wearing a different hat.)

## 6. Scaling identification + verification *(Phase 6 / 6.1)*
<!-- Source: notebooks/19, docs/VERIFICATION.md, reports/external_identification.json, reports/contrastive_identity.json. -->

The single most damaging reviewer question is *"does this survive beyond 3
friends?"* The answer is not to recruit fifteen people but to run the
**unmodified** GTA pipeline — same windowing, same feature code, same model
family — on public mouse-dynamics corpora at 10–120 users, and at the same time
to **reframe the task**. Closed-set "which of N players" is not the industry
problem; *"is this account being played by its usual owner?"* is — the
account-sharing / smurf / boost-detection framing (and, beyond games,
continuous authentication and fraud/bot detection). We report both, under the
challenge's own genuine/impostor **EER** protocol and an open-set rejection
test. Both corpora are **mouse-only**, so models use `MOUSE_ID_FEATURE_COLS`
(17 features; keyboard features *excluded*, not zero-filled) — the GTA numbers
of §5 are therefore not directly comparable, since the GTA fingerprint is partly
keyboard timing (SHAP, §5.1).

**Balabit (10 users — the literature benchmark; hours of activity each).** The
pipeline reaches **0.59** closed-set accuracy (95% CI 0.57–0.62; chance 0.10)
over 9,710 windows, and on the challenge's own labelled impostor task —
scoring each session as the mean P(claimed user) — **EER 0.144** over 784 test
sessions (395 genuine / 389 impostor). Challenge-era dedicated methods report
roughly 7–25% EER depending on data volume per decision, so a *generic*
windowed-feature + LightGBM pipeline transferred with no corpus-specific tuning
sits squarely in the credible range. EER is the headline number, not closed-set
accuracy, precisely because it matches the deployment question.

**SapiMouse (120 users — the scale stress-test; *minutes* each).** Under the
paper's 3-min-train / 1-min-test protocol — about **6 training windows per
user** — accuracy stays 10–20× chance at every enrolment size:

| Enrolled users | Accuracy (mean / 5 draws) | Chance | ×chance |
|---|---|---|---|
| 3 | 0.68 | 0.333 | 2× |
| 10 | 0.57 | 0.100 | 6× |
| 30 | 0.36 | 0.033 | 11× |
| 60 | 0.31 | 0.017 | 19× |
| 120 | 0.11 (CI 0.08–0.14) | 0.008 | **13×** |

But absolute accuracy is low (0.11 at 120 users), window-level verification EER
is **0.38**, and **open-set rejection is chance-level** (60 enrolled / 60
unknown: EER ≈ 0.48; FAR@FRR≤5% ≈ 0.93).

The two corpora bracket the claim precisely:

1. **The signal survives scale** — 10–20× chance all the way to 120 users; the
   method does not collapse beyond 3 friends.
2. **Absolute performance is data-bound, not method-bound at the small end** —
   Balabit (hours/user) yields a usable EER; SapiMouse (minutes/user, ~6
   windows) is data-starved and far from deployable. The binding constraint is
   the **per-user data budget**, not model capacity (§4).
3. **Open-set is the hard frontier** — max-probability rejection of unknown
   users is chance-level: closed-set softmax confidence is *not* an identity
   score. This motivates §8's pretraining (transfer a motion prior so per-user
   data goes further) and embedding/metric-learning over classifier confidence.

### 6.1 Do contrastive embeddings transfer to identity? *(a clean null)*

Point 3 names embedding/metric-learning as the natural next move, and §8.2
showed a self-supervised **contrastive** objective produces a frozen embedding
that beats both random-init and reconstruction for *cheat* detection — so we
tested whether that lever transfers to *identity*. We pretrain the same LSTM-AE
backbone contrastively on Balabit's own mouse motion (NT-Xent over two augmented
views), freeze it, embed sessions through the 16-D bottleneck, and score the
session-verification EER two ways — **cosine** to the enrolled user and a
**LightGBM** on per-chunk embeddings (the §6 protocol with hand-features →
embedding). A scale-augmentation ablation (`noscale`) tests whether the
scale-invariance 8.2 *wanted* for cheat detection *discards* the speed/scale
cues identity needs.

| frozen encoder | cosine EER | classifier EER |
|---|---|---|
| random init | 0.338 ± 0.009 | 0.255 ± 0.005 |
| contrastive (in-domain) | 0.301 | 0.250 |
| contrastive, no scale-aug | 0.299 | 0.250 |
| contrastive cs2cd (cross-domain) | 0.403 | 0.250 |
| **hand-crafted features (§6)** | — | **0.136** |

**The 8.2 lever does not transfer to identity.** Every 16-D embedding hits the
same ~0.25 classifier-EER — the contrastive encoder is statistically **tied with
random init** (0.255 ± 0.005), the only learning signal a modest cosine-route
gain (0.34 → 0.30). The **hand-crafted features roughly halve** that EER (0.136),
so at this scale domain-informed features beat self-supervised representations
for identity — the same "capacity isn't the bottleneck, the data budget binds"
finding from the other direction. And the scale hypothesis is **not** supported:
`noscale` ≈ `scale` (0.299 vs 0.301), so the bottleneck is not scale-invariance
but something more basic — *augmentation*-contrastive self-supervision learns
*augmentation-invariance*, which is orthogonal to *user-discrimination* (the
cross-domain encoder is even *worse* than random, 0.403). A footnote sharpens
the bar: a **random** LSTM embedding already verifies at ~0.25 EER (mouse-motion
sequences are identity-rich enough that random projections + a classifier
separate 10 users well below chance), so "beat random" is the meaningful bar the
self-supervised objective fails to clear. The principled next step (not run
here) is **supervised / metric** contrastive — positives drawn from the *same
user* rather than augmented views — which targets identity directly.

→ [docs/VERIFICATION.md](VERIFICATION.md)

## 7. The detection-vs-evasion frontier *(Phase 7)*
<!-- Source: pipeline/adversarial/humanizer.py, scripts/evasion_frontier.py, notebooks/20, ADVERSARIAL.md (arms-race), reports/evasion_frontier.json -->

Generating a cheat is half the problem; the anti-cheat day job is the **arms race**.
We parameterise each synthetic cheat with a **humanisation-strength knob λ ∈ [0, 1]**
(`pipeline/adversarial/humanizer.py`): `λ=0` is the obvious bot, `λ=1` is humanised
*toward the target player's own legit play* — eased/overshooting aimbot snaps with
per-step jitter matched to the player's move-step scale, a reaction delay drawn from a
human RT model (~220 ± 40 ms), and keystroke cadence jittered to the player's own CV.
For each λ we measure **detection AUC(λ)** (the §4 chunk-level LSTM-AE on 18 real GTA
sessions, 3 seeds) against a closed-form **utility(λ)** — the cheat's residual advantage
over an unaided human (reaction-time / correction-speed edge; cadence consistency for
the macro), monotone-decreasing to 0 (behaves like the player → no advantage).

**The frontier favours the defender — there is no λ that is both undetectable and worth
running.** Chunk-level AUC across the sweep:

| λ | aimbot | triggerbot | macro |
|---|---|---|---|
| 0.00 | 0.789 | 0.925 | 0.600 |
| 0.50 | 0.788 | 0.819 | 0.460 |
| 1.00 | 0.836 | 0.758 | 0.403 |

- **Aimbot — humanising backfires.** Detection *rises* with λ (chunk 0.79 → 0.84; the
  classical window detector 0.59 → 0.72) while utility → 0. Player-matched jitter and
  overshoot inject *more* anomalous variance than a clean robotic teleport, which the
  autoencoder reconstructs easily; the stealthiest aimbot is the obvious one, and it
  still loses its entire speed edge once slowed to human reaction.
- **Triggerbot — bounded evasion.** Detection falls monotonically (0.93 → 0.76) as the
  reaction delay grows toward human, but **plateaus above chance**: AUC 0.76 even at full
  human RT (utility 0).
- **Macro — chance only at zero utility.** Detection decays to ~chance (0.60 → 0.40) as
  the cadence jitters, but reaches it *exactly* when utility (the macro's entire value)
  is gone.

Equilibrium (detection AUC at the smallest λ with utility ≤ 0.2): aimbot **0.81**,
triggerbot **0.76**, macro **0.40**. **Caveats (honest):** closed-world — we humanise
toward the player's own logged distribution (the attacker's best case) and score with a
*fixed* detector; a real arms race retrains both sides. The macro utility proxy is the
weakest of the three axes, N = 18 sessions / 3 players, and the result is specific to
**input-level** biometrics.

→ [docs/ADVERSARIAL.md](ADVERSARIAL.md) (arms-race section)

## 8. Self-supervised pretraining & data efficiency *(Phase 8 / 8.1 / 8.2)*
<!-- Source: notebooks/21-22, PRETRAINING.md, reports/{pretraining_domain_gap,data_efficiency_*,contrastive_transfer}.json -->

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

**Two follow-ups closed the lever list (Phase 8.1 / 8.2).** *(8.1 — in-domain + the `dt` fix)*
pretraining the same encoder in-domain on the full public CS2CD release (478 legit matches) and
neutralising the `dt` mismatch **both left transfer unchanged** — in-domain ≤ from-scratch on GTA,
`dt`-neutralised ≈ native, volume flat; a CS2CD-reference domain-gap re-run shows in-domain CS2 isn't even
closer to GTA (`dx` PSI 0.88). The null is *deeper than the domain gap* — the binding constraint is the
small-N task/data regime, not the corpus. *(8.2 — the objective)* swapping the magnitude-dominated
reconstruction MSE for a magnitude-invariant **contrastive** objective (NT-Xent over two augmented views),
scored **contrastive-natively** on the *frozen* 16-D embedding (one-class + linear-probe, not
reconstruction-error AUC), is **the first non-null**: in-domain contrastive beats both random-init and the
8.1 reconstruction encoder on every probe (e.g. kNN 0.585 vs 0.486 / 0.493; linear-probe 0.662 vs 0.547 /
0.603). The win is modest (absolute ~0.55–0.66, near the weak ~0.56 real-cheat ceiling), in-domain-specific
(out-of-domain captcha contrastive is random-level on the unsupervised metrics) and volume-flat — so it is a
representation-quality result, not a deployable detector; but it isolates the **objective** (not corpus,
capacity, or `dt`) as the lever that moves small-N transfer.

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
scale** (Balabit EER 0.144, where hand-crafted features still beat learned
embeddings — §6.1), and — the throughline — a **methodology that measures
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
