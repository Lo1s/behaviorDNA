# Adversarial Bot Generation & Detection Benchmark

> Methodology and results write-up for [Phase 3](ROADMAP.md#phase-3--adversarial-bot-generation--detection-benchmark) of the BehaviorDNA roadmap.

> **Data status (2026-05-30).** The headline results are now measured on **18 real GTA sessions** (3 players) — see [On real data](#on-real-data-2026-05-30--18-real-gta-sessions-3-players). The chunk-level LSTM-AE detects aimbot at AUC 0.79 and triggerbot at 0.93 on real gameplay; classical window features stay at chance for aimbot. The earlier "evolution" tables (Phase 3 baseline → Phase 1) are retained as **historical mock-data baselines** that motivated each phase — they're labelled as such.

## Why this exists

A behavioural anti-cheat system that has not been benchmarked against synthetic cheats is making an untested claim. Real-world cheating data is scarce, often unlabelled, and frequently dirty. Synthetic cheats — generated from explicit, controllable mechanisms — let us answer two questions a hiring manager will ask:

1. **What does your detector catch?** (per-cheat-type detection rate, ROC AUC)
2. **What does it miss, and do you know why?** (failure analysis, gap to operational threshold)

This module provides both.

---

## Cheat models

Three cheat types are simulated. Each maps to a controllable signature in the input event stream.

### Aimbot

**Mechanism.** Automated cursor lock onto a target right before firing.

**Simulation.** For every `mouse_click` press event in a legit session, the trajectory in the `snap_duration_ms` window leading up to the click is overwritten with a smoothed interpolation from the cursor's actual position at `t - snap_duration_ms` to the click coordinates. The smoothing exponent controls how detectable the snap is:

$$
x(t) = x_0 + (x_{\text{target}} - x_0)\,\left(\frac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}}\right)^{1 - \text{smoothing}}
$$

| Preset | Smoothing | Behaviour |
|---|---|---|
| `obvious` | 0.0 | Linear ramp — most obvious aimbot |
| `medium` | 0.5 | Ease-out curve |
| `soft` | 0.85 | Fast initial movement then slow settle — hardest to detect |

**Measurable signature.** Mean turn-angle (curvature) along the snap window drops from ≈ 1.5 rad (human, with overshoot/correction) to ≈ 0.0 rad (perfectly straight or smoothly curved trajectory).

### Triggerbot

**Mechanism.** Fires the instant the crosshair crosses a target.

**Simulation.** For every click, the timestamp of the most recent preceding `mouse_move` event is shifted so the gap to the click becomes `reaction_time_ms` (default 3 ms). The mouse position itself is preserved — only the timing changes.

**Measurable signature.** Click reaction-time distribution collapses to a sharp spike near 0 ms. Humans show a broad distribution centred at 100–250 ms.

### Macro

**Mechanism.** Repeated keystroke pattern at perfectly regular intervals (e.g. recoil-control scripts).

**Simulation.** A contiguous time window of length `duration_ms` is selected (default starts at 30 % of session duration, lasts 8 s). All existing key events in that window are removed and replaced with a periodic press/release cycle through `keys` at exact `interval_ms` spacing.

**Measurable signature.** FFT of the binarised key-press signal shows a sharp spectral peak at `1000 / interval_ms` Hz. Human keystrokes produce broadband noise with no dominant frequency.

---

## Hybrid session construction

A "hybrid session" is a legit recording with one cheat type overlaid. The output preserves the BehaviorDNA recorder JSON schema and adds two fields:

- `cheat_label` ∈ {`legit`, `aimbot`, `triggerbot`, `macro`}
- `cheat_segments` — list of `[start_ms, end_ms]` ranges that were modified

This means **synthetic sessions are drop-in compatible with the ingestion pipeline**: `python -m pipeline.ingestion.run` will parse them without modification, and the `cheat_label` field flows through to `sessions.parquet` for downstream analysis.

The dataset generator (`pipeline.adversarial.generate_dataset`) produces 6 variants per legit session: 1 legit passthrough + 3 aimbot difficulties + 1 triggerbot + 1 macro.

---

## Benchmark

`pipeline.adversarial.benchmark` trains three unsupervised detectors **on legit-only feature rows** (the realistic production setup — detectors never see a cheat during training), then scores every feature row in the dataset.

**Detectors evaluated:**
- IsolationForest (`n_estimators=200, contamination=0.05`)
- LocalOutlierFactor (`n_neighbors=20, novelty=True, contamination=0.05`)
- OneClassSVM (`kernel="rbf", nu=0.05`)

**Features:** the 18 production `FEATURE_COLS` aggregated over 30-second windows.

**Metrics per detector × cheat type:**
- ROC AUC
- PR AUC
- Detection rate at FPR ≤ 5 %
- Mean anomaly score per class

> **In-sample vs held-out (review finding H1).** The default `run_benchmark`
> (and the per-type tables below) is an **in-sample** diagnostic: the scaler is
> fit on *all* rows (legit **and** cheat), the detectors on *all* legit rows,
> and those same legit rows are the negative baseline. That measures how
> separable a perturbation is *in the fitted data* — an approach/sanity proof,
> **not** a generalisation number. The held-out counterpart
> (`run_benchmark_heldout`, `python -m pipeline.adversarial.benchmark --heldout`
> → `reports/adversarial_benchmark_heldout.json`) splits **base recordings**,
> fits scaler + detectors on the *train* split's legit rows only, and reports a
> repeated-split 95% interval — see [Held-out base-session benchmark](#held-out-base-session-benchmark-classical-features) below.

---

## Results

### Baseline — original 18 features + per-window evaluation (Phase 3)

ROC AUC heatmap (every cell ≈ 0.5 = random chance):

| Detector | aimbot | macro | triggerbot |
|---|---|---|---|
| IsolationForest | 0.50 | 0.50 | 0.50 |
| LocalOutlierFactor | 0.50 | 0.51 | 0.50 |
| OneClassSVM | 0.50 | 0.55 | 0.50 |

**Read:** the original feature set fails to discriminate any cheat from legit play. See [notebooks/10_adversarial_bots.ipynb](../notebooks/10_adversarial_bots.ipynb).

This is **not** a failure of the synthetic data — the same data, scored at the event level (curvature, click reaction time, FFT coefficient of variation), separates cleanly. The failure was twofold:

1. **Aggregation dilution.** A 150 ms aimbot snap is 0.5 % of a 30 s window. `speed_mean`, `accel_mean`, `jitter` averaged across the whole window discard the snap signal.
2. **Cheat localization.** Each synthetic file is labelled `aimbot`/`macro`/`triggerbot` but only some of its windows actually contain cheat events. Per-window evaluation mixes cheat-containing windows with legit-looking siblings.

### After Phase 1 — 25 features + per-session aggregation

Phase 1 added 7 trajectory and timing features (see [FEATURES.md](FEATURES.md)) and the benchmark gained a `session_max` aggregation mode that flags whole sessions on their *maximum* per-window anomaly score — the realistic production decision.

| Detector | aimbot | macro | triggerbot |
|---|---|---|---|
| IsolationForest | 0.50 | 0.47 | 0.49 |
| LocalOutlierFactor | 0.49 | 0.54 | 0.48 |
| OneClassSVM | 0.53 | 0.68 | **0.87** |

**Read:** triggerbot detection jumped from chance to AUC 0.87 (`click_reaction_mean` collapses to ~0 ms in the cheat window, and per-session-max catches that window even though it lives in a mostly-normal session). Macro climbed to 0.68 via `keystroke_periodicity`. Aimbot remained stubbornly close to chance — the 150 ms snap signal is still buried by per-window mean aggregation.

**The aimbot gap is the case for Phase 2 — LSTM autoencoder on raw event sequences.** A sequence model bypasses window aggregation entirely and can flag the snap directly.

---

### On real data (2026-05-30) — 18 real GTA sessions, 3 players

The headline result. Synthetic cheats injected into the **18 real legit recordings**; LSTM-AE loaded from the persisted artifact; combined detector on an honest stratified 54/54-session split:

| Detector | aimbot | macro | triggerbot |
|---|---|---|---|
| IsolationForest | 0.50 | 0.49 | 0.50 |
| LocalOutlierFactor | 0.50 | 0.51 | 0.50 |
| OneClassSVM | 0.58 | 0.63 | 0.76 |
| **LSTMAutoencoder (chunk)** | **0.79** | 0.60 | **0.93** |
| LSTMAutoencoder (session-p95) | 0.51 | 0.51 | 0.51 |
| Combined (Phase 4 aggregator) | 0.42 | 0.52 | 0.61 |

**Read:** the thesis holds on real data. The **chunk-level LSTM-AE is the only detector that decisively beats chance on aimbot (0.79)** and is strongest on triggerbot (0.93) — exactly the cheats whose signal lives in short bursts the hand-crafted window features average away (classical detectors stay at ~0.50 for aimbot). Macro is the hardest (0.60): its periodic-click signature is partly visible to `keystroke_periodicity` (OneClassSVM 0.63) but subtle to the autoencoder. See `reports/figures/phase4_chunk_detection.png` for the per-chunk error distributions behind these AUCs.

The **Combined (aggregator) row is *below* chance for aimbot (0.42)** — it does *not* beat the best individual detector. The aggregator math is correct (`tests/test_aggregator.py`), but it combines *session-level* detector scores, and on real data those are near-chance (session-p95 ≈ 0.50, classical ≈ 0.50): a sparse cheat touches a minority of a session's chunks, and the isotonic calibrators are fit on only 18 legit sessions. Combining near-chance signals over a tiny calibration set is worse than just trusting the chunk-level detector. **The discriminative power is at the chunk level, not in the session-level combination** — recalibrating the aggregator (or aggregating the chunk signal directly) is tracked as Phase 4.1. See [docs/STREAMING.md](STREAMING.md) for the full write-up + the normalisation bug fixed this round.

<details><summary>Historical: same benchmark on the old mock dataset (kept for comparison)</summary>

| Detector | aimbot | macro | triggerbot |
|---|---|---|---|
| OneClassSVM | 0.53 | 0.68 | 0.87 |
| LSTMAutoencoder (chunk) | 0.78 | 0.70 | 0.96 |
| Combined (Phase 4) | 0.41 | 0.66 | 0.56 |

The mock numbers looked *higher* on triggerbot/macro because mock "legit" (idle desktop mouse) is trivially distinguishable from injected cheats; real gameplay is a harder, more honest baseline.
</details>

### Held-out base-session benchmark (classical features)

The tables above are **in-sample** (see the note under *Benchmark*). The held-out
counterpart (`--heldout`) splits the 18 base recordings, fits the scaler **and**
detectors on the train split's legit rows only, and evaluates held-out legit vs
held-out cheat variants per type, over 20 repeated 60/40 splits
(`reports/adversarial_benchmark_heldout.json`):

| Detector | aimbot | macro | triggerbot |
|---|---|---|---|
| IsolationForest | 0.50 [0.47–0.51] | 0.49 [0.46–0.52] | 0.43 [0.33–0.51] |
| LocalOutlierFactor | 0.49 [0.46–0.52] | 0.51 [0.50–0.53] | 0.49 [0.37–0.60] |
| OneClassSVM | 0.50 [0.48–0.52] | 0.50 [0.48–0.52] | 0.48 [0.27–0.59] |

*(ROC AUC, mean [2.5–97.5 percentile over splits]; per-session-max aggregation.)*

**Read:** out-of-sample, the classical window features are **at chance for every
cheat type** — the intervals straddle 0.50. This is the honest version of the
in-sample tables (where leakage and per-session-max on the fitted data can nudge
triggerbot up): hand-crafted *window* aggregates simply don't carry the
short-burst cheat signal once you can't peek at the test sessions. It is exactly
the result that motivates the **chunk-level LSTM-AE** (aimbot 0.79, triggerbot
0.93 on the same real sessions) — the gap is the aggregation window, not the
detector family. A held-out *LSTM-AE* number (retrain the autoencoder on train
bases only) is tracked in [ROADMAP](ROADMAP.md) as the remaining H1 step.

## What closes the gap

| Phase | What it adds | Closes which gap | Status |
|---|---|---|---|
| Phase 1 — Trajectory & temporal features | `mouse_curvature_*`, `path_efficiency`, `direction_changes_per_sec`, `click_reaction_mean`, `inter_click_movement`, `keystroke_periodicity` + per-session aggregation | Magnitude-only features → geometric features; per-window evaluation → per-session evaluation | ✅ done — triggerbot 0.87, macro 0.68 |
| Phase 2 — LSTM autoencoder | Sequence model on raw events, chunk-level scoring | Aimbot detection (window aggregation can't capture the 150 ms snap) | ✅ done — **real-data** aimbot chunk AUC 0.79, triggerbot 0.93 |
| Phase 4 — Bayesian session aggregator + streaming | Naive-Bayes log-odds combination, WebSocket API, live dashboard | Single-detector session aggregation diluting the cheat signal | ⚠️ infrastructure done; combined session-level score saturates on real data → recalibration is **Phase 4.1** |

Phase 2 added the LSTM autoencoder (see [docs/LSTM_AE.md](LSTM_AE.md)). At the **chunk level** it reaches AUC 0.79 on aimbot, 0.93 on triggerbot *on real data* — the model clearly learns to flag short cheat segments. At the **session level**, however, single-detector aggregation underperforms (≈ 0.50); the cheat signal exists in a handful of chunks but a percentile aggregator across hundreds of chunks dilutes it. Combining the chunk-level LSTM-AE signal directly (rather than per-session detector maxima) is the open Phase 4.1 problem.

---

## The arms race — detection vs evasion (Phase 7)

> [Phase 7](ROADMAP.md#phase-7--detection-vs-evasion-frontier). Generating a cheat is half the story; the day job at an anti-cheat company is the **arms race** — make the cheat *evade*, then characterise where the detector breaks. Code: [`pipeline/adversarial/humanizer.py`](../pipeline/adversarial/humanizer.py), [`scripts/evasion_frontier.py`](../scripts/evasion_frontier.py), [notebook 20](../notebooks/20_evasion_frontier.ipynb). Numbers below are from `reports/evasion_frontier.json` (18 real GTA legit sessions, 3 players, chunk-level LSTM-AE, 3 seeds).

### The humanizer (one knob λ ∈ [0, 1])

`humanizer.py` attaches a single **humanisation-strength** knob to each Phase-3 cheat. `λ=0` is the obvious bot (identical to the "obvious" preset); `λ=1` is humanised **toward the target player's own legit play**, sampled from a `PlayerBaseline` built from *that player's* recordings:

| Cheat | `λ=0` | `λ=1` | Detector signal it tries to erase |
|---|---|---|---|
| **aimbot** | linear teleport snap, no jitter | eased + overshooting + per-step jitter matched to the player's own move-step scale, stretched to a human flick time | the straight, super-smooth snap trajectory |
| **triggerbot** | ~3 ms sub-human click reaction | reaction sampled from a human RT model (~220 ± 40 ms) | the impossibly short click reaction |
| **macro** | perfectly periodic cadence | inter-key jitter matched to the player's own keystroke CV | the sharp periodicity / FFT peak |

Each `λ` produces (a) a humanised session the detector scores → **detection AUC(λ)**, and (b) a closed-form **utility(λ)** — the cheat's residual advantage over an unaided human (reaction-time / correction-speed edge; cadence consistency for the macro), which is monotone-decreasing by construction (utility 0 = behaves like the player = no advantage).

### The frontier

![Detection-vs-evasion frontier](../reports/figures/phase7_evasion_frontier.png)

Chunk-level detection AUC as the cheat is humanised (mean over 3 seeds; utility in the last column):

| λ | aimbot AUC | triggerbot AUC | macro AUC | utility (all cheats ≈) |
|---|---|---|---|---|
| 0.00 | 0.789 | **0.925** | 0.600 | ~1.0 |
| 0.25 | 0.782 | 0.866 | 0.518 | ~0.75 |
| 0.50 | 0.788 | 0.819 | 0.460 | ~0.50 |
| 0.75 | 0.805 | 0.783 | 0.414 | ~0.25 |
| 1.00 | **0.836** | 0.758 | 0.403 | **0.00** |

(The classical 30 s-window detector, OneClassSVM session-max, tells the same story: aimbot **rises** 0.59 → 0.72, triggerbot stays 0.70–0.78, macro hovers ~0.56.)

### The finding — the frontier favours the defender

Reading detection against utility, **there is no λ that is both undetectable and worth running:**

- **Aimbot — humanising backfires.** Detection *rises* with λ (chunk 0.79 → 0.84; window 0.59 → 0.72) while utility → 0. Player-matched jitter, overshoot and a longer eased snap inject *more* anomalous variance than a clean robotic teleport — which the autoencoder reconstructs easily because it is smooth and low-variance. The stealthiest aimbot is the obvious one, and it still loses all its speed edge once slowed to human reaction. No evasion path.
- **Triggerbot — bounded evasion.** Detection falls monotonically (0.93 → 0.76) as the reaction delay grows toward human, but **plateaus well above chance**: at full human RT (utility 0) it is still **0.758**. The compressed-reaction signature fades, but the residual click-timing structure stays visible.
- **Macro — the only one that reaches chance, at zero utility.** Detection decays to ~chance (0.60 → 0.40) as the cadence is jittered, but it gets there *exactly* when utility hits 0 — the macro's entire value **is** its perfect cadence, so jittering it away removes the signal and the advantage together.

**Equilibrium** (detection AUC at the smallest λ where utility has decayed to ≤ 0.2, "barely worth running"): aimbot **0.81** (λ=0.75), triggerbot **0.76** (λ=1.0), macro **0.40** (λ=1.0). For the two cheats that carry a real input-level advantage, the detector is still firmly above chance at the point the cheat stops being useful.

**Honest caveats.** This is a **closed-world** result: we humanise toward the player's *own logged* distribution (the attacker's best case) and score with a **fixed** detector — a true arms race retrains both sides, and a detector retrained on humanised cheats would shift these curves. The macro utility proxy (cadence consistency) is the least rigorous of the three axes. N is 18 sessions / 3 players. And the result is specific to **input-level** biometrics — outcome/aim-quality signals (Phase 9) are a different frontier.

---

## Production implications

The benchmark is unsupervised. In production:

- **False-positive cost dominates.** AUC = 0.95 sounds great but at 5 % FPR you ban 1-in-20 innocent players. Real deployments tune to FPR ≤ 0.1 %.
- **Defence in depth.** No single detector wins for all cheat types. A production stack runs multiple detectors (one tuned for trajectory, one for timing, one for keystroke patterns) and aggregates their scores via the [Phase 4](ROADMAP.md#phase-4--session-level-risk-aggregation--streaming-api) Bayesian aggregator.
- **Adversarial drift.** Soft aimbot variants are designed to evade naive geometric detectors. Continuous data collection + periodic retraining + drift monitoring ([Phase 5](ROADMAP.md#phase-5--statistical-rigor--mlops-polish)) is non-negotiable.

---

## Files

| Path | What |
|---|---|
| `pipeline/adversarial/bot_generator.py` | Aimbot / Triggerbot / Macro generators + derived metrics |
| `pipeline/adversarial/generate_dataset.py` | Builds the labelled dataset from `data/raw/` into `data/synthetic/` |
| `pipeline/adversarial/benchmark.py` | Runs detectors against synthetic data, writes `reports/adversarial/benchmark_results.csv` |
| `data/synthetic/` | Generated hybrid sessions — gitignored, regenerated deterministically (not stored) |
| `notebooks/10_adversarial_bots.ipynb` | Step-by-step tutorial with all visualizations |
| `reports/figures/adversarial_*.png` | Saved figures used in the notebook |
| `reports/adversarial/benchmark_results.csv` | Latest benchmark output |

---

## Reproducing

The synthetic dataset is **not stored in the repo** — it is ~520 MB of *derived*
data, regenerated deterministically (`bot_generator` seeds `random.Random(42)`)
from the versioned legit recordings. The LSTM-AE weights are GPU-trained (not
bit-reproducible), so they **are** versioned in DVC and pulled. A reviewer can
run all of the below from a fresh clone — no GPU or author workstation required
(step 0 needs a DagsHub token with read access; anonymous DVC pull is not
enabled on the remote — see the README Quickstart's DVC-remote setup).

```bash
source .venv/bin/activate

# 0. Pull the versioned artifacts: legit recordings (data/raw) + LSTM-AE model
dvc pull

# 1. Regenerate the labelled synthetic set (18 legit recordings → 108 hybrid
#    sessions); deterministic (~520 MB), written to data/synthetic/
python -m pipeline.adversarial.generate_dataset

# 2. Reproduce the headline detection AUCs (aimbot ~0.79, triggerbot ~0.93)
python -m pipeline.adversarial.benchmark

# 3. (optional) Re-execute the full tutorial notebook end-to-end
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/10_adversarial_bots.ipynb
```

Output appears in `reports/adversarial/benchmark_results.csv` and `reports/figures/adversarial_*.png`.
