# Adversarial Bot Generation & Detection Benchmark

> Methodology and results write-up for [Phase 3](ROADMAP.md#phase-3--adversarial-bot-generation--detection-benchmark) of the BehaviorDNA roadmap.

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

---

## Results (current pipeline, 18 window features)

ROC AUC heatmap (every cell ≈ 0.5 = random chance):

| Detector | aimbot | macro | triggerbot |
|---|---|---|---|
| IsolationForest | 0.50 | 0.50 | 0.50 |
| LocalOutlierFactor | 0.50 | 0.51 | 0.50 |
| OneClassSVM | 0.50 | 0.55 | 0.50 |

**Read:** the current feature set fails to discriminate any of these cheats from legit play. See [notebooks/10_adversarial_bots.ipynb](../notebooks/10_adversarial_bots.ipynb) for the ROC grid and detection-rate heatmap.

This is **not** a failure of the synthetic data — the same data, scored at the event level (curvature, click reaction time, FFT coefficient of variation), separates cleanly. The failure is in aggregation:

1. **Temporal averaging** — a 150 ms aimbot snap is 0.5 % of a 30 s window
2. **Magnitude-only features** — `speed_mean`, `accel_mean`, `jitter` discard direction and geometry
3. **No timing features** — none of the 18 features measure the gap between `mouse_move` and the click that follows

---

## What closes the gap

This is exactly the motivation for the next two phases of the roadmap:

| Phase | What it adds | Closes which gap |
|---|---|---|
| Phase 1 — Trajectory & temporal features | `mouse_curvature_mean/std`, `flick_count`, `flick_precision`, `direction_changes_per_sec`, `path_efficiency`, `inter_click_movement`, `keystroke_overlap_ratio` | Magnitude-only features → geometric features |
| Phase 2 — LSTM autoencoder | Sequence model trained directly on raw events | Aggregation entirely |

After Phase 1, re-running this benchmark should light up the detection heatmap. After Phase 2, it should light up even more — especially for the soft aimbot variant.

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
| `data/synthetic/` | Generated hybrid sessions (drop-in compatible with ingestion) |
| `notebooks/10_adversarial_bots.ipynb` | Step-by-step tutorial with all visualizations |
| `reports/figures/adversarial_*.png` | Saved figures used in the notebook |
| `reports/adversarial/benchmark_results.csv` | Latest benchmark output |

---

## Reproducing

```bash
source .venv/bin/activate

# 1. Generate the synthetic dataset (15 legit recordings → 90 hybrid sessions)
python -m pipeline.adversarial.generate_dataset

# 2. Run the benchmark
python -m pipeline.adversarial.benchmark

# 3. Re-execute the notebook end-to-end
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/10_adversarial_bots.ipynb
```

Output appears in `reports/adversarial/benchmark_results.csv` and `reports/figures/adversarial_*.png`.
