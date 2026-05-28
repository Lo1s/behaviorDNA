# Monitoring & Data Drift

> Phase 5c of the [BehaviorDNA roadmap](ROADMAP.md). Per-feature drift detection plus the runbook for when real recordings arrive.

This doc has two audiences:
1. Anyone who hasn't used drift metrics before — the **plain-English explanation** below.
2. Future-you when the GTA recordings land — the **Recording Arrival Runbook** at the bottom.

---

## What is data drift?

**Data drift** is when the data a model sees in production stops looking like the data it was trained on. The model silently gets worse because the world moved and the model didn't.

BehaviorDNA has a concrete, near-term case: every model so far is trained on **mock recordings** (mouse moving on a desktop, idle keystrokes). Real GTA gameplay will be statistically different — faster bursts, real aiming, WASD movement, combat clicks. Drift detection lets us *measure* exactly how different, per feature, so we know which features moved and whether the models need retraining rather than guessing.

We report two complementary metrics per feature: the **KS test** (is the change statistically significant?) and **PSI** (how big is the change, on an interpretable scale?).

---

## KS test (Kolmogorov–Smirnov, two-sample)

A statistical test answering: *"are these two samples drawn from the same distribution?"*

It works on the **cumulative distribution** of each sample — the running total of "what fraction of points are ≤ x". Plot both running totals on the same axes; the **KS statistic** is the single widest vertical gap between the two curves (a number from 0 = identical to 1 = completely separated). The test also returns a **p-value**:

- p-value < 0.05 → the gap is too big to be chance → the feature has **drifted**.
- p-value ≥ 0.05 → no statistically significant difference (often just because we don't have enough samples yet).

KS is great for significance but says nothing about *magnitude* in an interpretable unit — that's what PSI is for.

---

## PSI (Population Stability Index)

A single, interpretable number for *"how much did this feature's distribution shift?"*. It's the standard tool credit-risk and fraud teams use to decide when a model has gone stale.

Procedure:

1. Bin the **reference** sample into `bins` quantile buckets (deciles by default — each reference bucket holds ~10% of the reference data).
2. Compute the fraction of **reference** points and **current** points that land in each bucket.
3. Sum across buckets:

   ```
   PSI = Σ_i  (cur%_i − ref%_i) · ln(cur%_i / ref%_i)
   ```

Interpretation (rule of thumb):

| PSI | meaning |
|---|---|
| < 0.10 | no significant shift |
| 0.10 – 0.25 | moderate shift — keep an eye on it |
| > 0.25 | significant shift — investigate / retrain |

### Worked PSI example (by hand)

Reference splits evenly across two bins → `ref% = [0.5, 0.5]`.
Current shifts toward the second bin → `cur% = [0.3, 0.7]`.

```
PSI = (0.3 − 0.5)·ln(0.3/0.5) + (0.7 − 0.5)·ln(0.7/0.5)
    = (−0.2)·(−0.511)         + (0.2)·(0.336)
    = 0.102                   + 0.067
    = 0.169    → "moderate shift"
```

A bigger move (`cur% = [0.1, 0.9]`) gives PSI ≈ 0.64 → "significant". The unit test `tests/test_drift.py::TestPsi::test_worked_example_two_bins` checks exactly this number.

### Why report both KS and PSI?

KS gives a principled significance test; PSI gives an interpretable magnitude with battle-tested thresholds. With small samples KS often can't reach significance even when PSI is large — so the two together tell the fuller story (e.g. "PSI says this moved a lot, but we don't have enough data yet for KS to confirm").

---

## Usage

```bash
source .venv/bin/activate

# Default: compare the train split against the test split (sanity check on current data)
python -m pipeline.monitoring.drift

# Compare any two feature parquets, write the report somewhere specific
python -m pipeline.monitoring.drift \
    --reference data/splits/train.parquet \
    --current   data/splits/test.parquet \
    --out       reports/drift_report.csv
```

Output is a per-feature table sorted by PSI descending (worst-drifting first):

```
feature                ks_stat  ks_pvalue  ks_drifted   psi  psi_severity  n_ref  n_cur
scroll_direction_ratio    0.57       0.40       False  11.3   significant      7      3
wasd_rhythm               0.33       0.94       False   8.48  significant     12      3
...
```

Programmatic use:

```python
from pipeline.monitoring.drift import compute_drift_report
from pipeline.features.run import FEATURE_COLS

report = compute_drift_report(reference_df, current_df, FEATURE_COLS)
```

---

## Recording Arrival Runbook

**When the real GTA recordings land, do this in order:**

1. **Validate before ingesting.**
   ```bash
   python -m scripts.validate_recordings --dir <incoming_dir>
   ```
   Fix any `FAIL` rows before going further (schema problems, corrupt `event_count`, too-short sessions). `WARN` rows (missing activity label, mixed polling rates) are worth a look but won't block ingestion.

2. **Ingest + rebuild features.**
   ```bash
   cp <incoming_dir>/*.json data/raw/
   dvc repro                # ingestion → features (polling-rate normalisation now active) → split → train → evaluate
   ```

3. **Quantify the mock → real shift.** Snapshot the mock-era features first if you want a clean before/after; otherwise compare the existing train baseline against the new data:
   ```bash
   python -m pipeline.monitoring.drift \
       --reference <mock features.parquet> \
       --current  data/processed/features.parquet \
       --out reports/drift_mock_vs_real.csv
   ```
   Expect **significant PSI on several features** — that's the confirmation that the mock baseline was unrepresentative and that retraining on real data was necessary.

4. **Retrain the deep model + re-benchmark.**
   ```bash
   python -m scripts.train_lstm_ae                      # retrain LSTM-AE on real legit data
   python -m pipeline.adversarial.benchmark             # refresh detector AUCs
   ```
   Then refresh the results tables in [`docs/ADVERSARIAL.md`](ADVERSARIAL.md), [`docs/LSTM_AE.md`](LSTM_AE.md), and [`docs/STREAMING.md`](STREAMING.md).

5. **Regenerate the demo.**
   ```bash
   python -m scripts.build_phase4_demo
   ```

6. **Revisit the mock-data caveats.** Once real numbers are in, remove or update the "mock-data caveat" notes scattered across the docs and the README hero caption.

---

## See also

- [docs/FEATURES.md](FEATURES.md) — the 25 features the drift report runs over (incl. polling-rate normalisation)
- [docs/ROADMAP.md](ROADMAP.md) — Phase 5 plan + tooling backlog (CI pre-ingestion hook)
- `pipeline/monitoring/drift.py` — the implementation, with the same explanation inline
- `scripts/validate_recordings.py` — the QC gate referenced in step 1 of the runbook
