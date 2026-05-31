# MLOps — drift monitoring, model registry, CI gates

> Phase 5e of the [roadmap](ROADMAP.md). The production-maturity layer: how a model
> gets *monitored*, *promoted*, and *gated* — not just trained.

Three pieces, each a small, honest slice of a real MLOps loop:

1. **Drift monitoring** — is the live data still like the training data?
2. **Model registry + promotion** — which trained model is the one we'd serve?
3. **CI gates** — what stops a bad model or a bad recording from shipping?

---

## 1. Drift monitoring

When new recordings arrive, quantify how far they've moved from the training
distribution before trusting the model on them:

```bash
python -m pipeline.monitoring.drift --reference <baseline.parquet> --current <new.parquet>
```

Per-feature KS test + PSI, with industry-standard severity thresholds. Full
explanation, worked example, and the visual mock→real walkthrough:
[docs/MONITORING.md](MONITORING.md) and `notebooks/14_drift.ipynb`. This is the
trigger for retraining — we retrained on real GTA data only after the drift report
showed 20/25 features had shifted significantly.

---

## 2. Model registry + promotion

**Training registers a candidate.** Every `dvc repro` (training stage) logs a run to
the DagsHub-hosted MLflow and — for identification models — logs a servable
`scaler + classifier` pipeline and **registers it as a new version** of
`behaviordna-identifier` (`IDENTIFIER_REGISTRY_NAME`). Skipped gracefully when no
MLflow credentials are present (CI/offline).

**Promotion is a separate, deliberate step.** A trained version is a *candidate*, not
automatically the served model:

```bash
python -m scripts.promote_model            # promote best version → Production
python -m scripts.promote_model --metric val_accuracy --stage Production
```

`promote_model.py` looks up every registered version, finds the one whose run has the
best `val_accuracy`, and transitions **that** version to the **Production** stage
(archiving the previous Production version). Selection logic (`select_best_run`) is
pure and unit-tested (`tests/test_promote_model.py`); the live registry write degrades
gracefully if the backend rejects it.

**Verified live (2026-05-31):** training registered `behaviordna-identifier` v1 on
DagsHub; `promote_model` selected it (val_accuracy 0.739) and promoted it to Production.

> **Note on stages vs aliases.** We use the classic `Staging → Production` *stages* the
> roadmap calls for. MLflow ≥ 2.9 deprecates stages in favour of *aliases* (you'll see a
> `FutureWarning`); `promote_model.py` falls back to a `production` **alias** automatically
> if a backend has already removed stage transitions, so it keeps working either way.

---

## 3. CI gates (`.github/workflows/ci.yml`)

Two jobs guard `main`:

- **`lint-and-test`** (every push/PR): `ruff` + `black --check` + `pytest` with a
  **65% coverage floor** (`--cov-fail-under=65`).
- **`dvc-repro`** (main only, needs DagsHub creds): pulls data, **validates the raw
  recordings**, reproduces the full pipeline, pushes results. The validation step is the
  **pre-ingestion gate**:

  ```yaml
  - name: Validate recordings (pre-ingestion gate)
    run: python -m scripts.validate_recordings --dir data/raw
  ```

  `validate_recordings.py` exits non-zero on any malformed recording (bad schema, corrupt
  `event_count`, too-short session), so a bad batch **fails the build before it can poison
  the pipeline** — rather than surfacing as a confusing training error downstream.

---

## Honest notes

- The registry path is verified on DagsHub but is intentionally **best-effort**: every
  MLflow/registry call degrades gracefully (logs + continues / exits 0) so neither an
  offline dev box nor CI without credentials is ever broken by it.
- Promotion currently ranks by `val_accuracy`. With only 46 validation windows that's a
  noisy criterion (see `notebooks/13_calibration.ipynb` for the small-N caveats) — once
  more recordings land, promote on a more stable held-out metric.
