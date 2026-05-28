# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

BehaviorDNA is a game-agnostic ML system for player behavioural biometrics — identifying players and detecting automation from raw mouse/keyboard telemetry. It is built as a portfolio piece targeting AI/ML roles at anti-cheat companies (Anybrain, Irdeto, BattlEye, Riot, etc.). Roadmap and current phase status live in [docs/ROADMAP.md](docs/ROADMAP.md).

## Common commands

```bash
# Always activate the venv first (most commands fail without it)
source .venv/bin/activate

# Full pipeline (only re-runs stages whose inputs changed)
dvc repro

# Re-run one stage
dvc repro features

# Tests
pytest -q                                 # full suite (~3s, ~100 tests)
pytest tests/test_features.py::test_x     # single test
pytest -q --no-header -k adversarial      # by keyword

# Lint + format (pre-commit hooks run these automatically)
ruff check . && black --check .
ruff check --fix .                        # auto-fix safe issues
black .                                   # apply formatting

# Execute a notebook end-to-end
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
  --inplace notebooks/<name>.ipynb

# Run the API locally
uvicorn api.main:app --reload --port 8000

# Run the dashboard
streamlit run dashboard/app.py            # http://localhost:8501

# Generate the synthetic-cheat dataset + benchmark detectors
python -m pipeline.adversarial.generate_dataset
python -m pipeline.adversarial.benchmark
```

## Architecture (the parts you can't see from `ls`)

The system is a **5-stage DVC pipeline** defined in `dvc.yaml`. Each stage is a Python module under `pipeline/` and produces a Parquet (or model artifact) that the next stage depends on. Files in `data/raw/`, `data/processed/`, `data/splits/`, `data/external/`, `data/synthetic/`, and `models/` are all DVC-managed (gitignored except `.gitkeep`).

```
collector/ (Windows-side .exe)            recorder_gui.py / record_session.py
       │
       ▼ JSON files
data/raw/
       │
       ▼  pipeline/ingestion/run.py   →  sessions.parquet + events.parquet
data/processed/
       │
       ▼  pipeline/features/run.py    →  features.parquet  (one row per 30s window)
data/processed/features.parquet
       │
       ▼  pipeline/features/split.py  →  train.parquet / val.parquet / test.parquet
data/splits/
       │
       ▼  pipeline/training/run.py    →  models/model.pkl + models/model.onnx
       │  pipeline/evaluation/run.py  →  reports/eval_metrics.json + confusion_matrix.csv
       │
       ├──► api/main.py             (FastAPI batch inference: /predict/player, /predict/anomaly)
       └──► dashboard/app.py        (Streamlit; 4 tabs)
```

### Critical design choices that aren't obvious from reading any single file

1. **Window-based features.** All 25 production features (`FEATURE_COLS` in `pipeline/features/run.py`) are aggregates over **30-second non-overlapping windows** (`WINDOW_MS = 30_000`). The constant is hardcoded — changing it requires re-running `dvc repro`. Full per-feature documentation in `docs/FEATURES.md`.

2. **Sens/DPI + polling-rate normalization.** Mouse kinematics (`speed_*`, `accel_*`) are divided by `norm_factor = sensitivity * dpi / 800.0` before aggregation, so different hardware setups are comparable. This is applied in `compute_mouse_kinematics()` and must NOT be applied again downstream. Separately, the three polling-rate-proportional features (`event_rate`, `mouse_key_ratio`, `direction_changes_per_sec`) are multiplied by `rate_norm = polling_rate_norm(polling_rate)` (= 1000 / polling_rate) so a 125 Hz and a 1000 Hz mouse give comparable values. `rate_norm` is threaded through `process_session_windows` and applied consistently in `pipeline/features/run.py:run`, `pipeline/adversarial/benchmark.py`, and `pipeline/inference/streaming.py`. Defaults to 1.0 when `polling_rate` is missing. Full rationale + which features are deliberately left alone: `docs/FEATURES.md`.

3. **No z-score scaling in the feature stage.** `StandardScaler` is deliberately applied only inside `pipeline/training/run.py` (on the training fold), never in `pipeline/features/run.py`. This prevents train/test leakage.

4. **Session-level splits.** `pipeline/features/split.py` uses `GroupShuffleSplit` by `session_id` so all windows from one recording stay in one fold. Players with fewer than `min_sessions_per_player` sessions (configured in `configs/training.yaml`, default 3) are dropped. With very few sessions, empty splits are written (pipeline still passes).

5. **Model selection is config-driven.** `pipeline/training/run.py` reads `model.type` and `model.task` from `configs/training.yaml`. Identification: lightgbm/random_forest/xgboost/svc. Anomaly: isolation_forest/lof/one_class_svm. ONNX export only happens for sklearn-compatible classifiers.

5b. **LSTM autoencoder lives outside the sklearn dispatch.** `pipeline/models/lstm_ae.py` is a PyTorch model trained on raw event sequences (Phase 2). It's invoked via `pipeline/adversarial/benchmark.py:run_lstm_ae_benchmark` for synthetic-cheat evaluation. The DVC pipeline does NOT currently dispatch to it (deferred to a future Phase 2.1 — the chunk-level benchmark already proves the model). GPU is auto-detected; the code runs on CUDA when available (`torch.cuda.is_available()`) and falls back to CPU silently. The persisted artifact lives at `models/lstm_ae.pt` + `models/lstm_ae_meta.json`; regenerate via `python -m scripts.train_lstm_ae`. See `docs/LSTM_AE.md`.

5c. **Phase 4 streaming pipeline is event-driven and transport-independent.** `pipeline/inference/streaming.py:SessionStreamState` is one in-memory state machine pushed by either the `/stream` WebSocket endpoint (`api/streaming.py`), the replay CLI (`scripts/replay_session.py`), or the demo generator (`scripts/build_phase4_demo.py`). The aggregator (`pipeline/inference/aggregator.py:RiskAggregator`) is a Naive-Bayes log-odds combiner with per-detector isotonic calibration and a configurable cheat-rate prior — the prior_logit math is documented inline in the module docstring with a worked example. The dashboard's "📡 Live Session" tab replays a session offline (no WS needed) and updates a Plotly chart as events flow through. **Mock-data caveat applies**: all current AUC numbers are limited by the mock-data legit baseline; absolute risk magnitudes will tighten once real GTA recordings land (see `[memory: project_recording_status]`). See `docs/STREAMING.md` for the architecture diagram + plain-English aggregator math.

6. **Optional MLflow.** Training and feature stages log to DagsHub's hosted MLflow if `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are in `.env`. Missing credentials degrade silently — the pipeline still produces all artifacts. Never hardcode credentials.

7. **Session JSON schema is forward-compatible.** Recordings carry extra metadata fields (`activity`, `polling_rate`, `resolution`, `grip_style`, `dominant_hand`, `warmup`) that are read via `data.get()` in the ingestion pipeline. Old session files without these fields still ingest successfully.

8. **Adversarial module produces drop-in synthetic sessions.** `pipeline/adversarial/bot_generator.py` injects aimbot / triggerbot / macro signatures into legit recordings while preserving the recorder JSON schema, so the full pipeline accepts them unchanged. The key finding (documented in `docs/ADVERSARIAL.md`): the current 18 window features fail at AUC ≈ 0.5 — this motivates Phases 1 and 2 of the roadmap.

## Conventions

- **Git commits**: do not include a `Co-Authored-By` trailer (user preference).
- **Plan mode**: substantial features (anything touching multiple modules) should be planned via the existing roadmap in `docs/ROADMAP.md`. Each completed phase updates its checklist and status.
- **Tutorial-style notebooks**: notebooks 09, 10 (and future 11–15) are intentionally written as step-by-step tutorials with diagrams and visualizations — the user uses them as study material, so verbosity is a feature not a bug.
- **Pre-commit hooks** auto-run ruff + black + trailing-whitespace + end-of-file-fixer on commit. If a hook modifies files, re-stage and re-commit; don't use `--no-verify`.

## Where to look

- [README.md](README.md) — public-facing project overview
- [docs/ROADMAP.md](docs/ROADMAP.md) — 5-phase portfolio roadmap with status
- [docs/ETHICS.md](docs/ETHICS.md) — data collection methodology, anti-cheat compatibility
- [docs/ADVERSARIAL.md](docs/ADVERSARIAL.md) — Phase 3 methodology + key finding
- [docs/RECORDING_INSTRUCTIONS.md](docs/RECORDING_INSTRUCTIONS.md) — player-facing guide
- `configs/training.yaml` — model + data split configuration (the only knob users tune)
- `notebooks/01_*` through `notebooks/07_*` — foundational data + model analysis
- `notebooks/10_adversarial_bots.ipynb` — current largest tutorial (32 cells)
