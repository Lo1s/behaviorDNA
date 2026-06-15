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
pytest -q                                 # full suite (478 tests)
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

The system is a **5-stage DVC pipeline** defined in `dvc.yaml`. Each stage is a Python module under `pipeline/` and produces a Parquet (or model artifact) that the next stage depends on. `data/raw/`, `data/processed/`, `data/splits/`, `data/external/`, `data/synthetic/`, and `models/` are DVC-managed (gitignored). **`data/raw/` is a whole-dir DVC output → do NOT git-track any file inside it** (a `.gitkeep`/README there makes `dvc add/commit`/CI's `dvc pull` fail with "output already tracked by SCM"); its layout/recording-separation doc lives in [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md). Inside `data/raw/`: legit recordings at the top level, real cheat recordings in `data/raw/cheat/` (see design choice #9).

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
       └──► dashboard/app.py        (Streamlit; 5 tabs)
```

### Critical design choices that aren't obvious from reading any single file

1. **Window-based features.** All 30 production features (`FEATURE_COLS` in `pipeline/features/run.py`) are aggregates over **30-second non-overlapping windows** (`WINDOW_MS = 30_000`). The constant is hardcoded — changing it requires re-running `dvc repro`. The bank is split into two task-specific slices: `ID_FEATURE_COLS` (25, player identifier) and `CHEAT_FEATURE_COLS` (30, cheat detectors / adversarial benchmark / streaming) — see `docs/SIGNALS.md`. Each model artifact carries its own `feature_cols`, which downstream code must treat as authoritative. Full per-feature documentation in `docs/FEATURES.md`.

2. **Sens/DPI + polling-rate normalization.** Mouse kinematics (`speed_*`, `accel_*`) are divided by `norm_factor = sensitivity * dpi / 800.0` before aggregation, so different hardware setups are comparable. This is applied in `compute_mouse_kinematics()` and must NOT be applied again downstream. Separately, the three polling-rate-proportional features (`event_rate`, `mouse_key_ratio`, `direction_changes_per_sec`) are multiplied by `rate_norm = polling_rate_norm(polling_rate)` (= 1000 / polling_rate) so a 125 Hz and a 1000 Hz mouse give comparable values. `rate_norm` is threaded through `process_session_windows` and applied consistently in `pipeline/features/run.py:run`, `pipeline/adversarial/benchmark.py`, and `pipeline/inference/streaming.py`. Defaults to 1.0 when `polling_rate` is missing. Full rationale + which features are deliberately left alone: `docs/FEATURES.md`.

3. **No z-score scaling in the feature stage.** `StandardScaler` is deliberately applied only inside `pipeline/training/run.py` (on the training fold), never in `pipeline/features/run.py`. This prevents train/test leakage.

4. **Player-stratified, session-level splits.** `pipeline/features/split.py` does a **per-player whole-session holdout** (not `GroupShuffleSplit`): every retained player appears in train/val/test, and all windows from one session stay in one fold. Players with fewer than `min_sessions_per_player` sessions (`configs/training.yaml`, default 3) are dropped; with very few sessions, empty-but-valid splits are written. **Cheat sessions are excluded** from the identification split (`is_cheat_session` flag — see #9), so the identifier trains on legit play only.

5. **Model selection is config-driven.** `pipeline/training/run.py` reads `model.type` and `model.task` from `configs/training.yaml`. Identification: lightgbm/random_forest/xgboost/svc. Anomaly: isolation_forest/lof/one_class_svm. ONNX export (sklearn-compatible classifiers only) goes through `pipeline/onnx_export.py` — a **bit-faithful float64 export** with a CI parity gate (fixes the earlier onnxmltools multiclass-converter fidelity bug; see `docs/FINDINGS.md`).

5b. **LSTM autoencoder lives outside the sklearn dispatch.** `pipeline/models/lstm_ae.py` is a PyTorch model trained on raw event sequences (Phase 2). It's invoked via `pipeline/adversarial/benchmark.py:run_lstm_ae_benchmark` for synthetic-cheat evaluation. The DVC pipeline does NOT currently dispatch to it (deferred to a future Phase 2.1 — the chunk-level benchmark already proves the model). GPU is auto-detected; the code runs on CUDA when available (`torch.cuda.is_available()`) and falls back to CPU silently. The persisted artifact lives at `models/lstm_ae.pt` (**DVC-tracked** — `dvc pull` to fetch; `models/lstm_ae.pt.dvc` is the git-tracked pointer) + `models/lstm_ae_meta.json` (small, git-tracked); regenerate from scratch via `python -m scripts.train_lstm_ae`. The 523 MB synthetic-cheat dataset is deliberately **not** versioned — it's regenerated deterministically (seed 42) via `python -m pipeline.adversarial.generate_dataset` (see `docs/ADVERSARIAL.md` "Reproducing"). See `docs/LSTM_AE.md`.

5c. **Phase 4 streaming pipeline is event-driven and transport-independent.** `pipeline/inference/streaming.py:SessionStreamState` is one in-memory state machine pushed by either the `/stream` WebSocket endpoint (`api/streaming.py`), the replay CLI (`scripts/replay_session.py`), or the demo generator (`scripts/build_phase4_demo.py`). The aggregator (`pipeline/inference/aggregator.py:RiskAggregator`) is a Naive-Bayes log-odds combiner with per-detector isotonic calibration and a configurable cheat-rate prior — the prior_logit math is documented inline in the module docstring with a worked example. The dashboard's "📡 Live Session" tab replays a session offline (no WS needed) and updates a Plotly chart as events flow through. **Real-data status (2026-05-30):** benchmarked on 18 real GTA sessions — the chunk-level LSTM-AE detector works (aimbot AUC 0.79, triggerbot 0.93), but the *session-level combined risk* saturates (near-chance session inputs + only 18 calibration sessions) → recalibration is Phase 4.1. Real data also exposed (and this round **fixed**) a normalisation bug: `SessionStreamState` must be told the session's sens/DPI + polling rate via `configure_for_session(...)` before pushing events, else real-hardware sessions are mis-scaled — now wired into **all** transports (the WS `__session__` first message, the dashboard tab, and the replay CLI, not just `replay_offline`). `finalize()` flushes the trailing partial window and discards the partial chunk (the LSTM needs a full `chunk_length`). Serving no longer fits at startup: it loads a **versioned bundle** (`models/serving_bundle.pkl`, DVC-tracked) via `load_or_build_stream_state` (~1 s, no `data/synthetic` needed), built by `python -m scripts.build_serving_bundle`. See `docs/STREAMING.md`.

6. **Optional MLflow.** Training and feature stages log to DagsHub's hosted MLflow if `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are in `.env`. Missing credentials degrade silently — the pipeline still produces all artifacts. Never hardcode credentials.

7. **Session JSON schema is forward-compatible.** Recordings carry extra metadata fields (`activity`, `polling_rate`, `resolution`, `grip_style`, `dominant_hand`, `warmup`) that are read via `data.get()` in the ingestion pipeline. Old session files without these fields still ingest successfully.

8. **Adversarial module produces drop-in synthetic sessions.** `pipeline/adversarial/bot_generator.py` injects aimbot / triggerbot / macro signatures into legit recordings while preserving the recorder JSON schema, so the full pipeline accepts them unchanged. The original Phase-3 finding (`docs/ADVERSARIAL.md`): the *first* 18 window features detected aimbots at AUC ≈ 0.5 — which motivated Phase 1 (trajectory/reaction features → 30) and Phase 2 (the LSTM-AE on raw sequences). **Phase 7** adds `pipeline/adversarial/humanizer.py` — a humanisation knob `λ ∈ [0,1]` that interpolates each cheat from obvious bot → humanised toward the player's *own* legit play (per-player `PlayerBaseline`), plus a closed-form `utility(λ)`; `scripts/evasion_frontier.py` sweeps λ (chunk-LSTM-AE + OneClassSVM detectors, reuses the persisted `models/lstm_ae.pt`, **inference-only/CPU-fine**) → `reports/evasion_frontier.json` + the frontier figure. Finding: the frontier is **defender-favoured** (no λ is both undetectable and worth running; humanising the aimbot *raises* detection). Experiments are scripts, not DVC stages (like Phases 8). See `docs/ADVERSARIAL.md` (arms-race section) + [nb 20].

10. **Self-supervised pretraining (Phase 8) lives outside the DVC pipeline.** `pipeline/pretraining/` masked-denoising-pretrains the *same* `LSTMAutoencoder` on CaptchaSolve30k, then transfers the full weights into fine-tuning. The linchpin: **all three corpora map onto the one 8-D event-tensor schema** (`pipeline/pretraining/corpora.py` — captcha & CS2CD are *sampled* per-tick streams re-encoded into the schema `session_to_event_tensor` produces for GTA). Artifact `models/pretrained_encoder.pt` (**DVC-tracked**, git-tracked `.dvc` pointer) + `_meta.json`; regenerate via `python -m scripts.pretrain_encoder` (CUDA). Experiments are scripts, not DVC stages: `scripts/domain_gap_report.py` + `scripts/data_efficiency.py --domain {cs2cd,gta}` → `reports/{pretraining_domain_gap,data_efficiency_*}.json`. **Outcome = a rigorous null** (pretrained ≈ scratch; the captcha→game domain gap, esp. the `dt` channel, dominates). The big captcha parquet is git-ignored/un-DVC'd & re-downloadable; the loader streams it via `pyarrow.iter_batches` to avoid OOM. **Phase 8.1** tests this in-domain: `pipeline/pretraining/cs2cd_full.py` ingests the full public CS2CD release (478 legit matches → 16 GB per-match shard cache, lazy LRU dataset + `ShardGroupedSampler`; label = subdir — the full release has **no** `cheater_present` column) for the `scripts/indomain_transfer.py` grid (arms scratch/frozen/finetune × sources `s1` native-`dt` / `s2` `dt`-neutralised × volume 50/200/382, + a captcha comparison; `scripts/cs2cd_diversity_probe.py` is the Step-0 gate → release is **player-anonymised**, so volume ≠ player diversity). **Verdict = the null holds, deeper than the domain gap:** in-domain ≤ from-scratch on GTA, `s2`≈`s1`, volume flat (`scripts/domain_gap_report.py --reference cs2cd` shows in-domain CS2 isn't even closer to GTA). The 6 `models/pretrained_cs2cd_{s1,s2}_{50,200,382}.pt` encoders are DVC-tracked. **Phase 8.2** swaps the *objective*: `pipeline/pretraining/{augment,contrastive,embed_eval}.py` contrastively pretrain the *same* encoder (SimCLR/TS2Vec **NT-Xent** over two augmented views), evaluated **contrastive-natively** on the *frozen* 16-D embedding (Mahalanobis/OCSVM/kNN/linear-probe — NOT reconstruction-error AUC) via `scripts/contrastive_transfer.py --phase {pretrain,eval}` → `reports/contrastive_transfer.json` + 4 DVC-tracked `models/pretrained_contrastive_{cs2cd_50,cs2cd_200,cs2cd_382,captcha}.pt` (reuses the 8.1 shard pipeline: `CS2CDShardChunkDataset` → contrastive subclass + `ShardGroupedSampler`). **Verdict = the first non-null:** in-domain contrastive beats both random-init and the 8.1 reconstruction encoder on every probe (modest ~0.55–0.66, in-domain-specific, volume-flat) — the *objective* (magnitude-invariant vs magnitude-dominated MSE) was the lever, not corpus/capacity/`dt`. See `docs/PRETRAINING.md`.

11. **Outcome telemetry (Phase 9) is a *feasibility spike*, outside the DVC pipeline + data-gated.** `pipeline/outcome/cs2_demo.py` (+ CLI `scripts/parse_cs2_demo.py`) parses a CS2 `.dem` via `demoparser2` into per-window `OUTCOME_FEATURE_COLS` (13 features: kills/deaths/shots/hits/damage/accuracy/headshot-ratio/… + per-tick view-angle dynamics). `OUTCOME_FEATURE_COLS` is a **separate, additive slice** in `pipeline/features/run.py` — **NOT** part of `FEATURE_COLS` (not computable from the recorder stream), aligned onto the `WINDOW_MS` grid so it *joins* to the input features on `(session_id, window_idx)` once dual-capture data exists. The demo↔recorder **clock-sync** (`estimate_offset_by_xcorr`) is marker-free: cross-correlate recorder mouse-motion vs demo view-angle-motion; `peak_corr` self-validates (low ⇒ reject). `demoparser2` is imported **lazily** so the module/tests run without the native dep or a `.dem` (tests use synthetic frames). Validated on a real public demo (`data/external/cs2_demo/`, gitignored/re-downloadable). The **dual-capture ingest pipeline** (`pipeline/outcome/dual_capture.py` + CLI `scripts/ingest_dual_capture.py`, `tests/test_dual_capture.py`) wraps these primitives: recorder JSON + `.dem` → one **clock-synced, window-joined** table (input features ⨝ outcome on `(session_id, window_idx)`), reusing `process_session_windows` verbatim and **correcting the xcorr offset back to the window anchor** (`min(t)` over all events, not the first mouse-move) so the grids align exactly; `has_outcome` marks demo coverage (filter combat via `shots_fired>0`). Still data-gated: the supervised detector + Phase 4.1 re-attempt **await dual-capture sessions** (recorder + demo recorded simultaneously; cheat positives need offline `cheat_sim`). See `docs/CHEAT_DATA_COLLECTION.md` "Phase 9".

9. **Cheat vs legit recordings are separated by folder AND flag.** Legit recordings live at the top of `data/raw/`; real cheat recordings (cheat_sim-injected) live in `data/raw/cheat/`. A session is *also* flagged by `pipeline.ingestion.run._is_cheat_session` (typed/untyped cheat spans or a non-`legit` `cheat_label`). Non-recursive `*.json` globs (ingestion, `train_lstm_ae`, `compare_architectures._load_legit_tensors`, validate, dashboard, `generate_dataset`) see legit only; the cheat-detection evaluators (`compare_architectures --eval-data real`, `benchmark --data-dir data/raw`) additionally scan `cheat/`. **Authoritative cheat labels** (`cheat_segments_typed`) come from in-band F8/F9/F10 toggle keys in the recording itself via `scripts/label_cheat_segments.py` — not from any external log. Full map: `docs/DATA_LAYOUT.md` / `docs/CHEAT_DATA_COLLECTION.md`.

## Conventions

- **Git commits**: do not include a `Co-Authored-By` trailer (user preference).
- **Push to BOTH remotes**: `git push origin main && git push dagshub main` (DagsHub renders the public README). For DVC-tracked data/model changes, `dvc push` to the DagsHub remote too.
- **README results are self-updating**: headline numbers come from `scripts/generate_results.py` (CI-gates `--check`). Don't hand-edit the results block / metrics — regenerate instead. Likewise the *structural* facts (test count, dashboard tab count, notebook/doc counts) are owned by `scripts/generate_metadata.py` → `reports/repo_metadata.json` (also CI-gated `--check`); don't hand-edit those numbers in README/CLAUDE — run `python -m scripts.generate_metadata` and re-stage.
- **Keep this file current (after every milestone).** When a milestone lands — a roadmap phase, a structural refactor (new pipeline stage / model, data-layout or feature-set change), or anything that changes the design choices / commands / "where to look" above — update the affected lines here **in the same commit**, and sync the status in `docs/ROADMAP.md`. CLAUDE.md is loaded into context every session, so stale guidance actively misleads (e.g. a wrong `split.py` description sends the next agent down the wrong path).
- **Plan mode**: substantial features (anything touching multiple modules) should be planned via the existing roadmap in `docs/ROADMAP.md`. Each completed phase updates its checklist and status.
- **Tutorial-style notebooks**: notebooks 09, 10, 16, 17, 18, 19, 20 (and 11–15) are intentionally written as step-by-step tutorials — the user uses them as study material, so verbosity is a feature not a bug. Notebooks 16/17 are GPU-live (seeded; absolute AUCs may wobble ~±0.01 run-to-run, ranking is stable). Notebooks 18 (CS2CD signal importance), 19 (public-corpus ID, Phase 6), 20 (Phase 7 evasion frontier — loads `scripts/evasion_frontier.py`'s output), 21 (Phase 8 pretraining — loads the GPU scripts' outputs), and 22 (Phase 8.2 contrastive pretraining — loads `scripts/contrastive_transfer.py`'s output) are CPU-fast.
- **Pre-commit hooks** auto-run ruff + black + trailing-whitespace + end-of-file-fixer on commit. If a hook modifies files, re-stage and re-commit; don't use `--no-verify`. The EOF-fixer commonly rewrites JSON reports → re-stage the JSON + re-commit. (black skips notebooks everywhere — enforced by `pyproject.toml` `[tool.black] force-exclude = '\.ipynb$'`, so `black .` / `black --check .` agree with CI whether or not the `black[jupyter]` extra is installed locally; **ruff is the notebook lint gate**.)

## Where to look

- [README.md](README.md) — public-facing overview (results block is auto-generated)
- [docs/ROADMAP.md](docs/ROADMAP.md) — roadmap + status: Phases 1–5 done, 6 done, 7 done (defender-favoured evasion frontier), 8 done (null result), 8.1 done (in-domain CS2CD null holds — deeper than the domain gap), 8.2 done (contrastive objective — the **first non-null**: in-domain beats random + reconstruction on the frozen embedding, modest/in-domain-specific), 1.5 partial, 9 **spike done** (CS2 outcome telemetry + clock-sync feasible; execution awaits dual-capture data)
- [docs/FINDINGS.md](docs/FINDINGS.md) — the honest results in one page (small-N rigor, ONNX bug, etc.)
- [docs/SIGNALS.md](docs/SIGNALS.md) — signal/feature research + ID-vs-cheat feature-set decoupling + data-collection roadmap
- [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) — `data/raw/` legit/cheat layout + which consumer reads what
- [docs/ARCHITECTURE_COMPARISON.md](docs/ARCHITECTURE_COMPARISON.md) · [docs/VERIFICATION.md](docs/VERIFICATION.md) · [docs/STREAMING.md](docs/STREAMING.md) · [docs/MLOPS.md](docs/MLOPS.md) · [docs/ADVERSARIAL.md](docs/ADVERSARIAL.md) · [docs/PRETRAINING.md](docs/PRETRAINING.md) · [docs/ETHICS.md](docs/ETHICS.md) · [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) · [docs/DATASET_CARDS.md](docs/DATASET_CARDS.md)
- [docs/REPORT.md](docs/REPORT.md) — arXiv-style tech report, **full draft** (§§1–6,8,9,10 + abstract/appendix drafted; §7 evasion flagged planned; only the manual arXiv submission remains)
- `configs/training.yaml` — model + data-split configuration (the main knob users tune)
- `notebooks/01_*`–`07_*` foundational analysis · `10` adversarial bots · `12` explainability · `16` architecture deep-dive · `17` ID at scale (CS2) · `18` signal importance · `19` public-corpus ID/verification · `20` evasion frontier (Phase 7) · `21` self-supervised pretraining (Phase 8) · `22` contrastive pretraining (Phase 8.2)
