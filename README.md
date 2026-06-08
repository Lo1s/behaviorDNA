# BehaviorDNA 🎮🧬

> **Player behavioral biometrics from raw input telemetry.**
> Can we identify *who* is playing — or detect automation — purely from mouse and keyboard patterns?

[![CI](https://github.com/Lo1s/behaviorDNA/actions/workflows/ci.yml/badge.svg)](https://github.com/Lo1s/behaviorDNA/actions/workflows/ci.yml)
[![Experiment Tracking](https://img.shields.io/badge/Experiment_Tracking-DagsHub-f76c6c?logo=mlflow)](https://dagshub.com/Lo1s/behaviorDNA)

---

![BehaviorDNA chunk-level cheat detection](reports/figures/phase4_chunk_detection.png)

*Chunk-level cheat detection — **synthetic cheats injected into 18 real GTA legit sessions** (the approach proof). The LSTM autoencoder's reconstruction error separates cheat chunks (coloured) from legit-behaviour chunks (green): triggerbot ROC AUC 0.94, aimbot 0.80, macro 0.61 — while hand-crafted window features stay at chance for aimbot. The approach is **also validated on real recorded cheats and on a second game (CS2)** — see [Results](#results-at-a-glance). Reproduce with `python -m scripts.build_phase4_demo`. See [docs/ADVERSARIAL.md](docs/ADVERSARIAL.md) and [docs/STREAMING.md](docs/STREAMING.md).*

> **Data status.** Measured on real GTA gameplay (18 legit sessions, 3 players) + real **recorded** cheats (a controllable offline harness, [docs/CHEAT_DATA_COLLECTION.md](docs/CHEAT_DATA_COLLECTION.md)) + an **external** CS2 cheat dataset. The chunk-level detector works on real data and transfers across games; the *session-level* live-risk aggregator saturates on the current single-recorder calibration set and is gated pending more (cross-player) data (Phase 4.1 — see [docs/STREAMING.md](docs/STREAMING.md#what-works-and-what-doesnt-honest)).

---

## Highlights — what this demonstrates

- **End-to-end MLOps on *real* data:** custom Windows telemetry recorder → DVC pipeline → training → calibration → drift monitoring → MLflow model registry → FastAPI + ONNX serving + Streamlit dashboard, all CI-tested (**317 tests**).
- **Deep model where it earns its place:** a sequence autoencoder detects cheats hand-crafted features can't (triggerbot **0.93** chunk AUC vs aimbot **≈ chance** for window features) — and it **transfers to a second game** (Counter-Strike 2, ~0.72) on data I didn't create.
- **Honest validation over flattering numbers:** caught a real **ONNX-export fidelity bug** via a probability-parity check; *verified* (not assumed) a session-level detection ceiling before building on it; an **ablation** showing the model is over-parameterised at this N; an **architecture study** finding LSTM/TCN/Transformer statistically tied.
- **Anti-cheat framing throughout:** false-positive/ban-cost reasoning, calibrated probabilities (ECE/Brier), and deliberate, audited model promotion — see the **[Model Card](MODEL_CARD.md)**.

---

## Results at a glance

**Player identification** (behavioural biometric, real GTA, per 30 s window):

| Setting | Result |
|---|---|
| 3 players | **0.853** acc / 0.862 F1 |
| same-hardware pair *(no hardware confound)* | **0.75** acc (0.65 baseline) — the honest biometric |

**Cheat detection** — chunk-level ROC AUC, validated across three independent settings (hand-crafted window features ≈ 0.50 = chance for aimbot):

| Setting | Result |
|---|---|
| Synthetic cheats on real legit — *approach proof* | aimbot 0.79 · **triggerbot 0.93** · macro 0.60 |
| **External game (CS2CD, 10 players, different engine)** — *generalisation* | **~0.72** on real CS2 cheats |
| Own *recorded* real cheats (1 player) — *hardest, most honest* | aimbot 0.52 · triggerbot 0.60 · macro 0.57 |

LSTM-AE vs TCN-AE vs Transformer-AE are **statistically tied in every setting** → capacity isn't the bottleneck, data is ([ARCHITECTURE_COMPARISON.md](docs/ARCHITECTURE_COMPARISON.md)).

**Engineering:** sklearn inference **1.4 ms**/window (~89k windows/s — real-time) · mock→real **drift 20/25 features** significant (KS+PSI) · MLflow registry + CI + 317 tests.

🧭 **Start here:** **[Findings](docs/FINDINGS.md)** (the honest results in one page) → **[Model Card](MODEL_CARD.md)** (intended use, limits, ban-cost) → `notebooks/12_explainability.ipynb` (SHAP + per-channel attribution) → `notebooks/16_architecture_comparison.ipynb` (GPU-live LSTM/TCN/Transformer deep-dive + "is it the split or a real ceiling?" experiments) → `notebooks/17_identification_at_scale.ipynb` (10-player ID on CS2 + "does identity survive cheating?") → `notebooks/18_signal_importance_cs2.ipynb` (what signals to monitor — behavioural + non-behavioural — and which earn promotion) → `docs/ARCHITECTURE_COMPARISON.md` · `docs/SIGNALS.md`. *Numbers are directional at this data scale (one cheat recorder, 3 players), not production guarantees.*

---

## What is this?

BehaviorDNA is a game-agnostic ML system that:

1. **Collects** raw input telemetry (mouse, keyboard) during gameplay sessions
2. **Engineers** behavioral features — rhythm, timing, micro-jitter, reaction patterns
3. **Builds** per-player behavioral fingerprints across sessions
4. **Detects** anomalies and automation-like behavior (bots, macros, scripts)
5. **Identifies** players by their behavioral signature alone

Designed as a portfolio project demonstrating end-to-end MLOps — from data collection to deployed inference API.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Windows (Gaming Host)                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │  collector/  — lightweight input listener        │    │
│  │  outputs session JSON → data/raw/               │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────┘
                           │ sync / copy
┌──────────────────────────▼──────────────────────────────┐
│                   WSL / Linux (Dev)                      │
│                                                          │
│  pipeline/ingestion/   raw JSON → Parquet               │
│  pipeline/features/    feature engineering               │
│  pipeline/training/    model training (LightGBM, AE)    │
│  pipeline/evaluation/  metrics, reports                  │
│                                                          │
│  models/               saved model artifacts             │
│  api/                  FastAPI inference endpoint        │
│  dashboard/            MLflow / visualization            │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data collection | Python, `pynput` (Windows) |
| Data versioning | DVC |
| Experiment tracking | MLflow + DagsHub |
| Feature engineering | Pandas, NumPy |
| ML models | LightGBM, Scikit-learn (Isolation Forest), PyTorch (LSTM/AE) |
| Pipeline orchestration | DVC pipelines + GitHub Actions |
| Model export | ONNX |
| Inference API | FastAPI |
| CI/CD | GitHub Actions |

---

## Project Structure

```
behaviorDNA/
├── collector/          # Windows-side input telemetry recorder
├── pipeline/
│   ├── ingestion/      # Raw JSON → structured Parquet
│   ├── features/       # Feature extraction & engineering
│   ├── training/       # Model training scripts
│   └── evaluation/     # Metrics, reports, comparison
├── models/             # Saved model artifacts (.pkl, .onnx)
├── api/                # FastAPI inference service
├── dashboard/          # Visualization & MLflow helpers
├── configs/            # Hydra / YAML configuration
├── scripts/            # Utility & setup scripts
├── tests/              # Unit & integration tests
├── docs/               # Architecture diagrams, notes
└── data/
    ├── raw/            # Raw session JSON files (DVC-tracked)
    ├── processed/      # Parquet feature tables (DVC-tracked)
    └── splits/         # Train/val/test splits (DVC-tracked)
```

---

## Quickstart

### 1. Clone & set up (WSL/Linux)

```bash
git clone https://github.com/YOUR_USERNAME/behaviorDNA.git
cd behaviorDNA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up DVC remote (DagsHub)

```bash
dvc remote add origin https://dagshub.com/YOUR_USERNAME/behaviorDNA.dvc
dvc pull
```

### 3. Configure MLflow credentials (optional)

Copy `.env.example` to `.env` and fill in your DagsHub credentials to enable experiment tracking:

```bash
cp .env.example .env
# edit .env — set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
```

Training runs log automatically to DagsHub when credentials are present. Without `.env`, training still works — MLflow logging is silently skipped.

### 4. Record a session (Windows)

The recommended way is the compiled GUI (see `collector/recorder_gui.py` → PyInstaller). For CLI use:

```bash
# On Windows (native Python, not WSL)
cd collector
python record_session.py \
  --player your_name \
  --game gta \
  --activity combat \
  --polling-rate 1000 \
  --resolution 1920x1080 \
  --grip palm \
  --hand right \
  --warmup no \
  --sens 0.35 \
  --dpi 800
```

See [docs/RECORDING_INSTRUCTIONS.md](docs/RECORDING_INSTRUCTIONS.md) for the full player guide (activity schedule, how to look up hardware values, data quality rules).

### 5. Run the pipeline

```bash
dvc repro
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` — four tabs: Overview, Player Profiles, Predict, Session Explorer.

### 7. Or run the whole stack in Docker

```bash
docker compose up --build      # API → :8000 (/docs) · dashboard → :8501
```

API + dashboard from one image; mounts your local `models/` + `data/`, or
`dvc pull`s them with a DagsHub token. Hosted-demo (Streamlit Cloud) + deploy
notes: **[docs/DEPLOY.md](docs/DEPLOY.md)**.

---

## Roadmap

- [x] Project structure & repo setup
- [x] Data collector (Windows, pynput) — GUI + standalone .exe via PyInstaller
- [x] Ethics & safety documentation
- [x] Ingestion pipeline (JSON → Parquet)
- [x] Feature engineering module
- [x] Anomaly detection model (Isolation Forest / Autoencoder)
- [x] Player identification model (LightGBM)
- [x] MLflow experiment tracking
- [x] ONNX model export
- [x] FastAPI inference endpoint
- [x] Test suite (features, split, training, evaluation, API)
- [x] GitHub Actions CI/CD
- [x] DagsHub integration

---

## Portfolio Roadmap

A 5-phase roadmap targeting anti-cheat ML/AI roles is tracked in detail in [docs/ROADMAP.md](docs/ROADMAP.md). Current status:

| Phase | Goal | Status |
|---|---|---|
| 1. [Trajectory & temporal features](docs/ROADMAP.md#phase-1--trajectory--temporal-features) | 7 anti-cheat-targeted window features | ✅ Done — triggerbot AUC 0.50 → 0.87, macro 0.55 → 0.68 |
| 1.5. [Feature expansion (backlog)](docs/ROADMAP.md#phase-15--feature-expansion-optional) | Further window-feature ideas | 📝 Backlog |
| 2. [LSTM autoencoder](docs/LSTM_AE.md) | Deep-learning sequence model on raw events | ✅ Done — real-data aimbot chunk AUC **0.79**, triggerbot **0.93** |
| 3. [Adversarial bots](docs/ADVERSARIAL.md) | Synthetic cheat generator + detection benchmark | ✅ Done — 90 labelled hybrid sessions, full ROC grid |
| 4. [Streaming + risk aggregation](docs/STREAMING.md) | Naive-Bayes log-odds aggregator + WebSocket API + live dashboard tab | ⚠️ Infra end-to-end; session-level aggregator saturates on real data → Phase 4.1 (see [doc](docs/STREAMING.md#what-works-and-what-doesnt-honest)) |
| 4.1. [Live recorder + multi-user backlog](docs/ROADMAP.md#phase-41--live-recorder--multi-user-backlog) | Phase 4 follow-ups | 📝 Backlog |
| 5. [Statistical rigor & MLOps](docs/ROADMAP.md#phase-5--statistical-rigor--mlops-polish) | SHAP, calibration, drift, registry | ⬜ Not started |

Legend: ✅ Done · 🚧 In progress · ⬜ Not started · 📝 Backlog

## TODO / Research Directions

- [x] **External dataset exploration** — CS2CD cheat detection + CaptchaSolve30k mouse kinematic analysis (`notebooks/05_external_datasets.ipynb`)
- [x] **Multi-model comparison** — benchmark RandomForest, XGBoost, SVC vs LightGBM for identification; LOF, One-Class SVM vs IsolationForest for detection (`notebooks/06_model_comparison.ipynb`)
- [x] **Promote best models to pipeline** — RandomForest, XGBoost, SVC, LOF, OneClassSVM now selectable via `configs/training.yaml`
- [x] **Behavioral differentiation analysis** — deep dive into how cheater/bot trajectories differ from legit behavior using CS2CD and CaptchaSolve30k (`notebooks/07_behavioral_differentiation.ipynb`)
- [x] **Adversarial bot generation + detection benchmark** — synthetic aimbot/triggerbot/macro generator, 90 labelled hybrid sessions, per-detector ROC grid (`notebooks/10_adversarial_bots.ipynb`, `docs/ADVERSARIAL.md`)
- [x] **Trajectory & temporal features** — 7 anti-cheat-targeted features (curvature, path efficiency, click reaction time, keystroke periodicity, …) closing the triggerbot + macro detection gap (`notebooks/08_trajectory_features.ipynb`, `docs/FEATURES.md`)
- [x] **LSTM autoencoder on raw event sequences** — PyTorch sequence model, GPU-accelerated (RTX 3070), solves the aimbot detection gap at the chunk level (real-data AUC 0.79). 11-step tutorial in `notebooks/09_lstm_autoencoder.ipynb`; full architecture write-up in `docs/LSTM_AE.md`
- [x] **Streaming inference + Bayesian session aggregation** — `/stream` WebSocket endpoint, `pipeline/inference/aggregator.py` (Naive-Bayes log-odds + isotonic calibration), `scripts/replay_session.py` with synthetic-cheat injection, "📡 Live Session" dashboard tab, reproducible PNG + GIF demo artifacts via `scripts/build_phase4_demo.py`. Full architecture in [docs/STREAMING.md](docs/STREAMING.md).
- [ ] **Calibration + SHAP + drift monitor + MLflow registry** — production polish (Phase 5)
- [x] **Real-time dashboard** — four-tab Streamlit app in `dashboard/app.py`

---

## Why this project?

Built as a portfolio piece targeting the behavioral biometrics / anti-cheat domain.
Demonstrates: data engineering, feature design, MLOps pipelines, model deployment — not just a notebook.

---

## Ethics & safety

This project operates entirely at the OS input level — no game memory reading, no packet sniffing, no anti-cheat bypass. All data is collected with explicit participant consent for research purposes.

See [docs/ETHICS.md](docs/ETHICS.md) for full details on data collection methodology, anti-cheat compatibility per game, consent process, and data privacy.
