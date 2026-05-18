# BehaviorDNA 🎮🧬

> **Player behavioral biometrics from raw input telemetry.**
> Can we identify *who* is playing — or detect automation — purely from mouse and keyboard patterns?

[![CI](https://github.com/Lo1s/behaviorDNA/actions/workflows/ci.yml/badge.svg)](https://github.com/Lo1s/behaviorDNA/actions/workflows/ci.yml)
[![Experiment Tracking](https://dagshub.com/Lo1s/behaviorDNA.svg)](https://dagshub.com/Lo1s/behaviorDNA)

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

```bash
# On Windows (native Python, not WSL)
cd collector
python record_session.py --player your_name --game valorant
```

### 5. Run the pipeline

```bash
dvc repro
```

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

## TODO / Research Directions

- [ ] **External dataset exploration** — feature importance analysis on CS2CD and CaptchaSolve30k (`notebooks/05_external_datasets.ipynb`)
- [ ] **Multi-model comparison** — benchmark RandomForest, XGBoost, SVC vs LightGBM for identification; LOF, One-Class SVM vs IsolationForest for detection (`notebooks/06_model_comparison.ipynb`)
- [ ] **Promote best models to pipeline** — wire winning models from notebook comparison into `pipeline/training/run.py` and `configs/training.yaml`
- [ ] **Autoencoder / LSTM** — deep learning behavioral fingerprinting (placeholder in config)
- [ ] **Real-time dashboard** — populate the empty `dashboard/` directory with a Streamlit or Gradio demo

---

## Why this project?

Built as a portfolio piece targeting the behavioral biometrics / anti-cheat domain.
Demonstrates: data engineering, feature design, MLOps pipelines, model deployment — not just a notebook.

---

## Ethics & safety

This project operates entirely at the OS input level — no game memory reading, no packet sniffing, no anti-cheat bypass. All data is collected with explicit participant consent for research purposes.

See [docs/ETHICS.md](docs/ETHICS.md) for full details on data collection methodology, anti-cheat compatibility per game, consent process, and data privacy.
