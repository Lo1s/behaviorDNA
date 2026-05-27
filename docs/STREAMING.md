# Streaming Inference + Risk Aggregation

> Phase 4 of the [BehaviorDNA roadmap](ROADMAP.md). Combines every detector built in Phases 1–3 into one calibrated session-level risk score and exposes it as a live WebSocket-driven dashboard.

> **Mock-data caveat.** All numbers, plots, and dashboard demos in this doc are measured against the current 15-session mock dataset (mouse-on-desktop, not real gameplay). Real GTA recordings from 3 players are pending. The infrastructure is data-independent — absolute risk magnitudes will tighten once real gameplay lands.

---

![Phase 4 live demo](../reports/figures/phase4_live_demo.gif)

*Live cheat-risk timeline. A synthetic aimbot is injected at t=10s. The LSTM-AE chunk signal is active throughout, but the combined risk doesn't cross the alert threshold until t=30s when the first classical-detector window completes and adds its evidence. The aggregator combines four independent signals via Naive-Bayes log-odds.*

---

## Architecture

```
              ┌──────────────────────────────────────────────────────┐
              │                                                      │
events ──►   ─┤    SessionStreamState (pipeline/inference)           ├─► ScoreUpdate
              │                                                      │
              │   ┌─────────────────────────┐    ┌─────────────────┐ │
              │   │ window buffer (30s)     │    │ chunk buffer    │ │
              │   │   → classical detectors │    │   → LSTM-AE     │ │
              │   │   (Phase 1, 25 features)│    │   (Phase 2, L=64)│ │
              │   └────────────┬────────────┘    └────────┬────────┘ │
              │                │                          │          │
              │                ▼                          ▼          │
              │      max-so-far per detector       p95 of chunk MSEs │
              │                │                          │          │
              │                └─────────────┬────────────┘          │
              │                              ▼                       │
              │             RiskAggregator (Naive-Bayes log-odds)    │
              │                              │                       │
              │                              ▼                       │
              │                 session_risk ∈ [0, 1]                │
              └──────────────────────────────────────────────────────┘
                             ▲                          │
                             │                          ▼
                ┌────────────┴──────────┐    ┌──────────────────────┐
                │ api/streaming.py      │    │ Dashboard "Live"     │
                │   /stream WebSocket   │    │ tab + GIF generator  │
                └───────────────────────┘    └──────────────────────┘
                          ▲
                          │
                ┌─────────┴─────────┐
                │ scripts/replay_   │
                │ session.py        │
                │   --inject-cheat  │
                └───────────────────┘
```

All five components share one in-memory state machine (`SessionStreamState`) that ingests events one at a time. Transport is decoupled — the same engine is driven by WebSocket events in production and by `replay_offline()` calls in tests and the demo generator.

---

## Plain-English aggregator math

The combination layer is a textbook **Naive Bayes log-odds sum** with calibration. Every step is documented in code (`pipeline/inference/aggregator.py`) and is small enough to keep in your head:

1. **Calibrate every detector.** Each detector emits scores on its own scale. We fit a monotonic *isotonic regression* per detector that maps raw score → estimated cheat probability `p_i ∈ (0, 1)`.

2. **Convert probabilities to log-odds (`logit`).** Logit is the function that turns `p` into `log(p / (1−p))`. Probability 0.5 becomes 0; 0.9 becomes +2.2; 0.99 becomes +4.6. This is the natural scale for adding evidence: independent signals add linearly here.

3. **Add the prior.** Without a prior, the formula assumes 50% base-rate cheating, which is absurd. For an expected 5% cheat rate the prior in logit space is `log(0.05 / 0.95) ≈ -2.94`. Detectors then have to provide enough evidence to drag the posterior back above 0 (= 50%).

4. **Sigmoid back to a probability.** The combined logit goes through `sigmoid(x) = 1 / (1 + exp(-x))` to produce the final `session_risk ∈ [0, 1]`.

### Worked example

Three detectors fire on a session, all calibrated, prior = 5%:

| detector | raw score | p_i (calibrated) | logit(p_i) |
|---|---|---|---|
| IsolationForest | 0.42 | 0.10 | −2.20 |
| OneClassSVM | −0.05 | 0.30 | −0.85 |
| LSTM-AE | 2.10 | 0.85 | +1.73 |

```
Σ logit(p_i)   =  −2.20 − 0.85 + 1.73  =  −1.32
prior_logit    =  log(0.05 / 0.95)     ≈  −2.94
posterior_logit =  −1.32 + (−2.94)     =  −4.26
posterior_risk =  sigmoid(−4.26)        ≈  0.014   ← low risk
```

Now the same scenario but every detector says 0.9:

```
Σ logit(p_i)   =  3 × 2.20             =  +6.59
posterior_logit =  +6.59 + (−2.94)     =  +3.65
posterior_risk =  sigmoid(3.65)         ≈  0.975   ← clearly flag
```

Three independent strong signals override the conservative prior; one or two medium signals do not. **That conservatism is what keeps the false-positive rate low** — production anti-cheat needs ≤ 0.1 % FPR, which the prior enforces by default.

---

## API surface

### WebSocket endpoint

```
GET ws://<host>:8000/stream
```

**Client → server messages** (one JSON per event):

```json
{"t": 12345.6, "type": "mouse_move", "x": 100, "y": 200, "dx": 1, "dy": 0}
{"t": 12350.0, "type": "mouse_click", "x": 100, "y": 200, "pressed": true}
{"t": 12500.0, "type": "key_press", "key": "w"}
{"type": "__end__"}
```

**Server → client messages** (only when a window or chunk boundary fires):

```json
{
  "t": 30000.0,
  "n_events": 1234,
  "n_windows": 1,
  "n_chunks": 3,
  "per_detector": {
    "IsolationForest": 0.42,
    "LSTMAutoencoder": 1.83
  },
  "session_risk": 0.18,
  "detector_logits": {
    "IsolationForest": -1.2,
    "LSTMAutoencoder": 0.8
  },
  "triggered_by": "window"
}
```

The server emits at boundary events only (every 30 s for the classical detectors, every `chunk_length=64` events for the LSTM-AE). No per-event echoing.

### Replay client

```bash
# WebSocket against a running API
python -m scripts.replay_session data/raw/<file>.json \
    --speed 5 \
    --inject-cheat aimbot \
    --inject-at 30 \
    --out /tmp/replay_scores.jsonl

# Offline (no server, drives the engine in-process)
python -m scripts.replay_session data/raw/<file>.json \
    --offline \
    --inject-cheat aimbot \
    --inject-at 30 \
    --out /tmp/replay_scores.jsonl
```

### Dashboard "Live Session" tab

`streamlit run dashboard/app.py` → 📡 **Live Session** tab. Pick a session, configure cheat injection, click "Run live replay". The chart updates as the engine processes events.

### Programmatic demo

```bash
python -m scripts.build_phase4_demo \
    --cheat aimbot \
    --inject-at 10
```

Produces `reports/figures/phase4_live_replay.png` (static) and `reports/figures/phase4_live_demo.gif` (animated).

---

## Implementation map

| Path | What |
|---|---|
| `pipeline/inference/aggregator.py` | `IsotonicCalibrator`, `RiskAggregator`, `fit_aggregator_from_synthetic` |
| `pipeline/inference/streaming.py` | `SessionStreamState`, `ScoreUpdate`, `build_stream_state` |
| `api/streaming.py` | `/stream` WebSocket endpoint, mounted from `api/main.py` |
| `scripts/train_lstm_ae.py` | Persists `models/lstm_ae.pt` + `models/lstm_ae_meta.json` |
| `scripts/replay_session.py` | CLI replay (WebSocket or offline) with optional cheat injection |
| `scripts/build_phase4_demo.py` | Generates the PNG + GIF demo artifacts |
| `dashboard/app.py` | 📡 Live Session tab |
| `tests/test_aggregator.py` | 15 tests covering calibrator monotonicity, NaN handling, log-odds combination, explain output |
| `tests/test_streaming.py` | 14 tests covering state machine + WebSocket via FastAPI TestClient |
| `tests/test_replay_session.py` | 4 tests for cheat injection + offline replay JSONL output |

187 unit tests total. The streaming pipeline runs entirely on the RTX 3070 via the persisted LSTM-AE artifact and falls back to CPU automatically.

---

## What works and what doesn't (honest)

**What works**

- The math: per-detector calibration is monotonic, the log-odds sum behaves correctly under independence, the prior pulls posterior risk in the right direction (verified in `tests/test_aggregator.py`).
- The plumbing: live WebSocket events produce a session_risk in real time, the dashboard renders it as a growing timeline, the GIF demo is fully reproducible.
- Per-detector explainability: `RiskAggregator.explain()` returns per-detector logit contributions, surfaced in the dashboard as a separate panel and visible in the GIF's bottom plot.

**What doesn't (yet)**

- Combined session-level AUC. With the current mock-data legit baseline, every classical detector over-fires on any "active" session, so the calibrated probabilities sit near 0.7–0.9 even for legit replay. The combined risk is dominated by these over-firing detectors, not by the chunk-level LSTM-AE signal. We expect this to flip once real gameplay recordings replace the mock baseline.

The honest current numbers (stratified 50/50 split, mock data) are in `docs/ADVERSARIAL.md`. The streaming demo's value at this stage is the **infrastructure proof** — it shows the math, transport, calibration, dashboard, and aggregator all working end-to-end. Real-data validation comes once the GTA recordings land.

---

## WSL-host networking notes

The recorder runs on the Windows host; the API runs in WSL2. If you ever wire the recorder to push events directly to `/stream` (Phase 4.1 backlog), the WS URL from Windows looks like:

```
ws://localhost:8000/stream    # works if you started uvicorn with --host 0.0.0.0
```

The dashboard's Live tab runs **inside WSL** and uses the in-process engine via `replay_offline()`, so it doesn't depend on the WS layer at all. WebSocket networking is only needed when you want a non-WSL client (the live recorder, a deployed dashboard, an external test harness).

---

## See also

- [docs/LSTM_AE.md](LSTM_AE.md) — the sequence model that produces the chunk-level signal
- [docs/ADVERSARIAL.md](ADVERSARIAL.md) — synthetic cheat methodology + per-detector results
- [docs/FEATURES.md](FEATURES.md) — the 25-feature classical detector input
- [docs/ROADMAP.md](ROADMAP.md) — full 5-phase plan with the Phase 4.1 backlog
