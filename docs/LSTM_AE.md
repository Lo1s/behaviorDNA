# LSTM Autoencoder for Sequence Anomaly Detection

> Phase 2 of the [BehaviorDNA roadmap](ROADMAP.md). Unsupervised sequence model that operates on the raw event stream and solves the Phase 1 aimbot detection gap.

> **Data status (2026-05-30).** Numbers below are now measured on **18 real GTA sessions** (3 players, 1.35M events). Retrain via `python -m scripts.train_lstm_ae` (persisted to `models/lstm_ae.pt`). Real-data chunk-level AUC: aimbot 0.79, triggerbot 0.93, macro 0.60. The chunk-level result is the headline; session-level aggregation stays near chance (see `docs/STREAMING.md` → Phase 4.1).

## Motivation

[Phase 1](ROADMAP.md#phase-1--trajectory--temporal-features) added 7 window-level trajectory + timing features and closed the detection gap for triggerbots (AUC 0.50 → 0.87) and macros (0.55 → 0.68). **Aimbot stayed at AUC 0.53** — the 150 ms snap signal is too brief to survive 30-second window aggregation, no matter how clever the per-window features are.

Phase 2 sidesteps window aggregation entirely with an LSTM autoencoder trained on raw event sequences.

## Architecture

```
Input  (B, L=64, 8)
   │
   ▼  Bidirectional LSTM encoder (hidden=64, num_layers=2, dropout=0.2)
   │   final hidden state h_T → (B, 2 × num_layers × hidden) = (B, 256)
   ▼
   Linear → bottleneck z ∈ ℝ^16
   │
   ▼  broadcast z across L timesteps
   LSTM decoder (hidden=64, num_layers=2)
   │
   ▼
   Linear → reconstruction (B, L, 8)
   │
   ▼
   Loss = MSE(input, reconstruction)
```

| | value |
|---|---|
| Parameters | ~196,000 |
| Training | Adam, lr=1e-3, weight_decay=0, 30 epochs |
| Batch size | 256 |
| Device | CUDA (RTX 3070) auto-detected, CPU fallback |
| Wall-clock | ~2 min for 30 epochs on the RTX 3070 |
| Reproducibility | `torch.manual_seed(42)`, `numpy.random.seed(42)` |

## Input representation

Each event becomes one row in an `(N, 8)` float32 tensor:

| ch | feature | semantics |
|---|---|---|
| 0 | `log1p(dt_ms)` | log-compressed time since previous event |
| 1 | `dx_norm` | mouse delta x ÷ (sensitivity × DPI / 800) |
| 2 | `dy_norm` | mouse delta y ÷ (sensitivity × DPI / 800) |
| 3 | `is_mouse_move` | one-hot |
| 4 | `is_mouse_click_press` | one-hot (button-down only) |
| 5 | `is_mouse_scroll` | one-hot |
| 6 | `is_key_press` | one-hot |
| 7 | `is_key_release` | one-hot |

Sens/DPI normalisation mirrors `compute_mouse_kinematics` in [`pipeline/features/run.py`](../pipeline/features/run.py). The `log1p` on `dt_ms` compresses the dynamic range (idle gaps vs sub-ms event bursts).

## Training

- **Train-fold-only normalisation.** Mean/std computed across the train-fold session tensors; saved alongside model weights for inference. Mirrors `StandardScaler` in `pipeline/training/run.py`.
- **Chunking.** Each session tensor is sliced into chunks of length `L=64` with stride `S=32` (50% overlap during training; non-overlapping at inference).
- **Loss.** Per-element MSE between input chunk and its reconstruction, averaged over batch, length, and feature axes.
- **Early stopping.** Best val-loss weights restored at end of training (`val_loss = 0.28` typical on the current 15-session dataset).

## Anomaly scoring

Two evaluation granularities are reported in the benchmark:

### Chunk-level (LSTMAutoencoder/chunk)

Per-chunk reconstruction MSE. Each chunk is labelled cheat-positive if it overlaps any `cheat_segment` from the synthetic file, otherwise legit. AUC computed against the entire pool of legit chunks (from legit files) vs cheat chunks.

This is **what the model actually learnt to flag**. Headline numbers:

| Cheat | Classical best AUC (real) | LSTM-AE chunk AUC (real) |
|---|---|---|
| Aimbot | 0.58 | **0.79** |
| Macro | 0.63 | 0.60 |
| Triggerbot | 0.76 | **0.93** |

*(real GTA data, 2026-05-30; the `reports/figures/phase4_chunk_detection.png` figure shows the per-chunk error distributions behind these.)*

### Session-level (LSTMAutoencoder/session)

Each session's chunk scores aggregated to their 95th percentile, then AUC computed across sessions.

| Cheat | LSTM-AE session AUC (real) |
|---|---|
| Aimbot | ~0.51 |
| Macro | ~0.51 |
| Triggerbot | ~0.51 |

This is **noticeably worse than chunk-level** and worse than the classical detectors at session level. Why:

- Each synthetic file has ~600 chunks; the aimbot affects ~38 of them
- Cheat chunks score 7× higher than clean chunks (mean MSE 2.75 vs 0.40)
- But the p95 across 600 chunks puts us in the natural-variance tail of the legit baseline
- The signal is *present* but a single-detector percentile aggregation dilutes it

### The gap is the case for Phase 4

Multi-detector Bayesian aggregation will combine:
- LSTM-AE per-chunk scores (catches aimbot, near-perfect on triggerbot)
- Window-level `click_reaction_mean` (catches triggerbot at session-level AUC 0.87)
- Window-level `keystroke_periodicity` (catches macro at session-level AUC 0.68)

into a single session-level risk score. No single detector wins for every cheat type; the combination should.

## Reproducing

```bash
source .venv/bin/activate

# Generate the synthetic dataset (idempotent)
python -m pipeline.adversarial.generate_dataset

# Run the full benchmark including LSTM-AE training + scoring
python -m pipeline.adversarial.benchmark

# Re-execute the tutorial notebook end-to-end
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
  --inplace notebooks/09_lstm_autoencoder.ipynb
```

Output appears in:
- `reports/adversarial/benchmark_results.csv` — full AUC table
- `reports/figures/lstm_ae_step*.png` — every figure in the tutorial

## WSL2 + CUDA setup notes

The RTX 3070 is visible inside WSL2 via the NVIDIA WSL driver (installed on the Windows host; **no separate driver inside WSL**). PyTorch with `cu130` wheels handles the rest. Verification:

```python
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 3070'
```

If CUDA isn't visible, the code falls back to CPU automatically — training takes ~10× longer but produces the same model.

## Implementation map

| Path | What |
|---|---|
| `pipeline/sequences/preprocessing.py` | `session_to_event_tensor`, `fit_normalizer`, `apply_normalizer` |
| `pipeline/sequences/dataset.py` | `EventSequenceDataset` (PyTorch Dataset with chunking) |
| `pipeline/models/lstm_ae.py` | `LSTMAutoencoder`, `train_lstm_ae`, `score_sequences` |
| `pipeline/adversarial/benchmark.py` | `run_lstm_ae_benchmark` (chunk + session granularity) |
| `notebooks/09_lstm_autoencoder.ipynb` | 11-step tutorial with all visualisations |
| `tests/test_sequences.py`, `tests/test_lstm_ae.py` | 38 unit tests covering shape, normalisation, training loop, inference |

## Limitations + path forward

1. **Aggregation gap.** Single-detector session-level scoring underperforms chunk-level. Phase 4's Bayesian multi-detector aggregator is the planned fix.
2. **Small training set.** Only 15 legit sessions yield ~19k training chunks. The legit baseline has high natural variance because the model hasn't seen enough \"normal\" patterns. Retraining with the incoming GTA recordings should tighten the baseline.
3. **No teacher forcing.** The current decoder takes a broadcast bottleneck vector at every step. A teacher-forced or autoregressive decoder might reconstruct longer sequences more sharply, but adds complexity.
4. **Per-chunk decisions only.** The model produces no calibrated probability — just an unnormalised MSE. Phase 5 calibration (isotonic / Platt) is the natural place to make these comparable to classical detector outputs.

## See also

- [docs/ADVERSARIAL.md](ADVERSARIAL.md) — synthetic-cheat methodology (Phase 3)
- [docs/FEATURES.md](FEATURES.md) — classical feature catalogue (Phase 1)
- [docs/ROADMAP.md](ROADMAP.md) — the full 5-phase plan
