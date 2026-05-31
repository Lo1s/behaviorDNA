"""
pipeline/constants.py
=====================
Structural constants shared across the pipeline.

These are **not** runtime-tunable knobs — they are baked into trained
artifacts (the feature schema, the LSTM-AE weights, the persisted scaler).
Changing one requires re-deriving everything downstream (`dvc repro`, and
retraining the LSTM-AE / aggregator). That is exactly why they live here in
one place and are *imported* rather than re-declared per module: a value that
must agree between the offline feature pipeline and the online streaming engine
should have a single source of truth, not a copy in each file.

Genuinely tunable settings (model type, split sizes, hyperparameters) live in
`configs/training.yaml`, not here.
"""

# Non-overlapping analysis window for all classical window features, in
# milliseconds. Used by both pipeline/features/run.py (offline feature
# extraction) and pipeline/inference/streaming.py (online window flushing);
# the two MUST agree or streamed features won't match what the model trained on.
WINDOW_MS = 30_000  # 30 seconds
