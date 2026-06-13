"""
pipeline.pretraining
=====================
Self-supervised pretraining of the sequence encoder (Phase 8).

Pretrains the existing :class:`pipeline.models.lstm_ae.LSTMAutoencoder` with a
masked-denoising objective on a large unlabelled human-mouse corpus
(CaptchaSolve30k), then transfers the full encoder+decoder weights into the
downstream chunk-level cheat-detection autoencoder to measure data-efficiency.

The linchpin is that **all three corpora share one 8-D event-tensor schema**
(see :mod:`pipeline.sequences.preprocessing`), so the pretrained weights load
straight into the fine-tuning model. See ``docs/PRETRAINING.md``.
"""
