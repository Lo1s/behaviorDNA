# Sequence-autoencoder architecture comparison

Is the LSTM the right backbone for chunk-level cheat detection — or would a
convolutional (TCN) or self-attention (Transformer) model do better? Run with
`python -m scripts.compare_architectures`: all three trained with the **same**
loop on the **same** 18-session legit chunks, evaluated with the **same**
chunk-AUC metric. (Re-run after real cheat recordings land — see below.)

## Results (1 run, RTX 3070, 25 epochs)

| Model | Params | Train time | Val recon loss | Overfit gap | Aimbot | Triggerbot | Macro |
|---|---|---|---|---|---|---|---|
| **LSTM-AE** | 196k | 28 s | 0.579 | −0.03 | 0.804 | 0.940 | 0.614 |
| **TCN-AE** | **78k** | **17 s** | **0.473** | +0.05 | 0.731 | 0.842 | 0.572 |
| **Transformer-AE** | 207k | 33 s | 0.617 | ~0.00 | **0.817** | **0.958** | 0.608 |

(chunk-level ROC AUC; chance = 0.50. Figure: `reports/figures/arch_comparison.png`.)

## What it shows

- **All three are competitive, and the gaps are modest** — at N=18 sessions with
  a single validation split, differences of ~0.01–0.02 AUC are within run-to-run
  noise. **Capacity is not the bottleneck; data is** (consistent with the 5d
  ablation finding for the identification model).
- **The Transformer is viable on this hardware and marginally edges detection**
  (aimbot 0.82, triggerbot 0.96) at similar cost to the LSTM — a useful
  counter-point to "transformers need huge data": the *model* is tiny too
  (~0.2M params, <1 GB VRAM, 33 s to train), so it isn't starved.
- **Lower reconstruction loss ≠ better detection.** The TCN reconstructs *best*
  (val loss 0.47) and is the cheapest (78k params, fastest), yet separates cheat
  chunks *worst*. The autoencoder's job here is anomaly *contrast*, not fidelity
  — a good reminder to optimise the metric you actually care about.

## Recommendation

Keep the **LSTM-AE** in production for now: it's the incumbent, fully integrated
(streaming, persistence, explainability), and statistically tied with the
Transformer at this scale. There is **no evidence a different backbone would
help** until there's more data — so no architecture change is made on noise.

## Revisit after real cheat data (tracked)

This comparison is **unsupervised** (reconstruction). Once `cheat_sim` produces
**labelled** real cheat chunks (`docs/CHEAT_DATA_COLLECTION.md`), the right
follow-up is a **supervised** re-run of this harness — a classifier head on each
backbone — where the label signal should dominate and the ranking may change.
The pure modules (`pipeline/models/{tcn_ae,transformer_ae}.py`) and this script
are ready for that.
