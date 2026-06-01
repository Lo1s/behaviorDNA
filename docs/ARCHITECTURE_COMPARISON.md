# Sequence-autoencoder architecture comparison

Is the LSTM the right backbone for chunk-level cheat detection — or would a
convolutional (TCN) or self-attention (Transformer) model do better? Run with
`python -m scripts.compare_architectures`: all three trained with the **same**
loop on the **same** 18-session legit chunks, evaluated with the **same**
chunk-AUC metric. `--eval-data real` evaluates on the **real labelled cheats**
instead of synthetic (see [Real-cheat results](#real-cheat-results-2026-06-02)).

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

## Real-cheat results (2026-06-02)

Re-run with `--eval-data real` — same training loop and legit chunks, but the
cheat eval set is now the **3 real labelled hydRa sessions** (per-type via
`cheat_segments_typed`), not synthetic. 1 run, 25 epochs, RTX 3070.

| Model | Params | Val recon loss | Aimbot | Triggerbot | Macro |
|---|---|---|---|---|---|
| **LSTM-AE** | 196k | 0.579 | 0.525 | **0.603** | **0.566** |
| **TCN-AE** | **78k** | **0.473** | 0.512 | 0.590 | 0.559 |
| **Transformer-AE** | 207k | 0.617 | **0.527** | **0.603** | **0.566** |

(`reports/figures/arch_comparison_real.png`. Legit baseline = the 18 legit
sessions **plus the cheat sessions' own clean chunks**, so it's a harder, more
honest contrast than the chunk-benchmark's legit-only baseline — which is why
these sit a touch below the `--lstm-chunk-only` numbers.)

**What it confirms:**
- **All three are tied within ~0.015 on real cheats** — an even tighter spread
  than synthetic, and far below the synthetic AUCs (0.73–0.96). The thesis holds
  *more* strongly on real data: **capacity is not the bottleneck, data is.**
  Swapping the backbone is tuning on noise.
- **LSTM ≈ Transformer** (identical to 3 dp on triggerbot/macro); **TCN is
  marginally worst again** despite reconstructing best (val loss 0.47) — the same
  "fidelity ≠ contrast" pattern as the synthetic run.
- This is unsupervised reconstruction. The lever that could actually move the
  ranking is **supervised** training on the real labels — but with 3 cheat
  sessions / 1 player it would overfit; it waits on more (cross-player) data.

## Recommendation

Keep the **LSTM-AE** in production: it's the incumbent, fully integrated
(streaming, persistence, explainability), and statistically tied with the
Transformer on **both** synthetic and now **real** cheats. There is **no
evidence a different backbone would help** — the gap is data, not architecture,
so no change is made on noise.

## Revisit after real cheat data (tracked)

✅ **Unsupervised real-cheat re-run done** (above) — the ranking did **not**
change; all three stay tied, confirming data (not capacity) is the limit.

The remaining open follow-up is a **supervised** re-run of this harness — a
classifier head on each backbone — where the real labels should dominate and the
ranking *may* change. That's **gated on more cheat data**: with 3 cheat sessions
from 1 player a supervised classifier overfits immediately, so it waits on the
cross-player batch (`docs/CHEAT_DATA_COLLECTION.md` → "Second batch: cross-player").
The pure modules (`pipeline/models/{tcn_ae,transformer_ae}.py`) and this script
are ready for it.
