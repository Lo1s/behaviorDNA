# Self-supervised pretraining (Phase 8 / 8.1 / 8.2)

> *A small "foundation model for human input motion" — and an honest measurement of why it doesn't
> (yet) transfer to game-input biometrics.*
>
> Code: [`pipeline/pretraining/`](../pipeline/pretraining) · scripts `pretrain_encoder` /
> `domain_gap_report` / `data_efficiency` / `indomain_transfer` / `contrastive_transfer` · notebooks
> [21](../notebooks/21_pretraining.ipynb) (8/8.1) & [22](../notebooks/22_contrastive_pretraining.ipynb) (8.2) ·
> tests `tests/test_pretraining.py`, `tests/test_contrastive.py`.

## Why

Notebooks [16](../notebooks/16_architecture_comparison.ipynb) (capacity isn't the bottleneck) and
[19](../notebooks/19_identification_at_scale_public.ipynb) ("the signal survives; the data budget is
the binding constraint") agree the limiting factor here is **data, not model**. Phase 8 attacks that:
pretrain the sequence encoder self-supervised on a large *unlabelled* human-mouse corpus, then test
whether it buys **data-efficiency** on the downstream chunk-level cheat detector — the most "modern
ML" headline available, and one that targets the actual constraint.

The discipline (the project's honest-positioning rule): **a null is a publishable result.** We
measure the captcha→game domain gap *first* so the verdict is interpretable either way.

## Method

**Objective — masked-step denoising.** Zero a random 15 % of timesteps in an input chunk; train the
existing [`LSTMAutoencoder`](../pipeline/models/lstm_ae.py) to reconstruct the **clean** chunk
(`MSE(model(masked), clean)`). Filling masked steps from context forces the bottleneck to learn the
human-motion manifold. The full encoder **and** decoder transfer into fine-tuning (max reuse). Code:
[`pipeline/pretraining/masking.py`](../pipeline/pretraining/masking.py),
[`pretrain.py`](../pipeline/pretraining/pretrain.py).

**One shared 8-D schema — the precondition for transfer.** Transfer is only valid if pretrain /
fine-tune / eval share the encoder's input dim *and channel semantics*. Everything maps to the
[8-D event tensor](../pipeline/sequences/preprocessing.py) `[dt, dx, dy, is_mouse_move,
is_mouse_click_press, is_mouse_scroll, is_key_press, is_key_release]`:

| Corpus | Role | Encoding |
|---|---|---|
| **CaptchaSolve30k** (≈17.7k mouse sessions) | pretrain | per-tick `{x,y,isDown}` → `dx/dy` deltas, `is_mouse_move=1`, click on rising edge, `dt=log1p(~4.2 ms)` |
| **CS2CD** (10 players, real cheats) | fine-tune + eval | `usercmd_mouse_dx/dy`, `FIRE`→click rising edge, `dt=log1p(~15.6 ms)`; `RIGHTCLICK`/scope **dropped** (no native channel) |
| **GTA** (N=18, synthetic cheats) | fine-tune + eval | native `session_to_event_tensor` (event-driven) |

Captcha/CS2 are **sampled** streams (one sample per tick); GTA is an **event** stream. That distinction
is itself part of the domain gap. Each domain is z-scored on its own train fold, so raw scale is
removed and the residual gap lives in temporal/geometric *shape*. Adapters:
[`pipeline/pretraining/corpora.py`](../pipeline/pretraining/corpora.py).

## The captcha → game domain gap (measured before claiming transfer)

Per-channel KS + PSI (`pipeline/monitoring/drift.py`), reference = captcha
(`reports/pretraining_domain_gap.json`, figure `reports/figures/phase8_domain_gap.png`):

| channel | captcha→CS2CD PSI | captcha→GTA PSI |
|---|---|---|
| `dt` | **11.8** (sig) | **9.8** (sig) |
| `dx` | 0.08 (none) | **0.37** (sig) |
| `dy` | 0.06 (none) | 0.04 (none) |
| `is_mouse_click_press` | 0.00 | 0.00 |
| `is_mouse_move` | ~const (both) | drifted |

**Reading:** the **`dt` channel is wildly mismatched** (the sampled-vs-event distinction made concrete:
captcha fixed ~4.2 ms tick vs CS2 ~15.6 ms vs GTA event-driven with idle gaps) — the temporal encoding
barely overlaps. Movement geometry transfers *well to CS2* (`dx/dy` PSI < 0.1) but *poorly to GTA*
(`dx` PSI 0.37; game aim is a different regime from drag-to-target). The gap predicts **weak
transfer**, strongest exactly where it matters least (`dt`).

## Headline — data-efficiency curves

Budget (number of legit fine-tuning units) × {pretrained-init, scratch-init} × 5 seeds; fine-tune the
AE, score a **fixed** legit-vs-cheat chunk-AUC eval set (`scripts/data_efficiency.py`,
`reports/data_efficiency_{cs2cd,gta}.json`, figures `phase8_data_efficiency_{cs2cd,gta}.png`).

| Dataset | Budget | Pretrained | Scratch | Δ |
|---|---|---|---|---|
| **CS2CD** | 1 / 2 / 5 / 10 streams | 0.703 / 0.702 / 0.698 / 0.699 | 0.702 / 0.702 / 0.698 / 0.698 | **≈ 0.000** |
| **GTA** | 2 / 5 / 10 / 15 sessions | 0.538 / 0.549 / 0.557 / 0.556 | 0.539 / 0.553 / 0.562 / 0.561 | **−0.001 … −0.005** |

(±std ≈ 0.00–0.02 over seeds — all Δ are within noise.)

## Verdict — a rigorous null

**At this scale, with masked-denoising on the 8-D event tensors, captcha-pretraining does not buy
data-efficiency on either downstream cheat-detection task.** The domain-gap report explains *why*: the
prior we learned (fixed-tick, drag-to-target human motion) is mismatched to the game-aim regime on the
channel carrying the most structure (`dt`, plus `dx` for GTA).

Two honesty caveats that sharpen the result:

1. **CS2CD is near-separable at random init** (untrained chunk-AUC ≈ 0.70): the cheat snaps have
   larger deltas → higher reconstruction error even with random weights. So neither fine-tuning nor
   pretraining moves the CS2CD number — CS2CD is a *weak discriminator* of the transfer question.
2. **GTA fine-tuning itself helps** (AUC climbs 0.54→0.56 with budget) — so the curve is meaningful;
   it's *pretraining* specifically that adds nothing.

This is the outcome the [roadmap](ROADMAP.md#phase-8--self-supervised-pretraining) flagged as valid:
*"pretraining-doesn't-help = domain gap dominates, also a real result."* It's a useful negative result
for the field — **a generic human-mouse corpus is not a drop-in foundation for game-input
biometrics.**

### What would change the verdict
1. **Match the temporal encoding** — resample game streams to a fixed tick (or drop `dt`) so the
   prior isn't fighting a 10-PSI mismatch on its most-structured channel.
2. **An in-domain (game-mouse) pretraining corpus** — the geometry gap to GTA (`dx` PSI 0.37) says
   out-of-domain *motion* is the problem, not the method.
3. **A contrastive objective** (Phase 8 stretch goal) — reconstruction is dominated by input
   magnitude (see caveat 1); a contrastive prior would be less magnitude-bound. ✅ **Tested in Phase 8.2
   (below) — and this one is *not* a null:** in-domain contrastive beats both random-init and the
   reconstruction encoder on the frozen embedding. The objective was the lever.

## Phase 8.1 — in-domain pretraining (does closing the domain gap rescue the null?)

Phase 8's verdict named two fixes that *should* change it: **(1) match the temporal encoding** and
**(2) pretrain on an in-domain game-mouse corpus**. Phase 8.1 ran both — and **neither moved the
needle.**

**Setup.** Pretrain the *same* LSTM-AE **in-domain** on the full public **CS2CD release** (795 matches),
legit-only (`no_cheater_present`), then transfer to the **same GTA cheat-detection target as Phase 8**
(directly comparable). Three crossed axes + a captcha comparison source (Phase 8's encoder on the same
GTA pool) and from-scratch:
- **Arm:** scratch (A) · **frozen** encoder (B, decoder-only — the condition Phase 8 skipped) · fine-tuned (C).
- **Source:** `s1` = native CS2 tick `dt` · `s2` = **dt-neutralised** (zeroed in *both* domains) — Phase 8's
  fix #1. CS2's `dt` is a literal constant, so the naive "resample to the GTA grid" is a no-op after
  z-scoring; neutralising the channel is the clean causal test of the temporal mismatch.
- **Pretraining volume:** 50 / 200 / 382 matches. **Step 0** (`scripts/cs2cd_diversity_probe.py`) found
  the release is **player-anonymised** (`Player_1..10` *per match*, not linkable across matches), so this
  is a *stream-volume* axis, **not** player diversity — and CS2CD splits are match-disjoint only.

**Result (GTA chunk-AUC, fine-tune budget 15, mean of 3 seeds).** Reference: **scratch = 0.562**. Every
pretrained config sits **at or below** it:

| source | frozen | fine-tune |
|---|---|---|
| captcha (out-of-domain) | 0.557 | 0.557 |
| cs2cd `s1` @382 (in-domain, native `dt`) | 0.549 | 0.555 |
| cs2cd `s2` @382 (in-domain, `dt`-neutralised) | 0.553 | 0.559 |

- **In-domain pretraining does not beat scratch** (or captcha) — marginally *worse*.
- **`s2` ≈ `s1`** → the `dt` mismatch was **not** the binding constraint (fix #1 fails).
- **Volume 50→200→382 is flat** → more in-domain data doesn't help.
- **frozen ≤ fine-tune ≤ scratch** → the in-domain embedding carries no transferable structure for the task.

**Why — the domain-gap re-run, CS2CD-as-reference** (`domain_gap_report.py --reference cs2cd`): in-domain
CS2 is **not** closer to GTA than captcha was. Its spatial gap is *worse* (`dx` PSI **0.88** vs captcha's
0.37) and the temporal gap persists (`dt` KS **0.95** — CS2's fixed tick is as mismatched to GTA's
event-driven `dt` as captcha's was; `dt` PSI degenerates to 0 on the constant channel, so KS is the
honest metric there). The in-domain corpus simply isn't on the GTA manifold.

**Verdict — the null is deeper than the domain gap** (pre-registered outcome (b)). Closing the domain gap
*and* removing the temporal mismatch both leave transfer unchanged. The binding constraint is the
**task/data regime**, not the corpus: the real-cheat GTA chunk signal is weak (~0.56 — the same ceiling
scratch reaches) and lives in obvious per-event anomalies, not a learnable motion *prior*, so no
pretraining flavour helps at N≈18 sessions / 3 players. The one Phase-8 lever 8.1 did **not** test is a
**contrastive objective**; the rest of the "what would change the verdict" list is now closed (negative).
Figure: `reports/figures/phase8_1_indomain_transfer_gta.png`.

Honest scope notes: the player-diversity axis the roadmap hoped for doesn't exist (anonymised release); a
*player-disjoint* GTA target was tried but floored every arm at chance (cross-player shift dominates at 2
training players), so the comparable Phase-8 non-disjoint pool is the target; the CS2CD cheat-detection
sanity arm is omitted (the full release carries no recoverable per-player cheat label — the match-level
"not-cheater" label is ~56% precise per the dataset card).

## Phase 8.2 — contrastive pretraining (does the *objective* matter?)

> ✅ **Done (2026-06-15) — the project's first non-null pretraining result.** Swapping masked-denoising
> reconstruction for a **contrastive** objective (SimCLR/TS2Vec-style NT-Xent over two augmented views),
> evaluated **contrastive-natively** on the *frozen* 16-D embedding, beats **both** random-init *and* the
> Phase-8.1 reconstruction encoder on every probe — modestly, but outside the seed bands. The lever was the
> **objective**, not the corpus, capacity, or the `dt` encoding.

**Why.** Phase 8 / 8.1 both used *reconstruction* (MSE), which is **magnitude-dominated** — exactly the
Phase 8.1 caveat (CS2CD is near-separable at random init because cheat snaps have larger deltas → larger
reconstruction error even untrained). A contrastive prior is **magnitude-invariant by construction** (one
view is a random rescale of the other) and is scored on the embedding *directly* (kNN / one-class /
linear-probe), not by reconstruction error — sidestepping that caveat. It was the one item left on Phase 8's
"what would change the verdict" list.

**Method.** Two augmented views per 8-D chunk (`pipeline/pretraining/augment.py`: jitter + random scale on
`dx/dy`, time-mask, crop-resize) → `LSTMAutoencoder.encode` (the *same* 16-D bottleneck as 8/8.1) → a
projection head → **NT-Xent** (`pipeline/pretraining/contrastive.py`). Reuses the 8.1 in-domain CS2CD shard
pipeline verbatim (`CS2CDShardChunkDataset` → a contrastive two-view subclass + `ShardGroupedSampler`),
volumes 50/200/382, plus an out-of-domain captcha contrastive encoder. **Eval** = freeze the encoder, embed
GTA legit/cheat chunks, and score with Mahalanobis / OCSVM / kNN (unsupervised one-class, fit on legit only)
+ a cross-validated linear probe (`pipeline/pretraining/embed_eval.py`, `scripts/contrastive_transfer.py`),
mean over the GTA fine-tune-budget × seed grid. The **random-init** encoder under the same probe is the
apples-to-apples baseline (it also retires Phase 8.1's "near-separable at random init" caveat).

**Result (GTA cheat-detection ROC AUC on the *frozen* embedding; mean over budget × seed).**

| source | Mahalanobis | OCSVM | kNN | linear-probe |
|---|---|---|---|---|
| random init | 0.481 | 0.477 | 0.486 | 0.547 |
| recon cs2cd@382 (Phase 8.1) | 0.511 | 0.528 | 0.493 | 0.603 |
| **contrastive cs2cd@382 (in-domain)** | **0.550** | **0.540** | **0.585** | **0.662** |
| contrastive captcha (out-of-domain) | 0.482 | 0.474 | 0.530 | 0.605 |

(seed±std ≈ 0.01–0.04; the contrastive-vs-baseline gaps exceed it.)

- **In-domain contrastive beats both baselines on every probe** — Δ vs reconstruction ≈ +0.04 (Maha),
  +0.01 (OCSVM), +0.09 (kNN), +0.06 (linear-probe); Δ vs random ≈ +0.07 / +0.06 / +0.10 / +0.12. This is the
  first time *any* pretraining flavour cleared random-init on this target.
- **The gain is in-domain-specific.** Out-of-domain captcha contrastive matches reconstruction on the
  linear-probe (0.605) but is **random-level on the unsupervised one-class metrics** (Maha 0.482) — so the
  contrastive *objective* lifts representation quality even OOD, but *unsupervised detection* needs the
  in-domain corpus.
- **Volume is flat** (50/200/382 → Maha 0.537/0.522/0.550, kNN 0.568/0.556/0.585) — the gain **saturates by
  ≈50 matches**, echoing 8.1's flat volume axis (but now well *above* baseline, not at it).
- **Unsupervised < supervised:** the cheat is more *linearly separable* (probe 0.66) than a simple
  density/distance detector extracts (Maha/OCSVM ~0.54–0.55; kNN 0.585 is the best unsupervised scorer).

**Verdict — modest but real; the objective was the lever.** Reconstruction pretraining was a null on this
target across two corpora *and* the `dt` fix (Phase 8 / 8.1); the **contrastive objective is not** —
in-domain it yields a frozen embedding measurably better than both random-init and reconstruction. The
absolute ceiling stays modest (~0.55–0.66, i.e. near the weak ~0.56 real-cheat chunk signal — the task/data
regime still binds), so it is **not a deployable detector on its own**, and the headline is *directional*: at
small N, swapping a magnitude-dominated objective for a magnitude-invariant one is what moved the needle.
Honest caveat (as in 8.1): the legit eval baseline includes the one-class fit's own sessions (mildly
optimistic), but it is identical for every source, so the *comparison* is fair. Figure:
`reports/figures/phase8_2_contrastive_transfer_gta.png`; study notebook
[22](../notebooks/22_contrastive_pretraining.ipynb).

## Reproduce (CUDA desktop)

```bash
source .venv/bin/activate
python -m scripts.pretrain_encoder --max-sessions 6000 --epochs 30   # → models/pretrained_encoder.pt (DVC-tracked)
python -m scripts.domain_gap_report                                  # → reports/pretraining_domain_gap.json
python -m scripts.data_efficiency --domain cs2cd                     # → reports/data_efficiency_cs2cd.json
python -m scripts.data_efficiency --domain gta
jupyter nbconvert --to notebook --execute --inplace notebooks/21_pretraining.ipynb   # CPU-fast (loads the above)

# Phase 8.1 — in-domain (CS2CD) pretraining (needs the ~48 GB full-release download)
python -m scripts.cs2cd_diversity_probe                               # Step-0 gate → PLAYER_THIN verdict
python -m pipeline.pretraining.cs2cd_full --step all                  # full-release shards (478 legit) + manifest
python -m scripts.indomain_transfer --phase all --num-workers 6       # 6 encoders + transfer grid + figure
python -m scripts.domain_gap_report --reference cs2cd                 # → reports/pretraining_domain_gap_cs2cd_ref.json

# Phase 8.2 — contrastive pretraining (reuses the 8.1 shard cache + manifest; no re-download)
python -m scripts.contrastive_transfer --phase pretrain              # 4 contrastive encoders (cs2cd 50/200/382 + captcha)
python -m scripts.contrastive_transfer --phase eval                  # → reports/contrastive_transfer.json + figure
jupyter nbconvert --to notebook --execute --inplace notebooks/22_contrastive_pretraining.ipynb   # CPU-fast
```

`models/pretrained_encoder.pt` is DVC-tracked (`dvc pull` to fetch; `_meta.json` is git-tracked).
CaptchaSolve30k / CS2CD parquets live in `data/external/` (re-downloadable — see
[`data/external/README.md`](../data/external/README.md)). Seeds fixed (42).

See also: [REPORT.md §8](REPORT.md) · [SIGNALS.md](SIGNALS.md) · [FINDINGS.md](FINDINGS.md) ·
[ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) · [DATASET_CARDS.md](DATASET_CARDS.md).
