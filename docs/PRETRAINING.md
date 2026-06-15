# Self-supervised pretraining (Phase 8 / 8.1)

> *A small "foundation model for human input motion" — and an honest measurement of why it doesn't
> (yet) transfer to game-input biometrics.*
>
> Code: [`pipeline/pretraining/`](../pipeline/pretraining) · scripts `pretrain_encoder` /
> `domain_gap_report` / `data_efficiency` · notebook
> [21](../notebooks/21_pretraining.ipynb) · tests `tests/test_pretraining.py`.

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
   magnitude (see caveat 1); a contrastive prior would be less magnitude-bound.

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
```

`models/pretrained_encoder.pt` is DVC-tracked (`dvc pull` to fetch; `_meta.json` is git-tracked).
CaptchaSolve30k / CS2CD parquets live in `data/external/` (re-downloadable — see
[`data/external/README.md`](../data/external/README.md)). Seeds fixed (42).

See also: [REPORT.md §8](REPORT.md) · [SIGNALS.md](SIGNALS.md) · [FINDINGS.md](FINDINGS.md) ·
[ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) · [DATASET_CARDS.md](DATASET_CARDS.md).
