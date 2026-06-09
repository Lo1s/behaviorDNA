# Work review — everything since commit `4bfd6cc`

> Personal review aid (local-only, not committed — like `REVIEWER_NOTES.md`).
> Covers the 6 commits after `4bfd6cc` ("typed cheat labels"), oldest → newest.
> Generated 2026-06-08.

## TL;DR

Six increments, all pushed to **both** remotes (origin + dagshub), HEAD = `4098dab`:

| # | Commit | Date | One-liner |
|---|---|---|---|
| 1 | `27b0c1f` | 06-02 | Architecture comparison on **real recorded cheats** (LSTM/TCN/Transformer-AE) |
| 2 | `6ff0262` | 06-02 | Architecture comparison on **external CS2CD** dataset |
| 3 | `e3b086e` | 06-04 | **Recruiter-facing README** polish (breadth + honesty) |
| 4 | `d6eac4b` | 06-07 | **GPU-live notebooks 16 & 17** + notebook-07 bridge |
| 5 | `bf3f8a2` | 06-08 | **CS2CD signal-importance research** (nb 18) + promote 5 features |
| 6 | `4098dab` | 06-08 | **Fix**: exclude cheat sessions from the identification split |

**Through-line:** we now have a 3-setting cheat-detection story (synthetic GTA / real GTA / external CS2), an honest verdict that **architecture is not the bottleneck — data is**, a windowed signal-importance study that promoted 5 new features, and a correctness fix that keeps cheat play out of the identifier. Status: **332 tests pass**, ruff/black clean.

**Recurring honesty spine:** validate cheat features on CS2's *real* cheats (our GTA cheat data is `cheat_sim`-simulated); separate causal signal from collection artifacts; report neutral/negative results straight.

---

## 1. `27b0c1f` — Architecture comparison on real recorded cheats

**Why:** the LSTM/TCN/Transformer-AE comparison existed only on *synthetic* cheats; we now had 3 real `cheat_sim` recordings (typed segments) to test on.

**What:** extended `scripts/compare_architectures.py` with `--eval-data real` — same training loop + chunk-AUC metric, but the cheat eval set is the real labelled hydRa sessions (per-type via `cheat_segments_typed`). Outputs `reports/architecture_comparison_real.json` + figure.

**Key numbers (chunk ROC AUC, aimbot / triggerbot / macro):**
- LSTM-AE 0.525 / 0.603 / 0.566 · TCN-AE 0.512 / 0.590 / 0.559 · Transformer-AE 0.527 / 0.603 / 0.566
- All three **tied within ~0.015** — a *tighter* spread than synthetic, and far below the synthetic AUCs (0.73–0.96).

**Finding:** on real cheats the thesis holds *more* strongly — capacity isn't the bottleneck, data is. (Lower absolute AUC because the legit baseline includes the cheat sessions' own clean chunks → harder, more honest contrast.)

---

## 2. `6ff0262` — External CS2CD dataset AE comparison

**Why:** an *independent*, cross-game check — our cheat data is single-game/single-source (GTA, one player).

**What:** new `scripts/benchmark_cs2cd_ae.py` (+ `tests/test_benchmark_cs2cd.py`). Trains the 3 backbones on CS2CD's legit mouse stream (`dx, dy, fire, rightclick`), recovers contiguous same-label streams by grouping `(steamid, cheater_present)`, chunks to 64 ticks, scores cheat AUC. 10 players, 390 legit + 390 cheat chunks.

**Key numbers:** LSTM-AE **0.723** · TCN-AE 0.722 · Transformer-AE 0.722 — tied to **0.001**.

**Finding:** the reconstruction approach **transfers to a different game/engine** (~0.72), and the architecture tie holds across synthetic GTA, real GTA, and CS2 — the strongest evidence yet that data, not backbone, is the lever. Caveats documented (legit eval overlaps training → mildly optimistic; `cheater_present` is match-level/coarse).

---

## 3. `e3b086e` — Recruiter-facing README polish

**Why:** the README undersold the breadth (only showed synthetic results) ahead of sending to a recruiter.

**What:**
- Added a **"Highlights — what this demonstrates"** TL;DR block.
- Rebuilt **"Results at a glance"** to show cheat detection across **all three settings** (synthetic / external CS2 / own recorded cheats) + the "backbones tied → data is the limit" line.
- Added a **"Start here"** guided reading path.
- Clarified the hero caption (synthetic cheats on real legit) + data-status.
- Fixed **stale test counts** (234 / ~100 → 317) in `docs/STREAMING.md` + `CLAUDE.md`.

No code/logic changes.

---

## 4. `d6eac4b` — GPU-live notebooks 16 & 17 + notebook-07 bridge

The biggest increment. Three tutorial notebooks, all executed live on the RTX 3070 (seeded; AUCs may wobble ±0.01, ranking stable).

**`notebooks/16_architecture_comparison.ipynb` (new)** — sequence-model deep dive:
- 3 backbones × 3 settings, trained live. **Backbone spread: real-GTA 0.014, CS2 0.001** (synthetic's ~0.12 is just the cheap TCN trailing; LSTM ≈ Transformer).
- Beyond the bar chart: training curves, reconstruction-error distributions, bottleneck-embedding PCA, per-channel attribution.
- **Two experiments that turned hand-waves into measurements:**
  - *Split-sensitivity:* strict held-out-legit baseline barely moves AUC (**|Δ| ≈ 0.007**) → the modest real-GTA numbers are **not** a split artifact.
  - *Supervised lever:* a player-held-out supervised CS2 classifier (**0.714**) ≈ the unsupervised AE (**0.723**) → labels don't lift the ceiling at this data scale.

**`notebooks/17_identification_at_scale.ipynb` (new)** — identity at scale on CS2's 10 players:
- 10-player ID: shallow LightGBM **0.61** vs deep LSTM **0.44** (chance 0.10; shallow competitive at this N).
- **Standout finding:** identity is largely **erased by cheating** — legit ID accuracy 0.61 collapses to ~chance (**0.05**) on the same players' cheat windows.
- Accuracy-vs-roster-size scaling curve (2→10 players).

**`notebooks/07` Section F (added)** — pays off its own cliffhanger: per-tick `yaw` r ≈ 0.09 → **chunk-AUC ≈ 0.72** windowed; links to nb 16.

Plus `reports/figures/arch_comparison_synthesis.png` + README/CLAUDE pointers.

---

## 5. `bf3f8a2` — CS2CD signal-importance research + promote 5 features

**Why:** you asked for new fingerprinting/cheat features. The repo's own Phase-5d ablation says the GTA model is over-parameterised at N=18, so we validated on **CS2CD (real cheats, 10 players)** where we don't overfit — windowed importance, which nobody had done (nb 05/07 only ranked *per-tick*).

**`notebooks/18_signal_importance_cs2.ipynb` (new):**
- Per-tick vs windowed explainer; confound-aware importance (separates causal signal from collection artifacts).
- **Windowed behavioural cheat AUC 0.735** (player-held-out) — matches the AE.
- **Incremental value:** existing-analogs 0.709 → +new 0.736; **new-features-only 0.739 > existing** → the proposed features carry *more* signal. Top drivers: **`speed_p99`** (peak-flick) and **`fast_segment_straightness`**.
- **Artifact trap, live:** `ping` scores 0.83 cheat / 0.94 ID alone but is a per-session/per-player *network* confound (same lesson as nb07's rank/match-type).
- **Outcome features dead** on this sample (`damage = 0` across all 20 streams) → the strongest cheat signals (headshot ratio, dmg/shot, recoil) need combat-dense telemetry we don't have.

**Promoted 5 GTA-computable winners** to `FEATURE_COLS` (25 → 30): `speed_p50/p90/p99`, `fast_segment_straightness`, `click_reaction_p5`. Computed in `pipeline/features/run.py` (auto-synced through `process_session_windows` to streaming + benchmark); `api/main.py` `FeatureVector` schema + test fixtures updated to track `len(FEATURE_COLS)`; 5 unit tests added. Same-data GTA ID effect: 25-feat 0.600 → 30-feat 0.625 (neutral-to-better, within small-N noise).

**`docs/SIGNALS.md` (new):** the "what to monitor + how to capture it" research catalogue — behavioural / outcome / context / hardware / system signals, each tagged *helps ID? · helps cheat? · causal/artifact · do we collect it? · how to capture* — plus a prioritised **data-collection roadmap** and the architectural finding that ID and cheat detection should have **separate feature sets**.

---

## 6. `4098dab` — Fix: exclude cheat sessions from the identification split

**Why (bug found during #5):** hydRa's 3 cheat sessions were being ingested as ordinary `hydra` identification windows (hydra 214 windows vs dninix 108), which depressed the identification metric to ~0.60 — literally nb17's "identity erased by cheating" leaking into the ID model.

**What:**
- `pipeline/ingestion/run.py`: flag `is_cheat_session` per session (typed/untyped cheat spans or non-legit `cheat_label`; legit + old/mock → False).
- `pipeline/features/run.py`: carry the flag into `features.parquet` (defaults False for old frames).
- `pipeline/features/split.py`: **exclude cheat sessions** from train/val/test (guarded for backward compatibility). Cheat sessions remain available to the benchmark path (reads `data/raw` directly).
- 8 new tests (flag logic + split exclusion + backward-compat).

**Proof:** `dvc repro` now logs `Excluding 3 cheat session(s) / 142 window(s)` and **identification recovered 0.60 → 0.765 acc / 0.780 F1** (legit-only, 30 features). No-op on the published legit-only data.

---

## Key findings rollup (for the review)

1. **Architecture is not the bottleneck — data is.** LSTM ≈ TCN ≈ Transformer-AE across synthetic GTA, real GTA, and external CS2 (backbone spread ≤ 0.014 on real/CS2). More params buy nothing consistent.
2. **The approach transfers across games** (CS2 ~0.72 on data we didn't create).
3. **The modest real-data AUCs are honest, not artifacts** — split-sensitivity Δ ≈ 0.007, and supervised ≈ unsupervised. The ceiling is the behavioural signal at this data scale.
4. **The strongest cheat signals are non-behavioural** (outcome/performance, system/process) and we can't measure them yet → that's the concrete data-collection roadmap.
5. **Cheating erases the behavioural fingerprint** (ID 0.61 → ~0.05 on cheat windows) — a real anti-cheat-relevant result, and the reason cheat sessions don't belong in the identifier.

---

## Open decisions (need your call — not code)

1. **`dvc push`** the local cheat recordings (~70 MB, 3 hydRa sessions + `cheat_activity.jsonl`). They were recorded locally (Jun 1) and **never dvc-pushed** — DagsHub still has the legit-only 35-file dataset. Until pushed, the published pipeline stays legit-only.
2. **README `0.853` reconciliation.** That number is a pre-cheat-data snapshot at 25 features; the current legit-only number at 30 features is ~**0.765**. If you push the data, we should reconcile the headline (the same-hardware 0.75 framing already in the README handles the honest story).
3. **Decouple cheat-detection features from identification** (next increment). `FEATURE_COLS` is shared by the identifier and the cheat detectors, so cheat features trade against ID at small N. A separate cheat-feature set / model head is the principled fix.

---

## Repo / data state notes (so nothing surprises you)

- **Source-only commits for #5 and #6.** I committed code/docs/notebooks/tests but **reverted the local pipeline outputs** (`dvc.lock`, `reports/*`, `data/raw.dvc`) because they reference your un-pushed local cheat data. CI's `dvc-repro` job regenerates the canonical pipeline on the clean published data and `dvc push`es it.
- Your **local working tree** therefore shows `dvc status` drift (local model = 30-feature legit-only; committed lock = older). Running `dvc repro` locally rebuilds it; nothing is broken.
- I installed **`black[jupyter]` locally only** as a formatting aid — it is **not** in `requirements.txt`, and CI's black skips notebooks as before (ruff is the notebook gate, and all notebooks pass it).
- New files this span: `notebooks/16,17,18`, `scripts/benchmark_cs2cd_ae.py`, `docs/SIGNALS.md`, `tests/test_benchmark_cs2cd.py`, several `reports/figures/*` + `reports/architecture_comparison_{real,cs2cd}.json`.

## Suggested review order
1. `README.md` (top screen) → 2. `docs/SIGNALS.md` → 3. `notebooks/18` (signal research) → 4. `notebooks/16` (architecture + the two experiments) → 5. `notebooks/17` (ID at scale + identity-under-cheating) → 6. the `4098dab` diff (cheat-exclusion fix). `docs/ARCHITECTURE_COMPARISON.md` holds the canonical committed numbers.
