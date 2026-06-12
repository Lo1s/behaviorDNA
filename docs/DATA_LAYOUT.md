# `data/raw/` recording layout

`data/raw/` is a **DVC-managed directory** (`data/raw.dvc`; pull it with `dvc pull`). It is
purely DVC-tracked — git tracks nothing inside it (that's required: a whole-dir DVC output
can't contain SCM-tracked files, or `dvc add/commit` and CI's `dvc pull`/`repro` error with
*"output already tracked by SCM"*). **Cheat and legit recordings are separated by folder
*and* by an in-file flag.**

```
data/raw/                 (dvc pull to populate)
  *.json          ← LEGIT recordings — the active identification dataset (18 GTA sessions)
  cheat/          ← REAL cheat recordings (cheat_sim-injected) + cheat_activity.jsonl
  mock/           ← old mock/desktop batch (excluded)
  real_data/      ← earlier real batch (excluded)
```

## Why top-level = legit

Most consumers scan **`data/raw/*.json` non-recursively**, so anything in a subfolder is
excluded by default. Keeping legit recordings at the top level means the identification and
legit-manifold models train/evaluate on legit play only, automatically:

| Consumer | Reads | Effect of this layout |
|---|---|---|
| `pipeline/ingestion/run.py` → features → split → train (LightGBM identifier) | top-level `*.json` | identification = 18 legit sessions |
| `scripts/train_lstm_ae.py` (LSTM-AE) | top-level `*.json` | legit-manifold AE trains on legit only |
| `scripts/compare_architectures.py` `_load_legit_tensors` (LSTM/TCN/Transformer-AE) | top-level `*.json` | sequence AEs train on legit only |
| `scripts/validate_recordings.py`, `dashboard/app.py` | top-level `*.json` | operate on legit |
| `pipeline/adversarial/generate_dataset.py` | top-level `*.json` | injects synthetic cheats into **legit** only (never re-cheats the real cheat sessions) |

The **cheat-detection** evaluators additionally scan `cheat/`:

| Consumer | Reads | Purpose |
|---|---|---|
| `scripts/compare_architectures.py --eval-data real` | top-level + `cheat/` | legit baseline + real cheat chunks |
| `pipeline/adversarial/benchmark.py` (`--lstm-chunk-only --data-dir data/raw`) | top-level + `cheat/` | per-cheat-type chunk-AUC on real cheats |

## Belt-and-suspenders: the `is_cheat_session` flag

Folder placement is the primary separation, but a session is *also* identified as cheat by
`pipeline.ingestion.run._is_cheat_session` (typed/untyped cheat spans, or a non-`legit`
`cheat_label`). The identification split (`pipeline/features/split.py`) and the sequence-AE
loaders also exclude flagged cheat sessions, so a cheat recording accidentally left at the top
level is still kept out of the legit models. **New cheat recordings belong in `data/raw/cheat/`.**

## DVC workflow

After adding/moving recordings: `dvc commit data/raw && dvc push` (versions + publishes the new
state to the DagsHub remote, cheat recordings included). Don't `dvc checkout data/raw` against an
older `data/raw.dvc` if it would drop un-pushed recordings. Because `data/raw` is a whole-dir DVC
output, **don't git-track files inside it** (no `.gitkeep`/README there) — keep layout docs here
in `docs/` instead.
