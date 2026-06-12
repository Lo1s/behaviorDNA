# `data/raw/` — recording layout

Raw recorder session JSONs (DVC-managed; gitignored except this README + `.gitkeep`s).
**Cheat and legit recordings are separated by folder *and* by an in-file flag.**

```
data/raw/
  *.json          ← LEGIT recordings — the active identification dataset (18 GTA sessions)
  cheat/          ← REAL cheat recordings (cheat_sim-injected) + cheat_activity.jsonl
  mock/           ← old mock/desktop batch (excluded)
  real_data/      ← earlier real batch (excluded)
  README.md, .gitkeep
```

## Why top-level = legit

Most consumers scan **`data/raw/*.json` non-recursively**, so anything in a subfolder is
excluded by default. Keeping legit recordings at the top level means the identification
pipeline trains/evaluates on legit play only, automatically:

| Consumer | Reads | Effect of this layout |
|---|---|---|
| `pipeline/ingestion/run.py` → features → split → train | top-level `*.json` | identification = 18 legit sessions |
| `scripts/train_lstm_ae.py` | top-level `*.json` | legit-manifold AE trains on legit only |
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
`cheat_label`). The identification split (`pipeline/features/split.py`) and the LSTM-AE loader
both exclude flagged cheat sessions, so a cheat recording accidentally left at the top level is
still kept out of the legit models. A cheat recording belongs in `cheat/`.

## A note on DVC

`data/raw` is a DVC-tracked directory. After adding/moving recordings, run
`dvc commit data/raw && dvc push` to version + publish the new state (the `cheat/` recordings
are not pushed by default). Don't `dvc checkout data/raw` against an older `data/raw.dvc` if it
would drop un-pushed recordings.
