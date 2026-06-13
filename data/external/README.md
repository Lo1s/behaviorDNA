# data/external/ — public corpora layout

External datasets are **not** in git (only this README and `.gitkeep`s are; see
`.gitignore`). Download each corpus into its directory below, then the adapters
in `pipeline/external/` (Phase 6) and the notebooks can find them. Once a corpus
is in use, `dvc add` its directory so it's versioned like the rest of the data.

**Per-corpus audits** (what each supports, its limits, schema mapping, leakage,
domain shift): [docs/DATASET_CARDS.md](../../docs/DATASET_CARDS.md).

| Directory | Dataset | Used by | Where to get it |
|---|---|---|---|
| `balabit/` | **Balabit Mouse Dynamics Challenge** — 10 users, per-user session CSVs (`record timestamp, client timestamp, button, state, x, y`), plus test-set impostor labels (`is_illegal`) | Phase 6 — `pipeline/external/balabit.py`, notebook 19 (literature-comparable EER) | github.com/balabit/Mouse-Dynamics-Challenge (archived repo; clone/download the `training_files` + `test_files` trees here) |
| `sapimouse/` | **SapiMouse** — 120 users, 1-min + 3-min mouse sessions (`client timestamp, button, state, x, y`) | Phase 6 — `pipeline/external/sapimouse.py`, notebook 19 (the 120-user scale claim) | SapiMouse release page (Antal et al., Sapientia University — see the SapiMouse paper for the download link); unpack the per-user session CSVs here |
| `cs2cd/` | **CS2CD** — CS2 cheat detection dataset (10 players, real cheats); convention: `cs2cd_balanced_25000.parquet` | notebooks 16/17/18 (already in use) | per notebook 05 |
| `captchasolve30k/` | **CaptchaSolve30k** — ~20k human mouse sessions | notebook 05; Phase 8 pretraining corpus | per notebook 05 |
| `cs2_demo/` | **CS2 SourceTV demo** (`test_demo.dem`) — a public sample for the Phase 9 outcome-telemetry spike (extraction + clock-sync validation only; not a training corpus) | Phase 9 — `pipeline/outcome/cs2_demo.py`, `scripts/parse_cs2_demo.py` | `github.com/LaihoE/demoparser` → `src/parser/test_demo.dem` (the demoparser2 repo's own test demo); or any `.dem` from your own CS2 match (Steam → My Game Stats) |

**Licensing note:** all four are released for research use; check each source's
terms before redistributing anything derived from them. Don't commit raw corpus
files to git — this directory is gitignored for a reason (size + licence).

**Expected layout sanity check (Phase 6):** after downloading, the adapter
stubs' module docstrings (`pipeline/external/balabit.py`, `sapimouse.py`)
document the exact per-row CSV contract the parsers will be implemented
against — if the downloaded files don't match, update the docstring first,
then the parser.
