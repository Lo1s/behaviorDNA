# Dataset Cards — public corpora

Short, honest audits of the three public corpora this project uses, so a reader
can judge *what each one supports and what it can't*. Raw corpus files are not in
git (size + licence — see [`data/external/README.md`](../data/external/README.md));
these cards describe what the adapters/notebooks actually consume.

All three are released **for research use** — verify each source's terms before
redistributing anything derived from them.

---

## Balabit Mouse Dynamics Challenge

| | |
|---|---|
| **Source / retrieval** | `github.com/balabit/Mouse-Dynamics-Challenge` (archived). Download the `training_files/` + `test_files/` trees + `public_labels.csv` into `data/external/balabit/`. |
| **Unit of observation** | One **mouse event** row: `record-ts, client-ts, button, state ∈ {Move,Drag,Pressed,Released}, x, y`. Sessions are per-user CSVs. |
| **Counts** | **10 users**; training sessions (legit) + test sessions; **784 labelled** test sessions used for verification. Mouse-only — **no keyboard**. |
| **Labels** | Test sessions carry `is_illegal` (1 = impostor session claimed to be user *N*). Training sessions are genuine. |
| **Schema mapping → BehaviorDNA** | Clean: the adapter ([`pipeline/external/balabit.py`](../pipeline/external/balabit.py)) emits recorder-schema JSON, so the normal ingestion→features pipeline runs. **Only the 17 mouse-only features** (`MOUSE_ID_FEATURE_COLS`, a subset of the local 25) apply; keyboard/reaction features are absent. |
| **Data quality** | Contains `(65535, 65535)` uint16 sentinel-glitch rows (~1/session) — filtered by a coordinate-sanity ceiling in the adapter. |
| **Leakage risk / split unit** | Split **by session** (a user's sessions must not straddle folds); the challenge ships a train/test division by design. |
| **Collection confounds** | Per-user capture apparatus is uncontrolled/undocumented, so hardware could partially leak identity — standard caveat for this corpus, shared by the literature we compare against. |
| **Domain shift vs local data** | Large but **not separately quantified**: general desktop mouse activity vs in-game GTA aiming. Treat absolute numbers as literature-comparable, not transferable. |
| **Suitable for** | Literature-comparable **mouse-only identification + impostor verification (EER)** at small N. |
| **Not suitable for** | Keyboard signals, cheat detection, or claims about in-game behaviour. |
| **Used by** | Phase 6 — notebook 19; results in [VERIFICATION.md](VERIFICATION.md) (closed-set ≈ 0.59 acc, impostor **EER 0.144**). |

---

## SapiMouse

| | |
|---|---|
| **Source / retrieval** | `ms.sapientia.ro` (Antal et al., Sapientia University — see the SapiMouse paper). Unpack per-user session CSVs into `data/external/sapimouse/`. |
| **Unit of observation** | One **mouse event** row: `client-ts, button, state, x, y`. |
| **Counts** | **120 users**, exactly **one 1-min + one 3-min** session each (2 sessions/user — deliberately thin). Mouse-only. |
| **Labels** | Player identity (the user). No cheat labels. A `protocol` field tags each session `1min`/`3min`. |
| **Schema mapping → BehaviorDNA** | Same mouse-only adapter path as Balabit ([`pipeline/external/sapimouse.py`](../pipeline/external/sapimouse.py)); 17 mouse features apply. |
| **Leakage risk / split unit** | **Session-disjoint by construction** via the paper's own protocol: train on the 3-min session, test on the 1-min session. Only 2 sessions/user, so enrollment is thin. |
| **Collection confounds** | Per-user hardware undocumented; short, prompted tasks (not free gameplay). |
| **Domain shift vs local data** | Large; short controlled mouse tasks vs GTA gameplay. Not separately quantified. |
| **Suitable for** | A **scale stress-test** — does the windowed identifier hold at 100+ users? |
| **Not suitable for** | Anything needing many windows/user (it is **data-starved**: ~6 train windows/user → open-set rejection ≈ chance), or cheat detection. |
| **Used by** | Phase 6 — notebook 19; 120-user acc ≈ 0.11 (chance 0.008, ~13× chance) but open-set ≈ chance → the motivation for self-supervised pretraining (Phase 8). |

---

## CS2CD (Counter-Strike 2 cheat detection)

| | |
|---|---|
| **Source / retrieval** | CS2 cheat-detection dataset (retrieval per notebook 05). The repo uses a **50,000-row balanced sample** of the much larger public release (~735 M ticks); stored as `data/external/cs2cd/cs2cd_balanced_25000.parquet`. |
| **Unit of observation** | **One game tick** — *not* a mouse event. **226 columns** of server-side game state: `aim_punch_angle(_vel)`, recoil indices, view/round/inventory state, `usercmd_input_history`, etc. |
| **Counts** | 50,000 ticks, **balanced** `cheater_present` 25k/25k. |
| **Labels** | `cheater_present` ∈ {0,1} per tick (balanced). It is **cheat-labelled, not player-labelled**. |
| **Schema mapping → BehaviorDNA** | **Fundamentally different sensor.** CS2CD exposes game-state / view-angle telemetry, *not* OS cursor `x/y`, so the local collector's mouse-kinematic features **do not transfer directly**. The sequence models consume CS2CD's own derived signals; bridging to the local schema needs a deliberately-shared subset or an explicit adapter (not yet built). |
| **Data quality / continuity** | The **balanced 50k sample breaks temporal continuity** of the full release — ticks are sampled for class balance, not contiguous gameplay. The full-release continuity/match structure is lost in this sample. |
| **Leakage risk / split unit** | Ticks within a round/match are highly correlated → split by **match (and player) **, never by random tick. The balanced sample does not preserve match grouping, so it supports signal-importance / approach-proof work better than a clean generalisation estimate. |
| **Collection confounds** | Server-side game state (demo-parsed), a different layer from client OS input entirely. |
| **Domain shift vs local data** | **Maximal** — different game, different sensor layer (game state vs OS input). A "different game" generalisation datapoint, not a same-sensor transfer. |
| **Suitable for** | Within-CS2CD **cheat detection** (LSTM-AE chunk AUC ≈ 0.72) and **signal-importance** analysis (notebook 18); a second-game generalisation datapoint. |
| **Not suitable for** | Cross-corpus *feature* transfer without an adapter; player identification; or reading the balanced-sample AUC as a continuity-preserving production estimate. |
| **Used by** | Notebooks 16/17/18; [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md), [SIGNALS.md](SIGNALS.md). |

---

> A fourth corpus, **CaptchaSolve30k** (~20k human mouse sessions), is staged for
> Phase 8 self-supervised pretraining and used exploratorily in notebook 05; a
> card will be added when it moves into a measured experiment.

See also: [VERIFICATION.md](VERIFICATION.md) · [SIGNALS.md](SIGNALS.md) · [FINDINGS.md](FINDINGS.md) · [`data/external/README.md`](../data/external/README.md)
