# Verification & identification at scale — public-corpus results (Phase 6)

> Produced by [`scripts/run_external_identification.py`](../scripts/run_external_identification.py)
> → [`reports/external_identification.json`](../reports/external_identification.json)
> (seed 42, fully reproducible; the README results block regenerates from the JSON).
> Walkthrough notebook: [notebooks/19](../notebooks/19_identification_at_scale_public.ipynb).

## The reframe

Closed-set *"which of 3 players is this?"* is not the industry problem.
**Verification** is: *"is this account being played by the person who usually
plays it?"* — account sharing, smurfing, boosting. Same models, harder and more
honest protocol (EER over genuine/impostor trials, open-set rejection of users
never enrolled). It also generalises beyond games: continuous authentication and
web fraud/bot detection are the same problem with a different skin.

This phase runs the **exact GTA pipeline** — same windowing, same feature code,
same model family — on two public mouse-dynamics corpora, answering the killer
question (*does it survive beyond 3 friends?*) with no new data collection.

**Mouse-only caveat:** both corpora have no keyboard channel, so models use
`MOUSE_ID_FEATURE_COLS` (17 features; keyboard features *excluded*, not
zero-filled). GTA numbers are therefore not directly comparable — the GTA
fingerprint is partly keyboard timing (SHAP, notebook 12).

## Results

### Balabit (10 users — the literature benchmark)

Hours of desktop activity per user; sessions segmented at idle gaps before
windowing (`split_on_idle` — desktop captures are not continuous gameplay).

| Task | Result |
|---|---|
| Closed-set ID, session-held-out | **0.59** acc (95% CI 0.57–0.62; chance 0.10) — 9,710 windows |
| **Impostor detection (the challenge's own task)** | **EER 0.144** over 784 labelled test sessions (395 genuine / 389 impostor), scoring each session as mean P(claimed user) |

The EER is the headline: with *generic* windowed features + LightGBM — no
corpus-specific tuning — the same pipeline that fingerprints GTA players
detects Balabit's real impostor sessions at 14.4% EER. Challenge-era dedicated
methods report roughly 7–25% depending on data volume per decision, so this
sits squarely in the credible range for a transfer of an existing pipeline.

### SapiMouse (120 users — the scale stress-test)

Paper protocol: train on each user's 3-minute session, test on their 1-minute
session. That is **~6 training windows per user** — a deliberate stress test of
how far 30-second aggregate windows stretch on seconds of data.

| Enrolled users | Accuracy (mean over 5 user-draws) | Chance | ×chance |
|---|---|---|---|
| 3 | 0.68 | 0.333 | 2× |
| 10 | 0.57 | 0.100 | 6× |
| 30 | 0.36 | 0.033 | 11× |
| 60 | 0.31 | 0.017 | 19× |
| 120 | 0.11 (CI 0.08–0.14) | 0.008 | **13×** |

Window-level verification EER at 120 users: **0.38**. Open-set (60 enrolled /
60 unknown): **EER ≈ 0.48 — chance**; FAR@FRR≤5% ≈ 0.93.

### What this says (the honest read)

1. **The signal survives scale.** Accuracy stays 10–20× chance all the way to
   120 users — the behavioural fingerprint is real and the method doesn't
   collapse beyond 3 friends.
2. **Absolute performance is data-bound, not method-bound at the small end.**
   Balabit (hours/user) → usable EER; SapiMouse (minutes/user) → far from
   deployable. With ~6 windows per user, 30 s aggregate features are starved.
3. **Open-set is the hard frontier.** Max-probability rejection over 60 unknown
   users is chance-level at this data volume — closed-set softmax confidence is
   not an identity score. This is precisely the gap **Phase 8 pretraining**
   targets (transfer a human-motion prior so per-user data goes further), and
   it motivates embedding/metric-learning approaches over classifier confidence.

## Protocol notes

- Idle-gap segmentation: gaps > 10 s split a session; segments ≥ 30 s kept.
- Windows: the standard 30 s non-overlapping pipeline windows.
- Balabit verification scores: mean over a test session's windows of the
  claimed user's probability; 32 labelled sessions skipped (too short to
  produce a window).
- LightGBM: `num_leaves=31, n_estimators=200, min_child_samples=3`,
  class-balanced; StandardScaler fit on train only. Single seed (42).
- Metrics code: [`pipeline/verification.py`](../pipeline/verification.py)
  (EER / DET / FAR@FRR / closed-set→verification trial conversion; unit-tested).
