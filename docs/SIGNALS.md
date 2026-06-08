# Signal & telemetry research — what to monitor for fingerprinting + cheat detection

> Research output of [notebooks/18_signal_importance_cs2.ipynb](../notebooks/18_signal_importance_cs2.ipynb)
> (windowed importance on the external **CS2CD** dataset), building on the per-tick studies in
> [notebooks/05](../notebooks/05_external_datasets.ipynb) and [notebooks/07](../notebooks/07_behavioral_differentiation.ipynb).
> The question: *which signals — behavioural and non-behavioural — actually help identify players
> and detect cheats, and what do we need to **monitor** to obtain them?*

## Three guard-rails

1. **Per-tick vs windowed.** A single instant is weak (CS2 `yaw` rank-biserial r ≈ 0.09); the signal
   lives in **windowed temporal patterns** (percentiles, periodicity, geometry, sequence shape). The
   same CS2 stream that is near-chance per tick reaches ~0.72 chunk-AUC once windowed (nb16).
2. **Causal vs collection-artifact.** A feature can score high because of *how the data was collected*,
   not because it detects cheating. On CS2CD, `ping`/`rank`/`match_making_mode` "predict" cheating at
   r ≈ 0.8–1.0 — but they are per-session/per-match confounds, **not** cheat signals. Promotion is
   gated on *causal* incremental value, not raw importance.
3. **Simulated vs real cheats.** Our GTA cheat recordings are `cheat_sim`-injected (simulated), so the
   **primary** evidence for cheat-detection features is CS2CD's **real** third-party cheats.

## Signal catalogue

| Signal | helps ID? | helps cheat? | causal / artifact | do we collect it? | how to capture it |
|---|---|---|---|---|---|
| **Mouse-speed percentiles** (`speed_p50/p90/p99`) | weak | **yes** (p99 = peak-flick) | causal | ✅ raw input | **promoted** → `pipeline/features/run.py` |
| **`fast_segment_straightness`** | — | **yes** (aimbot snap) | causal | ✅ raw input | **promoted** |
| **`click_reaction_p5`** | — | **yes** (triggerbot) | causal | ✅ raw input | **promoted** (percentile principle) |
| View-angle aim dynamics (ang. velocity p99, flicks) | weak | **yes** (~0.70 alone) | causal | ⚠️ CS2 has view-angles; GTA only mouse Δ | GTA analog = `speed_p99` + curvature; native needs game view-angle hook |
| Distribution percentiles for ID (speed/curvature) | **yes** | — | causal | ✅ raw input | promoted (speed); extend to curvature/reaction when data grows |
| Handedness / L-R asymmetry, pause signature | **yes** | weak | causal | ✅ raw input | feature ideas — promote when ID is decoupled from cheat (see below) |
| **Outcome / performance** (headshot ratio, dmg/shot, recoil, accuracy) | — | **strong, causal** | causal | ❌ | **needs combat-dense, outcome-labelled telemetry** — game-event hook or demo/log parser + labels |
| Game-state context (weapon, health, velocity, scoped) | context | enables conditioning | causal (as conditioner) | ❌ (GTA) / ✅ (CS2CD) | recorder hook into game state; lets us condition behaviour (reaction *in combat*) |
| **Hardware / setup** (`sensitivity`, `dpi`, `polling_rate`, `resolution`, `grip_style`, `dominant_hand`) | **yes** (cross-setup) | tells (impossible rates) | causal-ish | ✅ **recorded, unused** | already in `sessions.parquet` — wire as features for *cross-setup* ID (exclude for same-hardware biometric) |
| Network dynamics (within-session ping/jitter) | weak | lag-switch | causal | ❌ (only session-mean) | per-tick network telemetry |
| **System / process** (injected modules, handles, frame-time anomalies) | — | **strong** (kernel AC) | causal | ❌ | a separate **agent** (à la BattlEye/EAC) |
| `ping` (session-mean), `rank`, `match_making_mode`, warmup/round meta | (spurious) | (spurious) | **artifact** | ✅ (CS2CD) | **exclude** — high score, non-causal (collection confound) |

## Promoted this round (CS2CD-validated, GTA-computable)

`speed_p50/p90/p99`, `fast_segment_straightness`, `click_reaction_p5` — added to `FEATURE_COLS`
(now 30). Evidence (nb18, player-held-out on CS2 real cheats): **new-features-only cheat AUC 0.74 >
existing-analog 0.71**, with `speed_p99` and `fast_segment_straightness` the top behavioural drivers.
GTA effect is neutral-to-slightly-better (0.600 → 0.625 acc) and within small-N noise — see the
"shared feature set" caveat below.

## Needs-data — the collection roadmap (prioritised)

1. **Combat-dense, outcome-labelled telemetry** (highest payoff). Outcome stats (headshot ratio,
   damage/shot, recoil control) are the *causally strongest* cheat signals but are **untestable on
   what we have** — the CS2CD sample is action-sparse (`damage_total = 0` across all 20 streams).
   *Capture:* a game-event hook or demo/log parser that records kills/damage/hits/shots per window,
   plus ground-truth cheat labels. *Unlocks:* supervised outcome-based detection.
2. **Cross-player real cheat recordings** (unblocks the supervised lever — nb16 D2 — and Phase 4.1).
   See [docs/CHEAT_DATA_COLLECTION.md](CHEAT_DATA_COLLECTION.md).
3. **System/process agent** (the kernel-anti-cheat surface): injected modules, suspicious handles,
   frame-time anomalies. Non-behavioural, central to real cheat detection, needs a dedicated agent.
4. **Per-tick network telemetry** (lag-switch detection) — within-session ping/jitter, not the
   session-mean confound.
5. **Pretraining corpus:** [CaptchaSolve30k](../notebooks/05_external_datasets.ipynb) (20k human mouse
   sessions, already cached) to pretrain the AE's human-motion manifold, then transfer to GTA.

## Architectural finding — decouple identification from cheat detection

`FEATURE_COLS` is shared by the **player identifier** (training/eval) and the **cheat detectors**
(streaming anomaly models, benchmark). At N=18 GTA sessions this forces a trade-off: cheat-oriented
features add little to identification (and the model is already over-parameterised — Phase 5d). The
clean fix is a **separate cheat-detection feature set / model head** so cheat features can be added
without touching the identifier. Tracked in [docs/ROADMAP.md](ROADMAP.md) Phase 1.5.

## Data-hygiene note (discovered here)

`data/processed/features.parquet` currently includes hydRa's 3 **cheat** sessions as ordinary
`hydra` identification windows (hydra = 214 windows vs dninix = 108). Per nb17's "identity is erased
by cheating" finding, these depress the *identification* metric (current ≈ 0.60 vs the README's
**0.853**, which is a pre-cheat-data snapshot at n_test = 34). The identification split should likely
**exclude cheat sessions** (you fingerprint players from legit play). Flagged as a follow-up — not
changed here, since it touches the headline biometric number.
