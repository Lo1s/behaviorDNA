# Collecting real cheat data (GTA5, offline)

> Proposal / runbook for capturing **real** cheat telemetry to replace the synthetic
> cheats. Motivated by the Phase 4.1 verification: synthetic *sparse* injection can't
> be lifted to a session-level risk (legit gameplay's natural high-error chunks are
> indistinguishable from a few injected cheat chunks). **Real continuous cheating**
> removes that artefact and is the unblock for session-level detection + honest
> within-session localization. See [STREAMING.md](STREAMING.md) → Phase 4.1.

---

## Ethics & safety — offline only (non-negotiable)

- Record in **GTA5 Story Mode / offline only**. No GTA Online, no FiveM, no public lobbies.
- Offline ⇒ **BattlEye/anti-cheat is not running** (no ban risk) and — the real point —
  **no other players are affected**. This is defensive ML research on your own machine,
  consistent with [docs/ETHICS.md](ETHICS.md): we study cheat *signatures* to detect them,
  we do not deploy cheats against anyone.
- Never run these mods online or against real players. The cheat artefacts stay in
  `data/` for training and are never used to gain an advantage over a human.

---

## The critical decision: which aimbot produces a detectable signal

Input-biometric detection only sees what the **OS input layer** sees — `pynput` in the
recorder captures mouse moves / clicks / key events. So the *mechanism* of the cheat
decides whether it's even visible:

| Cheat mechanism | Does the recorder see it? | Use as cheat data? |
|---|---|---|
| **Input-level aimbot** (driver-level, `SendInput`, or "move the cursor") | **Yes** — the superhuman snap is a real mouse movement | ✅ **record these** |
| **Internal / memory aimbot** (writes aim to game memory; no OS mouse motion) | **No** — telemetry looks fully human | ❌ not useful as positive data |
| **Triggerbot** (auto-fire when crosshair is on target) | **Yes** — superhuman click reaction/timing as click events | ✅ record |
| **Macros** (rapid-fire, recoil scripts, auto-walk/strafe) | **Yes** — keyboard/mouse input | ✅ record |

**So: pick mods that drive mouse/keyboard *input*, not pure memory aimbots.** Verify by
recording a 30 s test and confirming the JSON shows the expected signature (e.g. near-
instant large `dx/dy` snaps before a click, or sub-50 ms click reactions).

> **This is itself an anti-cheat finding worth documenting:** a memory-only aimbot that
> never touches the OS input layer is *invisible* to input biometrics — which is exactly
> why production anti-cheats also do memory/integrity scanning. Input biometrics is one
> layer, not the whole stack.

---

## Labeling — the gold standard: log the cheat toggle

The recorder already logs every `key_press`. So **bind the cheat on/off to a known
hotkey** and you recover *exactly* when the cheat was active from the recording itself —
no manual annotation:

- Note the toggle key (e.g. `F8`) in the session metadata.
- A small post-processing step turns the toggle-key timeline into `cheat_segments`
  (start/end ms pairs) in the session JSON — the **same schema the synthetic generator
  uses**, so the rest of the pipeline ingests it unchanged.
- This gives **per-chunk ground truth**, which is what enables honest *within-session*
  localization (the thing synthetic data could only fake).

Two complementary recording styles:
1. **Continuous-cheat sessions** — cheat on the whole time (combat/sniping). Most chunks
   are cheat-like ⇒ should finally separate at the **session** level (the Phase 4.1 unblock).
2. **Toggled sessions** — start legit, toggle cheat mid-session, toggle off. Gives a clean
   within-session label timeline for localization + a realistic "cheater who toggles".

---

## Recording protocol (suggested)

- **Same hardware / sens / DPI / polling rate** as the legit baseline. (We already learned
  the hardware-confound lesson with shotik's different DPI — keep the cheat signal from
  being confounded with a hardware change.)
- **Same player(s)** ideally → "this player cheating vs not" is the *pure* cheat signal,
  with no identity confound.
- Per cheat type (input-level aimbot / triggerbot / macro): ~5 continuous-cheat sessions in
  aim-relevant activities (`combat`, `sniping`) + matched legit sessions.
- Record metadata: `cheat_type`, `cheat_toggle_key`, `activity` (reuse the recorder's
  existing metadata fields; add the two cheat fields).

---

## The harness — `cheat_sim.py` (built)

Rather than download a real aimbot, generate the cheat **input signature** under full
control. The tooling is built and unit-tested:

- **`pipeline/adversarial/live_cheat.py`** — pure planners (shared difficulty presets with
  the synthetic generator): aimbot micro-correction snaps (eased; overshoot + jitter for the
  evasive *soft* case), sub-human triggerbot bursts, recoil/rapid-fire macros. Unit-tested.
- **`collector/cheat_sim.py`** — Windows-side `SendInput` actuator + hotkey loop. **No target
  acquisition, no memory reads, no networking** — the human does the coarse aim, the harness
  only performs the inhuman *final correction* / fire timing, so it can't function as a
  competitive cheat.
- **`scripts/label_cheat_segments.py`** — turns the in-band toggle keys into
  `cheat_label` + `cheat_segments`.

**Workflow (Windows host, offline Story Mode):**

```bash
# 1. start the recorder as usual (records mouse/keyboard + the F8/F9/F10 toggles in-band)
python recorder_gui.py          # or record_session.py ...

# 2. in parallel, arm the harness (offline confirmation required)
python cheat_sim.py --difficulty medium --i-am-offline
#    F8 aimbot · F9 triggerbot · F10 macro · F12 quit
#    aimbot: aim (right-click) → one superhuman correction snap
#    triggerbot: hold aim over target → sub-human auto-fire
#    macro: hold fire → periodic fire + recoil compensation

# 3. after recording, derive labels from the in-band toggles + strip control keys
python -m scripts.label_cheat_segments data/raw/<session>.json
```

Run **continuous-cheat** sessions (cheat on through combat) for the session-level signal,
and **toggled** sessions for within-session localization labels.

---

## Solo first-batch recording protocol & rules

The first batch is recorded by **one player (you)**. That shapes what's worth
capturing: cheat **detection** is a *within-player* problem (legit chunks vs
cheat chunks), so solo data is genuinely valuable for it — but player
*identification* needs more people, so don't expect this batch to move that.

### The one principle that matters most: change ONE thing at a time

The detector must learn **cheat vs legit**, not "session A vs B" or "combat vs
driving" or "warmed-up vs not". So **match conditions** between your legit and
cheat data — same hardware, sens/DPI, polling rate, activity, map area, warmup
state. The cleanest way to guarantee that is a **toggled session** (below):
legit and cheat chunks come from the *same* session, so the cheat is the only
variable.

### The single most important legit data: your *best, tryhard* aim

The detector's expensive failure is flagging a skilled human as a cheater. So
record some **maximum-effort legit combat** — your fastest flicks, quickest
reactions — with the cheat OFF. That elite-human play is the hard negative that
teaches the boundary between "very good human" and "superhuman cheat". Without
it, the model learns "fast = cheat".

### Suggested session matrix (~10–12 sessions, one sitting)

All combat/sniping (aim-cheats need targets — **don't** record aimbot/triggerbot
in driving/free-roam, there's no cheat signal there).

**A. Toggled sessions — highest value (controlled contrast + exact labels):**
play legit ~2 min → toggle cheat ON ~2 min → toggle OFF ~2 min, same fight.
- 2× combat, **aimbot** (F8), `--difficulty medium`
- 2× combat, **triggerbot** (F9), medium
- 1× combat, **macro** (F10), medium

**B. Continuous-cheat sessions — unblock session-level detection:**
cheat ON the whole session (this is what makes *most* chunks cheat-like, which
the sparse synthetic data couldn't).
- 1× combat aimbot (medium), 1× combat triggerbot (medium)
- difficulty spread: 1× aimbot `--difficulty obvious` (easy positive),
  1× aimbot `--difficulty soft` (the hard, evasive case)

**C. Matched legit — the false-positive boundary:**
- 2–3× combat **tryhard legit** (cheat sim not running), same gear/map as A/B.

### Rules / checklist

1. **`python cheat_sim.py --selftest`** first — confirm all rows PASS (the
   recording will capture the input). Then a 30 s pilot: record → run
   `label_cheat_segments` → eyeball that `cheat_segments` look right *before* the
   full batch.
2. **Same hardware all batch** — don't change sens/DPI/polling/resolution
   mid-session-set (avoids the confound we hit with different DPI).
3. **One difficulty per session** (it's a `cheat_sim` run-level flag); the
   `cheat_activity.jsonl` log records it.
4. **Play naturally** — real movement and aiming; let the bot do only the inhuman
   correction/fire. Don't stand still or spin aimlessly.
5. **Clean toggles** — tap F8/F9/F10 once to flip; leave a couple of seconds
   after toggling before/after the action (clean label boundaries). F12 quits.
6. **≥ ~6 min per session** (keeps it past the QC duration floor; more chunks).
7. **Consistent metadata** — same player name spelling, correct `activity`,
   `polling_rate`, sens, dpi in the recorder every time.
8. **Offline Story Mode only.**
9. **Label immediately after each session:**
   `python -m scripts.label_cheat_segments data/raw/<session>.json` and glance at
   the printed `cheat_label` / segment count.

### Expectation-setting (solo)

- **Detection benchmark + legit baseline:** strong gains. Real-cheat AUC replaces
  the synthetic caveat; continuous-cheat sessions let you re-attempt Phase 4.1.
- **Identification:** unchanged (still 3 players) — that waits for more people.
- A detector trained on *one* player's legit may over-fit your style (it could
  call a *different* human "anomalous"). That broadens as more players' legit
  data lands; fine for a first real-cheat benchmark.

---

## Pipeline integration (drop-in)

Real cheat sessions carry `cheat_label` + `cheat_segments` → identical schema to
`data/synthetic/` → `pipeline.adversarial.benchmark`, the LSTM-AE, and the streaming engine
ingest them **unchanged**. Then:

1. QC with `scripts/validate_recordings.py` (extend it to accept/validate the cheat fields).
2. Re-run `python -m pipeline.adversarial.benchmark` → chunk- **and** session-level AUC, now
   on real cheats (the headline number drops the "synthetic" caveat).
3. **Re-attempt Phase 4.1:** with continuous real cheating, session-level aggregation of the
   chunk signal should separate (most chunks elevated, not a sparse few) — build the live
   session-risk then, on data that supports it.
4. Label with `python -m scripts.label_cheat_segments <session>.json` (built) — derives
   `cheat_label` + `cheat_segments` from the in-band toggle keys and strips the control keys.

---

## What this unblocks

- **Phase 4.1** live session-risk (real continuous cheat → session separates; verified
  impossible on synthetic sparse injection).
- **Honest within-session localization** (toggle labels = per-chunk ground truth).
- A **real** adversarial benchmark — the cheat-detection results stop carrying the
  "synthetic cheat" caveat entirely.
