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
