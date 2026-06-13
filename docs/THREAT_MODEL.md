# Threat Model — what input biometrics can and cannot see

BehaviorDNA observes exactly one layer: **operating-system mouse and keyboard
telemetry** (timestamped moves, clicks, key up/down). Every capability and every
blind spot below follows from that single sensing choice. The useful framing for
an anti-cheat team is not "what AUC does it hit" but *where this layer has signal,
where it is structurally blind, and how each signal is evaded* — input biometrics
is **one layer of a defence-in-depth stack**, not a complete anti-cheat.

## Scope of the sensor

- **Sees:** cursor trajectories and kinematics, click timing, key timing/rhythm,
  event rates. Normalised for sens/DPI + polling rate, these are comparable across
  hardware ([FEATURES.md](FEATURES.md)).
- **Cannot see:** game memory, rendered frames, network traffic, process/driver
  integrity, or any assist that never moves the OS cursor. No amount of modelling
  recovers a signal the sensor never captured.

## Threat table

| Threat | Observable at the OS-input layer? | Signal it leaves | How BehaviorDNA detects it (measured) | Evasion | Status |
|---|---|---|---|---|---|
| **Input macro** — scripted key/click sequences | Often | Timing periodicity; inhumanly regular intervals | Keystroke-periodicity + window timing features; LSTM-AE chunk error | chunk AUC ≈ **0.60** — the weakest of the three; macros perturb *shape* least | Implemented |
| **Triggerbot** — auto-fire on target | Sometimes | Click-reaction timing below the human floor; low channel error | LSTM-AE chunk reconstruction error | chunk AUC ≈ **0.93** — the strongest signal | Implemented |
| **Cursor-driving aimbot** — snaps/drives the OS cursor | Sometimes | Trajectory anomaly; non-human acceleration/curvature | LSTM-AE on raw event sequences | chunk AUC ≈ **0.79** — hand-crafted window features sit at **chance (~0.50)** here, which is *why* the sequence model exists | Implemented |
| **Memory-only aim assistance** — reads game state, never moves the OS cursor | **No** | None at the OS-input layer | — | **Inherent blind spot** — needs memory/integrity scanning, a different sensor entirely | Out of scope (by sensor) |
| **Account sharing / smurfing** — a different human plays the account | Potentially | Identity drift: kinematics stop matching the enrolled fingerprint | Player identification / open-set verification | impostor **EER 0.144** @ 10 users (Balabit); degrades with player count + thin enrollment | Implemented (identification track) |

> AUCs are **chunk-level on synthetic cheats injected into 18 real GTA sessions** —
> the *approach proof*, small-N. Read them as "the signal exists and separates",
> not as a production operating point ([ADVERSARIAL.md](ADVERSARIAL.md),
> [FINDINGS.md](FINDINGS.md)). The EER is from the public Balabit corpus
> ([VERIFICATION.md](VERIFICATION.md)).

## The evasion frontier

Every signal above is a *timing/shape* statistic, so every one has a humanisation
counter: randomised macro intervals, humanised trigger delay, trajectory
smoothing/easing. Detection is therefore not a fixed AUC but a
**detector-vs-evasion curve** — how fast AUC decays as the cheat is made more
human-like, and where the equilibrium sits. Quantifying that curve is the explicit
goal of [Phase 7](ROADMAP.md#phase-7--detection-vs-evasion-frontier); the
`pipeline/adversarial/live_cheat.py` humaniser already exposes the
easing/overshoot/jitter knobs a λ-strength sweep would turn.

## Why this is one layer, not the system

- The **memory-only** row is the load-bearing honesty: an OS-input sensor *cannot*
  see an assist that only reads memory and lets the human aim. Catching that needs
  memory/integrity scanning (kernel anti-cheat) — a different layer.
- Input biometrics' real strengths are (a) **passive identity** — account sharing,
  ban evasion, smurf detection — and (b) **cheap, ubiquitous coverage** of the
  cursor-driving and timing-automation classes, as *one* corroborating signal
  feeding a layered decision, **never** an autonomous ban. The false-positive cost
  of acting on a single behavioural signal is covered in [MODEL_CARD.md](../MODEL_CARD.md).

See also: [FINDINGS](FINDINGS.md) · [ADVERSARIAL](ADVERSARIAL.md) · [VERIFICATION](VERIFICATION.md) · [ETHICS](ETHICS.md) · [MODEL_CARD](../MODEL_CARD.md)
