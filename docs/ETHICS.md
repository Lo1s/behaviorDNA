# Ethics & Safety

## How data is collected

BehaviorDNA records mouse and keyboard events using the Windows OS input API
(`SetWindowsHookEx` via `pynput`) — the same mechanism used by accessibility
tools, screen recorders, and peripheral software like Logitech G HUB or Razer
Synapse. It does **not**:

- read or modify game memory
- inject code into any process
- intercept or modify network traffic
- bypass, interfere with, or circumvent any anti-cheat system
- modify any game files

All data is collected at the OS level, entirely outside the game process.

---

## Consent

All session data in this project was collected with the **explicit consent** of
the participants. No data was collected covertly or without the player's
knowledge. Each participant:

- ran the recorder themselves on their own machine
- was informed of what data is captured (mouse/keyboard events only)
- was told how the data would be used (ML research / portfolio project)

---

## Anti-cheat compatibility

Because the recorder uses a global OS input hook, it is worth being transparent
about compatibility with each game's anti-cheat:

| Game | Anti-cheat | Assessment |
|---|---|---|
| CS2 | VAC | Safe — VAC targets memory cheats, not input hooks |
| Tarkov | BattlEye | Safe — BattlEye focuses on process/memory injection |
| Valorant | Vanguard (kernel) | Low risk but not zero — Vanguard is kernel-level and aggressive; tested on secondary accounts first |
| Arc Raiders | TBD | **Avoid** — anti-cheat not fully documented, but AutoHotkey (which uses the same global OS input hook mechanism as this recorder) is known to be blocked; do not use the recorder with Arc Raiders until confirmed safe |

No bans or flags were received during data collection for this project.
Participants using Valorant were advised to test on a secondary account first.

The recorder is **always stopped before launching** any game with a kernel-level
anti-cheat (e.g. Vanguard), and **never run concurrently** with such games in
production data collection.

---

## Data storage & privacy

- Session files contain **no personally identifiable information** beyond the
  player name/alias chosen by the participant themselves
- No IP addresses, hardware IDs, or account identifiers are recorded
- All raw session files are stored locally and shared only via a private DVC
  remote with access limited to project contributors
- Participants can request deletion of their sessions at any time

---

## Intended use

This project is a **research and portfolio demonstration**. The techniques shown
(behavioral fingerprinting, anomaly detection from input telemetry) are intended
to illustrate how such systems can be built ethically — with consent, at the OS
level, and without interfering with any game or platform.

This codebase is not intended for use in production anti-cheat systems,
surveillance, or any application where data is collected without consent.
