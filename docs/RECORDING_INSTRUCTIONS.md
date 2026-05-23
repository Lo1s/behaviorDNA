# BehaviorDNA — Recording Instructions

Thank you for helping with this project! You only need about **30 minutes of gameplay** split across **5 short sessions**. Everything is recorded through a simple app — no technical setup required.

---

## What you need

- The **BehaviorDNA_Recorder.exe** file (provided by hydra)
- GTA V running in **offline / story mode** (no online, no anti-cheat risk)
- About 6 minutes per session

---

## Before you start — look up these values once

You will need to enter these in the app. Look them up before your first session so you have them ready.

### Mouse polling rate (Hz)

Your mouse's report rate. Check your mouse software:

| Software | Where to find it |
|---|---|
| Logitech G HUB | Select your mouse → Performance → Reports per second |
| Razer Synapse | Your mouse → Performance → Polling rate |
| SteelSeries GG | Your mouse → Settings → Polling rate |
| No software | Almost certainly **1000 Hz** for any gaming mouse |

### Screen resolution

Windows key → Settings → System → Display → **Display resolution**
(e.g. 1920×1080, 2560×1440, 3840×2160)

### Grip style

How you hold your mouse:

- **Palm** — your whole hand rests on the mouse, fingers lie mostly flat
- **Claw** — palm touches the back of the mouse, fingers are arched upward
- **Fingertip** — only your fingertips touch the mouse, your palm doesn't rest on it

---

## How to record a session

1. **Launch GTA V** and load into story mode
2. **Open BehaviorDNA_Recorder.exe**
3. Fill in all the fields (see below)
4. Click **▶ START RECORDING** — then **immediately switch back to GTA**
5. Play for **~6 minutes**
6. **Alt-Tab back** to the recorder and click **⏹ STOP RECORDING**
7. A confirmation dialog will appear — click OK
8. The session file is saved automatically in a `sessions/` folder next to the .exe

> **Important:** Keep GTA as the active / focused window while recording. If you tab out or open the Windows taskbar, those mouse movements will also be captured and will corrupt the data.

---

## What to fill in the app

| Field | What to enter |
|---|---|
| **Your name** | Your nickname — use the same one every session |
| **Game** | GTA5 |
| **Activity** | What you will be doing during this session (see schedule below) |
| **Mouse polling rate** | From your mouse software (see above) |
| **Screen resolution** | From Windows display settings (see above) |
| **Grip style** | How you hold your mouse (see above) |
| **Dominant hand** | Which hand you use for the mouse |
| **Warmed up?** | **Yes** if you've been playing for 15+ minutes already. **No** if this is your first session of the day. |
| **Sensitivity** | Your exact in-game mouse sensitivity number |
| **DPI** | Your mouse DPI (same place as polling rate in mouse software) |

> **Your name, grip style, dominant hand, DPI, sensitivity, polling rate, and resolution stay the same every session.** Only **activity** and **warmed up** change between sessions.

---

## Recording schedule — 5 sessions

Record one session per activity. Try to spread them across **at least 2 different days**.

| Session | Activity to select | What to do in GTA |
|---|---|---|
| 1 | `on_foot` | Walk and run around the city. Minimal combat — just explore on foot. |
| 2 | `driving` | Drive around. Mix city traffic and open highway. No combat. |
| 3 | `combat` | Get into a gunfight with NPCs. Lots of shooting, fast mouse movement. |
| 4 | `sniping` | Find a rooftop or hill. Use a sniper rifle — slow, precise aiming. |
| 5 | `free_roam` | Play however feels natural. Don't think about it. |

**Ideal schedule:**
- Day 1: sessions 1 + 2
- Day 2: sessions 3 + 4
- Day 3: session 5

If you can only do it in one sitting, that is also fine — just take a 5-minute break between sessions.

---

## Rules that affect data quality

**Do:**
- Play normally — don't try to move your mouse in a special way
- Keep GTA as the active window for the full 6 minutes
- Use the same sensitivity and DPI settings you always play with

**Don't:**
- Sit in menus, the pause screen, inventory, or cutscenes during recording — no mouse movement means wasted data
- Change your sensitivity or DPI between sessions
- Record while doing something else at the same time (watching a video, texting, etc.)

---

## Sending the files

After all 5 sessions you will have **5 files** in the `sessions/` folder next to the .exe. File names look like:

```
20260610T143022_yourname_gta5_a3f1b2c4.json
```

Send all 5 files to hydra (Discord, Google Drive, USB — whatever is easiest).

---

## Questions?

Contact hydra on Discord.
