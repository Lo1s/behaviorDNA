"""
collector/cheat_sim.py
======================
Live cheat-signature harness — Windows-side, **offline GTA5 only**.

Runs *alongside* the recorder and injects superhuman input on toggle hotkeys so
the recording captures real, continuous, labelled cheat telemetry (the unblock
for session-level detection — see ``docs/CHEAT_DATA_COLLECTION.md``). The cheat
*logic* lives in the pure, unit-tested ``pipeline.adversarial.live_cheat``; this
file is only the Windows actuator (``SendInput``) + hotkey loop.

╔══════════════════════════════════════════════════════════════════════════╗
║  SAFETY / ETHICS — read this.                                             ║
║  • OFFLINE, single-player GTA5 (Story Mode) ONLY. Never online / FiveM.   ║
║  • This is defensive research: we generate cheat *signatures* to train a  ║
║    DETECTOR. It does NOT find or aim at enemies — no target acquisition,  ║
║    no memory reads, no networking. The human does the coarse aim; the     ║
║    harness only performs the inhuman *final correction* / fire timing.    ║
║  • It therefore cannot function as a competitive cheat.                   ║
╚══════════════════════════════════════════════════════════════════════════╝

Hotkeys (also captured in-band by the recorder → exact labels via
``scripts/label_cheat_segments.py``):
    F8  toggle AIMBOT     (then aim-key press → one superhuman correction snap)
    F9  toggle TRIGGERBOT (then hold aim-key over target → sub-human auto-fire)
    F10 toggle MACRO      (then hold fire-key → periodic fire + recoil comp)
    F12 quit

Usage (on the Windows host, offline):
    python cheat_sim.py --difficulty medium --i-am-offline
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Pure planning layer (no Windows deps) — repo root on path for direct run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.adversarial.live_cheat import (  # noqa: E402
    AIMBOT_PRESETS,
    CHEAT_AIMBOT,
    CHEAT_MACRO,
    CHEAT_TRIGGERBOT,
    MACRO_PRESET,
    TRIGGERBOT_PRESET,
    plan_aim_snap,
    plan_macro_tick,
    plan_trigger_burst,
)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
TOGGLE_KEYS = {"f8": CHEAT_AIMBOT, "f9": CHEAT_TRIGGERBOT, "f10": CHEAT_MACRO}
QUIT_KEY = "f12"
TRIGGER_COOLDOWN_MS = 150.0  # min gap between triggerbot auto-fires while held


# ---------------------------------------------------------------------------
# Windows SendInput actuator (relative mouse move + clicks). Game-friendly:
# SendInput with MOUSEEVENTF_MOVE works for DirectInput titles where SetCursorPos
# does not. Created only on win32.
# ---------------------------------------------------------------------------


class WinMouse:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        ULONG_PTR = ctypes.POINTER(ctypes.c_ulong)

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class _INPUTunion(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", _INPUTunion)]

        self._MOUSEINPUT = MOUSEINPUT
        self._INPUT = INPUT
        self._user32 = ctypes.windll.user32
        self.MOVE = 0x0001
        self.LDOWN, self.LUP = 0x0002, 0x0004
        self.RDOWN, self.RUP = 0x0008, 0x0010

    def _send(self, flags: int, dx: int = 0, dy: int = 0) -> None:
        mi = self._MOUSEINPUT(dx, dy, 0, flags, 0, None)
        inp = self._INPUT(0)  # type 0 == INPUT_MOUSE
        inp.u.mi = mi
        self._user32.SendInput(1, self._ctypes.byref(inp), self._ctypes.sizeof(inp))

    def move(self, dx: int, dy: int) -> None:
        if dx or dy:
            self._send(self.MOVE, dx, dy)

    def click(self, button: str, hold_ms: float) -> None:
        down, up = (
            (self.RDOWN, self.RUP) if button == "right" else (self.LDOWN, self.LUP)
        )
        self._send(down)
        time.sleep(max(0.0, hold_ms) / 1000.0)
        self._send(up)


def _run_actions(mouse: WinMouse, actions) -> None:
    """Execute a planned InputAction sequence via SendInput."""
    for a in actions:
        if a.delay_ms:
            time.sleep(a.delay_ms / 1000.0)
        if a.kind == "move":
            mouse.move(a.dx, a.dy)
        elif a.kind == "click":
            mouse.click(a.button, a.hold_ms)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


class State:
    def __init__(self, difficulty: str, seed: int) -> None:
        self.lock = threading.Lock()
        self.running = True
        self.active = {CHEAT_AIMBOT: False, CHEAT_TRIGGERBOT: False, CHEAT_MACRO: False}
        self.aim_held = False  # right mouse button
        self.fire_held = False  # left mouse button
        self.aim_edge = False  # right-button just pressed (consumed by aimbot)
        self.rng = np.random.default_rng(seed)
        self.aim_preset = AIMBOT_PRESETS[difficulty]
        self.last_trigger_ms = 0.0


def _log_activity(log_f, cheat: str, on: bool, difficulty: str) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": "toggle_on" if on else "toggle_off",
        "cheat": cheat,
        "difficulty": difficulty,
    }
    log_f.write(json.dumps(rec) + "\n")
    log_f.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _confirm_offline(args) -> bool:
    if args.i_am_offline:
        return True
    print("\nType 'offline' to confirm you are in single-player/Story Mode: ", end="")
    try:
        return input().strip().lower() == "offline"
    except (EOFError, KeyboardInterrupt):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live cheat-signature harness (offline)"
    )
    parser.add_argument("--difficulty", choices=list(AIMBOT_PRESETS), default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--i-am-offline",
        action="store_true",
        help="confirm offline single-player (skips the interactive prompt)",
    )
    args = parser.parse_args()

    print(__doc__.split("Usage")[0])  # banner
    print(f"  Difficulty  : {args.difficulty}")
    print(f"  Output log  : {OUTPUT_DIR / 'cheat_activity.jsonl'}")
    if not _confirm_offline(args):
        print("Not confirmed offline — aborting.")
        return 1

    if sys.platform != "win32":
        print(
            "[ERROR] cheat_sim drives Windows SendInput — run it on the Windows "
            "host where GTA5 + the recorder run, not under WSL/Linux."
        )
        return 2

    try:
        from pynput import keyboard, mouse
    except ImportError:
        print("[ERROR] pynput not found. On Windows: pip install pynput")
        return 1

    mouse_actuator = WinMouse()
    state = State(args.difficulty, args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(OUTPUT_DIR / "cheat_activity.jsonl", "w", encoding="utf-8")

    def on_press(key):
        name = getattr(key, "name", None)
        if name == QUIT_KEY:
            with state.lock:
                state.running = False
            return False  # stop keyboard listener
        cheat = TOGGLE_KEYS.get(name or "")
        if cheat:
            with state.lock:
                state.active[cheat] = not state.active[cheat]
                on = state.active[cheat]
            _log_activity(log_f, cheat, on, args.difficulty)
            print(f"  [{'ON ' if on else 'OFF'}] {cheat}")

    def on_click(x, y, button, pressed):
        with state.lock:
            if button == mouse.Button.right:
                state.aim_held = pressed
                if pressed:
                    state.aim_edge = True
            elif button == mouse.Button.left:
                state.fire_held = pressed

    kb = keyboard.Listener(on_press=on_press)
    ms = mouse.Listener(on_click=on_click)
    kb.start()
    ms.start()
    print("\n🔴 Harness armed. Switch to GTA5 (offline). F12 to quit.\n")

    # Actuator loop — reads state and performs the cheat the player is engaging.
    try:
        while True:
            with state.lock:
                if not state.running:
                    break
                aim_on = state.active[CHEAT_AIMBOT]
                trig_on = state.active[CHEAT_TRIGGERBOT]
                macro_on = state.active[CHEAT_MACRO]
                aim_edge = state.aim_edge
                aim_held = state.aim_held
                fire_held = state.fire_held
                state.aim_edge = False
                now_ms = time.perf_counter() * 1000.0

            if aim_on and aim_edge:
                _run_actions(mouse_actuator, plan_aim_snap(state.rng, state.aim_preset))
            if (
                trig_on
                and aim_held
                and (now_ms - state.last_trigger_ms) >= TRIGGER_COOLDOWN_MS
            ):
                _run_actions(
                    mouse_actuator, plan_trigger_burst(state.rng, TRIGGERBOT_PRESET)
                )
                state.last_trigger_ms = now_ms
            if macro_on and fire_held:
                _run_actions(mouse_actuator, plan_macro_tick(state.rng, MACRO_PRESET))

            time.sleep(0.004)  # ~250 Hz polling; planners own their own timing
    finally:
        kb.stop()
        ms.stop()
        log_f.close()
        print("\n⏹  Harness stopped. Activity log saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
