"""
collector/record_session.py
===========================
Windows-side input telemetry recorder.

Records mouse and keyboard events during a gameplay session and saves
them as a structured JSON file for later ingestion into the pipeline.

Usage:
    python record_session.py --player jiri --game valorant --sens 0.45 --dpi 800

Requirements (Windows host, NOT WSL):
    pip install pynput
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from pynput import keyboard, mouse
except ImportError:
    print("[ERROR] pynput not found. Install it on Windows with:")
    print("        pip install pynput")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Where to save session files.
# When running on Windows, output to a shared folder that WSL can also access.
# Default: same directory as this script — copy files to WSL manually or
# point OUTPUT_DIR at a path accessible from both sides, e.g.:
#   OUTPUT_DIR = Path("C:/Users/YOUR_USER/behaviorDNA/data/raw")
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Event buffer
# ---------------------------------------------------------------------------

events: list[dict] = []
recording = False
start_time: float = 0.0


def ts() -> float:
    """Milliseconds elapsed since recording started."""
    return round((time.perf_counter() - start_time) * 1000, 3)


# ---------------------------------------------------------------------------
# Mouse listeners
# ---------------------------------------------------------------------------

prev_x: int | None = None
prev_y: int | None = None


def on_move(x: int, y: int) -> None:
    if not recording:
        return
    global prev_x, prev_y
    dx = x - prev_x if prev_x is not None else 0
    dy = y - prev_y if prev_y is not None else 0
    prev_x, prev_y = x, y
    events.append(
        {
            "t": ts(),
            "type": "mouse_move",
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
        }
    )


def on_click(x: int, y: int, button, pressed: bool) -> None:
    if not recording:
        return
    events.append(
        {
            "t": ts(),
            "type": "mouse_click",
            "x": x,
            "y": y,
            "button": str(button),
            "pressed": pressed,
        }
    )


def on_scroll(x: int, y: int, dx: int, dy: int) -> None:
    if not recording:
        return
    events.append(
        {
            "t": ts(),
            "type": "mouse_scroll",
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
        }
    )


# ---------------------------------------------------------------------------
# Keyboard listeners
# ---------------------------------------------------------------------------


def _key_name(key) -> str:
    try:
        return key.char or str(key)
    except AttributeError:
        return str(key)


def on_key_press(key) -> None:
    if not recording:
        return
    events.append(
        {
            "t": ts(),
            "type": "key_press",
            "key": _key_name(key),
        }
    )


def on_key_release(key) -> None:
    if not recording:
        return
    events.append(
        {
            "t": ts(),
            "type": "key_release",
            "key": _key_name(key),
        }
    )


# ---------------------------------------------------------------------------
# Session save
# ---------------------------------------------------------------------------


def save_session(player: str, game: str, sensitivity: float, dpi: int) -> Path:
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{timestamp}_{player}_{game}_{session_id}.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    session = {
        "session_id": session_id,
        "player": player,
        "game": game,
        "sensitivity": sensitivity,
        "dpi": dpi,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": ts(),
        "event_count": len(events),
        "events": events,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            session, f, separators=(",", ":")
        )  # compact, no indent — files can be large

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a gameplay telemetry session.")
    parser.add_argument(
        "--player", required=True, help="Player identifier (e.g. 'jiri')"
    )
    parser.add_argument(
        "--game", required=True, help="Game name (e.g. 'valorant', 'cs2', 'tarkov')"
    )
    parser.add_argument("--sens", type=float, required=True, help="In-game sensitivity")
    parser.add_argument("--dpi", type=int, required=True, help="Mouse DPI")
    return parser.parse_args()


def main() -> None:
    global recording, start_time, prev_x, prev_y

    args = parse_args()

    print()
    print("=" * 50)
    print("  BehaviorDNA — Session Recorder")
    print("=" * 50)
    print(f"  Player      : {args.player}")
    print(f"  Game        : {args.game}")
    print(f"  Sensitivity : {args.sens}")
    print(f"  DPI         : {args.dpi}")
    print(f"  Output dir  : {OUTPUT_DIR.resolve()}")
    print("=" * 50)
    print()
    print("Press ENTER to start recording...")
    input()

    # Start listeners
    mouse_listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll,
    )
    keyboard_listener = keyboard.Listener(
        on_press=on_key_press,
        on_release=on_key_release,
    )
    mouse_listener.start()
    keyboard_listener.start()

    # Begin recording
    prev_x, prev_y = None, None
    start_time = time.perf_counter()
    recording = True

    print("🔴 Recording... switch to your game now.")
    print("   Press ENTER here to stop recording.\n")

    try:
        input()
    except KeyboardInterrupt:
        pass

    # Stop
    recording = False
    mouse_listener.stop()
    keyboard_listener.stop()

    duration_s = ts() / 1000
    print(f"\n⏹  Stopped. Duration: {duration_s:.1f}s | Events captured: {len(events)}")

    if len(events) < 100:
        print("⚠️  Very few events captured — did the game have focus?")

    output_path = save_session(
        player=args.player,
        game=args.game,
        sensitivity=args.sens,
        dpi=args.dpi,
    )

    print(f"✅ Session saved: {output_path.name}")
    print(f"   Full path    : {output_path.resolve()}")
    print()


if __name__ == "__main__":
    main()
