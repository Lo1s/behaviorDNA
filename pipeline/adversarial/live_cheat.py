"""
pipeline/adversarial/live_cheat.py
==================================
Pure planning layer for the **live** cheat-signature harness
(``collector/cheat_sim.py``).

This module decides *what* a cheat does to the input stream — it emits sequences
of abstract :class:`InputAction` objects — without touching the OS. The
Windows-side actuator translates those actions into ``SendInput`` calls. Keeping
the logic here means:

* it is unit-testable in CI (no ``pynput`` / ``ctypes`` dependency), and
* the **live** harness shares its cheat definitions with the **offline** synthetic
  generator (``pipeline/adversarial/bot_generator.py``) — both agree on "what an
  aimbot looks like", just one rewrites recorded events and the other drives live
  input.

**Safety (by construction).** There is deliberately *no target acquisition, no
memory reading, and no networking* anywhere in this workstream. The planners
produce the superhuman input *signature* — snap kinematics, sub-human click
reaction, periodic fire — for training a **detector**. They cannot find or aim at
an actual enemy on their own (the human does the coarse aim; the bot does only the
inhuman final correction). See ``docs/CHEAT_DATA_COLLECTION.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pipeline.adversarial.bot_generator import (
    CHEAT_AIMBOT,
    CHEAT_LEGIT,
    CHEAT_MACRO,
    CHEAT_TRIGGERBOT,
)

# ---------------------------------------------------------------------------
# Abstract input primitives
# ---------------------------------------------------------------------------


@dataclass
class InputAction:
    """One primitive the actuator executes. ``delay_ms`` is slept *before* it.

    kind="move"  → relative mouse move by (dx, dy) pixels.
    kind="click" → press ``button``, hold ``hold_ms``, release.
    """

    kind: str  # "move" | "click"
    delay_ms: float = 0.0
    dx: int = 0
    dy: int = 0
    button: str = "left"
    hold_ms: float = 0.0


# ---------------------------------------------------------------------------
# Difficulty presets — mirror pipeline/adversarial/generate_dataset.py so the
# live harness and the offline synthetic generator describe the same cheats.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AimbotPreset:
    """smoothing: 0 = single-step teleport (most detectable) → 1 = many small
    eased steps. overshoot/jitter add the *evasive* (soft) humanisation."""

    smoothing: float
    snap_duration_ms: float
    overshoot: float = 0.0  # fraction past target before settling back
    jitter_px: float = 0.0  # per-step gaussian jitter


AIMBOT_PRESETS = {
    "obvious": AimbotPreset(smoothing=0.0, snap_duration_ms=120.0),
    "medium": AimbotPreset(smoothing=0.5, snap_duration_ms=150.0),
    "soft": AimbotPreset(
        smoothing=0.85, snap_duration_ms=180.0, overshoot=0.15, jitter_px=2.0
    ),
}

# Triggerbot: sub-human reaction (humans 150–250 ms), short click hold.
TRIGGERBOT_PRESET = {"reaction_ms_lo": 3.0, "reaction_ms_hi": 20.0, "hold_ms": 30.0}

# Macro: periodic fire + downward recoil compensation.
MACRO_PRESET = {"interval_ms": 100.0, "hold_ms": 15.0, "recoil_dy": 6, "jitter_ms": 4.0}

_MAX_SNAP_STEPS = 20  # smoothing=1.0 → 21 eased steps


# ---------------------------------------------------------------------------
# Planners (rng-seeded → deterministic + testable)
# ---------------------------------------------------------------------------


def plan_aim_snap(
    rng: np.random.Generator,
    preset: AimbotPreset,
    target: tuple[float, float] | None = None,
) -> list[InputAction]:
    """Plan a superhuman **micro-correction** snap by a relative offset.

    ``target`` is the relative (dx, dy) to land on; if ``None`` a plausible
    correction offset is sampled (magnitude + random direction). The final step
    always lands *exactly* on the rounded integer target ("lock-on"), so the
    summed displacement is deterministic regardless of easing/overshoot/jitter.
    """
    if target is None:
        mag = float(rng.uniform(40.0, 300.0))
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        tx, ty = mag * math.cos(ang), mag * math.sin(ang)
    else:
        tx, ty = float(target[0]), float(target[1])

    n = max(1, 1 + int(round(preset.smoothing * _MAX_SNAP_STEPS)))
    # Ease-out cubic cumulative fraction at each step.
    fracs = [1.0 - (1.0 - (i + 1) / n) ** 3 for i in range(n)]
    peak = 1.0 + preset.overshoot
    xs = [f * peak * tx for f in fracs]
    ys = [f * peak * ty for f in fracs]
    if preset.overshoot > 0.0:  # settle back onto the true target
        xs.append(tx)
        ys.append(ty)
    if preset.jitter_px > 0.0:  # jitter all but the final lock-on step
        for i in range(len(xs) - 1):
            xs[i] += float(rng.normal(0.0, preset.jitter_px))
            ys[i] += float(rng.normal(0.0, preset.jitter_px))

    steps = len(xs)
    delay = preset.snap_duration_ms / steps
    tx_i, ty_i = int(round(tx)), int(round(ty))
    sent_x = sent_y = 0
    out: list[InputAction] = []
    for i in range(steps):
        if i == steps - 1:  # exact lock-on
            dx, dy = tx_i - sent_x, ty_i - sent_y
        else:
            dx, dy = int(round(xs[i])) - sent_x, int(round(ys[i])) - sent_y
        sent_x += dx
        sent_y += dy
        out.append(InputAction("move", delay_ms=delay, dx=dx, dy=dy))
    return out


def plan_trigger_burst(
    rng: np.random.Generator, preset: dict | None = None, n_clicks: int = 1
) -> list[InputAction]:
    """Plan ``n_clicks`` clicks with a **sampled sub-human reaction** before each.

    Reaction is drawn per click from ``[reaction_ms_lo, reaction_ms_hi]`` (not a
    constant) so the timing has realistic variance rather than a giveaway-perfect
    value.
    """
    p = preset or TRIGGERBOT_PRESET
    out: list[InputAction] = []
    for _ in range(max(1, n_clicks)):
        reaction = float(rng.uniform(p["reaction_ms_lo"], p["reaction_ms_hi"]))
        out.append(
            InputAction(
                "click", delay_ms=reaction, button="left", hold_ms=float(p["hold_ms"])
            )
        )
    return out


def plan_macro_tick(
    rng: np.random.Generator, preset: dict | None = None
) -> list[InputAction]:
    """Plan one macro tick: a click after a near-periodic interval, plus a
    downward recoil-compensation move (the classic no-recoil macro signature)."""
    p = preset or MACRO_PRESET
    jitter = float(rng.normal(0.0, p["jitter_ms"])) if p["jitter_ms"] else 0.0
    interval = max(1.0, float(p["interval_ms"]) + jitter)
    return [
        InputAction(
            "click", delay_ms=interval, button="left", hold_ms=float(p["hold_ms"])
        ),
        InputAction("move", delay_ms=0.0, dx=0, dy=int(p["recoil_dy"])),
    ]


# ---------------------------------------------------------------------------
# Labelling: toggle-key presses (captured in-band by the recorder) → segments
# ---------------------------------------------------------------------------


def toggle_log_to_segments(
    events: list[dict],
    toggle_keys: dict[str, str],
    session_end_ms: float | None = None,
) -> tuple[str, list[list[float]]]:
    """Derive ``(cheat_label, cheat_segments)`` from toggle-key presses.

    ``toggle_keys`` maps a key string (as the recorder logs it, e.g. ``"Key.f8"``)
    to a cheat type. Each press of a key **toggles** that cheat on/off — so
    presses pair up into on-spans (an odd final press runs to ``session_end_ms``
    or the last event's timestamp). ``cheat_segments`` is the union of all
    cheat-active ``[start_ms, end_ms]`` spans; ``cheat_label`` is the cheat type
    with the most total active time (``"legit"`` if nothing was toggled).
    """
    if session_end_ms is None:
        session_end_ms = max((float(e.get("t", 0.0)) for e in events), default=0.0)

    presses: dict[str, list[float]] = {ct: [] for ct in set(toggle_keys.values())}
    for ev in events:
        if ev.get("type") != "key_press":
            continue
        cheat = toggle_keys.get(ev.get("key"))
        if cheat is not None:
            presses[cheat].append(float(ev.get("t", 0.0)))

    segments: list[list[float]] = []
    active_ms: dict[str, float] = {}
    for cheat, ts in presses.items():
        ts = sorted(ts)
        total = 0.0
        for i in range(0, len(ts), 2):
            start = ts[i]
            end = ts[i + 1] if i + 1 < len(ts) else session_end_ms
            if end > start:
                segments.append([start, end])
                total += end - start
        if total > 0:
            active_ms[cheat] = total

    if not active_ms:
        return CHEAT_LEGIT, []
    label = max(active_ms, key=active_ms.get)
    return label, sorted(segments)


# Re-export the cheat-type constants so callers import them from one place.
__all__ = [
    "InputAction",
    "AimbotPreset",
    "AIMBOT_PRESETS",
    "TRIGGERBOT_PRESET",
    "MACRO_PRESET",
    "plan_aim_snap",
    "plan_trigger_burst",
    "plan_macro_tick",
    "toggle_log_to_segments",
    "CHEAT_AIMBOT",
    "CHEAT_TRIGGERBOT",
    "CHEAT_MACRO",
    "CHEAT_LEGIT",
]
