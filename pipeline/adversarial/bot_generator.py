"""
pipeline/adversarial/bot_generator.py
=====================================
Synthetic cheat trajectory generators.

Three cheat signatures are implemented:

  * ``AimbotGenerator``     — replaces the mouse trajectory leading into each
                              click with a fast smoothed snap toward the click
                              location. Humans show overshoot / micro-correction;
                              aimbots show a near-perfect curve to target.

  * ``TriggerbotGenerator`` — compresses the reaction-time gap between the
                              last mouse-movement update and the click event.
                              Humans: 150–250 ms. Triggerbot: 0–5 ms.

  * ``MacroGenerator``      — replaces a contiguous slice of keystrokes with a
                              perfectly periodic key-press / release pattern.
                              Humans produce noisy timing (broadband FFT);
                              macros produce a sharp spectral peak at 1/period.

All generators consume a session dict that follows BehaviorDNA's recorder
schema (``session_id``, ``player``, ``events``, …) and return a new session
dict with the same schema plus two extra keys:

  * ``cheat_label``   — the cheat name (or ``"legit"`` if none applied)
  * ``cheat_segments`` — list of ``[start_ms, end_ms]`` ranges that were
                         modified (so notebooks can highlight them)

The original session is not mutated — generators always return a deep copy.

These generators target *detection* — they're designed to produce signals
that a competent anti-cheat detector should be able to flag. They are
**not** designed to evade detection. If you want an adversarial training
loop where the bot adapts to fool the detector, that is future work.
"""

from __future__ import annotations

import copy
import logging
import math
import random
import uuid
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

CHEAT_LEGIT = "legit"
CHEAT_AIMBOT = "aimbot"
CHEAT_TRIGGERBOT = "triggerbot"
CHEAT_MACRO = "macro"

VALID_CHEAT_LABELS = {CHEAT_LEGIT, CHEAT_AIMBOT, CHEAT_TRIGGERBOT, CHEAT_MACRO}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _deepcopy_session(session: dict) -> dict:
    """Return a deep copy of the session so generators don't mutate input."""
    return copy.deepcopy(session)


def _click_press_events(events: list[dict]) -> list[tuple[int, dict]]:
    """Return ``(index, event)`` pairs for every ``mouse_click`` press."""
    out = []
    for i, ev in enumerate(events):
        if ev.get("type") == "mouse_click" and ev.get("pressed") is True:
            out.append((i, ev))
    return out


def _mouse_move_indices_between(
    events: list[dict], t_start: float, t_end: float
) -> list[int]:
    """Return indices of ``mouse_move`` events with ``t_start <= t <= t_end``."""
    return [
        i
        for i, ev in enumerate(events)
        if ev.get("type") == "mouse_move" and t_start <= ev["t"] <= t_end
    ]


# ---------------------------------------------------------------------------
# Aimbot
# ---------------------------------------------------------------------------


@dataclass
class AimbotGenerator:
    """Snap-to-target mouse trajectory before each click.

    For every click in the session the generator looks ``snap_duration_ms``
    backward, records the player's mouse position at that point, then replaces
    all intervening ``mouse_move`` events with a smoothed interpolation from
    that start position to the click ``(x, y)``.

    Parameters
    ----------
    smoothing:
        Smoothing exponent. ``0.0`` = pure linear interpolation (most obvious
        aimbot signature). Higher values bias the trajectory toward an
        exponential approach (more "humanlike" easing). Typical values:
          - ``0.0`` — obvious aimbot (no smoothing)
          - ``0.5`` — medium aimbot
          - ``0.9`` — soft aimbot (hardest to detect)
    snap_duration_ms:
        Time window before each click that is overwritten with the snap.
    target_fraction:
        Fraction of clicks in the session to convert into aimbot snaps.
        ``1.0`` = every click, ``0.5`` = half of them. Use < 1.0 to produce
        hybrid sessions with mixed human / aimbot behaviour.
    seed:
        Random seed for which clicks are selected when ``target_fraction < 1``.
    """

    smoothing: float = 0.3
    snap_duration_ms: float = 150.0
    target_fraction: float = 1.0
    seed: int = 42

    def apply(self, session: dict) -> dict:
        session = _deepcopy_session(session)
        events = session["events"]

        clicks = _click_press_events(events)
        if not clicks:
            log.warning("AimbotGenerator: session has no clicks, nothing to inject")
            session["cheat_label"] = CHEAT_LEGIT
            session["cheat_segments"] = []
            return session

        rng = random.Random(self.seed)
        n_target = max(1, int(round(len(clicks) * self.target_fraction)))
        selected = rng.sample(clicks, n_target)

        segments: list[list[float]] = []

        for click_idx, click_ev in selected:
            click_t = float(click_ev["t"])
            click_x = float(click_ev["x"])
            click_y = float(click_ev["y"])

            snap_start_t = click_t - self.snap_duration_ms
            mm_indices = _mouse_move_indices_between(events, snap_start_t, click_t)

            if len(mm_indices) < 2:
                continue

            start_ev = events[mm_indices[0]]
            x0, y0 = float(start_ev["x"]), float(start_ev["y"])

            self._rewrite_snap(events, mm_indices, x0, y0, click_x, click_y, click_t)

            segments.append([snap_start_t, click_t])

        session["cheat_label"] = CHEAT_AIMBOT
        session["cheat_segments"] = segments
        return session

    def _rewrite_snap(
        self,
        events: list[dict],
        mm_indices: list[int],
        x0: float,
        y0: float,
        x_target: float,
        y_target: float,
        t_target: float,
    ) -> None:
        """Overwrite the listed mouse_move events with a snap trajectory.

        The position at time ``t`` is computed by progressing from ``(x0, y0)``
        to ``(x_target, y_target)`` over the window. The ``smoothing`` field
        controls easing — at ``0.0`` motion is linear in time; higher values
        bias motion toward the end of the window (faster start, slow finish).
        """
        prev_x = x0
        prev_y = y0
        t_start = events[mm_indices[0]]["t"]
        duration = max(t_target - t_start, 1e-3)

        for idx in mm_indices:
            t = events[idx]["t"]
            u = (t - t_start) / duration
            u = min(max(u, 0.0), 1.0)
            # Easing: u^(1 - smoothing) accelerates the progress as smoothing
            # decreases. smoothing=0 -> linear (u^1); smoothing=0.9 -> u^0.1
            # which is very fast initial movement then plateau.
            eased = u ** (1.0 - self.smoothing) if self.smoothing < 1.0 else u

            new_x = x0 + (x_target - x0) * eased
            new_y = y0 + (y_target - y0) * eased
            dx = new_x - prev_x
            dy = new_y - prev_y

            events[idx]["x"] = int(round(new_x))
            events[idx]["y"] = int(round(new_y))
            events[idx]["dx"] = int(round(dx))
            events[idx]["dy"] = int(round(dy))

            prev_x = new_x
            prev_y = new_y


# ---------------------------------------------------------------------------
# Triggerbot
# ---------------------------------------------------------------------------


@dataclass
class TriggerbotGenerator:
    """Compress reaction-time gap between last mouse movement and click.

    For every click event, the most recent ``mouse_move`` event is shifted
    in time so it sits ``reaction_time_ms`` before the click. The original
    movement positions are preserved — only the timestamp is changed.

    Parameters
    ----------
    reaction_time_ms:
        Target latency in milliseconds between the immediately-preceding
        ``mouse_move`` event and the click. Triggerbots fire at ~0–5 ms;
        humans at 150–250 ms.
    target_fraction:
        Fraction of clicks to convert. Use < 1.0 for mixed sessions.
    seed:
        RNG seed for click selection.
    """

    reaction_time_ms: float = 3.0
    target_fraction: float = 1.0
    seed: int = 42

    def apply(self, session: dict) -> dict:
        session = _deepcopy_session(session)
        events = session["events"]
        clicks = _click_press_events(events)

        if not clicks:
            log.warning("TriggerbotGenerator: session has no clicks")
            session["cheat_label"] = CHEAT_LEGIT
            session["cheat_segments"] = []
            return session

        rng = random.Random(self.seed)
        n_target = max(1, int(round(len(clicks) * self.target_fraction)))
        selected = rng.sample(clicks, n_target)

        segments: list[list[float]] = []

        for click_idx, click_ev in selected:
            click_t = float(click_ev["t"])
            target_move_t = click_t - self.reaction_time_ms

            # Find the most recent mouse_move strictly before the click and
            # rewrite its timestamp so the gap to the click matches the target
            # reaction time. The (x, y) position is preserved.
            for j in range(click_idx - 1, -1, -1):
                if events[j].get("type") == "mouse_move":
                    events[j]["t"] = target_move_t
                    segments.append([target_move_t, click_t])
                    break

        # Re-sort events by time since we may have shuffled order
        events.sort(key=lambda e: e["t"])
        session["events"] = events
        session["cheat_label"] = CHEAT_TRIGGERBOT
        session["cheat_segments"] = segments
        return session


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------


@dataclass
class MacroGenerator:
    """Replace a slice of keyboard events with a perfectly periodic pattern.

    Picks a contiguous window in the session and overwrites all key events
    in that window with a fixed key-press / release cycle at a regular
    interval. Humans never produce keystrokes at perfectly periodic
    intervals, so an FFT of the inter-key-interval signal shows a sharp
    spectral peak that distinguishes macros from genuine play.

    Parameters
    ----------
    keys:
        Cycle of keys to press in order. Default mimics a movement macro.
    interval_ms:
        Fixed interval between consecutive presses. Smaller = faster macro.
    duration_ms:
        Length of the macro window. The macro starts at ``start_fraction``
        of session duration and runs for this long.
    start_fraction:
        Where in the session to place the macro (0.0 = beginning,
        0.5 = middle, 0.9 = near the end).
    seed:
        Unused for the default deterministic cycle but kept for API symmetry.
    """

    keys: tuple[str, ...] = ("w", "a", "s", "d")
    interval_ms: float = 250.0
    duration_ms: float = 5_000.0
    start_fraction: float = 0.3
    seed: int = 42

    def apply(self, session: dict) -> dict:
        session = _deepcopy_session(session)
        events = session["events"]

        if not events:
            session["cheat_label"] = CHEAT_LEGIT
            session["cheat_segments"] = []
            return session

        session_duration = float(session.get("duration_ms", events[-1]["t"]))
        macro_start = session_duration * self.start_fraction
        macro_end = min(macro_start + self.duration_ms, session_duration)

        # Strip existing key events from the macro window
        kept = [
            ev
            for ev in events
            if not (
                ev.get("type") in ("key_press", "key_release")
                and macro_start <= ev["t"] <= macro_end
            )
        ]

        # Generate periodic key events
        n_presses = int((macro_end - macro_start) / self.interval_ms)
        hold_ms = self.interval_ms * 0.3  # 30 % of interval = key down time

        for i in range(n_presses):
            t_press = macro_start + i * self.interval_ms
            t_release = t_press + hold_ms
            key = self.keys[i % len(self.keys)]
            kept.append({"t": t_press, "type": "key_press", "key": key})
            kept.append({"t": t_release, "type": "key_release", "key": key})

        kept.sort(key=lambda e: e["t"])

        session["events"] = kept
        session["cheat_label"] = CHEAT_MACRO
        session["cheat_segments"] = [[macro_start, macro_end]]
        return session


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


@dataclass
class CheatSpec:
    """Configuration for a single cheat injection."""

    cheat_type: str
    params: dict = field(default_factory=dict)

    def build(self) -> AimbotGenerator | TriggerbotGenerator | MacroGenerator:
        if self.cheat_type == CHEAT_AIMBOT:
            return AimbotGenerator(**self.params)
        if self.cheat_type == CHEAT_TRIGGERBOT:
            return TriggerbotGenerator(**self.params)
        if self.cheat_type == CHEAT_MACRO:
            return MacroGenerator(**self.params)
        raise ValueError(f"Unknown cheat_type: {self.cheat_type!r}")


def inject_cheat(session: dict, spec: CheatSpec, new_session_id: bool = True) -> dict:
    """High-level helper: take a legit session, return a labelled hybrid session.

    Parameters
    ----------
    session:
        BehaviorDNA recorder session dict.
    spec:
        Which cheat to inject + its parameters.
    new_session_id:
        If True, replace ``session_id`` with a freshly generated UUID prefix.
        This is recommended so the synthetic session does not collide with
        the original when ingested into the pipeline.
    """
    generator = spec.build()
    hybrid = generator.apply(session)

    if new_session_id:
        hybrid["session_id"] = str(uuid.uuid4())[:8]

    # Update top-level metadata that the ingestion pipeline checks
    hybrid["event_count"] = len(hybrid["events"])
    if hybrid["events"]:
        hybrid["duration_ms"] = float(max(ev["t"] for ev in hybrid["events"]))
    return hybrid


# ---------------------------------------------------------------------------
# Light derived metrics — useful for tests + notebook narration
# ---------------------------------------------------------------------------


def click_reaction_times(session: dict) -> list[float]:
    """Return time deltas (ms) between each click and the prior mouse_move.

    Triggerbot sessions show a tight near-zero distribution. Human sessions
    show a broad distribution centred on 100–250 ms.
    """
    events = session["events"]
    out = []
    last_move_t = None
    for ev in events:
        if ev.get("type") == "mouse_move":
            last_move_t = ev["t"]
        elif ev.get("type") == "mouse_click" and ev.get("pressed") is True:
            if last_move_t is not None:
                out.append(float(ev["t"] - last_move_t))
    return out


def key_press_intervals(session: dict) -> list[float]:
    """Return inter-key-press intervals in ms.

    Macro sessions show a sharp spike at the macro's ``interval_ms``.
    Human sessions show a noisy distribution.
    """
    presses = [ev["t"] for ev in session["events"] if ev.get("type") == "key_press"]
    return [float(presses[i + 1] - presses[i]) for i in range(len(presses) - 1)]


def aim_snap_curvature(session: dict, click_window_ms: float = 150.0) -> list[float]:
    """For every click, compute the mean turn-angle along the preceding window.

    Aimbot snaps produce near-zero curvature (straight or smoothly curved
    trajectories). Human aiming shows higher curvature due to micro-corrections.
    """
    events = session["events"]
    out = []
    for click_idx, click_ev in _click_press_events(events):
        click_t = float(click_ev["t"])
        mm = [
            events[i]
            for i in _mouse_move_indices_between(
                events, click_t - click_window_ms, click_t
            )
        ]
        if len(mm) < 3:
            continue
        angles = []
        for i in range(1, len(mm) - 1):
            ax = mm[i]["x"] - mm[i - 1]["x"]
            ay = mm[i]["y"] - mm[i - 1]["y"]
            bx = mm[i + 1]["x"] - mm[i]["x"]
            by = mm[i + 1]["y"] - mm[i]["y"]
            n_a = math.hypot(ax, ay)
            n_b = math.hypot(bx, by)
            if n_a < 1e-6 or n_b < 1e-6:
                continue
            cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (n_a * n_b)))
            angles.append(math.acos(cos))
        if angles:
            out.append(sum(angles) / len(angles))
    return out
