"""
pipeline/adversarial/humanizer.py
=================================
Phase 7 — the **detection-vs-evasion frontier**.

The Phase-3 generators (``bot_generator.py``) are designed to be *caught*. This
module makes them **evade**: every cheat gets a single humanisation-strength knob
``lam`` ∈ [0, 1] that interpolates from the obvious bot (``lam=0``, identical to the
Phase-3 "obvious" preset) to a maximally humanised variant (``lam=1``) whose input
signature is statistically close to the target player's own legit play.

Two quantities are produced per ``lam``:

* the **humanised session** (recorder JSON schema, drop-in for the ingestion +
  benchmark pipeline — populates ``cheat_label`` / ``cheat_segments``), which the
  detectors score → **detector AUC(λ)**, and
* a closed-form **utility(λ)** — the cheat's residual advantage over an unaided
  human. Utility is a deterministic function of ``lam`` and the player baseline
  (no sampling), so it is monotone-decreasing by construction.

The headline is the crossover: as ``lam`` rises the cheat is harder to detect
(AUC ↓) but also worth less (utility ↓) — *"humanised enough to evade ≈ no
longer worth running."* See ``scripts/evasion_frontier.py`` + ``docs/ADVERSARIAL.md``.

Humanisation levers (extending the easing/overshoot/jitter planners that already
exist in ``pipeline/adversarial/live_cheat.py:plan_aim_snap``):

* **reaction delay** — sampled from a *human* reaction-time model (visual-motor
  RT ≈ 220 ± 40 ms; the GTA click-to-move gap is sub-ms because moves are
  near-continuous, so it is *not* a usable RT proxy → we model it instead),
* **eased / minimum-jerk snaps** instead of linear teleports,
* **kinematic noise matched to the target player's own** move-step scale, and
* **keystroke-timing jitter matched to the player's own** inter-key CV.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass

import numpy as np

from pipeline.adversarial.bot_generator import (
    CHEAT_AIMBOT,
    CHEAT_LEGIT,
    CHEAT_MACRO,
    CHEAT_TRIGGERBOT,
    _click_press_events,
    _mouse_move_indices_between,
    key_press_intervals,
)

# ---------------------------------------------------------------------------
# Constants — the human reference model + the lam=0 "obvious" bot anchor.
# ---------------------------------------------------------------------------

# Human simple visual-motor reaction time (ms). Literature value; used for the
# reaction-delay injection and as the utility denominator.
HUMAN_RT_MS = 220.0
HUMAN_RT_STD_MS = 40.0
# Typical human aim-correction (flick) *movement* time after the reaction, ms.
HUMAN_FLICK_MS = 250.0

# lam=0 anchors (match bot_generator's "obvious" preset).
BOT_TRIGGER_RT_MS = 3.0  # sub-human triggerbot reaction
SNAP_FAST_MS = 120.0  # fast aimbot snap duration at lam=0
MAX_SMOOTHING = 0.9  # aimbot easing exponent reaches u**0.1 at lam=1
MAX_OVERSHOOT = 0.18  # fraction past target a human-like snap overshoots
MACRO_INTERVAL_MS = 200.0
MACRO_DURATION_MS = 8_000.0
MACRO_START_FRACTION = 0.35

VALID_CHEATS = (CHEAT_AIMBOT, CHEAT_TRIGGERBOT, CHEAT_MACRO)


# ---------------------------------------------------------------------------
# Player baseline — sampled from the player's OWN legit sessions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerBaseline:
    """Per-player kinematic / timing statistics used to humanise toward *them*.

    ``move_step_scale`` and ``key_interval_cv`` are estimated from the player's
    legit recordings; the reaction-time / flick-time fields default to the shared
    human model (kept here so utility can reference them per player).
    """

    move_step_scale: float  # px — typical legit mouse-move step magnitude
    key_interval_cv: float  # coefficient of variation of legit inter-key gaps
    human_rt_ms: float = HUMAN_RT_MS
    human_rt_std_ms: float = HUMAN_RT_STD_MS
    human_flick_ms: float = HUMAN_FLICK_MS


def _move_step_scale(sessions: list[dict]) -> float:
    """Median magnitude of nonzero ``(dx, dy)`` mouse-move steps across sessions."""
    mags: list[float] = []
    for s in sessions:
        for ev in s.get("events", []):
            if ev.get("type") != "mouse_move":
                continue
            dx = float(ev.get("dx") or 0.0)
            dy = float(ev.get("dy") or 0.0)
            m = float(np.hypot(dx, dy))
            if m > 0.0:
                mags.append(m)
    if not mags:
        return 2.0  # sane default jitter scale (px)
    return float(np.median(mags))


def _key_interval_cv(sessions: list[dict]) -> float:
    """Coefficient of variation of inter-key-press intervals across sessions."""
    intervals: list[float] = []
    for s in sessions:
        intervals.extend(key_press_intervals(s))
    arr = np.asarray([x for x in intervals if x > 0.0], dtype=np.float64)
    if arr.size < 2:
        return 0.6  # plausible human keystroke CV when data is thin
    mean = float(arr.mean())
    if mean <= 0.0:
        return 0.6
    return float(arr.std() / mean)


def player_baseline(sessions: list[dict]) -> PlayerBaseline:
    """Build a :class:`PlayerBaseline` from a player's legit recordings."""
    return PlayerBaseline(
        move_step_scale=_move_step_scale(sessions),
        key_interval_cv=_key_interval_cv(sessions),
    )


# ---------------------------------------------------------------------------
# lam → cheat parameters (shared by the rewrite + the utility, so they agree)
# ---------------------------------------------------------------------------


def _aimbot_params(lam: float) -> dict:
    lam = float(np.clip(lam, 0.0, 1.0))
    return {
        "smoothing": lam * MAX_SMOOTHING,
        "snap_duration_ms": SNAP_FAST_MS + lam * (HUMAN_FLICK_MS - SNAP_FAST_MS),
        "overshoot": lam * MAX_OVERSHOOT,
        "reaction_delay_ms": lam * HUMAN_RT_MS,
    }


def _trigger_rt_ms(
    lam: float, rng: np.random.Generator, baseline: PlayerBaseline
) -> float:
    """Per-click reaction time interpolated sub-human → sampled human RT."""
    lam = float(np.clip(lam, 0.0, 1.0))
    human = max(1.0, float(rng.normal(baseline.human_rt_ms, baseline.human_rt_std_ms)))
    return (1.0 - lam) * BOT_TRIGGER_RT_MS + lam * human


# ---------------------------------------------------------------------------
# Utility(λ) — closed form, monotone non-increasing
# ---------------------------------------------------------------------------


def aimbot_utility(lam: float, baseline: PlayerBaseline) -> float:
    """Residual time advantage of the snap over an unaided human correction."""
    p = _aimbot_params(lam)
    bot_time = p["reaction_delay_ms"] + p["snap_duration_ms"]
    human_time = baseline.human_rt_ms + baseline.human_flick_ms
    return float(max(0.0, (human_time - bot_time) / human_time))


def triggerbot_utility(lam: float, baseline: PlayerBaseline) -> float:
    """Reaction-time edge over the player's own reaction (the roadmap example)."""
    lam = float(np.clip(lam, 0.0, 1.0))
    rt = (1.0 - lam) * BOT_TRIGGER_RT_MS + lam * baseline.human_rt_ms
    return float(max(0.0, (baseline.human_rt_ms - rt) / baseline.human_rt_ms))


def macro_utility(lam: float, baseline: PlayerBaseline) -> float:
    """Timing consistency = the macro's value (perfect cadence → 1, human → 0)."""
    lam = float(np.clip(lam, 0.0, 1.0))
    # realised CV ≈ lam * human CV → consistency = 1 - lam
    return float(np.clip(1.0 - lam, 0.0, 1.0))


def cheat_utility(cheat_type: str, lam: float, baseline: PlayerBaseline) -> float:
    if cheat_type == CHEAT_AIMBOT:
        return aimbot_utility(lam, baseline)
    if cheat_type == CHEAT_TRIGGERBOT:
        return triggerbot_utility(lam, baseline)
    if cheat_type == CHEAT_MACRO:
        return macro_utility(lam, baseline)
    raise ValueError(f"Unknown cheat_type: {cheat_type!r}")


# ---------------------------------------------------------------------------
# Session rewrites — each returns a deep-copied, labelled recorder session
# ---------------------------------------------------------------------------


def _finalize(session: dict, label: str, segments: list[list[float]]) -> dict:
    session["cheat_label"] = label
    session["cheat_segments"] = segments
    session["session_id"] = str(uuid.uuid4())[:8]
    session["event_count"] = len(session["events"])
    if session["events"]:
        session["duration_ms"] = float(max(ev["t"] for ev in session["events"]))
    return session


def humanize_aimbot(
    session: dict,
    lam: float,
    baseline: PlayerBaseline,
    *,
    target_fraction: float = 1.0,
    seed: int = 42,
) -> dict:
    """Aimbot whose snap is eased / overshooting / jittered toward human at ``lam``.

    At ``lam=0`` this is a hard linear teleport snap (identical easing to
    ``AimbotGenerator(smoothing=0)``, no jitter/overshoot). At ``lam=1`` the snap
    is stretched to a human flick duration, eased (``u**0.1``), overshoots and
    settles, and carries per-step gaussian jitter scaled to the player's own
    move-step size. The final move always locks exactly onto the click (it is
    still an aimbot).
    """
    session = copy.deepcopy(session)
    events = session["events"]
    clicks = _click_press_events(events)
    if not clicks:
        return _finalize(session, CHEAT_LEGIT, [])

    p = _aimbot_params(lam)
    rng = np.random.default_rng(seed)
    n_target = max(1, int(round(len(clicks) * target_fraction)))
    sel_idx = rng.choice(len(clicks), size=min(n_target, len(clicks)), replace=False)
    selected = [clicks[i] for i in sorted(sel_idx)]

    jitter_sigma = lam * baseline.move_step_scale
    segments: list[list[float]] = []

    for _click_idx, click_ev in selected:
        click_t = float(click_ev["t"])
        click_x, click_y = float(click_ev["x"]), float(click_ev["y"])
        snap_start_t = click_t - p["snap_duration_ms"]
        mm = _mouse_move_indices_between(events, snap_start_t, click_t)
        if len(mm) < 2:
            continue

        start_ev = events[mm[0]]
        x0, y0 = float(start_ev["x"]), float(start_ev["y"])
        t_start = float(start_ev["t"])
        duration = max(click_t - t_start, 1e-3)
        smoothing = p["smoothing"]
        peak = 1.0 + p["overshoot"]

        prev_x, prev_y = x0, y0
        for j, idx in enumerate(mm):
            t = float(events[idx]["t"])
            u = min(max((t - t_start) / duration, 0.0), 1.0)
            eased = u ** (1.0 - smoothing) if smoothing < 1.0 else u
            is_last = j == len(mm) - 1
            if is_last:
                new_x, new_y = click_x, click_y  # exact lock-on
            else:
                # overshoot then settle back over the second half of the window
                settle = peak - (peak - 1.0) * max(0.0, (u - 0.5) / 0.5)
                new_x = x0 + (click_x - x0) * eased * settle
                new_y = y0 + (click_y - y0) * eased * settle
                if jitter_sigma > 0.0:
                    new_x += float(rng.normal(0.0, jitter_sigma))
                    new_y += float(rng.normal(0.0, jitter_sigma))
            events[idx]["x"] = int(round(new_x))
            events[idx]["y"] = int(round(new_y))
            events[idx]["dx"] = int(round(new_x - prev_x))
            events[idx]["dy"] = int(round(new_y - prev_y))
            prev_x, prev_y = new_x, new_y

        segments.append([snap_start_t, click_t])

    return _finalize(session, CHEAT_AIMBOT, segments)


def humanize_triggerbot(
    session: dict,
    lam: float,
    baseline: PlayerBaseline,
    *,
    target_fraction: float = 1.0,
    seed: int = 42,
) -> dict:
    """Triggerbot whose click reaction interpolates sub-human → the player's RT.

    Each selected click's immediately-preceding ``mouse_move`` is time-shifted so
    the gap to the click equals a per-click reaction time ``rt(lam)`` (sampled, so
    timing has natural variance at high ``lam``). Positions are preserved.
    """
    session = copy.deepcopy(session)
    events = session["events"]
    clicks = _click_press_events(events)
    if not clicks:
        return _finalize(session, CHEAT_LEGIT, [])

    rng = np.random.default_rng(seed)
    n_target = max(1, int(round(len(clicks) * target_fraction)))
    sel_idx = rng.choice(len(clicks), size=min(n_target, len(clicks)), replace=False)
    selected = [clicks[i] for i in sorted(sel_idx)]

    segments: list[list[float]] = []
    for click_idx, click_ev in selected:
        click_t = float(click_ev["t"])
        rt = _trigger_rt_ms(lam, rng, baseline)
        target_move_t = click_t - rt
        for j in range(click_idx - 1, -1, -1):
            if events[j].get("type") == "mouse_move":
                events[j]["t"] = target_move_t
                segments.append([target_move_t, click_t])
                break

    events.sort(key=lambda e: e["t"])
    session["events"] = events
    return _finalize(session, CHEAT_TRIGGERBOT, segments)


def humanize_macro(
    session: dict,
    lam: float,
    baseline: PlayerBaseline,
    *,
    keys: tuple[str, ...] = ("w", "a", "s", "d"),
    seed: int = 42,
) -> dict:
    """Recoil/movement macro whose cadence jitters from perfect (lam=0) toward
    the player's own keystroke CV (lam=1), smearing the FFT/periodicity peak."""
    session = copy.deepcopy(session)
    events = session["events"]
    if not events:
        return _finalize(session, CHEAT_LEGIT, [])

    rng = np.random.default_rng(seed)
    duration_ms = float(session.get("duration_ms", events[-1]["t"]))
    macro_start = duration_ms * MACRO_START_FRACTION
    macro_end = min(macro_start + MACRO_DURATION_MS, duration_ms)

    kept = [
        ev
        for ev in events
        if not (
            ev.get("type") in ("key_press", "key_release")
            and macro_start <= ev["t"] <= macro_end
        )
    ]

    hold_ms = MACRO_INTERVAL_MS * 0.3
    cv = lam * baseline.key_interval_cv
    t = macro_start
    i = 0
    while t < macro_end:
        key = keys[i % len(keys)]
        kept.append({"t": t, "type": "key_press", "key": key})
        kept.append({"t": t + hold_ms, "type": "key_release", "key": key})
        # jittered gap: mean MACRO_INTERVAL_MS, realised CV ≈ lam * human CV
        gap = MACRO_INTERVAL_MS * (1.0 + float(rng.normal(0.0, cv)))
        t += max(1.0, gap)
        i += 1

    kept.sort(key=lambda e: e["t"])
    session["events"] = kept
    return _finalize(session, CHEAT_MACRO, [[macro_start, macro_end]])


def humanize(
    session: dict,
    cheat_type: str,
    lam: float,
    baseline: PlayerBaseline,
    *,
    seed: int = 42,
) -> dict:
    """Dispatch to the per-cheat humaniser (parallels ``bot_generator.inject_cheat``)."""
    if cheat_type == CHEAT_AIMBOT:
        return humanize_aimbot(session, lam, baseline, seed=seed)
    if cheat_type == CHEAT_TRIGGERBOT:
        return humanize_triggerbot(session, lam, baseline, seed=seed)
    if cheat_type == CHEAT_MACRO:
        return humanize_macro(session, lam, baseline, seed=seed)
    raise ValueError(f"Unknown cheat_type: {cheat_type!r}")


__all__ = [
    "PlayerBaseline",
    "player_baseline",
    "humanize",
    "humanize_aimbot",
    "humanize_triggerbot",
    "humanize_macro",
    "cheat_utility",
    "aimbot_utility",
    "triggerbot_utility",
    "macro_utility",
    "VALID_CHEATS",
    "HUMAN_RT_MS",
    "HUMAN_FLICK_MS",
]
