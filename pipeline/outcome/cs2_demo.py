"""
pipeline/outcome/cs2_demo.py
============================
Parse a Counter-Strike 2 ``.dem`` (SourceTV demo) into **outcome-labelled,
window-aligned** telemetry — the Phase 9 feasibility spike (docs/ROADMAP.md).

Why this module exists
----------------------
docs/SIGNALS.md ranks the *causally strongest* cheat signals as outcome /
performance stats (headshot ratio, damage/shot, accuracy) plus view-angle aim
dynamics. None of them are derivable from the recorder's ``(dt, dx, dy)`` mouse
stream — they live in the game's own event log. A CS2 demo is that log, with
**per-tick ground truth** (every kill carries a ``headshot`` flag and hitgroup;
every shot is a ``weapon_fire``; every player's view angles are recorded each
tick).

What is solid vs. what the spike de-risks
-----------------------------------------
*Solid (validated against a real public demo):* ``demoparser2`` extracts kills,
damage, shots and per-tick view-angles cleanly, and ``aggregate_outcome_windows``
bins them onto the same ``WINDOW_MS`` grid the input-feature pipeline uses.

*The whole risk (per the roadmap):* a demo's clock is **server ticks**, the
recorder's clock is wall-time; the two processes start independently and share no
epoch. ``estimate_offset_by_xcorr`` solves this **without a manual marker** by
cross-correlating the recorder's mouse-motion (the *cause*) against the demo's
view-angle motion (the *effect*) — a self-validating sync (a sharp correlation
peak == good alignment). The cross-device correlation itself can only be measured
on a real dual-capture session; the algorithm is unit-tested with a known
injected offset.

Nothing here reads memory, touches the game process, or goes online — it parses a
recorded file offline (docs/ETHICS.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.constants import WINDOW_MS
from pipeline.features.run import OUTCOME_FEATURE_COLS

# Official CS2 matchmaking records at 64 tick; FACEIT/third-party at 128. The
# demo header does not expose it reliably across demo_version variants, so it is
# a configurable parameter (verify per-source) rather than a hidden constant.
CS2_DEFAULT_TICKRATE = 64.0

# A view-angle change faster than this counts as a "flick" (a deliberate fast
# re-aim). 500 deg/s is well above tracking/recoil-control motion but below the
# instantaneous teleport an unsmoothed aimbot snap produces.
FLICK_ANGVEL_DEG_S = 500.0


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
@dataclass
class DemoOutcomes:
    """Raw per-event tables pulled from a CS2 demo, keyed by server ``tick``.

    All four frames are demoparser2 outputs (pandas). ``aggregate_outcome_windows``
    turns them into per-window :data:`OUTCOME_FEATURE_COLS` rows for one player.

    Columns used downstream:
      kills  — player_death: tick, attacker_name, user_name, headshot, hitgroup
      hurts  — player_hurt:  tick, attacker_name, dmg_health, hitgroup
      fires  — weapon_fire:  tick, user_name, weapon
      angles — per (tick, name): pitch, yaw
    """

    kills: pd.DataFrame
    hurts: pd.DataFrame
    fires: pd.DataFrame
    angles: pd.DataFrame
    tickrate: float
    players: list[str]


def parse_demo_outcomes(
    dem_path: str, tickrate: float = CS2_DEFAULT_TICKRATE
) -> DemoOutcomes:
    """Parse a ``.dem`` into a :class:`DemoOutcomes`.

    ``demoparser2`` is imported lazily so the rest of the module (aggregation +
    sync, which operate on plain DataFrames) is usable / testable without the
    native parser or a demo file installed.
    """
    from demoparser2 import DemoParser  # lazy: native dep, only needed for real demos

    parser = DemoParser(dem_path)
    kills = parser.parse_event("player_death")
    hurts = parser.parse_event("player_hurt")
    fires = parser.parse_event("weapon_fire")
    angles = parser.parse_ticks(["pitch", "yaw"])

    players = sorted(
        n
        for n in pd.unique(angles.get("name", pd.Series(dtype=str)))
        if isinstance(n, str)
    )
    return DemoOutcomes(
        kills=kills,
        hurts=hurts,
        fires=fires,
        angles=angles,
        tickrate=float(tickrate),
        players=players,
    )


def tick_to_seconds(tick, tickrate: float = CS2_DEFAULT_TICKRATE):
    """Convert a server tick (or array of ticks) to seconds from demo start."""
    return np.asarray(tick, dtype=float) / float(tickrate)


# --------------------------------------------------------------------------- #
# View-angle kinematics
# --------------------------------------------------------------------------- #
def _wrap_deg(d: np.ndarray) -> np.ndarray:
    """Wrap an angle difference (deg) into (-180, 180] so yaw 179 -> -179 is 2 deg."""
    return (np.asarray(d, dtype=float) + 180.0) % 360.0 - 180.0


def view_angle_kinematics(
    angles: pd.DataFrame, player: str, tickrate: float = CS2_DEFAULT_TICKRATE
) -> dict:
    """Per-tick angular velocity (deg/s) for one player from their yaw/pitch track.

    Returns a dict with ``tick`` (the *later* tick of each consecutive pair),
    ``time_s`` and ``angvel`` (great-ish-circle angular speed, combining the
    wrapped yaw delta and the pitch delta). Gaps where the player is dead /
    not yet connected (dtick <= 0) are dropped.
    """
    sub = angles[angles["name"] == player].sort_values("tick")
    if len(sub) < 2:
        return {
            "tick": np.array([], dtype=float),
            "time_s": np.array([], dtype=float),
            "angvel": np.array([], dtype=float),
        }
    tick = sub["tick"].to_numpy(dtype=float)
    yaw = sub["yaw"].to_numpy(dtype=float)
    pitch = sub["pitch"].to_numpy(dtype=float)

    dtick = np.diff(tick)
    dyaw = _wrap_deg(np.diff(yaw))
    dpitch = np.diff(pitch)
    ang_dist = np.hypot(dyaw, dpitch)

    valid = dtick > 0
    dt_s = dtick[valid] / float(tickrate)
    angvel = ang_dist[valid] / dt_s
    return {
        "tick": tick[1:][valid],
        "time_s": tick[1:][valid] / float(tickrate),
        "angvel": angvel,
    }


# --------------------------------------------------------------------------- #
# Per-window aggregation (-> OUTCOME_FEATURE_COLS)
# --------------------------------------------------------------------------- #
def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def aggregate_outcome_windows(
    outcomes: DemoOutcomes,
    player: str,
    *,
    window_ms: int = WINDOW_MS,
    offset_s: float = 0.0,
    session_id: str = "demo",
) -> pd.DataFrame:
    """Bin one player's demo telemetry into per-:data:`WINDOW_MS`-window outcome rows.

    ``offset_s`` is the **demo time of recorder ``t=0``** (from
    :func:`estimate_offset_by_xcorr`): ``recorder_s = demo_s - offset_s``. With
    ``offset_s=0`` (the spike default, no recorder to sync to) windows are
    demo-relative. ``window_idx`` matches the input-feature pipeline's grid so the
    two tables join on ``(session_id, window_idx)``.

    Returns one row per window that has any activity, with columns
    ``[session_id, window_idx] + OUTCOME_FEATURE_COLS``.
    """
    tr = outcomes.tickrate
    win_s = window_ms / 1000.0

    def _win(tick_arr) -> np.ndarray:
        secs = tick_to_seconds(tick_arr, tr) - offset_s
        return np.floor(secs / win_s).astype(int)

    # --- events as (window_idx, payload) ---
    kills, hurts, fires = outcomes.kills, outcomes.hurts, outcomes.fires

    kill_win = _win(kills["tick"].to_numpy()) if len(kills) else np.array([], int)
    is_killer = (
        (kills["attacker_name"] == player).to_numpy()
        if len(kills)
        else np.array([], bool)
    )
    is_victim = (
        (kills["user_name"] == player).to_numpy() if len(kills) else np.array([], bool)
    )
    kill_hs = (
        kills.get("headshot", pd.Series(dtype=bool)).fillna(False).to_numpy(dtype=bool)
        if len(kills)
        else np.array([], bool)
    )

    hurt_win = _win(hurts["tick"].to_numpy()) if len(hurts) else np.array([], int)
    hurt_by_player = (
        (hurts["attacker_name"] == player).to_numpy()
        if len(hurts)
        else np.array([], bool)
    )
    hurt_dmg = (
        hurts.get("dmg_health", pd.Series(dtype=float)).to_numpy(dtype=float)
        if len(hurts)
        else np.array([], float)
    )
    hurt_head = (
        (hurts.get("hitgroup", pd.Series(dtype=str)) == "head").to_numpy()
        if len(hurts)
        else np.array([], bool)
    )

    fire_win = _win(fires["tick"].to_numpy()) if len(fires) else np.array([], int)
    fire_by_player = (
        (fires["user_name"] == player).to_numpy() if len(fires) else np.array([], bool)
    )

    ang = view_angle_kinematics(outcomes.angles, player, tr)
    ang_win = (
        np.floor((ang["time_s"] - offset_s) / win_s).astype(int)
        if ang["angvel"].size
        else np.array([], int)
    )

    # window universe
    all_idx = set()
    for arr, mask in (
        (kill_win, is_killer | is_victim),
        (hurt_win, hurt_by_player),
        (fire_win, fire_by_player),
        (ang_win, np.ones(ang_win.shape, bool)),
    ):
        if arr.size:
            all_idx.update(int(w) for w in arr[mask])
    if not all_idx:
        return pd.DataFrame(columns=["session_id", "window_idx", *OUTCOME_FEATURE_COLS])

    rows = []
    for w in sorted(all_idx):
        k_here = kill_win == w
        kills_w = int(np.sum(k_here & is_killer))
        deaths_w = int(np.sum(k_here & is_victim))
        hs_kills_w = int(np.sum(k_here & is_killer & kill_hs))

        h_here = (hurt_win == w) & hurt_by_player
        hits_w = int(np.sum(h_here))
        dmg_w = float(np.sum(hurt_dmg[h_here])) if hits_w else 0.0
        head_hits_w = int(np.sum(hurt_head[h_here]))

        shots_w = int(np.sum((fire_win == w) & fire_by_player))

        a_here = ang_win == w
        av = ang["angvel"][a_here] if a_here.any() else np.array([], float)

        rows.append(
            {
                "session_id": session_id,
                "window_idx": w,
                "kills": kills_w,
                "deaths": deaths_w,
                "shots_fired": shots_w,
                "hits_dealt": hits_w,
                "damage_dealt": dmg_w,
                "accuracy": _safe_ratio(hits_w, shots_w),
                "damage_per_shot": _safe_ratio(dmg_w, shots_w),
                # headshot ratio: prefer per-hit (denser) ground truth, fall back to per-kill
                "headshot_ratio": (
                    _safe_ratio(head_hits_w, hits_w)
                    if hits_w
                    else _safe_ratio(hs_kills_w, kills_w)
                ),
                "kills_per_shot": _safe_ratio(kills_w, shots_w),
                "view_angvel_p50": float(np.percentile(av, 50)) if av.size else 0.0,
                "view_angvel_p99": float(np.percentile(av, 99)) if av.size else 0.0,
                "view_angvel_max": float(np.max(av)) if av.size else 0.0,
                "flick_count": int(np.sum(av > FLICK_ANGVEL_DEG_S)),
            }
        )
    return pd.DataFrame(
        rows, columns=["session_id", "window_idx", *OUTCOME_FEATURE_COLS]
    )


# --------------------------------------------------------------------------- #
# Clock-sync: recorder <-> demo (the spike's whole risk)
# --------------------------------------------------------------------------- #
def _resample_uniform(
    t: np.ndarray, v: np.ndarray, grid_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a (time, value) signal onto a uniform grid by binned-mean."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    if t.size == 0:
        return np.array([]), np.array([])
    t0, t1 = float(t.min()), float(t.max())
    n = max(2, int(np.ceil((t1 - t0) * grid_hz)) + 1)
    grid = t0 + np.arange(n) / grid_hz
    idx = np.clip(np.floor((t - t0) * grid_hz).astype(int), 0, n - 1)
    out = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(out, idx, v)
    np.add.at(cnt, idx, 1.0)
    nz = cnt > 0
    out[nz] /= cnt[nz]
    return grid, out


def angular_speed_series(
    outcomes: DemoOutcomes, player: str, grid_hz: float = 16.0
) -> tuple[np.ndarray, np.ndarray]:
    """Demo-side motion signal: |view-angle velocity| (deg/s) on a uniform grid.

    Time axis is *demo seconds*. This is the *effect* the recorder mouse motion
    causes, so it is what we align the recorder against.
    """
    ang = view_angle_kinematics(outcomes.angles, player, outcomes.tickrate)
    return _resample_uniform(ang["time_s"], ang["angvel"], grid_hz)


def recorder_mouse_speed_series(
    events, grid_hz: float = 16.0
) -> tuple[np.ndarray, np.ndarray]:
    """Recorder-side motion signal: mouse speed (px/s) on a uniform grid.

    ``events`` is a recorder session's ``events`` list (dicts with ``t`` ms and,
    for ``mouse_move``, ``dx``/``dy``). The returned grid is anchored at the **first
    mouse-move** (``_resample_uniform`` anchors at ``t.min()``), so
    :func:`estimate_offset_by_xcorr`'s ``offset_s`` is the demo time of the first
    mouse-move. Dual-capture joins correct that back to the window anchor — see
    ``pipeline.outcome.dual_capture.ingest_dual_capture``. This is the *cause* we
    cross-correlate against the demo's :func:`angular_speed_series`.
    """
    t_ms, speed = [], []
    prev_t = None
    for e in events:
        if e.get("type") != "mouse_move":
            continue
        t = float(e.get("t", 0.0))
        dx = float(e.get("dx", 0.0))
        dy = float(e.get("dy", 0.0))
        if prev_t is not None:
            dt = (t - prev_t) / 1000.0
            if dt > 0:
                t_ms.append(t)
                speed.append(np.hypot(dx, dy) / dt)
        prev_t = t
    if not t_ms:
        return np.array([]), np.array([])
    t_s = (np.asarray(t_ms) - t_ms[0]) / 1000.0
    return _resample_uniform(t_s, np.asarray(speed), grid_hz)


def estimate_offset_by_xcorr(
    t_demo: np.ndarray,
    v_demo: np.ndarray,
    t_rec: np.ndarray,
    v_rec: np.ndarray,
    *,
    grid_hz: float = 16.0,
    max_lag_s: float = 30.0,
) -> dict:
    """Estimate the demo<->recorder clock offset by motion cross-correlation.

    Both signals must already be on the **same uniform grid spacing**
    (``1/grid_hz`` s); pass the outputs of :func:`angular_speed_series` and
    :func:`recorder_mouse_speed_series` with the same ``grid_hz``.

    Returns ``{offset_s, peak_corr, lag_samples}`` where ``offset_s`` is the demo
    time corresponding to recorder ``t=0`` — i.e. plug it straight into
    :func:`aggregate_outcome_windows`'s ``offset_s``. ``peak_corr`` in [-1, 1] is
    the **self-validation**: a sharp value near 1 means the alignment is real; a
    low/flat peak means the sync failed (wrong demo, wrong player, or no shared
    motion) and the offset should not be trusted.
    """
    a = np.asarray(v_demo, float)
    b = np.asarray(v_rec, float)
    if a.size < 2 or b.size < 2:
        return {"offset_s": 0.0, "peak_corr": 0.0, "lag_samples": 0}

    # zero-mean, unit-norm so the cross-correlation is a normalised correlation
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return {"offset_s": 0.0, "peak_corr": 0.0, "lag_samples": 0}
    a, b = a / na, b / nb

    full = np.correlate(a, b, mode="full")  # lag = (index - (len(b)-1))
    lags = np.arange(full.size) - (b.size - 1)
    max_lag = int(round(max_lag_s * grid_hz))
    keep = np.abs(lags) <= max_lag
    full, lags = full[keep], lags[keep]

    best = int(np.argmax(full))
    lag_samples = int(lags[best])
    # The peak lag aligns recorder sample n to demo sample (n + lag) — i.e.
    # recorder t=0 lands on demo sample `lag`. So the demo time of recorder t=0 is
    # t_demo[0] + lag/grid_hz. (Sign verified against injected-offset recovery.)
    t_demo0 = float(t_demo[0]) if np.asarray(t_demo).size else 0.0
    offset_s = t_demo0 + lag_samples / grid_hz
    return {
        "offset_s": float(offset_s),
        "peak_corr": float(full[best]),
        "lag_samples": lag_samples,
    }
