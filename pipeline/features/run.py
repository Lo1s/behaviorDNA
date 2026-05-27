"""
pipeline/features/run.py
========================
Stage 2 — Feature Engineering: structured Parquet events → behavioral feature windows.

Reads events.parquet and sessions.parquet from data/processed/, slices each session
into 30-second non-overlapping windows, and computes 25 behavioral features across
5 groups:

  Mouse kinematics    : speed, acceleration, jitter, click intervals
  Mouse trajectory    : curvature, path efficiency, direction changes  (Phase 1)
  Keyboard patterns   : hold duration, inter-key interval, burst rate, WASD rhythm
  Reaction timing     : click reaction latency, inter-click movement   (Phase 1)
  Keystroke geometry  : periodicity (CV of inter-key intervals)        (Phase 1)
  Session aggregates  : event rate, mouse/key ratio, active time %, scroll stats

Sens/DPI normalization is applied to speed/acceleration so sessions recorded at
different DPI settings are comparable. z-score scaling is deliberately NOT applied
here — it must be fit on the training fold only (lives in pipeline/training/run.py).

Output:
  data/processed/features.parquet   — one row per (session_id, window_idx)

Run via DVC:
    dvc repro features

Or directly:
    python -m pipeline.features.run
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
EVENTS_IN = PROCESSED_DIR / "events.parquet"
SESSIONS_IN = PROCESSED_DIR / "sessions.parquet"
FEATURES_OUT = PROCESSED_DIR / "features.parquet"

WINDOW_MS = 30_000  # 30 seconds
WASD_KEYS = {"w", "a", "s", "d", "Key.up", "Key.down", "Key.left", "Key.right"}

FEATURE_COLS = [
    # Mouse kinematics
    "speed_mean",
    "speed_std",
    "accel_mean",
    "accel_std",
    "jitter",
    "click_interval_mean",
    "click_interval_std",
    # Mouse trajectory (Phase 1 — anti-cheat-targeted)
    "mouse_curvature_mean",
    "mouse_curvature_std",
    "path_efficiency",
    "direction_changes_per_sec",
    # Keyboard patterns
    "hold_mean",
    "hold_std",
    "iki_mean",
    "iki_std",
    "burst_rate",
    "wasd_rhythm",
    # Reaction timing (Phase 1 — anti-cheat-targeted)
    "click_reaction_mean",
    "inter_click_movement",
    # Keystroke geometry (Phase 1 — anti-cheat-targeted)
    "keystroke_periodicity",
    # Session aggregates
    "event_rate",
    "mouse_key_ratio",
    "active_time_pct",
    "scroll_count",
    "scroll_direction_ratio",
]

META_COLS = [
    "session_id",
    "window_idx",
    "player",
    "game",
    "activity",
    "sensitivity",
    "dpi",
    "recorded_at",
    "duration_ms",
]


def compute_mouse_kinematics(
    mm: pd.DataFrame,
    mc: pd.DataFrame,
    norm_factor: float,
) -> dict:
    """Compute mouse speed, acceleration, jitter and click interval features.

    norm_factor = sensitivity * dpi / 800.0  — normalizes speed to a standard
    800 DPI @ 1.0 sensitivity baseline so different hardware is comparable.
    """
    result: dict = {
        "speed_mean": float("nan"),
        "speed_std": float("nan"),
        "accel_mean": float("nan"),
        "accel_std": float("nan"),
        "jitter": float("nan"),
        "click_interval_mean": float("nan"),
        "click_interval_std": float("nan"),
    }

    if len(mm) >= 2:
        mm = mm.sort_values("t")
        dx = mm["dx"].astype(float)
        dy = mm["dy"].astype(float)
        dt = mm["t"].astype(float).diff()

        dist = np.sqrt(dx**2 + dy**2)
        safe_dt = dt.replace(0, float("nan"))
        speed = dist / safe_dt / norm_factor

        result["speed_mean"] = float(speed.mean())
        result["speed_std"] = float(speed.std())

        accel = speed.diff() / safe_dt
        result["accel_mean"] = float(accel.mean())
        result["accel_std"] = float(accel.std())

        total_path = float(dist.sum())
        x_vals = mm["x"].astype(float)
        y_vals = mm["y"].astype(float)
        euclidean = math.sqrt(
            (x_vals.iloc[-1] - x_vals.iloc[0]) ** 2
            + (y_vals.iloc[-1] - y_vals.iloc[0]) ** 2
        )
        result["jitter"] = total_path / max(euclidean, 1.0)

    # Click intervals: time between successive button-down events
    presses = mc[mc["pressed"] == True].sort_values("t")  # noqa: E712
    if len(presses) >= 2:
        intervals = presses["t"].diff().dropna()
        result["click_interval_mean"] = float(intervals.mean())
        result["click_interval_std"] = float(intervals.std())

    return result


def compute_keyboard_patterns(
    kp: pd.DataFrame,
    kr: pd.DataFrame,
    window_duration_ms: float,
) -> dict:
    """Compute hold duration, inter-key interval, burst rate, and WASD rhythm."""
    result: dict = {
        "hold_mean": float("nan"),
        "hold_std": float("nan"),
        "iki_mean": float("nan"),
        "iki_std": float("nan"),
        "burst_rate": 0.0,  # 0 = no keys pressed, not undefined
        "wasd_rhythm": float("nan"),
    }

    if len(kp) == 0:
        return result

    result["burst_rate"] = len(kp) / (window_duration_ms / 1000.0)

    # IKI: inter-key interval between consecutive presses
    if len(kp) >= 2:
        iki = kp.sort_values("t")["t"].diff().dropna()
        result["iki_mean"] = float(iki.mean())
        result["iki_std"] = float(iki.std())

    # Hold duration: pair each press with the nearest subsequent release of same key
    if len(kr) > 0:
        hold_rows = []
        for key_val in kp["key"].unique():
            kp_key = (
                kp[kp["key"] == key_val]
                .sort_values("t")
                .rename(columns={"t": "press_t"})
            )
            kr_key = (
                kr[kr["key"] == key_val]
                .sort_values("t")
                .rename(columns={"t": "release_t"})
            )
            if kr_key.empty:
                continue
            paired = pd.merge_asof(
                kp_key[["press_t"]],
                kr_key[["release_t"]],
                left_on="press_t",
                right_on="release_t",
                direction="forward",
            ).dropna(subset=["release_t"])
            paired["hold_ms"] = paired["release_t"] - paired["press_t"]
            hold_rows.append(paired[paired["hold_ms"] > 0]["hold_ms"])

        if hold_rows:
            all_holds = pd.concat(hold_rows)
            result["hold_mean"] = float(all_holds.mean())
            result["hold_std"] = float(all_holds.std())

    # WASD rhythm: variance of IKI for movement keys only
    wasd_presses = kp[kp["key"].str.lower().isin(WASD_KEYS)].sort_values("t")
    if len(wasd_presses) >= 2:
        wasd_iki = wasd_presses["t"].diff().dropna()
        result["wasd_rhythm"] = float(wasd_iki.var())

    return result


def compute_session_aggregates(
    window: pd.DataFrame,
    w_start: float,
    window_duration_ms: float,
) -> dict:
    """Compute event rate, mouse/key ratio, active time %, and scroll stats."""
    n_total = len(window)
    event_rate = n_total / (window_duration_ms / 1000.0)

    et = window["event_type"]
    n_mouse = et.str.startswith("mouse").sum()
    n_key = et.isin(["key_press", "key_release"]).sum()
    mouse_key_ratio = n_mouse / (n_key + 1e-9)

    # Active time: fraction of 1-second buckets that contain at least one event
    n_buckets = max(1, round(window_duration_ms / 1000))
    bucket = ((window["t"] - w_start) // 1000).clip(upper=n_buckets - 1)
    active_pct = bucket.nunique() / n_buckets

    scrolls = window[et == "mouse_scroll"]
    scroll_count = len(scrolls)
    if scroll_count > 0:
        n_down = (scrolls["dy"].fillna(0) > 0).sum()
        scroll_direction_ratio = float(n_down / scroll_count)
    else:
        scroll_direction_ratio = float("nan")

    return {
        "event_rate": float(event_rate),
        "mouse_key_ratio": float(mouse_key_ratio),
        "active_time_pct": float(active_pct),
        "scroll_count": int(scroll_count),
        "scroll_direction_ratio": scroll_direction_ratio,
    }


def compute_trajectory_features(mm: pd.DataFrame, window_duration_ms: float) -> dict:
    """Mouse trajectory geometry: curvature, path efficiency, direction changes.

    These features are designed to survive 30-second aggregation while still
    carrying the geometric signal that distinguishes humans from aimbots:

    - ``mouse_curvature_mean/std`` — distribution of turn angles between
      consecutive 3-point segments. Aimbot snaps drive both metrics down.
    - ``path_efficiency`` — Euclidean displacement / total path length.
      1.0 = perfectly straight; 0.0 = highly self-intersecting.
    - ``direction_changes_per_sec`` — count of velocity-vector sign flips
      per second (combined dx + dy). Humans flip often (overshoot/correction);
      aimbots rarely do.
    """
    result: dict = {
        "mouse_curvature_mean": float("nan"),
        "mouse_curvature_std": float("nan"),
        "path_efficiency": float("nan"),
        "direction_changes_per_sec": float("nan"),
    }

    if len(mm) < 3:
        return result

    mm = mm.sort_values("t")
    xs = mm["x"].astype(float).to_numpy()
    ys = mm["y"].astype(float).to_numpy()

    # Vectors between consecutive points
    vx = np.diff(xs)
    vy = np.diff(ys)
    norms = np.hypot(vx, vy)

    # Curvature: angle between consecutive non-zero motion vectors
    valid = norms > 1e-6
    if valid.sum() >= 2:
        # Pair vector i with vector i+1, both must be valid
        pair_valid = valid[:-1] & valid[1:]
        if pair_valid.any():
            cos_theta = (vx[:-1] * vx[1:] + vy[:-1] * vy[1:]) / (
                norms[:-1] * norms[1:] + 1e-12
            )
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angles = np.arccos(cos_theta[pair_valid])
            if angles.size > 0:
                result["mouse_curvature_mean"] = float(np.mean(angles))
                result["mouse_curvature_std"] = float(np.std(angles))

    # Path efficiency: straight-line displacement / sum of segment lengths
    total_path = float(np.sum(norms))
    if total_path > 1e-6:
        displacement = math.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2)
        result["path_efficiency"] = float(displacement / total_path)

    # Direction changes per second (sign flips in either axis)
    if window_duration_ms > 0 and len(vx) >= 2:
        sx = np.sign(vx)
        sy = np.sign(vy)
        flips_x = int(np.sum((sx[:-1] != 0) & (sx[1:] != 0) & (sx[:-1] != sx[1:])))
        flips_y = int(np.sum((sy[:-1] != 0) & (sy[1:] != 0) & (sy[:-1] != sy[1:])))
        result["direction_changes_per_sec"] = float(
            (flips_x + flips_y) / (window_duration_ms / 1000.0)
        )

    return result


def compute_reaction_features(window: pd.DataFrame) -> dict:
    """Reaction timing: click latency and inter-click mouse displacement.

    - ``click_reaction_mean`` — average gap (ms) between each ``mouse_click``
      press and the most recent prior ``mouse_move``. Triggerbots collapse
      this to ~0; humans show 100–250 ms.
    - ``inter_click_movement`` — average Euclidean distance the mouse traveled
      between consecutive clicks. Cheats firing without repositioning produce
      near-zero values; humans move the cursor between shots.
    """
    result: dict = {
        "click_reaction_mean": float("nan"),
        "inter_click_movement": float("nan"),
    }

    if window.empty:
        return result

    w = window.sort_values("t")
    events_t = w["t"].astype(float).to_numpy()
    types = w["event_type"].to_numpy()
    xs = w["x"].astype(float).to_numpy()
    ys = w["y"].astype(float).to_numpy()
    pressed = w["pressed"].to_numpy()

    # Click-reaction time: time from last mouse_move before each click press
    reaction_times: list[float] = []
    last_move_t = None
    for i in range(len(w)):
        if types[i] == "mouse_move":
            last_move_t = events_t[i]
        elif types[i] == "mouse_click" and pressed[i] is True:
            if last_move_t is not None:
                reaction_times.append(events_t[i] - last_move_t)
    if reaction_times:
        result["click_reaction_mean"] = float(np.mean(reaction_times))

    # Inter-click movement distance
    click_mask = (types == "mouse_click") & (pressed == True)  # noqa: E712
    click_xs = xs[click_mask]
    click_ys = ys[click_mask]
    if click_xs.size >= 2:
        dx = np.diff(click_xs)
        dy = np.diff(click_ys)
        result["inter_click_movement"] = float(np.mean(np.hypot(dx, dy)))

    return result


def compute_keystroke_periodicity(kp: pd.DataFrame) -> dict:
    """Coefficient of variation of inter-key-press intervals.

    Macros press keys at perfectly regular intervals → CV → 0.
    Humans press keys irregularly → CV typically > 0.3.

    This is the time-domain analog of the FFT analysis in notebook 10
    (a sharp FFT peak ↔ a low-CV interval distribution).
    """
    result: dict = {"keystroke_periodicity": float("nan")}

    if len(kp) < 3:
        return result

    presses = kp.sort_values("t")["t"].astype(float).to_numpy()
    intervals = np.diff(presses)
    mean = float(np.mean(intervals))
    if mean > 1e-6:
        result["keystroke_periodicity"] = float(np.std(intervals) / mean)
    return result


def process_session_windows(
    session_events: pd.DataFrame,
    norm_factor: float,
) -> list[dict]:
    """Slice session events into 30s windows and extract features for each."""
    if session_events.empty:
        return []

    t_anchor = float(session_events["t"].min())
    t_max = float(session_events["t"].max())
    windows = []
    window_idx = 0

    while True:
        w_start = t_anchor + window_idx * WINDOW_MS
        w_end = w_start + WINDOW_MS

        if w_start > t_max:
            break

        mask = (session_events["t"] >= w_start) & (session_events["t"] < w_end)
        window = session_events[mask]

        if window.empty:
            break

        actual_ms = min(float(WINDOW_MS), t_max - w_start)

        mm = window[window["event_type"] == "mouse_move"]
        mc = window[window["event_type"] == "mouse_click"]
        kp = window[window["event_type"] == "key_press"]
        kr = window[window["event_type"] == "key_release"]

        features = {"window_idx": window_idx}
        features.update(compute_mouse_kinematics(mm, mc, norm_factor))
        features.update(compute_trajectory_features(mm, actual_ms))
        features.update(compute_keyboard_patterns(kp, kr, actual_ms))
        features.update(compute_reaction_features(window))
        features.update(compute_keystroke_periodicity(kp))
        features.update(compute_session_aggregates(window, w_start, actual_ms))
        windows.append(features)

        window_idx += 1

    return windows


def run() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(EVENTS_IN)
    sessions = pd.read_parquet(SESSIONS_IN)

    log.info(
        "Loaded %d events across %d session(s)",
        len(events),
        sessions["session_id"].nunique(),
    )

    feature_rows = []

    for _, sess in sessions.iterrows():
        sid = sess["session_id"]
        norm_factor = (float(sess["sensitivity"]) * float(sess["dpi"])) / 800.0
        sess_events = events[events["session_id"] == sid].sort_values("t")

        windows = process_session_windows(sess_events, norm_factor)

        for w in windows:
            row = {
                "session_id": sid,
                "player": sess["player"],
                "game": sess["game"],
                "activity": sess.get("activity"),
                "sensitivity": sess["sensitivity"],
                "dpi": sess["dpi"],
                "recorded_at": sess["recorded_at"],
                "duration_ms": sess["duration_ms"],
            }
            row.update(w)
            feature_rows.append(row)

        log.info(
            "  session %-10s  player=%-12s  windows=%d",
            sid,
            sess["player"],
            len(windows),
        )

    if not feature_rows:
        log.error("No feature windows produced. Exiting.")
        sys.exit(1)

    features_df = pd.DataFrame(feature_rows)[META_COLS + FEATURE_COLS]
    features_df.to_parquet(FEATURES_OUT, index=False)

    log.info("Wrote features: %s  (%d rows)", FEATURES_OUT, len(features_df))
    log.info("")
    log.info("=== Feature summary ===")
    log.info(
        "  Windows per session: min=%d  max=%d",
        features_df.groupby("session_id")["window_idx"].count().min(),
        features_df.groupby("session_id")["window_idx"].count().max(),
    )
    nan_rates = features_df[FEATURE_COLS].isna().mean()
    for col, rate in nan_rates[nan_rates > 0].items():
        log.info("  NaN  %-30s  %.0f%%", col, rate * 100)


if __name__ == "__main__":
    run()
