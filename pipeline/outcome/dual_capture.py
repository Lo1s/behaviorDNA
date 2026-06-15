"""
pipeline/outcome/dual_capture.py
================================
Phase 9 — ingest a **dual-capture** session (a recorder JSON + the CS2 SourceTV
``.dem`` recorded *at the same time*) into one clock-synced, window-joined table:

    input features  (recorder mouse/keyboard, per 30 s window)
        ⨝  on (session_id, window_idx)
    outcome features (demo kills/damage/accuracy/view-angles, per 30 s window)

The Phase 9 feasibility spike (``cs2_demo.py``) proved each half works and that the
two clocks can be aligned marker-free by motion cross-correlation. This module is
the **orchestration** that turns those primitives into a usable pipeline so that,
once real dual-capture sessions are recorded, the supervised outcome-based detector
(and the Phase 4.1 aggregator re-attempt) have a joined feature table to train on.

The one correctness subtlety is **window-grid alignment**. The input side
(``process_session_windows``) indexes windows from ``t_anchor = min(t)`` over *all*
recorder events; the outcome side indexes from ``offset_s`` (the demo time of
recorder ``t=0``). But the cross-correlation's ``offset_s`` is the demo time of the
**first mouse-move** (``recorder_mouse_speed_series`` → ``_resample_uniform`` anchors
its grid at ``t.min()``), which differs from the window anchor whenever the session
opens with a non-mouse event. So ``ingest_dual_capture`` corrects it back to the
anchor — ``offset_anchor = offset_xcorr − (first_mouse_t − t_anchor)`` — and feeds
*that* to ``aggregate_outcome_windows``. Both sides then count windows from the
identical origin and the ``(session_id, window_idx)`` join is exact.

Nothing here is online or touches a live game — it reads two recorded files
offline (docs/ETHICS.md).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from pipeline.features.run import (
    OUTCOME_FEATURE_COLS,
    polling_rate_norm,
    process_session_windows,
)
from pipeline.ingestion.run import parse_events
from pipeline.outcome.cs2_demo import (
    CS2_DEFAULT_TICKRATE,
    DemoOutcomes,
    aggregate_outcome_windows,
    angular_speed_series,
    estimate_offset_by_xcorr,
    parse_demo_outcomes,
    recorder_mouse_speed_series,
)

log = logging.getLogger(__name__)

# peak cross-correlation at or above this → trust the alignment (matches the
# spike CLI's STRONG/WEAK threshold). Below it the join may be window-shifted.
SYNC_STRONG_CORR = 0.5


@dataclass
class SyncResult:
    """Outcome of the recorder↔demo clock-sync for one dual-capture session."""

    offset_s: float  # demo time (s) of recorder t=0 (the window anchor)
    peak_corr: float  # cross-correlation peak ∈ [-1, 1] — the self-validation
    lag_samples: int
    grid_hz: float
    player: str
    tickrate: float
    verdict: str  # "STRONG" | "WEAK — do not trust"
    n_input_windows: int
    n_outcome_windows: int
    n_joined_windows: int  # input windows that also have demo-outcome coverage

    def as_dict(self) -> dict:
        return asdict(self)


def most_active_player(outcomes: DemoOutcomes) -> str:
    """Player with the most shots fired — usually the human behind the recorder."""
    fires = outcomes.fires
    if "user_name" in fires.columns and len(fires):
        counts = fires["user_name"].value_counts()
        if len(counts):
            return str(counts.index[0])
    return outcomes.players[0] if outcomes.players else ""


def _events_df(data: dict, session_id: str) -> pd.DataFrame:
    """Recorder dict → the events DataFrame the feature pipeline consumes."""
    return parse_events({**data, "session_id": session_id})


def recorder_input_windows(
    data: dict, *, session_id: str | None = None
) -> pd.DataFrame:
    """One recorder session dict → per-window **input** features.

    Reuses the production feature path verbatim (``parse_events`` +
    ``process_session_windows``) so the window grid *and* the feature columns match
    the DVC pipeline exactly. Returns ``[session_id, window_idx] + <input features>``
    (one row per active window), or an empty frame if the session has no valid events.
    """
    sid = session_id or str(data.get("session_id", "session"))
    events_df = _events_df(data, sid)
    if events_df.empty:
        return pd.DataFrame(columns=["session_id", "window_idx"])
    sensitivity = float(data.get("sensitivity", 1.0))
    dpi = float(data.get("dpi", 800.0))
    norm_factor = max(sensitivity * dpi / 800.0, 1e-6)
    rate_norm = polling_rate_norm(data.get("polling_rate"))
    rows = process_session_windows(events_df, norm_factor, rate_norm)
    if not rows:
        return pd.DataFrame(columns=["session_id", "window_idx"])
    df = pd.DataFrame(rows)
    df.insert(0, "session_id", sid)
    return df


def _recorder_anchor_ms(data: dict, session_id: str) -> float | None:
    """The window anchor ``min(t)`` over all valid events (matches the feature grid)."""
    events_df = _events_df(data, session_id)
    return None if events_df.empty else float(events_df["t"].min())


def _first_mouse_move_ms(data: dict) -> float | None:
    """Timestamp (ms) of the session's first mouse-move event, or None."""
    ts = [
        float(e.get("t", 0.0))
        for e in data.get("events", [])
        if e.get("type") == "mouse_move"
    ]
    return min(ts) if ts else None


def sync_recorder_to_demo(
    data: dict, outcomes: DemoOutcomes, player: str, *, grid_hz: float = 16.0
) -> dict:
    """Cross-correlate recorder mouse motion vs demo view-angle motion → offset dict.

    The returned ``offset_s`` is the demo time of the **first mouse-move** (the
    recorder series' grid anchor). :func:`ingest_dual_capture` corrects it to the
    window anchor before binning — see the module docstring.
    """
    t_demo, v_demo = angular_speed_series(outcomes, player, grid_hz=grid_hz)
    t_rec, v_rec = recorder_mouse_speed_series(data.get("events", []), grid_hz=grid_hz)
    return estimate_offset_by_xcorr(t_demo, v_demo, t_rec, v_rec, grid_hz=grid_hz)


def join_input_outcome(in_df: pd.DataFrame, out_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join input ⟕ outcome on ``(session_id, window_idx)``.

    Every input window is kept. ``has_outcome`` marks **demo coverage** — windows
    the demo has telemetry for (player alive/aiming). Windows with no demo telemetry
    at all (player dead/disconnected the whole window → no outcome row) get
    ``has_outcome=False`` and their :data:`OUTCOME_FEATURE_COLS` filled with 0. Note
    a covered window can still be combat-free (aim motion but no shots) — filter
    **combat** windows via ``shots_fired > 0``, not ``has_outcome``.
    """
    if in_df.empty:
        cols = list(in_df.columns) + [*OUTCOME_FEATURE_COLS, "has_outcome"]
        return pd.DataFrame(columns=cols)
    keep = ["session_id", "window_idx", *OUTCOME_FEATURE_COLS]
    right = out_df[keep] if not out_df.empty else pd.DataFrame(columns=keep)
    merged = in_df.merge(
        right, on=["session_id", "window_idx"], how="left", indicator=True
    )
    merged["has_outcome"] = merged["_merge"] == "both"
    merged = merged.drop(columns="_merge")
    for c in OUTCOME_FEATURE_COLS:
        merged[c] = merged[c].fillna(0.0)
    return merged


def ingest_dual_capture(
    recorder: str | Path | dict,
    demo: str | Path | DemoOutcomes,
    *,
    player: str | None = None,
    tickrate: float = CS2_DEFAULT_TICKRATE,
    grid_hz: float = 16.0,
    session_id: str | None = None,
) -> tuple[pd.DataFrame, SyncResult]:
    """Ingest one dual-capture session → ``(joined per-window table, SyncResult)``.

    ``recorder`` is a recorder JSON path (or an already-loaded dict); ``demo`` is a
    ``.dem`` path (parsed lazily via ``demoparser2``) or a pre-parsed
    :class:`DemoOutcomes` (so this is unit-testable without the native parser).

    Pipeline: input windows (anchor = ``min(t)`` over all events) → parse demo →
    pick player (most-active if unnamed) → clock-sync **at the same anchor** →
    outcome windows at the recovered offset → left-join on
    ``(session_id, window_idx)``. The returned :class:`SyncResult` carries
    ``peak_corr`` (the sync self-validation) and a STRONG/WEAK verdict; a WEAK sync
    is logged as a warning — the join may be window-shifted and should not be trusted.
    """
    data = (
        json.loads(Path(recorder).read_text())
        if isinstance(recorder, (str, Path))
        else recorder
    )
    sid = session_id or str(data.get("session_id", "session"))

    in_df = recorder_input_windows(data, session_id=sid)
    anchor_ms = _recorder_anchor_ms(data, sid)

    outcomes = (
        parse_demo_outcomes(str(demo), tickrate=tickrate)
        if isinstance(demo, (str, Path))
        else demo
    )
    player = player or most_active_player(outcomes)

    sync = sync_recorder_to_demo(data, outcomes, player, grid_hz=grid_hz)
    # The xcorr offset is the demo time of the first mouse-move; shift it back to
    # the window anchor (min event t) so outcome window_idx aligns to input.
    first_mouse_ms = _first_mouse_move_ms(data)
    anchor_lead_s = (
        (float(first_mouse_ms) - float(anchor_ms)) / 1000.0
        if first_mouse_ms is not None and anchor_ms is not None
        else 0.0
    )
    offset_anchor_s = sync["offset_s"] - anchor_lead_s
    out_df = aggregate_outcome_windows(
        outcomes, player, offset_s=offset_anchor_s, session_id=sid
    )
    joined = join_input_outcome(in_df, out_df)

    verdict = (
        "STRONG" if sync["peak_corr"] >= SYNC_STRONG_CORR else "WEAK — do not trust"
    )
    result = SyncResult(
        offset_s=float(offset_anchor_s),
        peak_corr=float(sync["peak_corr"]),
        lag_samples=int(sync["lag_samples"]),
        grid_hz=float(grid_hz),
        player=str(player),
        tickrate=float(outcomes.tickrate),
        verdict=verdict,
        n_input_windows=int(len(in_df)),
        n_outcome_windows=int(len(out_df)),
        n_joined_windows=int(joined["has_outcome"].sum()) if len(joined) else 0,
    )
    if result.peak_corr < SYNC_STRONG_CORR:
        log.warning(
            "clock-sync peak_corr=%.3f < %.2f — alignment unreliable; the "
            "(session_id, window_idx) join may be window-shifted",
            result.peak_corr,
            SYNC_STRONG_CORR,
        )
    return joined, result
