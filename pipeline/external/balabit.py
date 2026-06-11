"""
pipeline/external/balabit.py
============================
Adapter for the **Balabit Mouse Dynamics Challenge** dataset (10 users) — the
classic mouse-dynamics benchmark, so reported EER is literature-comparable.

Layout (as shipped by github.com/balabit/Mouse-Dynamics-Challenge, expected
under ``data/external/balabit/``)::

    training_files/user<N>/session_<id>     legit sessions, one CSV per session
    test_files/user<N>/session_<id>         sessions *claimed* to be user<N>
    public_labels.csv                       filename,is_illegal for the public
                                            subset of test sessions (1 = impostor)

Each session CSV row::

    record timestamp, client timestamp, button, state, x, y

``client timestamp`` is **seconds** (float, starts near 0) → converted to
session-relative milliseconds. ``state`` ∈ {Move, Drag, Pressed, Released,
Down, Up}; ``button`` ∈ {NoButton, Left, Right, Scroll} — mapped by
``base.rows_to_recorder_events`` (Down/Up are scroll ticks).

Training sessions feed closed-set identification; the labelled test sessions
give *real* genuine/impostor pairs for the Phase-6 verification task (EER on
"is this session really user<N>?").
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pipeline.external.base import (
    MAX_COORD,
    MIN_EVENTS,
    MouseCorpusAdapter,
    build_mouse_session,
    rows_to_recorder_events,
)

log = logging.getLogger(__name__)

# The corpus has no recording dates; use a fixed placeholder so sessions are
# valid recorder JSON without inventing fake per-session times.
RECORDED_AT = "2016-01-01T00:00:00Z"


def _parse_session_csv(path: Path) -> list[dict[str, Any]]:
    """One Balabit session CSV → recorder events (t in ms from session start).

    Rows are sorted by client timestamp before delta computation (the raw
    files are occasionally non-monotonic).
    """
    rows: list[tuple[float, str, str, int, int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t_s = float(row["client timestamp"])
                x = int(float(row["x"]))
                y = int(float(row["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= x <= MAX_COORD and 0 <= y <= MAX_COORD):
                continue  # sentinel glitch rows, e.g. (65535, 65535)
            rows.append((t_s, row["button"], row["state"], x, y))
    if not rows:
        return []
    rows.sort(key=lambda r: r[0])
    t0 = rows[0][0]
    return rows_to_recorder_events(
        ((t - t0) * 1000.0, button, state, x, y) for t, button, state, x, y in rows
    )


class BalabitAdapter(MouseCorpusAdapter):
    """Balabit Mouse Dynamics Challenge → recorder-schema sessions."""

    game = "balabit"

    def iter_sessions(self) -> Iterator[dict[str, Any]]:
        """Yield the **training** (legit, user-labelled) sessions.

        Sessions with fewer than MIN_EVENTS parsed events are skipped (some
        raw files are near-empty); the ≥60 s duration floor is left to the
        ingestion validator / experiment runner, which decide per use case.
        """
        train_dir = self.src / "training_files"
        for user_dir in sorted(train_dir.glob("user*")):
            player = user_dir.name
            for session_file in sorted(user_dir.glob("session_*")):
                events = _parse_session_csv(session_file)
                if len(events) < MIN_EVENTS:
                    log.debug("skip %s: %d events", session_file.name, len(events))
                    continue
                yield build_mouse_session(
                    session_id=f"balabit_{player}_{session_file.name}",
                    player=player,
                    mouse_events=events,
                    game=self.game,
                    recorded_at=RECORDED_AT,
                )

    def labels(self) -> dict[str, bool]:
        """public_labels.csv → {session filename: is_impostor}."""
        out: dict[str, bool] = {}
        with open(self.src / "public_labels.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["filename"]] = row["is_illegal"] == "1"
        return out

    def iter_test_sessions(self) -> Iterator[tuple[dict[str, Any], str, bool]]:
        """Yield (session, claimed_player, is_impostor) for labelled test files.

        ``claimed_player`` is the directory the session sits under (whose
        identity it claims); ``is_impostor`` comes from public_labels.csv.
        Unlabelled test sessions (outside the public subset) are skipped.
        """
        labels = self.labels()
        test_dir = self.src / "test_files"
        for user_dir in sorted(test_dir.glob("user*")):
            claimed = user_dir.name
            for session_file in sorted(user_dir.glob("session_*")):
                if session_file.name not in labels:
                    continue
                events = _parse_session_csv(session_file)
                if len(events) < MIN_EVENTS:
                    continue
                session = build_mouse_session(
                    session_id=f"balabit_test_{claimed}_{session_file.name}",
                    player=claimed,
                    mouse_events=events,
                    game=self.game,
                    recorded_at=RECORDED_AT,
                )
                yield session, claimed, labels[session_file.name]
