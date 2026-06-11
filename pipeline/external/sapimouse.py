"""
pipeline/external/sapimouse.py
==============================
Adapter for the **SapiMouse** dataset (120 users) — the scale claim: does the
windowed-feature identifier hold up at 100+ users?

Layout (sapimouse.zip from ms.sapientia.ro, expected under
``data/external/sapimouse/``)::

    sapimouse/user<N>/session_<date>_1min.csv
    sapimouse/user<N>/session_<date>_3min.csv

Each row::

    client timestamp, button, state, x, y

``client timestamp`` is **milliseconds** (int, starts at an arbitrary offset)
→ shifted to session-relative. ``state`` ∈ {Move, Drag, Pressed, Released};
no scroll events — mapped by ``base.rows_to_recorder_events``.

Protocol note: every user has exactly one 3-minute and one 1-minute session.
The SapiMouse paper's own split — **train on the 3-min session, test on the
1-min session** — is what the Phase-6 experiment uses (session-held-out by
construction). Sessions carry a ``protocol`` field ("1min" / "3min") so the
runner can split without filename parsing.
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

RECORDED_AT = "2020-05-14T00:00:00Z"  # collection date per the filenames


def _parse_session_csv(path: Path) -> list[dict[str, Any]]:
    """One SapiMouse session CSV → recorder events (t in ms from start)."""
    rows: list[tuple[float, str, str, int, int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t_ms = float(row["client timestamp"])
                x = int(float(row["x"]))
                y = int(float(row["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= x <= MAX_COORD and 0 <= y <= MAX_COORD):
                continue  # sentinel glitch rows, e.g. (65535, 65535)
            rows.append((t_ms, row["button"], row["state"], x, y))
    if not rows:
        return []
    rows.sort(key=lambda r: r[0])
    t0 = rows[0][0]
    return rows_to_recorder_events(
        (t - t0, button, state, x, y) for t, button, state, x, y in rows
    )


class SapiMouseAdapter(MouseCorpusAdapter):
    """SapiMouse → recorder-schema sessions (with a ``protocol`` tag)."""

    game = "sapimouse"

    def iter_sessions(self) -> Iterator[dict[str, Any]]:
        root = self.src / "sapimouse"
        if not root.exists():  # allow pointing directly at the unpacked dir
            root = self.src
        for user_dir in sorted(root.glob("user*")):
            player = user_dir.name
            for session_file in sorted(user_dir.glob("session_*.csv")):
                protocol = "1min" if session_file.stem.endswith("1min") else "3min"
                events = _parse_session_csv(session_file)
                if len(events) < MIN_EVENTS:
                    log.debug("skip %s: %d events", session_file.name, len(events))
                    continue
                yield build_mouse_session(
                    session_id=f"sapimouse_{player}_{session_file.stem}",
                    player=player,
                    mouse_events=events,
                    game=self.game,
                    recorded_at=RECORDED_AT,
                    extra={"protocol": protocol},
                )
