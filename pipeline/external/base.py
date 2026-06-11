"""
pipeline/external/base.py
=========================
Shared machinery for public mouse-dynamics corpus adapters.

The contract: an adapter turns one corpus into an iterator of **recorder-schema
session dicts** (the exact shape `pipeline/ingestion/run.py:validate_session`
accepts), which `write_sessions` dumps to JSON so the normal DVC pipeline can
ingest them. Mouse-only corpora carry no keyboard events — train/evaluate the
identifier on `MOUSE_ID_FEATURE_COLS` (see `pipeline/features/run.py`).

``build_mouse_session`` (the envelope assembly) is concrete and unit-tested.
Per-corpus CSV parsing lives in the subclasses (``balabit``, ``sapimouse``)
and is the remaining Phase-6 work.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

# Mouse-dynamics corpora are mostly absolute (x, y) + timestamp. We emit the
# recorder's event shape: mouse_move carries (x, y) and consecutive deltas
# (dx, dy); mouse_click carries (button, pressed). The feature stage derives
# kinematics from these. Keep timestamps in milliseconds from session start.

# Minimum events / duration the ingestion validator requires (mirror those
# thresholds here so an adapter can skip too-short corpus sessions cleanly).
MIN_EVENTS = 100
MIN_DURATION_MS = 60_000.0


def build_mouse_session(
    *,
    session_id: str,
    player: str,
    mouse_events: Sequence[dict[str, Any]],
    game: str,
    recorded_at: str,
    sensitivity: float = 1.0,
    dpi: int = 800,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one recorder-schema session dict from ordered mouse events.

    ``mouse_events`` must already be in the recorder event shape and sorted by
    ``t`` (ms from session start): each item is e.g.
    ``{"t": 12.0, "type": "mouse_move", "x": 800, "y": 410, "dx": 3, "dy": -1}``
    or ``{"t": 30.0, "type": "mouse_click", "button": "left", "pressed": true}``.

    Returns a dict that passes ``ingestion.run.validate_session``. ``duration_ms``
    is taken from the last event's timestamp; ``event_count`` from len. The
    public corpora have no hardware metadata, so ``sensitivity``/``dpi`` default
    to the normalisation reference (1.0 / 800 → norm_factor 1.0).
    """
    events = list(mouse_events)
    duration_ms = float(events[-1]["t"]) if events else 0.0
    session: dict[str, Any] = {
        "session_id": session_id,
        "player": player,
        "game": game,
        "sensitivity": float(sensitivity),
        "dpi": int(dpi),
        "recorded_at": recorded_at,
        "duration_ms": duration_ms,
        "event_count": len(events),
        "events": events,
    }
    if extra:
        session.update(extra)
    return session


def write_sessions(sessions: Iterator[dict], out_dir: Path) -> int:
    """Write each session dict to ``out_dir/<session_id>.json``. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for sess in sessions:
        path = out_dir / f"{sess['session_id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sess, f)
        n += 1
    return n


class MouseCorpusAdapter(abc.ABC):
    """Base class for a public mouse-dynamics corpus → recorder-schema adapter.

    Subclass and implement :meth:`iter_sessions`. ``game`` becomes the session
    ``game`` field (a corpus tag, e.g. ``"balabit"``), which keeps external
    sessions distinguishable from GTA in the combined parquet.
    """

    #: Short corpus tag used as the session ``game`` field.
    game: str = "external"

    def __init__(self, src: Path, game: str | None = None) -> None:
        self.src = Path(src)
        if game is not None:
            self.game = game

    @abc.abstractmethod
    def iter_sessions(self) -> Iterator[dict[str, Any]]:
        """Yield one recorder-schema session dict per corpus session.

        Implementations parse ``self.src`` and call :func:`build_mouse_session`.
        """
        raise NotImplementedError

    def export(self, out_dir: Path) -> int:
        """Materialise all sessions as JSON under ``out_dir``. Returns count."""
        return write_sessions(self.iter_sessions(), out_dir)
