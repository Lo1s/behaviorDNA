"""
pipeline/external/balabit.py
============================
Adapter for the **Balabit Mouse Dynamics Challenge** dataset (10 users) — the
classic mouse-dynamics benchmark, so reported EER is literature-comparable.

Status: STUB. The session-envelope assembly is inherited from
``base.MouseCorpusAdapter`` / ``build_mouse_session``; only the corpus-specific
CSV parsing below remains.

File format (to implement against)
----------------------------------
Balabit ships per-user directories of session CSV files. Each row is one mouse
event::

    record timestamp, client timestamp, button, state, x, y

- ``client timestamp`` (seconds, float) → session-relative ms for ``t``.
- ``state`` ∈ {Move, Pressed, Released, Drag, Down, Up}; ``button`` ∈
  {NoButton, Left, Right, Scroll}.
- Map Move/Drag → ``mouse_move`` (compute dx/dy as consecutive position
  deltas); Pressed/Down → ``mouse_click`` pressed=true; Released/Up →
  ``mouse_click`` pressed=false; Scroll → ``mouse_scroll``.
- The challenge has a *legit* training set per user and a *test* set with an
  ``is_illegal`` label per session (impostor) — useful directly for the Phase-6
  **verification / open-set** evaluation, not just closed-set ID.

TODO(Phase 6): implement ``_parse_session_csv`` + ``iter_sessions`` and verify
against ``ingestion.run.validate_session`` (≥100 events, ≥60 s). Drop sessions
that fail the thresholds.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pipeline.external.base import MouseCorpusAdapter


class BalabitAdapter(MouseCorpusAdapter):
    """Balabit Mouse Dynamics Challenge → recorder-schema sessions."""

    game = "balabit"

    def iter_sessions(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError(
            "Balabit parsing not implemented yet — see module docstring for the "
            "CSV format and the Phase-6 TODO."
        )
