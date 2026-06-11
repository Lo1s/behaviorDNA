"""
pipeline/external/sapimouse.py
==============================
Adapter for the **SapiMouse** dataset (120 users) — the scale claim: does the
windowed-feature identifier hold up at 100+ users?

Status: STUB. Envelope assembly is inherited; corpus-specific CSV parsing below
remains.

File format (to implement against)
----------------------------------
SapiMouse provides short mouse sessions per user (1-minute and 3-minute
protocols). Each session CSV row is one event::

    client timestamp, button, state, x, y

- ``client timestamp`` (ms) → session-relative ``t``.
- ``state`` ∈ {Move, Pressed, Released, Drag}; map as in the Balabit adapter.
- Sessions are short (1 min) — at the ingestion floor (≥60 s, ≥100 events), so
  expect to *keep the 3-minute sessions* and may need to relax/clearly document
  the threshold for the 1-minute set, or concatenate within-user.

TODO(Phase 6): implement ``_parse_session_csv`` + ``iter_sessions``; decide the
short-session policy and record it in docs/VERIFICATION.md. The large user count
is the point — make sure the users-curve goes to the full 120.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pipeline.external.base import MouseCorpusAdapter


class SapiMouseAdapter(MouseCorpusAdapter):
    """SapiMouse → recorder-schema sessions."""

    game = "sapimouse"

    def iter_sessions(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError(
            "SapiMouse parsing not implemented yet — see module docstring for "
            "the CSV format and the Phase-6 TODO."
        )
