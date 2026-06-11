"""
pipeline/external/
==================
Adapters that map public mouse-dynamics corpora into BehaviorDNA's recorder
JSON schema, so the existing pipeline (ingestion → features → split → train →
eval) runs on them unchanged — the same drop-in trick the synthetic cheat
generator uses (CLAUDE.md design choice #8).

Phase 6 (docs/ROADMAP.md): scale the identification claim beyond 3 friends and
reframe it as verification / open-set on Balabit (10 users) + SapiMouse (120).

Status: SCAFFOLD. ``base.build_mouse_session`` (envelope assembly) is concrete
and tested; the per-corpus event parsing in ``balabit`` / ``sapimouse`` is
stubbed with the file-format contract documented inline.
"""

from pipeline.external.base import (
    MouseCorpusAdapter,
    build_mouse_session,
    write_sessions,
)

__all__ = ["MouseCorpusAdapter", "build_mouse_session", "write_sessions"]
