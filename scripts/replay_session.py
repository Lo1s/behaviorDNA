"""
scripts/replay_session.py
=========================
Replay a recorded session JSON through the streaming pipeline, optionally
overlaying a synthetic cheat at a configured timestamp.

Two transport modes:

- **WebSocket** (default): connects to a running API at ``--api-url`` and
  pushes events with optional wall-clock pacing. The server returns
  ``ScoreUpdate`` snapshots which we write to ``--out`` as JSON Lines.

- **Offline**: with ``--offline``, the script drives a local
  ``SessionStreamState`` directly — no server, no network. Useful in
  tests, in the dashboard's launcher, and for the demo-artifact generator
  in ``scripts/build_phase4_demo.py``.

Usage::

    # Replay one session through a running API
    python -m scripts.replay_session data/raw/<file>.json \
        --speed 5 \
        --out /tmp/replay_scores.jsonl

    # Inject an aimbot at t=180s (after a 180s warm-up of the legit baseline)
    python -m scripts.replay_session data/raw/<file>.json \
        --inject-cheat aimbot \
        --inject-at 180 \
        --out /tmp/replay_scores.jsonl

    # Same but offline (no server)
    python -m scripts.replay_session data/raw/<file>.json \
        --offline --inject-cheat aimbot --inject-at 30 \
        --out /tmp/demo_scores.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cheat injection
# ---------------------------------------------------------------------------


def _maybe_inject_cheat(
    session: dict, cheat_type: str | None, inject_at_s: float | None
) -> dict:
    """If a cheat type is configured, slice the session and overlay synthetic events.

    We restrict the cheat to the portion of the session after ``inject_at_s``
    seconds — so the live demo shows a "clean" baseline ramp-up followed by
    a clear spike when the cheat starts. We do this by:

    1. Splitting the session events into before/after the inject timestamp.
    2. Building a tiny synthetic "after" session and running the chosen bot
       generator on it.
    3. Re-attaching the cheat events to the original "before" events.

    The result is a fully-formed session dict carrying ``cheat_label`` and
    ``cheat_segments`` so the dashboard knows when to draw the highlight.
    """
    if cheat_type is None or inject_at_s is None:
        # No injection — passthrough
        session = dict(session)
        session["cheat_label"] = "legit"
        session["cheat_segments"] = []
        return session

    from pipeline.adversarial.bot_generator import CheatSpec, inject_cheat

    inject_at_ms = float(inject_at_s) * 1000.0
    events = session.get("events", [])
    if not events:
        return session

    before = [e for e in events if e.get("t", 0.0) < inject_at_ms]
    after = [e for e in events if e.get("t", 0.0) >= inject_at_ms]
    if not after:
        log.warning(
            "Cheat inject_at=%.1fs is past session end (%.1fs) — skipping injection",
            inject_at_s,
            events[-1].get("t", 0.0) / 1000.0,
        )
        return session

    # Build a temporary session for the "after" portion and apply the generator
    tmp_session = dict(session)
    tmp_session["events"] = after
    spec = CheatSpec(cheat_type=cheat_type)
    tmp_cheat = inject_cheat(tmp_session, spec, new_session_id=False)

    out = dict(session)
    out["events"] = before + tmp_cheat["events"]
    out["event_count"] = len(out["events"])
    out["duration_ms"] = max(e["t"] for e in out["events"]) if out["events"] else 0.0
    out["cheat_label"] = tmp_cheat.get("cheat_label", cheat_type)
    out["cheat_segments"] = tmp_cheat.get("cheat_segments", [])
    return out


# ---------------------------------------------------------------------------
# Offline replay (no server)
# ---------------------------------------------------------------------------


def replay_offline(
    session: dict,
    *,
    speed: float = 5.0,
    out_path: Path | None = None,
    progress: Iterable[int] | None = None,
    state=None,
) -> list[dict]:
    """Drive a SessionStreamState directly. Returns the list of ScoreUpdates.

    Pass ``state`` to reuse a pre-built ``SessionStreamState`` (e.g. in tests,
    in the dashboard launcher, or in the demo-artifact generator) and avoid
    the ~45 s setup cost of ``build_stream_state``. With ``state=None`` (the
    default), one is built lazily.

    ``speed`` is informational only in offline mode (no sleeps); the field is
    kept for API parity with the WebSocket replay.
    """
    if state is None:
        from pipeline.inference.streaming import build_stream_state

        state = build_stream_state()
    updates: list[dict] = []
    out_f = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for i, ev in enumerate(session.get("events", [])):
            update = state.push_event(ev)
            if update is None:
                continue
            d = update.to_dict()
            updates.append(d)
            if out_f is not None:
                out_f.write(json.dumps(d) + "\n")
                out_f.flush()
            if progress is not None and i in progress:
                log.info(
                    "  event %d/%d  risk=%.3f",
                    i,
                    len(session["events"]),
                    update.session_risk,
                )
        final = state.finalize()
        if final is not None:
            d = final.to_dict()
            updates.append(d)
            if out_f is not None:
                out_f.write(json.dumps(d) + "\n")
                out_f.flush()
    finally:
        if out_f is not None:
            out_f.close()
    return updates


# ---------------------------------------------------------------------------
# WebSocket replay
# ---------------------------------------------------------------------------


async def replay_websocket(
    session: dict,
    *,
    api_url: str,
    speed: float = 5.0,
    out_path: Path | None = None,
) -> list[dict]:
    """Stream events over a WebSocket to a running API."""
    try:
        import websockets
    except ImportError as e:
        raise RuntimeError(
            "websockets package not installed — install with `pip install websockets`"
        ) from e

    updates: list[dict] = []
    out_f = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        async with websockets.connect(api_url) as ws:
            events = session.get("events", [])
            if not events:
                return updates
            t0 = float(events[0].get("t", 0.0))
            wall_start = asyncio.get_event_loop().time()
            recv_task = asyncio.create_task(_receive_loop(ws, out_f, updates))
            for ev in events:
                # Optional pacing: real-time when speed=1, faster as speed grows,
                # 0 means as-fast-as-possible (no sleeps).
                if speed > 0:
                    target_dt = (float(ev.get("t", t0)) - t0) / (1000.0 * speed)
                    elapsed = asyncio.get_event_loop().time() - wall_start
                    if target_dt > elapsed:
                        await asyncio.sleep(target_dt - elapsed)
                await ws.send(json.dumps(ev))
            # Signal end-of-stream
            await ws.send(json.dumps({"type": "__end__"}))
            try:
                await asyncio.wait_for(recv_task, timeout=5.0)
            except asyncio.TimeoutError:
                recv_task.cancel()
    finally:
        if out_f is not None:
            out_f.close()
    return updates


async def _receive_loop(ws, out_f, updates: list[dict]) -> None:
    try:
        async for msg in ws:
            try:
                d = json.loads(msg)
            except json.JSONDecodeError:
                continue
            updates.append(d)
            if out_f is not None:
                out_f.write(json.dumps(d) + "\n")
                out_f.flush()
    except Exception:
        return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a session through the streaming pipeline."
    )
    parser.add_argument("session_path", type=Path, help="Path to a session JSON")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run the stream engine in-process instead of using a WebSocket",
    )
    parser.add_argument("--api-url", default="ws://localhost:8000/stream")
    parser.add_argument(
        "--speed",
        type=float,
        default=5.0,
        help="Playback speed: 1.0 = real-time, 0.0 = as fast as possible (default 5×)",
    )
    parser.add_argument(
        "--inject-cheat",
        choices=["aimbot", "triggerbot", "macro"],
        default=None,
    )
    parser.add_argument(
        "--inject-at", type=float, default=None, help="Seconds into session"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Write JSONL of ScoreUpdates here"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    with open(args.session_path, encoding="utf-8") as f:
        session = json.load(f)
    log.info(
        "Loaded session: %s  events=%d  duration=%.1fs",
        args.session_path.name,
        session.get("event_count", len(session.get("events", []))),
        session.get("duration_ms", 0.0) / 1000.0,
    )

    session = _maybe_inject_cheat(session, args.inject_cheat, args.inject_at)
    if args.inject_cheat:
        log.info(
            "Injected cheat: %s starting at t=%.1fs  → %d segments",
            args.inject_cheat,
            args.inject_at or 0.0,
            len(session.get("cheat_segments", [])),
        )

    if args.offline:
        updates = replay_offline(session, speed=args.speed, out_path=args.out)
    else:
        updates = asyncio.run(
            replay_websocket(
                session, api_url=args.api_url, speed=args.speed, out_path=args.out
            )
        )

    log.info("Received %d ScoreUpdate snapshots", len(updates))
    if updates:
        last = updates[-1]
        log.info(
            "Final: session_risk=%.3f  n_events=%d  n_windows=%d  n_chunks=%d",
            last.get("session_risk", float("nan")),
            last.get("n_events", -1),
            last.get("n_windows", -1),
            last.get("n_chunks", -1),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
