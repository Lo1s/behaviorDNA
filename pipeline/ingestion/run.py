"""
pipeline/ingestion/run.py
=========================
Stage 1 — Ingestion: raw JSON session files → structured Parquet.

Reads all session JSON files from data/raw/, validates and normalises them,
and writes two Parquet files:

  data/processed/sessions.parquet   — one row per session (metadata)
  data/processed/events.parquet     — one row per event (all sessions combined)

Run via DVC:
    dvc repro ingest

Or directly:
    python -m pipeline.ingestion.run
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SESSIONS_OUT = PROCESSED_DIR / "sessions.parquet"
EVENTS_OUT = PROCESSED_DIR / "events.parquet"

SESSION_FIELDS = {
    "session_id": str,
    "player": str,
    "game": str,
    "sensitivity": float,
    "dpi": int,
    "recorded_at": str,
    "duration_ms": float,
    "event_count": int,
    "events": list,
}

VALID_EVENT_TYPES = {
    "mouse_move",
    "mouse_click",
    "mouse_scroll",
    "key_press",
    "key_release",
}


def validate_session(data: dict, filepath: Path) -> list:
    errors = []
    for field, expected_type in SESSION_FIELDS.items():
        if field not in data:
            errors.append(f"Missing field: {field}")
            continue
        if not isinstance(data[field], expected_type):
            errors.append(
                f"Field '{field}' expected {expected_type.__name__}, "
                f"got {type(data[field]).__name__}"
            )
    if "events" in data:
        if len(data["events"]) == 0:
            errors.append("Empty events list")
        elif len(data["events"]) < 100:
            errors.append(f"Suspiciously few events: {len(data['events'])}")
    if "duration_ms" in data and data.get("duration_ms", 0) < 60_000:
        errors.append(f"Session too short: {data['duration_ms']/1000:.1f}s (min 60s)")
    return errors


def parse_session_metadata(data: dict, filepath: Path) -> dict:
    return {
        "session_id": data["session_id"],
        "player": data["player"].strip().lower(),
        "game": data["game"].strip().lower().replace(" ", "_"),
        "activity": data.get("activity"),
        "sensitivity": float(data["sensitivity"]),
        "dpi": int(data["dpi"]),
        "recorded_at": pd.to_datetime(data["recorded_at"], utc=True),
        "duration_ms": float(data["duration_ms"]),
        "event_count": int(data["event_count"]),
        "source_file": filepath.name,
    }


def parse_events(data: dict) -> pd.DataFrame:
    rows = []
    session_id = data["session_id"]
    for ev in data["events"]:
        event_type = ev.get("type", "unknown")
        if event_type not in VALID_EVENT_TYPES:
            continue
        rows.append(
            {
                "session_id": session_id,
                "t": float(ev.get("t", 0)),
                "event_type": event_type,
                "x": ev.get("x"),
                "y": ev.get("y"),
                "dx": ev.get("dx"),
                "dy": ev.get("dy"),
                "button": ev.get("button"),
                "pressed": ev.get("pressed"),
                "key": ev.get("key"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["t"] = df["t"].astype("float32")
    for col in ["x", "y", "dx", "dy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")
    return df


def run() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(RAW_DIR.glob("*.json"))
    if not json_files:
        log.error("No JSON files found in %s", RAW_DIR)
        sys.exit(1)

    log.info("Found %d session file(s) in %s", len(json_files), RAW_DIR)

    session_rows = []
    event_frames = []
    skipped = 0

    for filepath in json_files:
        log.info("Reading %s", filepath.name)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            log.warning("  Skipping — JSON parse error: %s", e)
            skipped += 1
            continue

        errors = validate_session(data, filepath)
        if errors:
            log.warning("  Skipping — validation errors:")
            for err in errors:
                log.warning("    * %s", err)
            skipped += 1
            continue

        session_rows.append(parse_session_metadata(data, filepath))
        events_df = parse_events(data)
        if events_df.empty:
            log.warning("  No valid events parsed — skipping")
            skipped += 1
            continue

        event_frames.append(events_df)
        log.info(
            "  OK  player=%-12s  game=%-12s  events=%d  duration=%.1fs",
            data["player"],
            data["game"],
            len(events_df),
            data["duration_ms"] / 1000,
        )

    if not session_rows:
        log.error("No valid sessions after validation. Exiting.")
        sys.exit(1)

    sessions_df = pd.DataFrame(session_rows)
    sessions_df.to_parquet(SESSIONS_OUT, index=False)
    log.info(
        "Wrote sessions: %s  (%d rows, %d skipped)",
        SESSIONS_OUT,
        len(sessions_df),
        skipped,
    )

    events_all = pd.concat(event_frames, ignore_index=True)
    events_all.to_parquet(EVENTS_OUT, index=False)
    log.info(
        "Wrote events:   %s  (%d rows across %d sessions)",
        EVENTS_OUT,
        len(events_all),
        len(session_rows),
    )

    log.info("")
    log.info("=== Ingestion summary ===")
    log.info("  Sessions ingested : %d", len(sessions_df))
    log.info("  Sessions skipped  : %d", skipped)
    log.info("  Total events      : %d", len(events_all))
    log.info("  Players           : %s", sorted(sessions_df["player"].unique()))
    log.info("  Games             : %s", sorted(sessions_df["game"].unique()))
    log.info("  Event types       : %s", sorted(events_all["event_type"].unique()))


if __name__ == "__main__":
    run()
