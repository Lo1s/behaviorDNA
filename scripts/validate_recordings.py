"""
scripts/validate_recordings.py
==============================
Quality-control gate for incoming session recordings — run it **before**
`dvc repro` so a bad batch is caught at the door instead of halfway through
the pipeline.

It layers extra checks on top of the hard validation that ingestion already
does (`pipeline.ingestion.run.validate_session`):

  FAIL  — would break ingestion or training (missing required fields,
          corrupt event_count, empty/too-short session, bad JSON)
  WARN  — ingests fine but worth a look (missing/unknown activity label,
          mixed polling rates across the batch, non-monotonic timestamps,
          a player with fewer than min_sessions_per_player sessions,
          duration far from the ~6 min target)
  PASS  — clean

Exit code is 0 when there are no FAILs (1 otherwise), so this can gate a
future CI / pre-ingestion hook. With ``--strict`` warnings also fail.

Usage:
    python -m scripts.validate_recordings [--dir data/raw] [--strict]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from pipeline.ingestion.run import (
    VALID_EVENT_TYPES,
    parse_session_metadata,
    validate_session,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TRAINING_CONFIG = ROOT / "configs" / "training.yaml"

KNOWN_ACTIVITIES = {"on_foot", "driving", "combat", "sniping", "free_roam"}

# ~6 min recording target; flag sessions well outside a generous band
DURATION_MIN_S = 60.0
DURATION_MAX_S = 900.0


def _min_sessions_per_player(default: int = 3) -> int:
    """Read min_sessions_per_player from configs/training.yaml (best-effort)."""
    try:
        with open(TRAINING_CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return int(cfg.get("data", {}).get("min_sessions_per_player", default))
    except Exception:
        return default


def check_one(path: Path) -> dict:
    """Validate a single session JSON. Returns a result dict.

    Keys: file, status (PASS/WARN/FAIL), fails (list), warns (list),
    plus player / activity / polling_rate / duration_s for the summary.
    """
    fails: list[str] = []
    warns: list[str] = []
    player = activity = polling_rate = None
    duration_s = None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {
            "file": path.name,
            "status": "FAIL",
            "fails": [f"unreadable JSON: {e}"],
            "warns": [],
            "player": None,
            "activity": None,
            "polling_rate": None,
            "duration_s": None,
        }

    # Hard validation reused from ingestion (missing fields, empty/short, etc.)
    fails.extend(validate_session(data, path))

    # event_count integrity (catches truncated/corrupt files)
    events = data.get("events", [])
    declared = data.get("event_count")
    if declared is not None and declared != len(events):
        fails.append(f"event_count={declared} but len(events)={len(events)}")

    # Unknown event types
    if events:
        bad_types = {e.get("type") for e in events} - VALID_EVENT_TYPES
        if bad_types:
            warns.append(f"unknown event types: {sorted(bad_types)}")

        # Timestamp monotonicity (sorted, no large backward jumps)
        ts = [float(e.get("t", 0.0)) for e in events]
        backward = sum(1 for i in range(1, len(ts)) if ts[i] < ts[i - 1])
        if backward > 0:
            warns.append(f"{backward} out-of-order timestamps")

    # Metadata-derived checks (best-effort — only if required fields present)
    try:
        meta = parse_session_metadata(data, path)
        player = meta["player"]
        activity = meta["activity"]
        polling_rate = meta["polling_rate"]
        duration_s = meta["duration_ms"] / 1000.0
    except (KeyError, ValueError, TypeError):
        # Required-field problems already captured by validate_session
        pass

    # Activity label present + known
    if activity is None:
        warns.append("missing activity label")
    elif activity not in KNOWN_ACTIVITIES:
        warns.append(
            f"unknown activity '{activity}' (expected {sorted(KNOWN_ACTIVITIES)})"
        )

    # Polling rate present
    if polling_rate is None:
        warns.append("missing polling_rate (features won't be rate-normalised)")

    # Duration sanity
    if duration_s is not None:
        if duration_s < DURATION_MIN_S:
            warns.append(f"short session: {duration_s:.0f}s (< {DURATION_MIN_S:.0f}s)")
        elif duration_s > DURATION_MAX_S:
            warns.append(f"long session: {duration_s:.0f}s (> {DURATION_MAX_S:.0f}s)")

    status = "FAIL" if fails else ("WARN" if warns else "PASS")
    return {
        "file": path.name,
        "status": status,
        "fails": fails,
        "warns": warns,
        "player": player,
        "activity": activity,
        "polling_rate": polling_rate,
        "duration_s": duration_s,
    }


def validate_dir(directory: Path) -> list[dict]:
    """Validate every *.json in a directory + cross-file batch checks."""
    files = sorted(directory.glob("*.json"))
    results = [check_one(p) for p in files]

    # --- Cross-file batch checks ---
    # Polling-rate consistency
    rates = {r["polling_rate"] for r in results if r["polling_rate"] is not None}
    if len(rates) > 1:
        for r in results:
            r["warns"].append(f"mixed polling rates in batch: {sorted(rates)}")
            if r["status"] == "PASS":
                r["status"] = "WARN"

    # Per-player session count vs min_sessions_per_player
    min_sessions = _min_sessions_per_player()
    per_player: dict[str, int] = defaultdict(int)
    for r in results:
        if r["player"]:
            per_player[r["player"]] += 1
    for r in results:
        p = r["player"]
        if p and per_player[p] < min_sessions:
            r["warns"].append(
                f"player '{p}' has {per_player[p]} session(s) (< min {min_sessions} — "
                "would be dropped by the split stage)"
            )
            if r["status"] == "PASS":
                r["status"] = "WARN"

    return results


def _print_report(results: list[dict], strict: bool) -> int:
    """Print a per-file table + summary. Return process exit code."""
    print(f"\n{'STATUS':6}  {'FILE':52}  DETAILS")
    print("-" * 100)
    for r in results:
        detail = (
            "; ".join(r["fails"] + r["warns"]) if (r["fails"] or r["warns"]) else "ok"
        )
        print(f"{r['status']:6}  {r['file']:52.52}  {detail}")

    n_pass = sum(r["status"] == "PASS" for r in results)
    n_warn = sum(r["status"] == "WARN" for r in results)
    n_fail = sum(r["status"] == "FAIL" for r in results)

    print("\n=== Summary ===")
    print(f"  {len(results)} files: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")

    rates = Counter(r["polling_rate"] for r in results if r["polling_rate"] is not None)
    if rates:
        print(f"  Polling rates: {dict(rates)}")
    players = Counter(r["player"] for r in results if r["player"])
    if players:
        print(f"  Sessions per player: {dict(players)}")
    activities = Counter(r["activity"] for r in results if r["activity"])
    if activities:
        print(f"  Activities: {dict(activities)}")

    failed = n_fail > 0 or (strict and n_warn > 0)
    if failed:
        reason = "FAILs present" if n_fail else "warnings present (--strict)"
        print(f"\n  ✗ Validation failed ({reason}).")
        return 1
    print("\n  ✓ Validation passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QC-validate session recordings")
    parser.add_argument("--dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    if not args.dir.exists():
        log.error("Directory not found: %s", args.dir)
        return 1

    results = validate_dir(args.dir)
    if not results:
        log.warning("No *.json files in %s", args.dir)
        return 0

    return _print_report(results, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
