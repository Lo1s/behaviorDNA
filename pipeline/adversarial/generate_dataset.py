"""
pipeline/adversarial/generate_dataset.py
========================================
Generate a labelled synthetic-cheat dataset on top of existing legit sessions.

Reads every JSON file in ``data/raw/`` (recursively), and for each one writes:

  * 1 copy with no cheat injected  (label = ``legit``)
  * 1 aimbot variant per difficulty (``obvious``, ``medium``, ``soft``)
  * 1 triggerbot variant
  * 1 macro variant

Each output session is written to ``data/synthetic/`` with the ``cheat_label``
and ``cheat_segments`` fields populated. File names encode the cheat type:

    <original_basename>_<cheat_label>.json

Run:
    python -m pipeline.adversarial.generate_dataset
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from pipeline.adversarial.bot_generator import (
    CHEAT_AIMBOT,
    CHEAT_LEGIT,
    CHEAT_MACRO,
    CHEAT_TRIGGERBOT,
    CheatSpec,
    inject_cheat,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "synthetic"

# Difficulty presets — higher smoothing = harder to detect
AIMBOT_PRESETS = {
    "obvious": dict(smoothing=0.0, snap_duration_ms=120.0, target_fraction=1.0),
    "medium": dict(smoothing=0.5, snap_duration_ms=150.0, target_fraction=1.0),
    "soft": dict(smoothing=0.85, snap_duration_ms=180.0, target_fraction=0.7),
}

TRIGGERBOT_PRESET = dict(reaction_time_ms=3.0, target_fraction=1.0)

MACRO_PRESET = dict(interval_ms=200.0, duration_ms=8_000.0, start_fraction=0.35)


def _save(session: dict, src: Path, label: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{src.stem}_{label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(session, f, separators=(",", ":"))
    return out_path


def _passthrough_legit(session: dict) -> dict:
    """Add a cheat_label tag to a legit session without changing events."""
    session = dict(session)
    session["cheat_label"] = CHEAT_LEGIT
    session["cheat_segments"] = []
    return session


def process_one(src: Path) -> list[Path]:
    with open(src, encoding="utf-8") as f:
        session = json.load(f)

    written: list[Path] = []

    # 1. Legit copy
    written.append(_save(_passthrough_legit(session), src, "legit"))

    # 2. Aimbot at 3 difficulties
    for difficulty, params in AIMBOT_PRESETS.items():
        hybrid = inject_cheat(session, CheatSpec(CHEAT_AIMBOT, params))
        written.append(_save(hybrid, src, f"aimbot_{difficulty}"))

    # 3. Triggerbot
    hybrid = inject_cheat(session, CheatSpec(CHEAT_TRIGGERBOT, TRIGGERBOT_PRESET))
    written.append(_save(hybrid, src, "triggerbot"))

    # 4. Macro
    hybrid = inject_cheat(session, CheatSpec(CHEAT_MACRO, MACRO_PRESET))
    written.append(_save(hybrid, src, "macro"))

    return written


def run() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    # Only flat .json files in data/raw/ (skip subdirs: mock/, real_data/, cheat/).
    # Top-level is the legit dataset, so synthetic cheats are injected only into
    # legit recordings — never into the real cheat sessions in cheat/.
    sources = sorted(p for p in RAW_DIR.glob("*.json"))
    if not sources:
        log.warning("No JSON files in %s — nothing to do", RAW_DIR)
        return

    log.info("Generating synthetic dataset from %d legit session(s)", len(sources))

    total = 0
    for src in sources:
        try:
            written = process_one(src)
            total += len(written)
            log.info("  %s → %d variants", src.name, len(written))
        except Exception as e:
            log.error("  %s FAILED: %s", src.name, e)

    log.info("Done. Wrote %d files to %s", total, OUT_DIR)


if __name__ == "__main__":
    run()
