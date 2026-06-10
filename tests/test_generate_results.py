"""
tests/test_generate_results.py
==============================
The README "Results at a glance" block is generated from reports/*.json
(scripts/generate_results.py) and gated in CI with --check. These tests pin
the contract: the rendered block carries the pipeline's numbers and the
render is idempotent.
"""

import json
from pathlib import Path

from scripts.generate_results import (
    BEGIN,
    END,
    build_results_markdown,
    render_readme,
)

ROOT = Path(__file__).resolve().parents[1]


def test_block_carries_eval_metrics():
    with open(ROOT / "reports" / "eval_metrics.json") as f:
        ev = json.load(f)
    block = build_results_markdown()
    assert block.startswith(BEGIN)
    assert block.endswith(END)
    assert f"**{ev['test_accuracy']:.2f}** acc" in block
    lo, hi = ev["test_accuracy_ci95"]
    assert f"95% CI {lo:.2f}–{hi:.2f}" in block


def test_render_is_idempotent():
    once = render_readme((ROOT / "README.md").read_text())
    assert render_readme(once) == once
    assert once.count(BEGIN) == 1 and once.count(END) == 1
