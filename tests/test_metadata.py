"""
tests/test_metadata.py
======================
The repo's structural facts (test count, dashboard tabs, notebook/doc counts)
are generated from the git-tracked tree by scripts/generate_metadata.py and
gated in CI with --check. These tests pin the contract: the computed facts match
the tree independently, the owned spans still anchor, and the committed docs +
report are in sync (so a stale number fails locally, not just in CI).
"""

import ast
import json
from pathlib import Path

from scripts.generate_metadata import (
    REPORT,
    SPANS,
    compute_metadata,
    main,
    serialize_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_computed_facts_match_tree_independently():
    meta = compute_metadata()
    # Recount via the AST (a genuinely different method than the generator's
    # line regex, and immune to "def test_" appearing inside a string literal).
    n_defs = 0
    for p in (ROOT / "tests").rglob("test_*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test_"):
                n_defs += 1
    assert meta["n_test_functions"] == n_defs
    assert meta["n_notebooks"] == len(list((ROOT / "notebooks").glob("*.ipynb")))
    assert meta["n_docs"] == len(list((ROOT / "docs").glob("*.md")))
    assert meta["n_dashboard_tabs"] >= 1


def test_owned_spans_still_anchor():
    # If the prose is reworded so a pattern no longer matches, the generator
    # silently stops owning the number — this guards against that no-op drift.
    for span in SPANS:
        matches = span.pattern.findall(span.path.read_text())
        assert len(matches) == 1, f"span {span.name} matched {len(matches)} times"


def test_committed_report_is_in_sync():
    assert REPORT.exists(), "reports/repo_metadata.json missing — run generate_metadata"
    assert REPORT.read_text() == serialize_report(compute_metadata())
    on_disk = json.loads(REPORT.read_text())
    assert on_disk == compute_metadata()


def test_check_passes_on_committed_tree():
    # Mirrors the CI gate: the committed docs + report already match the tree.
    assert main(["--check"]) == 0
