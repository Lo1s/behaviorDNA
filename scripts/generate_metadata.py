"""
scripts/generate_metadata.py
============================
Single source of truth for the repo's *structural* facts — the brittle numbers
that live scattered across the README + CLAUDE.md and silently drift apart (test
count, dashboard tab count, notebook/doc counts).

The "Results at a glance" block already solves metric drift by generating itself
from ``reports/*.json`` (``scripts/generate_results.py``). This is the same idea
for facts that are *computable from the git-tracked tree itself*:

  1. ``compute_metadata()`` derives the facts from the source (no heavy deps —
     it greps files, so it runs in pre-commit and CI alike).
  2. it writes them to ``reports/repo_metadata.json`` (the committed SoT), and
  3. it rewrites a handful of *owned spans* in the prose so a stale number can
     never be committed: each owned span is a regex that must match exactly once
     (a no-match means the prose changed shape and the generator went stale — we
     raise rather than silently no-op).

CI runs ``--check`` to fail the build whenever the docs or the JSON no longer
match the tree, exactly like the results block.

Usage:
    python -m scripts.generate_metadata            # rewrite docs + JSON
    python -m scripts.generate_metadata --check    # exit 1 if stale (CI)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "repo_metadata.json"

_TEST_DEF = re.compile(r"^\s*(?:async\s+)?def\s+test_\w*\s*\(", re.MULTILINE)


def _count_test_functions() -> int:
    """Number of ``def test_*`` functions under tests/ — a dependency-free proxy
    for the suite size (pytest collects a few more once parametrize expands)."""
    total = 0
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        total += len(_TEST_DEF.findall(path.read_text()))
    return total


def _count_dashboard_tabs() -> int:
    """Count the labels passed to the dashboard's ``st.tabs([...])`` call."""
    src = (ROOT / "dashboard" / "app.py").read_text()
    m = re.search(r"st\.tabs\(\s*\[(.*?)\]", src, re.S)
    if m is None:
        raise SystemExit(
            "generate_metadata: could not find st.tabs([...]) in dashboard/app.py"
        )
    return len(re.findall(r"""["'][^"']*["']""", m.group(1)))


def compute_metadata() -> dict:
    """Derive the structural facts from the git-tracked tree. Deterministic."""
    return {
        "n_test_functions": _count_test_functions(),
        "n_test_files": len(list((ROOT / "tests").rglob("test_*.py"))),
        "n_notebooks": len(list((ROOT / "notebooks").glob("*.ipynb"))),
        "n_docs": len(list((ROOT / "docs").glob("*.md"))),
        "n_dashboard_tabs": _count_dashboard_tabs(),
    }


def serialize_report(meta: dict) -> str:
    """Stable JSON for the committed report (matches the pre-commit EOF fixer)."""
    return json.dumps(meta, indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class OwnedSpan:
    """A single prose fact this script keeps in sync. ``pattern`` must match
    exactly once in ``path``; ``render`` produces the replacement from metadata."""

    path: Path
    name: str
    pattern: re.Pattern
    render: Callable[[dict], str]


# Only the facts that have *demonstrably* drifted get a rewritten span; the rest
# are still guarded by the JSON-in-sync check below (so a 6th dashboard tab or a
# new notebook can't ship without the report noticing).
SPANS: list[OwnedSpan] = [
    OwnedSpan(
        path=ROOT / "README.md",
        name="readme_test_count",
        pattern=re.compile(r"\*\*\d+\+? tests\*\*"),
        render=lambda m: f"**{m['n_test_functions']} tests**",
    ),
    OwnedSpan(
        path=ROOT / "CLAUDE.md",
        name="claudemd_test_count",
        pattern=re.compile(r"# full suite \(~?\d+ tests\)"),
        render=lambda m: f"# full suite ({m['n_test_functions']} tests)",
    ),
]


def render_text(span: OwnedSpan, text: str, meta: dict) -> str:
    """Apply one owned span. Raises if the anchor pattern no longer matches —
    that means the prose was reworded and the generator silently stopped owning
    the number, which is the failure mode this whole script exists to prevent."""
    n = len(span.pattern.findall(text))
    if n != 1:
        raise SystemExit(
            f"generate_metadata: span {span.name!r} matched {n} times in "
            f"{span.path.relative_to(ROOT)} (expected exactly 1) — the prose "
            f"changed shape; update the pattern in scripts/generate_metadata.py."
        )
    replacement = span.render(meta)
    # Function replacement: not subject to backslash / \g<n> interpretation.
    return span.pattern.sub(lambda _m: replacement, text)


def _spans_by_path() -> dict[Path, list[OwnedSpan]]:
    out: dict[Path, list[OwnedSpan]] = {}
    for span in SPANS:
        out.setdefault(span.path, []).append(span)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any doc or the report is out of date instead of rewriting",
    )
    args = parser.parse_args(argv)

    meta = compute_metadata()
    rendered_report = serialize_report(meta)

    # (path -> (current_text, desired_text)) for every file we own.
    desired: dict[Path, tuple[str, str]] = {}
    for path, spans in _spans_by_path().items():
        text = path.read_text()
        new = text
        for span in spans:
            new = render_text(span, new, meta)
        desired[path] = (text, new)
    desired[REPORT] = (
        REPORT.read_text() if REPORT.exists() else "",
        rendered_report,
    )

    if args.check:
        stale = [p for p, (cur, new) in desired.items() if cur != new]
        if stale:
            names = ", ".join(str(p.relative_to(ROOT)) for p in stale)
            print(
                f"Repo metadata is stale ({names}) — run "
                "`python -m scripts.generate_metadata` and commit.",
                file=sys.stderr,
            )
            return 1
        print("Repo metadata is up to date.")
        return 0

    changed = []
    for path, (cur, new) in desired.items():
        if cur != new:
            path.write_text(new)
            changed.append(str(path.relative_to(ROOT)))
    print(
        "Repo metadata regenerated"
        + (f" ({', '.join(changed)})." if changed else " — already up to date.")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
