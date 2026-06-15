"""
scripts/check_dvc_deps.py
=========================
Guard against the "DVC stage reuses stale artifacts" failure mode (portfolio
review finding H2).

Each ``dvc.yaml`` stage declares an explicit list of file ``deps``. DVC only
re-runs a stage when one of those declared deps changes — so if the stage's
entrypoint *imports* a first-party module that is **not** declared, a change to
that imported module is invisible to ``dvc repro`` and the committed artifacts
silently go stale. This directly undercuts the repo's "generated results are the
source of truth" story.

This script statically computes, for every stage whose command is
``python -m pipeline.<...>``, the transitive **first-party** import closure of
its entrypoint (via ``ast`` — no imports are executed, so it is safe and cheap)
and asserts that every module in that closure is covered by the stage's declared
``deps`` (either listed directly, or under a declared directory dep). A missing
dep fails the build.

First-party = the resolved module file lives inside the repo root (i.e. not a
site-packages / stdlib module).

Usage:
    python -m scripts.check_dvc_deps            # report + exit 1 on any gap
    python -m scripts.check_dvc_deps --check    # alias (same behaviour; CI)
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DVC_YAML = ROOT / "dvc.yaml"


def _module_to_file(module: str) -> Path | None:
    """Resolve a dotted module name to a first-party file under ROOT, or None.

    Tries ``a/b/c.py`` then the package ``a/b/c/__init__.py``. Returns None when
    the module is third-party / stdlib (no matching file under the repo root).
    """
    parts = module.split(".")
    candidate = ROOT.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    pkg_init = ROOT.joinpath(*parts) / "__init__.py"
    if pkg_init.is_file():
        return pkg_init
    return None


def _imported_modules(file_path: Path) -> set[str]:
    """All dotted module names referenced by ``import`` / ``from`` in a file.

    For ``from a.b import c`` we yield both ``a.b`` and ``a.b.c`` so that
    submodule imports (``from pipeline.features import run``) resolve. AST walks
    nested nodes too, so function-local imports (e.g. the lazy ONNX import in
    ``pipeline/training/run.py``) are captured.
    """
    tree = ast.parse(file_path.read_text(), filename=str(file_path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # skip relative imports (none in this codebase)
                continue
            if node.module:
                modules.add(node.module)
                for alias in node.names:
                    modules.add(f"{node.module}.{alias.name}")
    return modules


def first_party_closure(entry_file: Path) -> set[Path]:
    """Transitive set of first-party files reachable from ``entry_file``.

    Includes ``entry_file`` itself. Recursion stops at third-party modules
    (``_module_to_file`` returns None for them).
    """
    seen: set[Path] = set()
    stack: list[Path] = [entry_file]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for module in _imported_modules(current):
            resolved = _module_to_file(module)
            if resolved is not None and resolved not in seen:
                stack.append(resolved)
    return seen


def _entrypoint_module(cmd: str) -> str | None:
    """Extract ``pipeline.x.y`` from a ``python -m pipeline.x.y`` command."""
    tokens = cmd.split()
    for i, tok in enumerate(tokens):
        if tok == "-m" and i + 1 < len(tokens):
            module = tokens[i + 1]
            return module if module.startswith("pipeline") else None
    return None


def _declared_py_deps(deps: list[str]) -> tuple[set[Path], list[Path]]:
    """Split declared deps into (.py files, directories), resolved to abs paths."""
    files: set[Path] = set()
    dirs: list[Path] = []
    for dep in deps or []:
        path = (ROOT / dep).resolve()
        if dep.endswith(".py"):
            files.add(path)
        elif path.is_dir():
            dirs.append(path)
    return files, dirs


def _covered(
    file_path: Path, declared_files: set[Path], declared_dirs: list[Path]
) -> bool:
    if file_path in declared_files:
        return True
    return any(d in file_path.parents for d in declared_dirs)


def check() -> int:
    spec = yaml.safe_load(DVC_YAML.read_text())
    stages = spec.get("stages", {})
    gaps: list[str] = []

    for name, stage in stages.items():
        cmd = stage.get("cmd", "")
        module = _entrypoint_module(cmd)
        if module is None:
            continue  # not a `python -m pipeline...` stage; nothing to check
        entry_file = _module_to_file(module)
        if entry_file is None:
            gaps.append(f"[{name}] cannot resolve entrypoint module {module!r}")
            continue

        closure = first_party_closure(entry_file)
        declared_files, declared_dirs = _declared_py_deps(stage.get("deps", []))

        for dep_file in sorted(closure):
            if not _covered(dep_file, declared_files, declared_dirs):
                rel = dep_file.relative_to(ROOT).as_posix()
                gaps.append(
                    f"[{name}] imports {rel} (transitively) but it is not a declared dep"
                )

    if gaps:
        print("DVC dependency gaps found (stale-artifact risk):\n", file=sys.stderr)
        for gap in gaps:
            print(f"  - {gap}", file=sys.stderr)
        print(
            "\nAdd the missing module(s) to the stage's `deps:` in dvc.yaml "
            "(or use a directory-level dep).",
            file=sys.stderr,
        )
        return 1

    print("dvc.yaml deps cover every first-party import closure. ✓")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # --check accepted for symmetry with the other generate_* CI gates; behaviour
    # is identical (this script never writes, it only verifies).
    parser.add_argument("--check", action="store_true", help="alias; same behaviour")
    parser.parse_args()
    return check()


if __name__ == "__main__":
    sys.exit(main())
