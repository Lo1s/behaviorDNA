"""
tests/test_check_dvc_deps.py
============================
Guards the H2 fix: every DVC stage's declared deps must cover the transitive
first-party import closure of its entrypoint, so a code change cannot leave
``dvc repro`` reusing stale artifacts.
"""

from __future__ import annotations

import textwrap

from scripts import check_dvc_deps
from scripts.check_dvc_deps import ROOT, check, first_party_closure


def test_current_dvc_yaml_passes():
    """The committed dvc.yaml must already cover every import closure."""
    assert check() == 0


def test_closure_captures_transitive_and_lazy_imports():
    """The train entrypoint's closure includes constants, features, and the
    function-local lazy ONNX import — exactly the deps H2 added."""
    closure = {
        p.relative_to(ROOT).as_posix()
        for p in first_party_closure(ROOT / "pipeline/training/run.py")
    }
    assert "pipeline/constants.py" in closure
    assert "pipeline/features/run.py" in closure
    assert "pipeline/onnx_export.py" in closure  # lazy import inside export_onnx


def test_missing_dep_is_detected(tmp_path, monkeypatch, capsys):
    """A stage that omits an imported first-party module must fail the check."""
    fake_yaml = tmp_path / "dvc.yaml"
    fake_yaml.write_text(textwrap.dedent("""\
            stages:
              features:
                cmd: python -m pipeline.features.run
                deps:
                  - pipeline/features/run.py
                outs:
                  - data/processed/features.parquet
            """))
    # pipeline/features/run.py imports pipeline/constants.py, which the fake
    # stage deliberately omits.
    monkeypatch.setattr(check_dvc_deps, "DVC_YAML", fake_yaml)
    assert check() == 1
    assert "pipeline/constants.py" in capsys.readouterr().err
