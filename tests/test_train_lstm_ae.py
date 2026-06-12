"""
tests/test_train_lstm_ae.py
===========================
Guard tests for scripts/train_lstm_ae.py — specifically that the training-data
loader keeps cheat sessions out of the legit-manifold LSTM-AE.
"""

import json

import scripts.train_lstm_ae as trainer


def _write_session(path, *, cheat: bool) -> None:
    data = {
        "session_id": path.stem,
        "player": "tester",
        "game": "gta5",
        "sensitivity": 1.0,
        "dpi": 800,
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "duration_ms": 120_000.0,
        "event_count": 5,
        "events": [
            {
                "t": i * 10.0,
                "type": "mouse_move",
                "x": 100 + i,
                "y": 200,
                "dx": 1,
                "dy": 0,
            }
            for i in range(5)
        ],
    }
    if cheat:
        data["cheat_label"] = "aimbot"
        data["cheat_segments_typed"] = [[1000, 2000, "aimbot"]]
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_legit_tensors_skips_cheat_sessions(tmp_path, monkeypatch):
    _write_session(tmp_path / "legit_a.json", cheat=False)
    _write_session(tmp_path / "legit_b.json", cheat=False)
    _write_session(tmp_path / "cheat_c.json", cheat=True)
    monkeypatch.setattr(trainer, "RAW_DIR", tmp_path)

    tensors, names = trainer._load_legit_tensors()

    assert names == ["legit_a.json", "legit_b.json"]
    assert "cheat_c.json" not in names
    assert len(tensors) == 2
