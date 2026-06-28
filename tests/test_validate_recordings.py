"""
tests/test_validate_recordings.py
=================================
Unit tests for scripts/validate_recordings.py — the pre-ingestion QC gate.
"""

from __future__ import annotations

import json

from scripts.validate_recordings import check_one, main, validate_dir


def _good_session(
    n_events: int = 500,
    player: str = "hydra",
    activity: str = "combat",
    polling_rate: int = 1000,
    duration_ms: float = 360_000.0,
) -> dict:
    events = [
        {
            "t": float(i * (duration_ms / n_events)),
            "type": "mouse_move",
            "x": 100 + i % 50,
            "y": 200,
            "dx": 1,
            "dy": 0,
        }
        for i in range(n_events)
    ]
    return {
        "session_id": "abcd1234",
        "player": player,
        "game": "gta5",
        "activity": activity,
        "polling_rate": polling_rate,
        "sensitivity": 0.5,
        "dpi": 800,
        "recorded_at": "2026-05-28T10:00:00+00:00",
        "duration_ms": duration_ms,
        "event_count": n_events,
        "events": events,
    }


def _write(tmp_path, name: str, data: dict):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# check_one
# ---------------------------------------------------------------------------


class TestCheckOne:
    def test_clean_session_passes(self, tmp_path):
        p = _write(tmp_path, "good.json", _good_session())
        r = check_one(p)
        assert r["status"] == "PASS", r["fails"] + r["warns"]

    def test_missing_activity_warns(self, tmp_path):
        s = _good_session()
        del s["activity"]
        p = _write(tmp_path, "no_activity.json", s)
        r = check_one(p)
        assert r["status"] == "WARN"
        assert any("activity" in w for w in r["warns"])

    def test_unknown_activity_warns(self, tmp_path):
        s = _good_session(activity="parkour")
        p = _write(tmp_path, "weird.json", s)
        r = check_one(p)
        assert r["status"] == "WARN"
        assert any("unknown activity" in w for w in r["warns"])

    def test_corrupt_event_count_fails(self, tmp_path):
        s = _good_session(n_events=500)
        s["event_count"] = 999  # mismatch
        p = _write(tmp_path, "corrupt.json", s)
        r = check_one(p)
        assert r["status"] == "FAIL"
        assert any("event_count" in f for f in r["fails"])

    def test_missing_required_field_fails(self, tmp_path):
        s = _good_session()
        del s["dpi"]
        p = _write(tmp_path, "no_dpi.json", s)
        r = check_one(p)
        assert r["status"] == "FAIL"

    def test_long_session_warns(self, tmp_path):
        # A short session would hard-FAIL ingestion's 60s floor, so to isolate
        # the duration-WARN path we use an over-long session (> 900s).
        s = _good_session(duration_ms=901_000.0)
        p = _write(tmp_path, "long.json", s)
        r = check_one(p)
        assert r["status"] == "WARN"
        assert any("long session" in w for w in r["warns"])

    def test_unreadable_json_fails(self, tmp_path):
        p = tmp_path / "broken.json"
        p.write_text("{not valid json")
        r = check_one(p)
        assert r["status"] == "FAIL"
        assert any("JSON" in f for f in r["fails"])

    def test_non_positive_sensitivity_fails(self, tmp_path):
        # sensitivity <= 0 → norm_factor 0/negative → div-by-zero / sign flip.
        s = _good_session()
        s["sensitivity"] = 0
        p = _write(tmp_path, "zero_sens.json", s)
        r = check_one(p)
        assert r["status"] == "FAIL"
        assert any("sensitivity" in f for f in r["fails"])

    def test_non_positive_dpi_fails(self, tmp_path):
        s = _good_session()
        s["dpi"] = -1
        p = _write(tmp_path, "neg_dpi.json", s)
        r = check_one(p)
        assert r["status"] == "FAIL"
        assert any("dpi" in f for f in r["fails"])


# ---------------------------------------------------------------------------
# validate_dir — cross-file batch checks
# ---------------------------------------------------------------------------


class TestValidateDir:
    def test_mixed_polling_rates_warn(self, tmp_path):
        _write(tmp_path, "a.json", _good_session(player="hydra", polling_rate=1000))
        _write(tmp_path, "b.json", _good_session(player="royik", polling_rate=125))
        results = validate_dir(tmp_path)
        assert all(any("mixed polling rates" in w for w in r["warns"]) for r in results)

    def test_consistent_polling_no_mix_warning(self, tmp_path):
        _write(tmp_path, "a.json", _good_session(player="hydra", polling_rate=1000))
        _write(tmp_path, "b.json", _good_session(player="royik", polling_rate=1000))
        results = validate_dir(tmp_path)
        assert not any(
            any("mixed polling rates" in w for w in r["warns"]) for r in results
        )

    def test_too_few_sessions_per_player_warns(self, tmp_path):
        # Single session for one player → below min_sessions_per_player (default 3)
        _write(tmp_path, "solo.json", _good_session(player="loner"))
        results = validate_dir(tmp_path)
        assert any(any("would be dropped" in w for w in r["warns"]) for r in results)

    def test_sensitivity_outlier_warns(self, tmp_path):
        # The ropyk bug: one session's sensitivity is a 100x units typo (0.25 vs
        # the batch's 25.0). The outlier should WARN; the in-range ones shouldn't.
        for i, sens in enumerate([25.0, 25.0, 0.25]):
            s = _good_session(player="hydra")
            s["sensitivity"] = sens
            _write(tmp_path, f"s{i}.json", s)
        results = validate_dir(tmp_path)
        outliers = [r for r in results if r["sensitivity"] == 0.25]
        in_range = [r for r in results if r["sensitivity"] == 25.0]
        assert outliers and all(
            any("off the batch median" in w for w in r["warns"]) for r in outliers
        )
        assert in_range and not any(
            any("off the batch median" in w for w in r["warns"]) for r in in_range
        )

    def test_normal_sensitivity_spread_no_warn(self, tmp_path):
        # A genuine spread inside the 10x factor must not false-positive.
        for i, sens in enumerate([20.0, 25.0, 30.0]):
            s = _good_session(player="hydra")
            s["sensitivity"] = sens
            _write(tmp_path, f"s{i}.json", s)
        results = validate_dir(tmp_path)
        assert not any(
            any("off the batch median" in w for w in r["warns"]) for r in results
        )


# ---------------------------------------------------------------------------
# main — exit codes
# ---------------------------------------------------------------------------


class TestMainExitCodes:
    def test_exit_zero_on_clean_batch(self, tmp_path):
        # 3 sessions for one player so the min-sessions check passes too
        for i in range(3):
            _write(tmp_path, f"s{i}.json", _good_session(player="hydra"))
        assert main(["--dir", str(tmp_path)]) == 0

    def test_exit_one_on_fail(self, tmp_path):
        s = _good_session()
        s["event_count"] = 1  # corrupt → FAIL
        _write(tmp_path, "bad.json", s)
        assert main(["--dir", str(tmp_path)]) == 1

    def test_strict_treats_warn_as_fail(self, tmp_path):
        # One session for a lone player → WARN (too few sessions). Non-strict
        # passes (exit 0); strict fails (exit 1).
        _write(tmp_path, "solo.json", _good_session(player="loner"))
        assert main(["--dir", str(tmp_path)]) == 0
        assert main(["--dir", str(tmp_path), "--strict"]) == 1
