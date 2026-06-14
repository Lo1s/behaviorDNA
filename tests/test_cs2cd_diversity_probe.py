"""
tests/test_cs2cd_diversity_probe.py
===================================
Unit gate for the Phase 8.1 Step-0 player-diversity probe (CPU-fast, no network).

The network probes (``list_match_files`` / ``probe_match``) hit HuggingFace and
are exercised by actually running the script; here we lock the *pure* verdict
logic (``classify`` + the id-scheme helpers) on synthetic probe dicts so a
regression in the DIVERSE/THIN decision is caught offline.
"""

from __future__ import annotations

from scripts.cs2cd_diversity_probe import (
    _is_steam64,
    _looks_anonymized,
    classify,
)


# ---------------------------------------------------------------------------
# id-scheme helpers
# ---------------------------------------------------------------------------
def test_looks_anonymized_matches_player_n():
    assert _looks_anonymized(["Player_1", "Player_2", "Player_10"])
    assert _looks_anonymized(["player1", "PLAYER_3"])  # case-insensitive, optional "_"
    # a single real id among Player_N stays "mostly anonymised"
    assert _looks_anonymized(["Player_1", "Player_2", "Player_3", "76561198000000001"])


def test_looks_anonymized_rejects_steam64_and_empty():
    assert not _looks_anonymized(["76561198000000001", "76561198000000002"])
    assert not _looks_anonymized([])


def test_is_steam64():
    assert _is_steam64(["76561198000000001", "76561198000000002"])
    assert not _is_steam64(["Player_1"])
    assert not _is_steam64(["7656"])  # too short
    assert not _is_steam64([])


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
def _anon_probe(labels):
    return {
        "steamid_all": labels,
        "steamid_looks_anonymized": _looks_anonymized(labels),
        "steamid_is_steam64": _is_steam64(labels),
    }


def test_classify_player_thin_when_player_n_recurs_across_matches():
    # The same Player_1..Player_10 label set in every match → union saturates.
    labels = [f"Player_{i}" for i in range(1, 11)]
    probes = [_anon_probe(labels) for _ in range(12)]
    out = classify(probes)
    assert out["verdict"] == "PLAYER_THIN"
    assert out["ids_are_per_match_anonymized"] is True
    assert out["global_distinct_steamid_labels_in_sample"] == 10


def test_classify_player_diverse_when_real_steam64_union_grows():
    # Distinct real Steam64 ids per match → union grows well past saturation.
    probes = []
    for m in range(12):
        labels = [f"765611980000{m:02d}{p:03d}" for p in range(10)]
        probes.append(_anon_probe(labels))
    out = classify(probes)
    assert out["verdict"] == "PLAYER_DIVERSE"
    assert out["ids_are_real_steam64"] is True
    assert out["global_distinct_steamid_labels_in_sample"] == 120


def test_classify_defaults_to_thin_on_empty():
    out = classify([])
    assert out["verdict"] == "PLAYER_THIN"
    assert out["n_matches_sampled"] == 0
