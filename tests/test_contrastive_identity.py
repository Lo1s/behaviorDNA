"""
tests/test_contrastive_identity.py
==================================
Unit + smoke tests for Phase 6.1 (contrastive embeddings for player identity).

Synthetic recorder sessions only — no Balabit download needed. Covers the
corpus→8-D chunk bridge (idle-split + chunking), the scale-augmentation ablation
toggle, the EER-based verification logic, and an end-to-end eval smoke (build a
2-user synthetic corpus → embed with an untrained encoder → both scoring routes
return a well-formed session-EER).
"""

from __future__ import annotations

import numpy as np

from pipeline.external.sequences import session_to_chunks, session_to_segment_tensors
from pipeline.models.lstm_ae import LSTMAutoencoder
from pipeline.pretraining.augment import Augmenter
from pipeline.sequences.preprocessing import (
    COL_DX,
    COL_DY,
    EVENT_FEATURE_DIM,
    fit_normalizer,
)
from pipeline.verification import eer
from scripts.contrastive_identity import ENC, _cosine, _eval_encoder, _segment_tensors


def _synth_session(player, sid, *, dx_scale=5.0, n_bursts=1, seed=0):
    """A recorder-schema session: ``n_bursts`` continuous 40 s mouse-move bursts
    (≥ the 30 s split_on_idle floor) separated by 20 s idle gaps. dx magnitude
    encodes a crude per-user 'speed' so users are separable in principle."""
    rng = np.random.default_rng(seed)
    events = []
    t = 0.0
    for _ in range(n_bursts):
        for _k in range(400):  # 400 events × 100 ms = 40 s burst
            events.append(
                {
                    "type": "mouse_move",
                    "t": t,
                    "x": 500,
                    "y": 500,
                    "dx": int(round(dx_scale * rng.normal())),
                    "dy": int(round(dx_scale * rng.normal())),
                }
            )
            t += 100.0
        t += 20_000.0  # idle gap > 10 s → split_on_idle breaks here
    return {
        "session_id": sid,
        "player": player,
        "sensitivity": 1.0,
        "dpi": 800,
        "events": events,
    }


# --------------------------------------------------------------------------- #
# corpus → chunks
# --------------------------------------------------------------------------- #
def test_session_to_chunks_shape_and_grid():
    s = _synth_session("A", "a1")
    ch = session_to_chunks(s, chunk_length=64, stride=64)
    assert ch.ndim == 3 and ch.shape[1:] == (64, EVENT_FEATURE_DIM)
    assert ch.dtype == np.float32 and len(ch) > 0


def test_idle_split_separates_bursts():
    s = _synth_session("A", "a2", n_bursts=2)
    segs = session_to_segment_tensors(s)
    assert len(segs) == 2  # two 40 s bursts split at the 20 s gap
    assert all(t.shape[1] == EVENT_FEATURE_DIM for t in segs)


def test_short_session_yields_no_segments():
    # 10 events over ~1 s → below split_on_idle's 30 s min_segment floor
    s = {
        "player": "A",
        "session_id": "x",
        "events": [
            {"type": "mouse_move", "t": i * 100.0, "dx": 1, "dy": 0} for i in range(10)
        ],
    }
    assert session_to_segment_tensors(s) == []
    assert len(session_to_chunks(s)) == 0


# --------------------------------------------------------------------------- #
# scale-augmentation ablation toggle
# --------------------------------------------------------------------------- #
def test_scale_prob_zero_disables_scaling():
    chunk = np.zeros((64, EVENT_FEATURE_DIM), dtype=np.float32)
    chunk[:, COL_DX] = 2.0
    chunk[:, COL_DY] = 3.0
    rng = np.random.default_rng(0)
    # all ops off → identity
    off = Augmenter(scale_prob=0.0, jitter_prob=0.0, mask_prob=0.0, crop_prob=0.0)
    np.testing.assert_array_equal(off(chunk, rng), chunk)
    # only scale on → dx/dy multiplied by a single constant factor
    only_scale = Augmenter(
        scale_prob=1.0, jitter_prob=0.0, mask_prob=0.0, crop_prob=0.0
    )
    out = only_scale(chunk, np.random.default_rng(1))
    ratio = out[:, COL_DX] / chunk[:, COL_DX]
    assert np.allclose(ratio, ratio[0]) and not np.isclose(ratio[0], 1.0)


# --------------------------------------------------------------------------- #
# verification logic (EER)
# --------------------------------------------------------------------------- #
def test_cosine_is_unit_normalised_dot():
    a = np.array([3.0, 4.0])
    assert np.isclose(_cosine(a, a), 1.0)
    assert np.isclose(_cosine(a, np.array([4.0, -3.0])), 0.0)  # orthogonal


def test_eer_separable_vs_overlapping():
    genuine = np.array([0.9, 0.95, 0.88, 0.92])
    impostor = np.array([0.1, 0.2, 0.05, 0.15])
    sep_eer, _ = eer(genuine, impostor)
    assert sep_eer < 0.05  # cleanly separable → ~0
    rng = np.random.default_rng(0)
    g = rng.normal(0.5, 0.2, 200)
    i = rng.normal(0.5, 0.2, 200)  # same distribution → chance
    chance_eer, _ = eer(g, i)
    assert 0.35 < chance_eer < 0.65


# --------------------------------------------------------------------------- #
# end-to-end eval smoke (untrained encoder; checks plumbing + schema)
# --------------------------------------------------------------------------- #
def test_eval_routes_end_to_end_smoke():
    train = [
        _synth_session(u, f"{u}_tr{k}", dx_scale=sc, seed=k)
        for u, sc in (("A", 8.0), ("B", 2.0))
        for k in range(3)
    ]
    test = [
        (_synth_session("A", "A_te", dx_scale=8.0, seed=9), "A", False),  # genuine
        (_synth_session("B", "imp_te", dx_scale=2.0, seed=8), "A", True),  # impostor
    ]
    stats = fit_normalizer(_segment_tensors(train))
    model = LSTMAutoencoder(**ENC)  # untrained — only the plumbing matters here
    res = _eval_encoder(model, train, test, stats, "cpu")

    assert set(res) == {"cosine", "classifier"}
    for route in res.values():
        assert "session_eer" in route
        e = route["session_eer"]
        assert np.isnan(e) or 0.0 <= e <= 1.0
        assert route["n_genuine"] == 1 and route["n_impostor"] == 1
