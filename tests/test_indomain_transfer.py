"""
tests/test_indomain_transfer.py
===============================
Unit gate for the Phase 8.1 in-domain transfer grid (CPU-fast).

Locks the parts that are easy to get subtly wrong: the frozen-encoder arm
(``_apply_arm`` freezes encoder + to_bottleneck and trains only the decoder; the
frozen weights must not move during training), the dt-neutralisation transform,
the config enumeration, and a tiny end-to-end ``_run_one`` smoke. The full grid
(real GTA pool + GPU pretraining) is exercised by actually running the script.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.models.lstm_ae import LSTMAutoencoder
from pipeline.sequences.preprocessing import COL_DT, EVENT_FEATURE_DIM
from scripts.data_efficiency import CHUNK
from scripts.indomain_transfer import (
    CAPTCHA_ENCODER,
    RAW,
    _apply_arm,
    _configs,
    _gta_pool,
    _run_one,
    _zero_dt,
)


def _tiny(**kw):
    return LSTMAutoencoder(hidden_dim=16, bottleneck_dim=8, num_layers=1, **kw)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def test_apply_arm_frozen_freezes_encoder_only(tmp_path):
    pth = tmp_path / "pre.pt"
    torch.save(_tiny().state_dict(), pth)
    m = _tiny()
    trainable = _apply_arm(m, "frozen", pth)
    assert all(not p.requires_grad for p in m.encoder.parameters())
    assert all(not p.requires_grad for p in m.to_bottleneck.parameters())
    assert all(p.requires_grad for p in m.decoder.parameters())
    assert all(p.requires_grad for p in m.from_decoder.parameters())
    n_dec = len(list(m.decoder.parameters())) + len(list(m.from_decoder.parameters()))
    assert trainable is not None and len(trainable) == n_dec


def test_apply_arm_scratch_and_finetune_all_trainable(tmp_path):
    m = _tiny()
    assert _apply_arm(m, "scratch", None) is None
    assert all(p.requires_grad for p in m.parameters())

    pth = tmp_path / "pre.pt"
    torch.save(_tiny().state_dict(), pth)
    m2 = _tiny()
    assert _apply_arm(m2, "finetune", pth) is None
    assert all(p.requires_grad for p in m2.parameters())


def test_frozen_encoder_weights_unchanged_after_train(tmp_path):
    from torch.utils.data import DataLoader

    from scripts.benchmark_cs2cd_ae import _ChunkDataset
    from scripts.compare_architectures import train_ae

    pth = tmp_path / "pre.pt"
    torch.save(_tiny().state_dict(), pth)
    m = _tiny()
    trainable = _apply_arm(m, "frozen", pth)
    enc_before = [p.detach().clone() for p in m.encoder.parameters()]
    dec_before = [p.detach().clone() for p in m.decoder.parameters()]

    chunks = (
        np.random.default_rng(0)
        .standard_normal((32, CHUNK, EVENT_FEATURE_DIM))
        .astype(np.float32)
    )
    loader = DataLoader(_ChunkDataset(chunks), batch_size=8, shuffle=True)
    train_ae(m, loader, loader, epochs=2, lr=1e-2, trainable_params=trainable)

    enc_after = list(m.encoder.parameters())
    dec_after = list(m.decoder.parameters())
    assert all(torch.equal(b, a.detach().cpu()) for b, a in zip(enc_before, enc_after))
    assert any(
        not torch.equal(b, a.detach().cpu()) for b, a in zip(dec_before, dec_after)
    )


# ---------------------------------------------------------------------------
# dt-neutralisation + config enumeration
# ---------------------------------------------------------------------------
def test_zero_dt_zeros_only_dt_channel():
    t = (
        np.random.default_rng(0)
        .standard_normal((10, EVENT_FEATURE_DIM))
        .astype(np.float32)
    )
    z = _zero_dt(t)
    assert np.allclose(z[:, COL_DT], 0.0)
    assert np.allclose(
        z[:, COL_DT + 1 :], t[:, COL_DT + 1 :]
    )  # other channels untouched
    assert not np.shares_memory(z, t)


def test_configs_enumeration():
    labels = [c[0] for c in _configs()]
    assert "scratch·native" in labels and "scratch·dt0" in labels
    assert any(label.startswith("cs2cd_s1·") for label in labels)
    assert any(label.startswith("cs2cd_s2·") for label in labels)
    # captcha config present iff the Phase-8 artifact is on disk
    assert (
        any(label.startswith("captcha·") for label in labels)
        == CAPTCHA_ENCODER.exists()
    )


# ---------------------------------------------------------------------------
# one fine-tune+eval run (synthetic; scratch arm so no encoder file needed)
# ---------------------------------------------------------------------------
def test_run_one_smoke():
    rng = np.random.default_rng(0)
    units = [
        rng.standard_normal((300, EVENT_FEATURE_DIM)).astype(np.float32)
        for _ in range(3)
    ]
    eval_legit = [
        rng.standard_normal((CHUNK, EVENT_FEATURE_DIM)).astype(np.float32)
        for _ in range(6)
    ]
    eval_cheat = [
        (rng.standard_normal((CHUNK, EVENT_FEATURE_DIM)) * 4).astype(np.float32)
        for _ in range(6)
    ]
    r = _run_one(
        units,
        eval_legit,
        eval_cheat,
        init="scratch",
        pretrained_path=None,
        budget=2,
        seed=0,
        epochs=1,
        lr=1e-3,
    )
    assert r is not None
    assert 0.0 <= r["auc"] <= 1.0
    assert r["budget"] == 2 and r["n_train_chunks"] > 0


# ---------------------------------------------------------------------------
# real GTA pool (only if recordings are present)
# ---------------------------------------------------------------------------
def test_gta_pool_smoke():
    if not list(RAW.glob("*.json")):
        pytest.skip("no GTA recordings present")
    units, eval_legit, eval_cheat = _gta_pool()
    assert len(units) > 0 and len(eval_cheat) > 0  # legit training + cheat eval chunks
    # dt-neutralised pool zeros the dt channel of its tensors
    units_z, _, _ = _gta_pool(neutralize_dt=True)
    assert np.allclose(units_z[0][:, COL_DT], 0.0)
