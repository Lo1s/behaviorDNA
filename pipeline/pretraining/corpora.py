"""
pipeline/pretraining/corpora.py
===============================
Map the external corpora onto the **shared 8-D event-tensor schema** so the
captcha-pretrained encoder transfers cleanly into CS2CD / GTA fine-tuning.

The schema is the one from :mod:`pipeline.sequences.preprocessing`:

    [dt, dx, dy, is_mouse_move, is_mouse_click_press,
     is_mouse_scroll, is_key_press, is_key_release]

GTA is already native (``session_to_event_tensor``). The two external corpora
here are **sampled** streams (one mouse sample per physics/engine tick) rather
than GTA's **event** stream, so we adopt one convention for both:

- every tick is a movement sample → ``is_mouse_move = 1`` with ``dx/dy`` deltas,
- ``is_mouse_click_press = 1`` on the **rising edge** of the button (co-occurs
  with the move — the model takes 8 floats, it does not require one-hot
  exclusivity),
- ``dt = log1p(ms_per_tick)`` (near-constant — these are fixed-rate captures),
- scroll / keyboard channels are 0.

This sampled-vs-event distinction is itself part of the captcha/CS2→GTA domain
gap (GTA's ``dt`` and ``is_mouse_move`` vary; here they are near-constant) and
is surfaced honestly by ``scripts/domain_gap_report.py``.

Scale is **not** normalised here — each domain is z-scored on its own train
fold downstream (``fit_normalizer``), so the residual gap lives in temporal /
geometric shape, not raw units.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.sequences.preprocessing import (
    COL_DT,
    COL_DX,
    COL_DY,
    COL_IS_MOUSE_CLICK_PRESS,
    COL_IS_MOUSE_MOVE,
    EVENT_FEATURE_DIM,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CAPTCHA_PARQUET = ROOT / "data" / "external" / "captcha30k" / "captcha30k.parquet"
CS2CD_PARQUET = ROOT / "data" / "external" / "cs2cd" / "cs2cd_balanced_25000.parquet"

# CS2 demos are recorded per engine tick; the exact rate barely matters because
# dt is constant within a contiguous run and gets z-scored to ~0 downstream.
CS2_MS_PER_TICK = 1000.0 / 64.0

# Channels worth comparing in the domain-gap report (scroll/key are all-zero
# for the sampled corpora → not meaningful to KS/PSI).
DRIFT_CHANNELS = ["dt", "dx", "dy", "is_mouse_move", "is_mouse_click_press"]
_CHANNEL_NAMES = [
    "dt",
    "dx",
    "dy",
    "is_mouse_move",
    "is_mouse_click_press",
    "is_mouse_scroll",
    "is_key_press",
    "is_key_release",
]


def _sampled_stream_to_tensor(
    x: np.ndarray, y: np.ndarray, down: np.ndarray, ms_per_tick: float
) -> np.ndarray:
    """Build an ``(N, 8)`` event tensor from a per-tick sampled mouse stream.

    ``x``/``y`` are absolute positions (or aim deltas already), ``down`` a
    boolean/0-1 button state. Deltas are first differences; the click channel
    fires on rising edges of ``down``.
    """
    n = len(x)
    out = np.zeros((n, EVENT_FEATURE_DIM), dtype=np.float32)
    if n == 0:
        return out

    dx = np.diff(x, prepend=x[:1]).astype(np.float32)
    dy = np.diff(y, prepend=y[:1]).astype(np.float32)
    out[:, COL_DX] = dx
    out[:, COL_DY] = dy
    out[:, COL_IS_MOUSE_MOVE] = 1.0

    # Constant tick interval; first row has no predecessor → dt 0 (matches
    # session_to_event_tensor's convention for the first event).
    out[1:, COL_DT] = np.log1p(max(ms_per_tick, 0.0))

    down = np.asarray(down).astype(bool)
    rising = np.zeros(n, dtype=bool)
    rising[1:] = down[1:] & ~down[:-1]
    rising[0] = down[0]  # a stream that starts pressed counts as a press
    out[rising, COL_IS_MOUSE_CLICK_PRESS] = 1.0
    return out


# ---------------------------------------------------------------------------
# CaptchaSolve30k (pretraining corpus)
# ---------------------------------------------------------------------------


def captcha_to_tensors(
    path: Path = CAPTCHA_PARQUET,
    *,
    max_sessions: int | None = None,
    mouse_only: bool = True,
    seed: int = 42,
    min_ticks: int = 64,
) -> list[np.ndarray]:
    """Load CaptchaSolve30k sessions as ``(N, 8)`` event tensors.

    Each session's ``tickInputs`` is an array of ``{x, y, isDown, sampleIndex}``
    sampled per physics tick (~4.17 ms). ``mouse_only`` keeps the non-touchscreen
    rows (the touchscreen captures are a different input modality). ``max_sessions``
    subsamples (seeded) for tractable pretraining; ``None`` uses all.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CaptchaSolve30k parquet not found: {path}. "
            "See notebooks/05_external_datasets.ipynb."
        )

    # The nested ``tickInputs`` column is ~50M structs — materialising it whole
    # OOMs. So: (1) read only the cheap ``touchscreen`` column to choose the
    # target row indices, then (2) stream the heavy columns in batches and parse
    # *only* the selected rows.
    touch = pd.read_parquet(path, columns=["touchscreen"])["touchscreen"]
    candidates = (
        np.flatnonzero(~touch.to_numpy().astype(bool))
        if mouse_only
        else np.arange(len(touch))
    )
    if max_sessions is not None and len(candidates) > max_sessions:
        rng = np.random.default_rng(seed)
        candidates = np.sort(rng.choice(candidates, size=max_sessions, replace=False))
    target = set(int(i) for i in candidates)

    pf = pq.ParquetFile(path)
    tensors: list[np.ndarray] = []
    offset = 0
    for batch in pf.iter_batches(
        batch_size=512, columns=["tickInputs", "duration", "physicsTickCount"]
    ):
        n_rows = batch.num_rows
        # Local indices of target rows in this batch; skip the (costly) nested
        # ``to_pylist`` conversion entirely for batches with no selected rows.
        local = [i for i in range(n_rows) if (offset + i) in target]
        if not local:
            offset += n_rows
            continue
        sub = batch.take(local)
        ti_col = sub.column("tickInputs").to_pylist()
        dur_col = sub.column("duration").to_pylist()
        ptc_col = sub.column("physicsTickCount").to_pylist()
        for ti, dur, ptc in zip(ti_col, dur_col, ptc_col):
            if ti is None or len(ti) < min_ticks:
                continue
            x = np.fromiter((s["x"] for s in ti), dtype=np.float32, count=len(ti))
            y = np.fromiter((s["y"] for s in ti), dtype=np.float32, count=len(ti))
            down = np.fromiter((s["isDown"] for s in ti), dtype=bool, count=len(ti))
            n_ticks = max(int(ptc or len(ti)), 1)
            ms_per_tick = float(dur) / n_ticks
            tensors.append(_sampled_stream_to_tensor(x, y, down, ms_per_tick))
        offset += n_rows
    log.info("captcha_to_tensors: %d sessions → tensors", len(tensors))
    return tensors


# ---------------------------------------------------------------------------
# CS2CD (fine-tune / eval corpus) — 8-D re-encoding
# ---------------------------------------------------------------------------

_CS2_FEATURES = ["usercmd_mouse_dx", "usercmd_mouse_dy", "FIRE", "RIGHTCLICK"]
_CS2_GAP = 2  # tick-gap threshold for splitting contiguous runs (matches benchmark)


def cs2cd_to_tensors_8d(
    path: Path = CS2CD_PARQUET, *, min_ticks: int = 64
) -> list[tuple[int, np.ndarray]]:
    """CS2CD contiguous same-label streams as ``(label, (N, 8))`` tuples.

    Groups by ``(steamid, cheater_present)`` and splits on tick gaps the same
    way as ``scripts/benchmark_cs2cd_ae._streams_from_df`` (the balanced file
    interleaves each player's cheat- and clean-match by tick), then re-encodes
    each run into the shared 8-D schema. ``RIGHTCLICK`` (scope) is dropped — it
    has no native 8-D channel. ``label`` is 1 for cheat, 0 for legit.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CS2CD parquet not found: {path}. See notebooks/05_external_datasets.ipynb."
        )
    df = pd.read_parquet(
        path, columns=["tick", "steamid", "cheater_present", *_CS2_FEATURES]
    )
    df = df.copy()
    df["steamid"] = df["steamid"].astype(str)

    streams: list[tuple[int, np.ndarray]] = []
    for (_sid, lab), g in df.groupby(["steamid", "cheater_present"]):
        g = g.drop_duplicates("tick").sort_values("tick")
        ticks = g["tick"].to_numpy()
        if len(ticks) == 0:
            continue
        run_id = np.concatenate([[0], (np.diff(ticks) > _CS2_GAP).cumsum()])
        feats = g[_CS2_FEATURES].to_numpy().astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        for rid in np.unique(run_id):
            arr = feats[run_id == rid]
            if len(arr) < min_ticks:
                continue
            tensor = _sampled_stream_to_tensor(
                arr[:, 0], arr[:, 1], arr[:, 2] > 0.5, CS2_MS_PER_TICK
            )
            streams.append((int(lab), tensor))
    log.info(
        "cs2cd_to_tensors_8d: %d streams (%d legit / %d cheat)",
        len(streams),
        sum(1 for s in streams if s[0] == 0),
        sum(1 for s in streams if s[0] == 1),
    )
    return streams


# ---------------------------------------------------------------------------
# Domain-gap helper
# ---------------------------------------------------------------------------


def channel_summary_frame(
    tensors: list[np.ndarray], *, max_rows: int = 200_000, seed: int = 42
) -> pd.DataFrame:
    """Stack event rows from ``tensors`` into a per-channel DataFrame.

    Columns are the 8 channel names; one row per event (subsampled to
    ``max_rows`` for a tractable KS/PSI comparison). Feeds
    ``pipeline.monitoring.drift.compute_drift_report``.
    """
    non_empty = [t for t in tensors if len(t) > 0]
    if not non_empty:
        return pd.DataFrame(columns=_CHANNEL_NAMES)
    stacked = np.concatenate(non_empty, axis=0)
    if len(stacked) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(stacked), size=max_rows, replace=False)
        stacked = stacked[idx]
    return pd.DataFrame(stacked, columns=_CHANNEL_NAMES)
