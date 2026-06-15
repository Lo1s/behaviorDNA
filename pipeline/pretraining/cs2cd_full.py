"""
pipeline/pretraining/cs2cd_full.py
==================================
Phase 8.1 — full-release CS2CD ingest → cached per-match tensor shards, plus a
lazy masked-denoising dataset over those shards.

Phase 8 pretrained on out-of-domain CaptchaSolve30k and got a null; Phase 8.1
pretrains *in-domain* on the full public CS2CD release (795 matches), reusing the
exact 8-D event-tensor schema and the Phase 8 masked-denoising loop. The Step-0
gate (``scripts/cs2cd_diversity_probe.py``) found the release is **PLAYER_THIN** —
players are per-match-anonymised (``Player_1..Player_10``) and NOT linkable across
matches — so:

  * the cheat/legit label is the **subdirectory** (``no_cheater_present`` → 0,
    ``with_cheater_present`` → 1), NOT a per-row column. The full-release parquet
    has **no** ``cheater_present`` column (unlike the balanced sample that
    ``corpora.cs2cd_to_tensors_8d`` reads), so grouping is by ``steamid`` alone
    within a match; and
  * splits are **match-disjoint only** (no cross-match player identity exists).

Design — the ROADMAP "dataloader-bound on a 16 GB box" mitigation: encode each
match **once** into a small per-match ``.pt`` shard (the 6 relevant columns
re-encoded to ``(N, 8)`` via the shared ``_sampled_stream_to_tensor``) plus a tiny
``.idx.json`` sidecar of run lengths, then sample masked chunks **lazily** per
epoch with an LRU over decoded shards so peak RAM ≈ a few shards, never the whole
corpus. The dataset yields ``(masked, clean)`` exactly like
``MaskedDenoisingDataset`` so ``pretrain_masked_ae`` is reused verbatim.
"""

from __future__ import annotations

import argparse
import json
import logging
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from pipeline.pretraining.corpora import (
    _CS2_GAP,
    CS2_MS_PER_TICK,
    _sampled_stream_to_tensor,
)
from pipeline.pretraining.masking import mask_chunk
from pipeline.sequences.dataset import _chunk_indices
from pipeline.sequences.preprocessing import (
    COL_DT,
    fit_normalizer,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "external" / "cs2cd_full"
CACHE_DIR = RAW_DIR / "_tensor_cache"
MANIFEST = ROOT / "reports" / "cs2cd_full_manifest.json"

REPO_ID = "CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection"
SUBDIR_LABEL = {"no_cheater_present": 0, "with_cheater_present": 1}
# The full-release parquet has NO ``cheater_present`` column → the match label is
# the subdir; ``RIGHTCLICK`` is projected for parity but has no 8-D channel.
PROJECT_COLS = [
    "tick",
    "steamid",
    "usercmd_mouse_dx",
    "usercmd_mouse_dy",
    "FIRE",
    "RIGHTCLICK",
]
_FEAT_COLS = ["usercmd_mouse_dx", "usercmd_mouse_dy", "FIRE", "RIGHTCLICK"]
MIN_TICKS = 64
SHARD_VERSION = 1

log = logging.getLogger("cs2cd_full")


# ---------------------------------------------------------------------------
# Download (idempotent / resumable)
# ---------------------------------------------------------------------------
def local_matches(subdir: str, raw_dir: Path = RAW_DIR) -> list[Path]:
    """Sorted local match parquet paths for a class subdir (numeric order)."""
    d = raw_dir / subdir
    if not d.is_dir():
        return []
    return sorted(
        d.glob("*.parquet"),
        key=lambda p: (int(p.stem) if p.stem.isdigit() else 1 << 30, p.stem),
    )


def download_cs2cd_full(
    *,
    max_matches_per_class: int | None = None,
    subdirs: tuple[str, ...] = tuple(SUBDIR_LABEL),
    repo_id: str = REPO_ID,
    dest: Path = RAW_DIR,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Download full-release match parquet files (idempotent + resumable).

    ``max_matches_per_class=None`` pulls every match (~48 GB). A cap fetches a
    seeded subset per class. Sidecar JSONs are intentionally NOT fetched — mouse
    motion + the subdir label are all Phase 8.1 needs. Returns local paths/subdir.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if max_matches_per_class is None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f"{s}/*.parquet" for s in subdirs],
            local_dir=str(dest),
            max_workers=8,
        )
    else:
        from huggingface_hub import HfApi, hf_hub_download

        files = HfApi().list_repo_files(repo_id, repo_type="dataset")
        rng = np.random.default_rng(seed)
        for s in subdirs:
            pqf = sorted(
                f for f in files if f.startswith(s + "/") and f.endswith(".parquet")
            )
            idx = rng.choice(
                len(pqf), size=min(max_matches_per_class, len(pqf)), replace=False
            )
            for i in sorted(idx):
                hf_hub_download(
                    repo_id, pqf[i], repo_type="dataset", local_dir=str(dest)
                )
    return {s: local_matches(s, dest) for s in subdirs}


# ---------------------------------------------------------------------------
# Per-match encode → shard (.pt) + run-length sidecar (.idx.json)
# ---------------------------------------------------------------------------
def _runs_from_match(
    parquet_path: Path, *, min_ticks: int = MIN_TICKS
) -> list[tuple[str, np.ndarray]]:
    """Read one match (projected cols), group by steamid, split on tick gaps, and
    re-encode each contiguous run to ``(N, 8)``.

    Mirrors ``corpora.cs2cd_to_tensors_8d`` but per-file and **label-free** (the
    label is the subdir, not a column). One match fits comfortably in RAM.
    """
    df = pd.read_parquet(parquet_path, columns=PROJECT_COLS)
    df["steamid"] = df["steamid"].astype(str)
    runs: list[tuple[str, np.ndarray]] = []
    for sid, g in df.groupby("steamid", sort=True):
        g = g.drop_duplicates("tick").sort_values("tick")
        ticks = g["tick"].to_numpy()
        if len(ticks) == 0:
            continue
        run_id = np.concatenate([[0], (np.diff(ticks) > _CS2_GAP).cumsum()])
        feats = np.nan_to_num(g[_FEAT_COLS].to_numpy().astype(np.float32))
        for rid in np.unique(run_id):
            arr = feats[run_id == rid]
            if len(arr) < min_ticks:
                continue
            tensor = _sampled_stream_to_tensor(
                arr[:, 0], arr[:, 1], arr[:, 2] > 0.5, CS2_MS_PER_TICK
            )
            runs.append((str(sid), tensor))
    return runs


def _shard_paths(parquet_path: Path, source: str, cache_dir: Path) -> tuple[Path, Path]:
    cls = parquet_path.parent.name
    base = cache_dir / source / cls / parquet_path.stem
    return base.with_suffix(".pt"), base.with_suffix(".idx.json")


def encode_match_to_shard(
    parquet_path: Path,
    *,
    label: int,
    cache_dir: Path = CACHE_DIR,
    source: str = "s1",
    min_ticks: int = MIN_TICKS,
    overwrite: bool = False,
) -> Path:
    """Encode one match into a per-match shard of ``(steamid, (N,8))`` runs.

    ``source="s1"`` = native CS2 tick encoding (``dt = log1p(CS2_MS_PER_TICK)``);
    the path encodes the source, so re-encoding a different source never clobbers.
    Writes ``<id>.pt`` (tensors) + ``<id>.idx.json`` (run lengths — read by the
    dataset to build its index without loading tensors). Idempotent unless
    ``overwrite``.
    """
    shard_path, idx_path = _shard_paths(parquet_path, source, cache_dir)
    if shard_path.exists() and idx_path.exists() and not overwrite:
        return shard_path

    runs = _runs_from_match(parquet_path, min_ticks=min_ticks)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard = {
        "version": SHARD_VERSION,
        "match_id": parquet_path.stem,
        "subdir": parquet_path.parent.name,
        "label": int(label),
        "source": source,
        "runs": [{"steamid": sid, "tensor": torch.from_numpy(t)} for sid, t in runs],
    }
    torch.save(shard, shard_path)
    idx_path.write_text(
        json.dumps(
            {
                "version": SHARD_VERSION,
                "label": int(label),
                "source": source,
                "steamids": [sid for sid, _ in runs],
                "lengths": [int(len(t)) for _, t in runs],
            }
        )
        + "\n"
    )
    return shard_path


def build_shard_cache(
    parquet_paths: list[Path],
    *,
    label: int,
    source: str = "s1",
    cache_dir: Path = CACHE_DIR,
    min_ticks: int = MIN_TICKS,
    overwrite: bool = False,
    log_every: int = 25,
) -> list[Path]:
    """Encode each match to a shard (single-process — honours the RAM ceiling)."""
    shards: list[Path] = []
    for i, p in enumerate(parquet_paths):
        shards.append(
            encode_match_to_shard(
                p,
                label=label,
                source=source,
                cache_dir=cache_dir,
                min_ticks=min_ticks,
                overwrite=overwrite,
            )
        )
        if log_every and (i + 1) % log_every == 0:
            log.info(
                "  encoded %d/%d matches (source=%s)", i + 1, len(parquet_paths), source
            )
    return shards


def _load_shard(shard_path: Path) -> dict:
    """Load a trusted local cache shard (our own file → weights_only=False)."""
    return torch.load(shard_path, map_location="cpu", weights_only=False)


def _read_run_lengths(shard_path: Path) -> list[int]:
    """Run lengths for a shard from its tiny ``.idx.json`` sidecar (no tensors)."""
    idx_path = shard_path.with_suffix("").with_suffix(".idx.json")
    if idx_path.exists():
        return [int(x) for x in json.loads(idx_path.read_text())["lengths"]]
    return [int(r["tensor"].shape[0]) for r in _load_shard(shard_path)["runs"]]


# ---------------------------------------------------------------------------
# Match-disjoint manifest (PLAYER_THIN: split unit = match; steamid = stream id)
# ---------------------------------------------------------------------------
def build_manifest(
    *,
    raw_dir: Path = RAW_DIR,
    split_unit: str = "match",
    seed: int = 42,
    frac: tuple[float, float, float] = (0.8, 0.1, 0.1),
    diversity_points: tuple[int, ...] = (50, 200),
    out: Path = MANIFEST,
) -> dict:
    """Build a match-disjoint split over the legit (``no_cheater_present``) matches
    for pretraining, plus nested volume subsets.

    PLAYER_THIN ⇒ ``split_unit='match'``; ``steamid`` is a per-match stream id, not
    a cross-match identity. ``diversity_subsets`` are nested prefixes of the
    shuffled pretrain order (the volume axis [50, 200, …, full]).
    """
    legit = local_matches("no_cheater_present", raw_dir)
    cheat = local_matches("with_cheater_present", raw_dir)
    ids = [p.stem for p in legit]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n = len(ids)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    tr_order = [ids[i] for i in perm[:n_tr]]  # shuffled → prefixes are random subsets
    va = sorted(ids[i] for i in perm[n_tr : n_tr + n_va])
    ho = sorted(ids[i] for i in perm[n_tr + n_va :])

    points = sorted({*diversity_points, n_tr})
    subsets = {str(k): sorted(tr_order[:k]) for k in points if 0 < k <= n_tr}

    manifest = {
        "branch": "THIN",
        "split_unit": split_unit,
        "seed": seed,
        "n_legit_matches": len(legit),
        "n_cheat_matches": len(cheat),
        "pretrain_matches": sorted(tr_order),
        "val_matches": va,
        "heldout_matches": ho,
        "cheat_matches": sorted(p.stem for p in cheat),
        "diversity_subsets": subsets,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info(
        "manifest: %d legit (%d pretrain / %d val / %d heldout), %d cheat; subsets=%s → %s",
        len(legit),
        len(tr_order),
        len(va),
        len(ho),
        len(cheat),
        list(subsets),
        out,
    )
    return manifest


def shards_for_matches(
    match_ids: list[str], *, subdir: str, source: str, cache_dir: Path = CACHE_DIR
) -> list[Path]:
    """Resolve cached shard paths for a list of match ids (skips un-encoded)."""
    out = []
    for mid in match_ids:
        sp = cache_dir / source / subdir / f"{mid}.pt"
        if sp.exists():
            out.append(sp)
    return out


# ---------------------------------------------------------------------------
# Normaliser (fit on a streamed subsample of shards)
# ---------------------------------------------------------------------------
def _apply_dt_override(tensor: np.ndarray, dt_override_ms: float) -> np.ndarray:
    """S2 hook (option a): overwrite the (constant) dt channel.

    NB: CS2 is fixed-tick so dt is already constant; overwriting the constant is
    ~a no-op after z-scoring. A *meaningful* S2 (matching GTA's variable dt) needs
    true resampling (option b) — see ``docs/PRETRAINING.md`` / the Phase 8.1 plan.
    """
    out = tensor.copy()
    if len(out) > 1:
        out[1:, COL_DT] = np.log1p(max(float(dt_override_ms), 0.0))
    return out


def fit_shard_normalizer(
    shard_paths: list[Path],
    *,
    max_shards: int = 24,
    max_runs: int = 600,
    seed: int = 42,
    dt_override_ms: float | None = None,
) -> dict:
    """Fit z-score stats on a seeded subsample of runs drawn from the shards."""
    rng = np.random.default_rng(seed)
    if len(shard_paths) > max_shards:
        pick = rng.choice(len(shard_paths), size=max_shards, replace=False)
        sample = [shard_paths[i] for i in sorted(pick)]
    else:
        sample = list(shard_paths)
    tensors: list[np.ndarray] = []
    for sp in sample:
        for run in _load_shard(sp)["runs"]:
            t = run["tensor"].numpy()
            if dt_override_ms is not None:
                t = _apply_dt_override(t, dt_override_ms)
            tensors.append(t)
            if len(tensors) >= max_runs:
                break
        if len(tensors) >= max_runs:
            break
    return fit_normalizer(tensors)


# ---------------------------------------------------------------------------
# Lazy masked-denoising dataset over cached shards
# ---------------------------------------------------------------------------
class CS2CDShardChunkDataset(Dataset):
    """Lazy ``(masked, clean)`` dataset over cached per-match shards.

    Memory-efficient: holds only a per-**run** index (one entry per stream) plus a
    cumulative chunk-count array for O(log n) global indexing, and an LRU of at
    most ``lru_shards`` decoded+normalised shards. Same masking/return contract as
    :class:`pipeline.pretraining.masking.MaskedDenoisingDataset`, so
    ``pretrain_masked_ae`` consumes it unchanged.
    """

    def __init__(
        self,
        shard_paths: list[Path],
        *,
        stats: dict,
        chunk_length: int = 64,
        stride: int = 32,
        mask_frac: float = 0.15,
        seed: int = 42,
        lru_shards: int = 8,
        dt_override_ms: float | None = None,
    ) -> None:
        if chunk_length <= 0 or stride <= 0:
            raise ValueError("chunk_length and stride must be > 0")
        if not 0.0 <= mask_frac < 1.0:
            raise ValueError(f"mask_frac must be in [0, 1), got {mask_frac}")
        self.shard_paths = list(shard_paths)
        self.stats = stats
        self.chunk_length = chunk_length
        self.stride = stride
        self.mask_frac = mask_frac
        self.seed = seed
        self.lru_shards = max(1, lru_shards)
        self.dt_override_ms = dt_override_ms

        # Per-run index from the tiny sidecars (no tensors loaded here).
        self._runs: list[tuple[Path, int]] = []
        self._cum: list[int] = []
        total = 0
        for sp in self.shard_paths:
            for ri, n in enumerate(_read_run_lengths(sp)):
                nc = len(_chunk_indices(int(n), chunk_length, stride))
                if nc == 0:
                    continue
                self._runs.append((sp, ri))
                total += nc
                self._cum.append(total)
        self._len = total
        self._lru: "OrderedDict[Path, list[np.ndarray]]" = OrderedDict()

    def __len__(self) -> int:
        return self._len

    def _decoded(self, shard_path: Path) -> list[np.ndarray]:
        """Return normalised run tensors for a shard, via the LRU cache."""
        cached = self._lru.get(shard_path)
        if cached is not None:
            self._lru.move_to_end(shard_path)
            return cached
        mean = np.asarray(self.stats["mean"], dtype=np.float32)
        std = np.asarray(self.stats["std"], dtype=np.float32)
        runs = []
        for run in _load_shard(shard_path)["runs"]:
            t = run["tensor"].numpy()
            if self.dt_override_ms is not None:
                t = _apply_dt_override(t, self.dt_override_ms)
            runs.append(((t - mean) / std).astype(np.float32))
        self._lru[shard_path] = runs
        self._lru.move_to_end(shard_path)
        while len(self._lru) > self.lru_shards:
            self._lru.popitem(last=False)
        return runs

    def _clean_window(self, idx: int) -> np.ndarray:
        """Return the normalised ``(chunk_length, 8)`` window for global chunk ``idx``.

        The pure window-extraction half of ``__getitem__`` (no masking) — factored
        out so the Phase 8.2 contrastive subclass can reuse the LRU + global index
        without re-implementing the bisect lookup.
        """
        if idx < 0:
            idx += self._len
        r = bisect_right(self._cum, idx)
        prev = self._cum[r - 1] if r > 0 else 0
        local = idx - prev
        shard_path, run_idx = self._runs[r]
        clean = self._decoded(shard_path)[run_idx]
        start = local * self.stride
        return clean[start : start + self.chunk_length]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self._clean_window(idx)
        rng = np.random.default_rng(self.seed + idx)
        masked = mask_chunk(window, self.mask_frac, rng)
        return (
            torch.from_numpy(masked).float(),
            torch.from_numpy(window.copy()).float(),
        )

    def shard_index_groups(self) -> list[list[int]]:
        """Global chunk indices grouped by shard (all of a shard's runs together).

        Lets :class:`ShardGroupedSampler` keep each shard resident while its
        chunks are consumed, so a shuffled epoch touches each shard once.
        """
        groups: "OrderedDict[Path, list[int]]" = OrderedDict()
        prev = 0
        for r, (shard_path, _ri) in enumerate(self._runs):
            groups.setdefault(shard_path, []).extend(range(prev, self._cum[r]))
            prev = self._cum[r]
        return list(groups.values())


class ShardGroupedSampler(Sampler[int]):
    """Shuffle shard *order* + chunks *within* a shard, but never interleave shards.

    With ``shuffle=True`` over many large shards, a global shuffle makes the
    dataset's LRU thrash (≈one ~30 MB shard reload per sample). Grouping by shard
    keeps each shard resident for the duration of its chunks → one load per shard
    per epoch, while still randomising both shard order and within-shard order
    (sufficient stochasticity for masked-denoising pretraining).
    """

    def __init__(
        self, dataset: CS2CDShardChunkDataset, *, shuffle: bool = True, seed: int = 0
    ):
        self.groups = dataset.shard_index_groups()
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._n = len(dataset)

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.groups)))
        if self.shuffle:
            rng.shuffle(order)
        for gi in order:
            idxs = list(self.groups[gi])
            if self.shuffle:
                rng.shuffle(idxs)
            yield from idxs


# ---------------------------------------------------------------------------
# CLI: build the manifest + (optionally) the legit S1 shard cache
# ---------------------------------------------------------------------------
def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 8.1 — CS2CD full-release ingest"
    )
    parser.add_argument(
        "--step",
        choices=["manifest", "encode-legit", "all"],
        default="all",
        help="manifest only, encode legit shards only, or both",
    )
    parser.add_argument(
        "--source", default="s1", help="tensor source variant (s1 native)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap matches encoded (smoke)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.step in ("manifest", "all"):
        build_manifest(seed=args.seed)

    if args.step in ("encode-legit", "all"):
        legit = local_matches("no_cheater_present")
        if args.limit:
            legit = legit[: args.limit]
        if not legit:
            log.warning(
                "no local legit matches under %s — run the download first", RAW_DIR
            )
            return 1
        log.info("encoding %d legit matches → %s shards…", len(legit), args.source)
        build_shard_cache(legit, label=0, source=args.source, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
