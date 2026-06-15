"""Phase 8.1 Step 0 — CS2CD full-release player-diversity gate.

Settles the single unverified assumption Phase 8.1 hinges on: is the full public
CS2CD release genuinely *player*-diverse (real, globally-unique steamids that are
linkable across matches) or *player-thin* (per-match ``Player_N`` anonymisation,
not linkable across matches)? The verdict branches the whole phase:

  PLAYER_DIVERSE -> pretraining-diversity (#distinct players) is the headline
                   axis; player- AND match-disjoint splits are possible.
  PLAYER_THIN    -> the S1<->S2 temporal-encoding variant is the primary
                   experiment; the match-count axis scales stream *volume*
                   (not trackable identities); splits are match-disjoint only.

Cheap by construction: lists the repo file tree (no download) and reads only the
``steamid`` column (+ the small sidecar JSON) for a small sample of matches via
range requests over ``HfFileSystem`` — never the full ~52 GB. See
``docs/ROADMAP.md`` -> "Phase 8.1" -> Step 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "reports" / "cs2cd_diversity_probe.json"

REPO_ID = "CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection"
SUBDIRS = ("no_cheater_present", "with_cheater_present")

# Real CS2 identities are 17-digit Steam64 ids ("7656..."); the release instead
# ships per-match anonymised labels of the form "Player_N".
_ANON_RE = re.compile(r"^Player_?\d+$", re.IGNORECASE)
_STEAM64_RE = re.compile(r"^7656\d{13}$")
_UNION_SATURATION = (
    16  # global distinct labels at/below this ⇒ not growing with matches
)

log = logging.getLogger("cs2cd_diversity_probe")


# ---------------------------------------------------------------------------
# Pure helpers (network-free → unit-tested)
# ---------------------------------------------------------------------------
def _looks_anonymized(ids: list[str]) -> bool:
    """True if (most of) the ids match the ``Player_N`` anonymisation pattern."""
    ids = [str(i) for i in ids]
    if not ids:
        return False
    anon = sum(1 for i in ids if _ANON_RE.match(i))
    return anon >= max(1, int(round(0.8 * len(ids))))


def _is_steam64(ids: list[str]) -> bool:
    """True if every id is a 17-digit Steam64 value (real, linkable identities)."""
    ids = [str(i) for i in ids]
    return bool(ids) and all(_STEAM64_RE.match(i) for i in ids)


def classify(probes: list[dict]) -> dict:
    """Derive the DIVERSE/THIN verdict from a list of per-match probe dicts.

    Each probe must carry ``steamid_all`` (the match's distinct steamid labels),
    ``steamid_looks_anonymized`` and ``steamid_is_steam64``. The discriminator is
    whether the *global* union of labels across matches grows with match count
    (real, linkable ids) or saturates near the per-match player count (per-match
    ``Player_N`` relabelling).
    """
    union: set[str] = set()
    for p in probes:
        union.update(str(x) for x in p.get("steamid_all", []))
    n = len(probes)
    all_anon = bool(probes) and all(p.get("steamid_looks_anonymized") for p in probes)
    any_anon = any(p.get("steamid_looks_anonymized") for p in probes)
    any_steam64 = any(p.get("steamid_is_steam64") for p in probes)
    union_saturates = len(union) <= _UNION_SATURATION

    if all_anon or (any_anon and union_saturates and not any_steam64):
        verdict = "PLAYER_THIN"
        reason = (
            "steamids are per-match anonymised (Player_N): the same small label "
            "set recurs in every match and is NOT linkable across matches, so the "
            "global distinct-identity union does not grow with the match count."
        )
    elif any_steam64 and not union_saturates:
        verdict = "PLAYER_DIVERSE"
        reason = (
            "steamids are real Steam64 ids whose global union grows with the "
            "number of matches → trackable, diverse player identities."
        )
    else:
        verdict = "PLAYER_THIN"  # conservative default for an ambiguous id scheme
        reason = (
            "id scheme is ambiguous (neither clearly Player_N nor clearly "
            "growing Steam64); defaulting to THIN (conservative)."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "n_matches_sampled": n,
        "global_distinct_steamid_labels_in_sample": len(union),
        "global_steamid_label_sample": sorted(union)[:24],
        "ids_are_per_match_anonymized": all_anon,
        "ids_are_real_steam64": any_steam64,
    }


# ---------------------------------------------------------------------------
# Network probes (HfFileSystem range reads — no full download)
# ---------------------------------------------------------------------------
def list_match_files(repo_id: str = REPO_ID) -> dict[str, list[str]]:
    """Return ``{subdir: [match parquet paths]}`` from the repo tree (no download)."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return {
        sub: sorted(
            f for f in files if f.startswith(sub + "/") and f.endswith(".parquet")
        )
        for sub in SUBDIRS
    }


def probe_match(repo_id: str, parquet_path: str) -> dict:
    """Probe ONE match: read only its ``steamid`` column + the sibling JSON.

    Uses ``pyarrow.parquet`` over an ``HfFileSystem`` handle (column projection +
    HTTP range requests) so only the tiny dict-encoded steamid column is fetched,
    not the whole match file.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{repo_id}"
    with fs.open(f"{base}/{parquet_path}") as fh:
        table = pq.read_table(fh, columns=["steamid"])
    ids = sorted({str(x) for x in table.column("steamid").unique().to_pylist()})

    sidecar_keys: list[str] = []
    sidecar_ids: list[str] = []
    json_path = parquet_path[: -len(".parquet")] + ".json"
    try:
        with fs.open(f"{base}/{json_path}") as fh:
            meta = json.load(fh)
        if isinstance(meta, dict):
            sidecar_keys = list(meta.keys())
            seen: set[str] = set()
            for events in meta.values():
                if isinstance(events, list):
                    for ev in events:
                        if isinstance(ev, dict) and "user_steamid" in ev:
                            seen.add(str(ev["user_steamid"]))
            sidecar_ids = sorted(seen)
    except Exception as exc:  # sidecar is optional metadata
        log.debug("sidecar read failed for %s: %r", json_path, exc)

    return {
        "match_path": parquet_path,
        "n_rows": int(table.num_rows),
        "distinct_steamid_in_match": len(ids),
        "steamid_all": ids,
        "steamid_sample": ids[:12],
        "steamid_looks_anonymized": _looks_anonymized(ids),
        "steamid_is_steam64": _is_steam64(ids),
        "sidecar_json_keys": sidecar_keys[:40],
        "sidecar_user_steamids": sidecar_ids[:12],
    }


def _sample_matches(tree: dict[str, list[str]], n_matches: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    per = max(1, n_matches // len([s for s in SUBDIRS if tree.get(s)]))
    sampled: list[str] = []
    for sub in SUBDIRS:
        files = tree.get(sub, [])
        if not files:
            continue
        idx = rng.choice(len(files), size=min(per, len(files)), replace=False)
        sampled.extend(files[i] for i in sorted(idx))
    return sampled


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 8.1 Step 0 — CS2CD full-release player-diversity gate"
    )
    parser.add_argument(
        "--n-matches",
        type=int,
        default=12,
        help="matches to sample across both subdirs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Quiet the per-request HTTP chatter from the HF download stack.
    for noisy in ("httpx", "huggingface_hub", "hf_transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    tree = list_match_files()
    n_total = {sub: len(v) for sub, v in tree.items()}
    log.info(
        "CS2CD release matches per class: %s (total %d)", n_total, sum(n_total.values())
    )

    probes: list[dict] = []
    for path in _sample_matches(tree, args.n_matches, args.seed):
        try:
            pr = probe_match(REPO_ID, path)
        except Exception as exc:
            log.warning("  probe failed for %s: %r", path, exc)
            continue
        probes.append(pr)
        log.info(
            "  %-34s rows=%-7d distinct=%-3d anon=%-5s steam64=%-5s sample=%s",
            pr["match_path"],
            pr["n_rows"],
            pr["distinct_steamid_in_match"],
            pr["steamid_looks_anonymized"],
            pr["steamid_is_steam64"],
            pr["steamid_sample"][:4],
        )

    cls = classify(probes)
    report = {
        "repo_id": REPO_ID,
        "n_matches_total_per_class": n_total,
        "n_matches_total": sum(n_total.values()),
        "seed": args.seed,
        **cls,
        "probes": probes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    log.info("")
    log.info("VERDICT: %s", cls["verdict"])
    log.info("  %s", cls["reason"])
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
