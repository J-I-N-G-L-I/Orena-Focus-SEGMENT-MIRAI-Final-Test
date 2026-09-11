"""Bounded, deterministic whole-video swaps for extremely rare Segment subtypes."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

LEGACY_SPLIT_VERSION = "video_hash_v1"
RARE_SPLIT_VERSION = "rare_subtype_video_swap_v1"
RARE_TOTAL_THRESHOLD = 20
MAX_SWAP_COMBINATIONS = 100_000
RARE_CAPABILITIES = frozenset(("aggregation", "complex", "event"))
VideoKey = tuple[str, str]
CellKey = tuple[str, str, str]


def _video(row: Mapping[str, Any]) -> VideoKey:
    return str(row["dataset"]), Path(str(row["video"])).name.lower()


def _video_records(keys: set[VideoKey]) -> list[dict[str, str]]:
    return [{"dataset": dataset, "video": video} for dataset, video in sorted(keys)]


def _shape(rows: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    # Matching this counter preserves both tracks' row counts and the existing
    # broad procedure composition. It does not add procedure text to training.
    return Counter((str(row["track"]), str(row.get("procedure_type", ""))) for row in rows)


def _cell(row: Mapping[str, Any]) -> CellKey:
    return str(row["track"]), str(row["dataset"]), str(row["primary_capability"])


def _pool_ok(pool: Mapping[CellKey, int], quotas: Mapping[CellKey, int], dataset: str) -> bool:
    for key, quota in quotas.items():
        track, source, capability = key
        if source != dataset:
            continue
        count = pool.get(key, 0)
        if track == "segment":
            # No repeat through step 2000; full pool covered by the end.
            if 5 * count < 2 * quota or count > quota:
                return False
            # All three rare capability pools must be covered by step 2500.
            if capability in RARE_CAPABILITIES and 2 * count > quota:
                return False
        elif track == "frame" and count < quota:
            return False
    return True


def select_protected_dev(
    rows: Sequence[dict[str, Any]],
    baseline_dev: set[VideoKey],
    seed: int,
    cell_quotas: Mapping[CellKey, int],
) -> tuple[set[VideoKey], dict[str, Any]]:
    """Protect every video containing a source/subtype with <=20 Segment rows.

    Only protected baseline-dev videos move to train. An equal number of safe
    train videos move back, minimizing source/capability-count changes under
    exact track/procedure row-count and fixed schedule constraints. Exhaustive
    selection is limited to 100k combinations per source; infeasibility fails.
    """
    by_video: dict[VideoKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[_video(row)].append(row)
    all_videos = set(by_video)
    baseline = {(str(dataset), Path(str(video)).name.lower()) for dataset, video in baseline_dev}
    if not baseline <= all_videos:
        raise ValueError("Baseline dev split contains unknown videos")
    segment = [row for row in rows if row["track"] == "segment"]
    totals = Counter((str(row["dataset"]), str(row["primary_capability_raw"])) for row in segment)
    rare = {key for key, count in totals.items() if count <= RARE_TOTAL_THRESHOLD}
    protected = {_video(row) for row in segment if (str(row["dataset"]), str(row["primary_capability_raw"])) in rare}
    incoming = baseline & protected
    outgoing: set[VideoKey] = set()
    source_audits: dict[str, dict[str, Any]] = {}
    for dataset in sorted({key[0] for key in incoming}):
        moved = sorted(key for key in incoming if key[0] == dataset)
        eligible = sorted(key for key in all_videos - baseline - protected if key[0] == dataset)
        needed = len(moved)
        if len(eligible) < needed:
            raise RuntimeError(f"Rare split has insufficient safe replacement videos for {dataset}")
        search_size = math.comb(len(eligible), needed)
        if search_size > MAX_SWAP_COMBINATIONS:
            raise RuntimeError(f"Rare split search for {dataset} needs {search_size} combinations, above {MAX_SWAP_COMBINATIONS}; freeze a reviewed policy instead")
        old_train_rows = [row for row in rows if row["dataset"] == dataset and _video(row) not in baseline]
        moved_rows = [row for key in moved for row in by_video[key]]
        base_pool = Counter(_cell(row) for row in old_train_rows)
        incoming_pool = Counter(_cell(row) for row in moved_rows)
        target_shape = _shape(moved_rows)
        vectors = {key: Counter(_cell(row) for row in by_video[key]) for key in eligible}
        shapes = {key: _shape(by_video[key]) for key in eligible}
        # Preserve original source spelling for the final seeded tie-break.
        names = {key: min(Path(str(row["video"])).name for row in by_video[key]) for key in eligible}
        hashes = {key: hashlib.sha256(f"{seed}:{dataset}:{names[key]}".encode()).hexdigest() for key in eligible}
        segment_cells = sorted(key for key in cell_quotas if key[:2] == ("segment", dataset))
        frame_cells = sorted(key for key in cell_quotas if key[:2] == ("frame", dataset))
        best: tuple[Any, ...] | None = None
        feasible = 0
        for candidate in combinations(eligible, needed):
            combined_shape: Counter[tuple[str, str]] = Counter()
            removed: Counter[CellKey] = Counter()
            for key in candidate:
                combined_shape.update(shapes[key])
                removed.update(vectors[key])
            if combined_shape != target_shape:
                continue
            pool = {key: base_pool[key] + incoming_pool[key] - removed[key] for key in base_pool.keys() | incoming_pool.keys() | removed.keys()}
            if not _pool_ok(pool, cell_quotas, dataset):
                continue
            feasible += 1
            delta = [pool.get(key, 0) - base_pool[key] for key in segment_cells]
            frame_delta = [pool.get(key, 0) - base_pool[key] for key in frame_cells]
            tie = tuple(sorted(hashes[key] for key in candidate))
            objective = (sum(abs(value) for value in delta), max((abs(value) for value in delta), default=0), sum(abs(value) for value in frame_delta), tie)
            if best is None or objective < best[0]:
                best = objective, candidate, pool
        if best is None:
            raise RuntimeError(f"No rare-video swap for {dataset} preserves row/procedure counts and fixed schedule coverage")
        objective, chosen, pool = best
        outgoing.update(chosen)
        source_audits[dataset] = {
            "required_swaps": needed, "safe_candidates": len(eligible),
            "combinations": search_size, "feasible_combinations": feasible,
            "objective": {"segment_capability_l1": objective[0], "segment_capability_max": objective[1], "frame_capability_l1": objective[2]},
            "train_cells_before": {":".join(key): base_pool[key] for key in sorted(base_pool)},
            "train_cells_after": {":".join(key): pool[key] for key in sorted(pool)},
        }
    selected = (baseline - incoming) | outgoing
    if selected & protected:
        raise RuntimeError("Rare-video protection left a protected video in dev")
    for dataset in sorted({key[0] for key in all_videos}):
        old_rows = [row for key in baseline if key[0] == dataset for row in by_video[key]]
        new_rows = [row for key in selected if key[0] == dataset for row in by_video[key]]
        if len([key for key in baseline if key[0] == dataset]) != len([key for key in selected if key[0] == dataset]) or _shape(old_rows) != _shape(new_rows):
            raise RuntimeError(f"Rare split changed dev video/track/procedure counts for {dataset}")
        # No-rare synthetic fixtures are deliberately identity transformations.
        if incoming:
            pool = Counter(_cell(row) for row in rows if row["dataset"] == dataset and _video(row) not in selected)
            if not _pool_ok(pool, cell_quotas, dataset):
                raise RuntimeError(f"Rare split violates fixed schedule coverage for {dataset}")
    new_dev_counts = Counter((str(row["dataset"]), str(row["primary_capability_raw"])) for row in segment if _video(row) in selected)
    rare_cells = [{"dataset": dataset, "subtype": subtype, "total": totals[(dataset, subtype)],
                   "train": totals[(dataset, subtype)] - new_dev_counts[(dataset, subtype)],
                   "dev": new_dev_counts[(dataset, subtype)],
                   "protected_videos": len({_video(row) for row in segment if (str(row["dataset"]), str(row["primary_capability_raw"])) == (dataset, subtype)})}
                  for dataset, subtype in sorted(rare)]
    audit = {
        "version": RARE_SPLIT_VERSION, "rare_total_threshold": RARE_TOTAL_THRESHOLD,
        "selection_rule": "protect_source_primary_subtype_le20_then_minimal_whole_video_swaps",
        "objective_order": ["segment_capability_l1", "segment_capability_max", "frame_capability_l1", "seeded_video_hash"],
        "seed": seed, "max_combinations_per_source": MAX_SWAP_COMBINATIONS,
        "baseline_dev_videos": _video_records(baseline), "selected_dev_videos": _video_records(selected),
        "dev_to_train": _video_records(incoming), "train_to_dev": _video_records(outgoing),
        "protected_videos": _video_records(protected), "rare_cells": rare_cells,
        "source_subtype_counts": {f"{dataset}:{subtype}": {"all": count, "train": count - new_dev_counts[(dataset, subtype)], "dev": new_dev_counts[(dataset, subtype)]}
                                  for (dataset, subtype), count in sorted(totals.items())},
        "train_only_cells": [f"{dataset}:{subtype}" for (dataset, subtype) in sorted(totals) if not new_dev_counts[(dataset, subtype)]],
        "source_subtype_dev_blind_spots": [{"dataset": dataset, "subtype": subtype, "total": count, "train": count, "dev": 0, "protected_rare_cell": (dataset, subtype) in rare}
                                             for (dataset, subtype), count in sorted(totals.items()) if not new_dev_counts[(dataset, subtype)]],
        "source_selection": source_audits,
        "preserves_source_track_row_counts": True, "preserves_source_procedure_distribution": True,
    }
    return selected, audit
