#!/usr/bin/env python3
"""Bind the two Taro configurations to one reproducible, capture-group holdout.

Preserve all 100 selected training episodes. Evaluation comes from full-only
capture groups; cleaned/original duplicates remain together. Does not train.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import h5py
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lerobot_conversion.zdata_pipeline.config import load_config
from scripts.lerobot_conversion.zdata_pipeline.taro_rgb_v5 import release_rows


REPO = Path(__file__).resolve().parents[2]


def capture_group(segment: str) -> str:
    if not segment or not segment.startswith("/"):
        raise ValueError("capture grouping requires a full source_segment path")
    return re.sub(r"_part_\d+$", "", segment)


def prepare() -> dict:
    configs = [
        load_config(REPO / f"configs/embodiments/taro_exp_{v}.yaml") for v in ("100", "full")
    ]
    if configs[0].train != configs[1].train:
        raise ValueError("ablation training hyperparameters must be identical")
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    inventories = {}
    expected_names = None
    for config in configs:
        root = config.source.root.parent
        projection = yaml.safe_load((root / "meta/projection.yaml").read_text())
        # Bind source-native hand order to the released projection declaration.
        rows = {}
        for path, release_row in release_rows(config).items():
            source = root / path
            with h5py.File(source, "r") as h5:
                meta = json.loads(h5["metadata/json"][()])
                identity = str(h5.attrs["source_episode_id"])
                session = capture_group(meta["source_segment"])
                union("episode:" + identity, "capture:" + session)
                declaration = projection["sources"][str(h5.attrs["dataset_source_id"])][
                    "end_effectors"
                ]["right"]
                for scope in ("state", "action"):
                    if declaration[scope + "_dof"] != 20 or declaration[
                        scope + "_destination_indices"
                    ] != list(range(20)):
                        raise ValueError("right hand projection is not native 20D")
                    names = declaration[scope + "_coordinate_names"]
                    expected_names = names if expected_names is None else expected_names
                    if (
                        names != expected_names
                        or declaration["canonicalization_mode"] != "identity_native"
                    ):
                        raise ValueError("inconsistent native hand semantics")
                stat = source.stat()
                rows[path] = {
                    "source_episode_id": identity,
                    "capture": session,
                    "source_stat": [stat.st_size, stat.st_mtime_ns],
                    "frames": release_row["frame_count"],
                    "pool": release_row["source_pool"],
                }
        inventories[config.name] = rows
    small_groups = {
        find("episode:" + r["source_episode_id"]) for r in inventories[configs[0].name].values()
    }
    full_groups = {
        find("episode:" + r["source_episode_id"]) for r in inventories[configs[1].name].values()
    }
    candidates = full_groups - small_groups
    # Stable hash ranking: reserve 10% of full-only capture groups (minimum one).
    selected = set(
        sorted(candidates, key=lambda s: hashlib.sha256(("42:" + s).encode()).hexdigest())[
            : max(1, len(candidates) // 10)
        ]
    )
    if not selected:
        raise ValueError("No leakage-free full-only captures available for evaluation")
    result = {
        "schema_version": 1,
        "seed": 42,
        "grouping": "source_episode_id union full source_segment without part suffix",
        "evaluation": "shared full-only capture groups; preserve all 100 training episodes",
        "right_hand_coordinate_names": expected_names,
        "variants": {},
    }
    for config in configs:
        rows = inventories[config.name]
        for row in rows.values():
            row["split"] = (
                "val" if find("episode:" + row["source_episode_id"]) in selected else "train"
            )
        result["variants"][config.name] = {
            "release_manifest_sha256": config.source.release_manifest_sha256,
            "projection_sha256": hashlib.sha256(
                (config.source.root.parent / "meta/projection.yaml").read_bytes()
            ).hexdigest(),
            "episodes": rows,
            "counts": dict(Counter(r["split"] for r in rows.values())),
            "pool_split_counts": dict(Counter(r["pool"] + ":" + r["split"] for r in rows.values())),
        }
    plan_path = configs[0].source.split_manifest
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if plan_path.exists() and plan_path.read_text() != payload:
        raise ValueError("Existing split differs; use a new preparation/output lineage")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(payload)
    print(plan_path)
    for name, variant in result["variants"].items():
        print(name, variant["counts"], variant["pool_split_counts"])
    return result


if __name__ == "__main__":
    prepare()
