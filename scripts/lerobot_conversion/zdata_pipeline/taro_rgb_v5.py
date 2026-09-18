"""Admission and mask-preserving adapter for the released Taro RGB-v5 surface.

This deliberately supports one verified projection, not arbitrary padded HDF5.
Source files remain untouched. Invalid values are neutralized for arithmetic only;
their masks must survive conversion, statistics, sampling, and the training loss.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .config import PipelineConfig


@lru_cache(maxsize=8)
def _release(root: str, digest: str, pointer_mtime: int) -> dict:
    root = Path(root)
    pointer = json.loads((root / "meta/merged_episode_manifest_v1.latest.json").read_text())
    payload = (root / "meta" / pointer["filename"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest or pointer["sha256"] != digest:
        raise ValueError("RGB-v5 release manifest differs from the configured digest")
    rows = [json.loads(line) for line in payload.splitlines()]
    result = {row["path"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate paths in RGB-v5 release")
    return result


def release_rows(config: PipelineConfig) -> dict:
    root = config.source.root.resolve().parent
    pointer = root / "meta/merged_episode_manifest_v1.latest.json"
    return _release(str(root), config.source.release_manifest_sha256, pointer.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _split_plan(path: str, mtime: int) -> dict:
    return json.loads(Path(path).read_text())


def split_for_source(config: PipelineConfig, source: Path) -> str:
    plan_path = config.source.split_manifest
    plan = _split_plan(str(plan_path.resolve()), plan_path.stat().st_mtime_ns)
    variant = plan["variants"][config.name]
    if variant["release_manifest_sha256"] != config.source.release_manifest_sha256:
        raise ValueError("split plan release differs from configuration")
    key = source.resolve().relative_to(config.source.root.resolve().parent).as_posix()
    row = variant["episodes"][key]
    stat = source.stat()
    if [stat.st_size, stat.st_mtime_ns] != row["source_stat"]:
        raise ValueError(f"source changed since split preparation: {source}")
    split = row["split"]
    if split not in {"train", "val"}:
        raise ValueError(f"invalid split for {source}")
    return split


def compact_slices(slices: dict) -> dict:
    result = dict(slices)
    start, end = result["right_hand"]
    if end - start != 25:
        raise ValueError("Taro RGB-v5 right_hand must have 25 padded coordinates")
    result["right_hand"] = (start, start + 20)
    return result


def validate_source(config: PipelineConfig, source: Path, h5: h5py.File) -> None:
    key = source.resolve().relative_to(config.source.root.resolve().parent).as_posix()
    row = release_rows(config)[key]
    expected = {
        "source_format": "zdata_hdf5_projected",
        "projection_id": "cross_embodiment_max26_v5",
        "embodiment_id": "flexiv_taro_20d_bimanual_v1",
        "stored_action_representation_id": "absolute_cartesian_rot6d_v1",
        "action_source": "action",
        "obs_based_action_offset": 0,
        "finalized": True,
        "frame_count": row["frame_count"],
        "episode_id": row["episode_id"],
        "dataset_source_id": row["dataset_source_id"],
    }
    for key, value in expected.items():
        if h5.attrs.get(key) != value:
            raise ValueError(f"RGB-v5 {key} differs from the admitted contract")
    if config.action.source != "expert" or config.action.observation_offset:
        raise ValueError("Taro RGB-v5 uses same-row action/expert command targets")
    if config.cameras != {
        "head": "view0_rgb",
        "left_wrist": "view1_rgb",
        "right_wrist": "view2_rgb",
    }:
        raise ValueError("Taro ablation requires exactly head and both wrist views")
    if config.train.rtc_training_max_prefix_steps:
        raise ValueError("masked RGB-v5 targets currently require prefix training disabled")
    split_for_source(config, source)


def gather_mask(h5: h5py.File, name: str, slices: dict, entries: tuple) -> np.ndarray:
    from .source import gather_fields

    raw = np.asarray(h5[f"extra/{name}"])
    if not np.isin(raw, [0, 1]).all():
        raise ValueError(f"non-binary {name}")
    return gather_fields(raw, slices, entries, transform_rot6d=False).astype(bool)


def projected_arrays(h5: h5py.File, layout) -> tuple:
    from .source import gather_fields, resolve_field_slices, transform_gathered_rot6d

    state_slices = compact_slices(resolve_field_slices(h5["state"]))
    action_slices = compact_slices(resolve_field_slices(h5["action"]))
    state = gather_fields(h5["state/flat"][:], state_slices, layout.state, transform_rot6d=False)
    action = gather_fields(
        h5["action/expert"][:], action_slices, layout.action, transform_rot6d=False
    )
    sm = gather_mask(h5, "state_value_mask", state_slices, layout.state)
    am = np.logical_and.reduce(
        [
            gather_mask(h5, name, action_slices, layout.action)
            for name in (
                "action_supervision_mask",
                "action_value_mask",
                "action_presence_mask",
                "action_ownership_mask",
            )
        ]
    )
    for scope, slices, mask_name in (
        ("state", state_slices, "state_value_mask"),
        ("action", action_slices, "action_supervision_mask"),
    ):
        end = slices["right_hand"][1]
        if np.asarray(h5[f"extra/{mask_name}"][:, end : end + 5]).any():
            raise ValueError(f"{scope} hand padding unexpectedly contains valid coordinates")
    # EEF rotation/translation transforms mix coordinates: supervise complete poses only.
    # Identity rot6d is only an arithmetic placeholder, never a training label.
    for values, mask, entries in ((state, sm, layout.state), (action, am, layout.action)):
        for entry in entries:
            if entry.rot6d_field:
                valid = mask[:, entry.start : entry.end].all(axis=1)
                mask[:, entry.start : entry.end] = valid[:, None]
                values[~valid, entry.start : entry.end] = 0
                offset = entry.start
                for field, width in entry.fields:
                    if field == entry.rot6d_field:
                        values[~valid, offset : offset + width] = [1, 0, 0, 0, 1, 0]
                    offset += width
            else:
                values[:, entry.start : entry.end] = np.where(
                    mask[:, entry.start : entry.end], values[:, entry.start : entry.end], 0
                )
    state = transform_gathered_rot6d(state, layout.state)
    # Masks on complete rot6d blocks are unchanged by the permutation.
    views = np.asarray(h5["extra/view_mask"][:, :3], dtype=bool)
    eligible = sm.all(axis=1) & views.all(axis=1)
    return state, action, sm, am, eligible


def validate_inventory(config: PipelineConfig) -> None:
    """Fail closed for missing/extra release members or a stale projection/split."""
    rows = release_rows(config)
    root = config.source.root.resolve().parent
    paths = {
        p.resolve().relative_to(root).as_posix()
        for subset in config.source.subsets
        for p in config.source.root.glob(f"{subset}/{config.source.episode_glob}")
    }
    if paths != set(rows):
        raise ValueError(
            f"RGB-v5 membership differs: missing={len(set(rows) - paths)}, extra={len(paths - set(rows))}"
        )
    path = config.source.split_manifest
    plan = _split_plan(str(path.resolve()), path.stat().st_mtime_ns)["variants"][config.name]
    if set(plan["episodes"]) != set(rows):
        raise ValueError("Split plan membership differs from released membership")
    if (
        hashlib.sha256((root / "meta/projection.yaml").read_bytes()).hexdigest()
        != plan["projection_sha256"]
    ):
        raise ValueError("Projection changed after split preparation")
    for relative in rows:
        split_for_source(config, root / relative)
