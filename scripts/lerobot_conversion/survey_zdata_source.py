#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Survey a ``zdata_hdf5`` source tree before converting it.

Emits one JSON record per episode covering schema, cameras, task, sample rates, continuity,
action/state relationships, and target motion. Run this *before* a conversion: it is how
schema drift gets caught early
rather than as corrupted output. Findings this surfaced on the semihumanoid corpus that
would otherwise have been silent:

* state dimension differs between subsets (32 vs 46 -- the v3+ subsets prepend a 7-dim
  joint array per arm, shifting every later field slice), which is why the converter
  resolves fields by name rather than by index;
* camera key sets differ between subsets, so only the intersection is safely mappable;
* a subset contained ``*_policy`` rollouts mixed in with ``*_expert`` demonstrations.

Subsets are auto-discovered from the source root, so a newly collected subset cannot be
silently omitted.

Usage::

    python scripts/lerobot_conversion/survey_zdata_source.py <out.json> [--config <yaml>]

Requires h5py, which is deliberately not a project dependency::

    uv run --no-sync --with h5py python scripts/lerobot_conversion/survey_zdata_source.py out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from zdata_pipeline.config import PipelineConfig, load_config, resolve_source_subsets
from zdata_pipeline.source import discover_episodes, resolve_field_slices


DEFAULT_ROOT = "~/ml_data/data/training_data/semihumanoid"
EPISODE_GLOB = "*/20*/*/*/*/episode.h5"
EXCLUDE = "_failed_recordings"


def discover_subsets(root: Path) -> list[str]:
    """Subset directory names, auto-discovered so the list cannot go stale."""
    return sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))


def _span(block: np.ndarray) -> float:
    """Largest per-axis peak-to-peak movement in a (T, D) block."""
    return float(np.abs(block.max(axis=0) - block.min(axis=0)).max())


def _step_summary(values: np.ndarray) -> dict[str, float]:
    steps = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(steps.mean()) if len(steps) else 0.0,
        "median": float(np.median(steps)) if len(steps) else 0.0,
        "p99": float(np.percentile(steps, 99)) if len(steps) else 0.0,
        "max": float(steps.max(initial=0.0)),
    }


def _source_rotations(values: np.ndarray) -> Rotation:
    array = np.asarray(values, dtype=np.float64)
    first = array[:, :3]
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = array[:, 3:]
    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    return Rotation.from_matrix(np.stack([first, second, np.cross(first, second)], axis=-1))


def _target_motion(
    config: PipelineConfig,
    target: np.ndarray,
    slices: dict[str, tuple[int, int]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for entry in config.action.keys:
        fields = [target[:, slice(*slices[name])] for name in entry.fields]
        values = np.concatenate(fields, axis=1)
        if entry.format == "XYZ_ROT6D":
            result[f"{entry.key}.position_step_m"] = _step_summary(
                np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1)
            )
            rotations = _source_rotations(values[:, 3:9])
            result[f"{entry.key}.rotation_step_rad"] = _step_summary(
                (rotations[:-1].inv() * rotations[1:]).magnitude()
            )
        else:
            result[f"{entry.key}.step_linf"] = _step_summary(
                np.max(np.abs(np.diff(values, axis=0)), axis=1)
            )
    return result


def survey_episode(path: Path, subset: str, config: PipelineConfig | None = None) -> dict:
    """One flat record per episode. Raises nothing; callers get an ``error`` key instead."""
    with h5py.File(path, "r") as h:
        attrs = h.attrs
        state = h["state/flat"][:]
        state_slices = resolve_field_slices(h["state"])
        action_slices = resolve_field_slices(h["action"])
        action_sources = {
            name: h[f"action/{name}"][:]
            for name in h["action"]
            if isinstance(h[f"action/{name}"], h5py.Dataset)
            and h[f"action/{name}"].ndim == 2
            and h[f"action/{name}"].shape[0] == len(state)
        }
        action = action_sources.get("executed", next(iter(action_sources.values())))
        elapsed_ms = np.asarray(h["frame/elapsed_ms"][:], dtype=np.float64)
        gaps = np.diff(elapsed_ms)
        camera_records = {}
        for camera_name, camera in h["images"].items():
            image_count = int(camera.attrs.get("image_count", 0))
            ages = np.asarray(camera["frame_age_ms"][:]) if "frame_age_ms" in camera else None
            camera_records[camera_name] = {
                "image_count": image_count,
                "coverage": image_count / len(state),
                "frame_age_ms_p99": (
                    float(np.percentile(ages, 99)) if ages is not None and len(ages) else None
                ),
            }

        record = {
            "subset": subset,
            "episode": path.parent.name,
            "robot": str(attrs.get("robot_id", "?")),
            "frames": int(attrs.get("frame_count", len(state))),
            "sampling_hz": float(attrs.get("sampling_hz", 0)),
            "image_sampling_hz": float(attrs.get("image_sampling_hz", 0)),
            "recording_mode": str(attrs.get("recording_mode", "?")),
            "rotation_mode": str(attrs.get("rotation_mode", "?")),
            "schema_version": str(attrs.get("schema_version", "?")),
            "task_uuid": str(attrs.get("task_uuid", "?")),
            "task_text": str(attrs.get("task_text", "?")),
            "valid_for_training": bool(attrs.get("valid_for_training", False)),
            "policy_type": path.parent.name.rsplit("_", 1)[-1],
            "cameras": camera_records,
            "state_dim": int(state.shape[1]),
            "action_dim": int(action.shape[1]),
            "state_fields": {
                name: {"start": start, "end": end, "width": end - start}
                for name, (start, end) in state_slices.items()
            },
            "action_fields": {
                name: {"start": start, "end": end, "width": end - start}
                for name, (start, end) in action_slices.items()
            },
            "state_present_all": (
                bool(h["state/present"][:].all()) if "present" in h["state"] else None
            ),
            "action_residual_max": (
                float(np.abs(h["action/residual"][:]).max()) if "residual" in h["action"] else None
            ),
            "action_state_equal": {
                name: bool(values.shape == state.shape and np.array_equal(values, state))
                for name, values in action_sources.items()
            },
            "frame_gap_ms": _step_summary(gaps),
            "gap_count_above_40_ms": int(np.count_nonzero(gaps > 40.0)),
        }

        if config is not None:
            target = (
                state
                if config.action.source == "observation"
                else action_sources[config.action.source]
            )
            target_slices = state_slices if config.action.source == "observation" else action_slices
            record["configured_target_source"] = config.action.source
            record["configured_target_offset"] = config.action.observation_offset
            record["target_motion"] = _target_motion(config, target, target_slices)

        # Coarse per-arm activity: how much did each arm actually move? On the semihumanoid
        # corpus this revealed that most episodes hold one arm still, which dominates the
        # relative-action distribution and therefore the training loss.
        for side in ("left", "right"):
            pos = state_slices.get(f"{side}_arm_pose_pos")
            grip = state_slices.get(f"{side}_gripper")
            ft = state_slices.get(f"{side}_ft")
            if pos:
                record[f"{side}_pos_span_m"] = _span(state[:, pos[0] : pos[1]])
            if grip:
                col = state[:, grip[0]]
                record[f"{side}_gripper_min"] = float(col.min())
                record[f"{side}_gripper_max"] = float(col.max())
            if ft:
                record[f"{side}_ft_absmax"] = float(np.abs(state[:, ft[0] : ft[1]]).max())
        return record


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out", type=Path, help="path to write the JSON records to")
    ap.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT), help="source tree root")
    ap.add_argument("--config", type=Path, help="embodiment config controlling discovery/targets")
    args = ap.parse_args()

    config = load_config(args.config) if args.config is not None else None
    root = (config.source.root if config is not None else args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1

    records: list[dict] = []
    subset_paths = (
        resolve_source_subsets(config)
        if config is not None
        else [root / name for name in discover_subsets(root)]
    )
    for subset_path in subset_paths:
        subset = subset_path.name
        episodes = (
            discover_episodes(config, subset_path)
            if config is not None
            else [
                path for path in sorted(subset_path.glob(EPISODE_GLOB)) if EXCLUDE not in path.parts
            ]
        )
        for path in episodes:
            try:
                records.append(survey_episode(path, subset, config))
            except Exception:
                records.append(
                    {
                        "subset": subset,
                        "episode": path.parent.name,
                        "error": traceback.format_exc(limit=3),
                    }
                )
        print(f"  {subset}: {len(episodes)} episodes", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=1))
    errors = sum(1 for r in records if "error" in r)
    print(f"surveyed {len(records)} episodes ({errors} errors) -> {args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
