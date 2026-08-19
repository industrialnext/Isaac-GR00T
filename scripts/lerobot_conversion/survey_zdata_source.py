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

Emits one JSON record per episode covering schema, cameras, task, sample rates and coarse
per-arm motion. Run this *before* a conversion: it is how schema drift gets caught early
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

    python scripts/lerobot_conversion/survey_zdata_source.py <out.json> [--root <dir>]

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


DEFAULT_ROOT = "~/ml_data/data/training_data/semihumanoid"
EPISODE_GLOB = "*/20*/*/*/*/episode.h5"
EXCLUDE = "_failed_recordings"


def discover_subsets(root: Path) -> list[str]:
    """Subset directory names, auto-discovered so the list cannot go stale."""
    return sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))


def _span(block: np.ndarray) -> float:
    """Largest per-axis peak-to-peak movement in a (T, D) block."""
    return float(np.abs(block.max(axis=0) - block.min(axis=0)).max())


def survey_episode(path: Path, subset: str) -> dict:
    """One flat record per episode. Raises nothing; callers get an ``error`` key instead."""
    with h5py.File(path, "r") as h:
        attrs = h.attrs
        state = h["state/flat"][:]
        action = h["action/executed"][:]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in h["state/field_names"][:]]
        slices = {n: (int(a), int(b)) for n, (a, b) in zip(names, h["state/field_slices"][:])}

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
            "valid_for_training": bool(attrs.get("valid_for_training", False)),
            "policy_type": path.parent.name.rsplit("_", 1)[-1],
            "cameras": sorted(h["images"].keys()),
            "state_dim": int(state.shape[1]),
            "action_dim": int(action.shape[1]),
            "state_fields": names,
            "state_present_all": bool(h["state/present"][:].all()),
            "action_residual_max": float(np.abs(h["action/residual"][:]).max()),
        }

        # Coarse per-arm activity: how much did each arm actually move? On the semihumanoid
        # corpus this revealed that most episodes hold one arm still, which dominates the
        # relative-action distribution and therefore the training loss.
        for side in ("left", "right"):
            pos = slices.get(f"{side}_arm_pose_pos")
            grip = slices.get(f"{side}_gripper")
            ft = slices.get(f"{side}_ft")
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
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1

    records: list[dict] = []
    for subset in discover_subsets(root):
        episodes = [p for p in sorted((root / subset).glob(EPISODE_GLOB)) if EXCLUDE not in p.parts]
        for path in episodes:
            try:
                records.append(survey_episode(path, subset))
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
