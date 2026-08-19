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

"""Measure per-camera health for ``zdata_hdf5`` episodes.

Two quantities per camera, and the conversion QC thresholds are derived from them:

* **coverage** = ``image_count / frame_count``. With a 30 Hz image clock against a 50 Hz
  state clock the ceiling is ~0.60. Materially *below* that means the camera dropped
  frames; *above 1.0* is impossible for a healthy recording and indicates a truncated
  state stream, so the gate needs an upper bound as well as a lower one.
* **age_p99_ms** = 99th percentile of ``frame_age_ms``, i.e. how stale the image a given
  state frame references is. Healthy episodes sit near one frame interval (~50 ms);
  a bad recording day produced episodes at several *seconds*, which would train the
  policy on observations from seconds earlier while the state moved.

Subsets and cameras are both auto-discovered, so new data cannot be silently skipped.

Usage::

    uv run --no-sync --with h5py python scripts/lerobot_conversion/camera_health_zdata.py out.json
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
    return sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))


def measure_episode(path: Path, subset: str) -> dict:
    """Per-camera coverage and staleness for one episode. RGB groups only."""
    with h5py.File(path, "r") as h:
        frames = int(h.attrs["frame_count"])
        record: dict = {
            "subset": subset,
            "episode": path.parent.name,
            "frames": frames,
            "policy_type": path.parent.name.rsplit("_", 1)[-1],
            "cameras": {},
        }
        for cam in sorted(h["images"].keys()):
            if not cam.endswith("_rgb"):
                continue  # depth is not a GR00T modality
            group = h["images"][cam]
            age = group["frame_age_ms"][:]
            images = int(group.attrs["image_count"])
            record["cameras"][cam] = {
                "image_count": images,
                "coverage": images / frames if frames else 0.0,
                "age_p50_ms": float(np.percentile(age, 50)),
                "age_p99_ms": float(np.percentile(age, 99)),
                "age_max_ms": int(age.max()),
            }
        return record


def summarize(records: list[dict]) -> None:
    """Print a per-subset table; this is what threshold choices should be read off."""
    ok = [r for r in records if "error" not in r]
    subsets = sorted({r["subset"] for r in ok})
    print("\nper-subset camera health (coverage median/p10, age_p99 median):")
    for subset in subsets:
        rows = [r for r in ok if r["subset"] == subset]
        cams = sorted({c for r in rows for c in r["cameras"]})
        print(f"  {subset}  ({len(rows)} episodes)")
        for cam in cams:
            cov = np.array([r["cameras"][cam]["coverage"] for r in rows if cam in r["cameras"]])
            age = np.array([r["cameras"][cam]["age_p99_ms"] for r in rows if cam in r["cameras"]])
            print(
                f"    {cam:26s} coverage med={np.median(cov):.2f} p10={np.percentile(cov, 10):.2f} "
                f"max={cov.max():.2f} | age_p99 med={np.median(age):5.0f}ms max={age.max():7.0f}ms"
            )


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
                records.append(measure_episode(path, subset))
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
    print(f"measured {len(records)} episodes ({errors} errors) -> {args.out}")
    summarize(records)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
