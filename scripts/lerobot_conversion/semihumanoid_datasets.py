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

"""Inspect a semihumanoid GR00T dataset root and emit the ``--dataset-path`` string.

The converted corpus grows over time, so training commands should never hardcode a list of
dataset directories -- that is how a newly-added subset silently gets left out of a run.

Usage::

    # what is in there right now
    python scripts/lerobot_conversion/semihumanoid_datasets.py --out-root <root>

    # paste straight into launch_finetune.py --dataset-path
    python scripts/lerobot_conversion/semihumanoid_datasets.py --out-root <root> --print train

Train datasets are every directory holding ``meta/info.json`` whose name does **not** end in
``_val``; the ``_val`` siblings are the held-out sets and are excluded from training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


VAL_SUFFIX = "_val"
STATS_FILES = ("stats.json", "relative_stats.json")


def find_datasets(root: Path) -> tuple[list[Path], list[Path]]:
    """Return (train, val) dataset directories, each sorted by name."""
    # glob yields <root>/<ds>/meta/info.json, so the dataset dir is two levels up.
    found = sorted(p.parent.parent for p in root.glob("*/meta/info.json"))
    train = [p for p in found if not p.name.endswith(VAL_SUFFIX)]
    val = [p for p in found if p.name.endswith(VAL_SUFFIX)]
    return train, val


def summarize(ds: Path) -> dict:
    info = json.loads((ds / "meta" / "info.json").read_text())
    missing = [f for f in STATS_FILES if not (ds / "meta" / f).exists()]
    return {
        "name": ds.name,
        "episodes": info.get("total_episodes", 0),
        "frames": info.get("total_frames", 0),
        "fps": info.get("fps"),
        "stats_missing": missing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument(
        "--print",
        dest="which",
        choices=["train", "val", "none"],
        default="none",
        help="emit an os.pathsep-joined path string on stdout and nothing else",
    )
    ap.add_argument(
        "--write-manifest",
        action="store_true",
        help="(re)write <out-root>/manifest.json from the current datasets, ledgers and reports",
    )
    args = ap.parse_args()

    root = args.out_root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1
    train, val = find_datasets(root)
    if not train:
        print(f"ERROR: no datasets with meta/info.json under {root}", file=sys.stderr)
        return 1

    if args.which != "none":
        chosen = train if args.which == "train" else val
        print(os.pathsep.join(str(p) for p in chosen))
        return 0

    print(f"root: {root}")
    for label, group in (("train", train), ("val", val)):
        print(f"\n{label} datasets ({len(group)}):")
        tot_e = tot_f = 0
        for ds in group:
            s = summarize(ds)
            warn = (
                f"   [stats missing: {', '.join(s['stats_missing'])}]" if s["stats_missing"] else ""
            )
            print(f"  {s['name']:22s} {s['episodes']:5d} eps  {s['frames']:8d} frames{warn}")
            tot_e += s["episodes"]
            tot_f += s["frames"]
        print(
            f"  {'TOTAL':22s} {tot_e:5d} eps  {tot_f:8d} frames  ({tot_f / 50 / 3600:.2f} h @50Hz)"
        )

    usable = sum(
        max(0, e["length"] - 39)
        for ds in train
        for e in (
            json.loads(line)
            for line in (ds / "meta" / "episodes.jsonl").read_text().splitlines()
            if line
        )
    )
    print(f"\ntrainable 40-step start indices: {usable}")
    stale = [ds.name for ds in train + val if summarize(ds)["stats_missing"]]
    if stale:
        print(
            f"\nNOTE: {len(stale)} dataset(s) have no cached stats and will regenerate on the next "
            f"stats.py run or at training start: {', '.join(stale)}"
        )
    print("\n--dataset-path for training:")
    print(f"  {os.pathsep.join(str(p) for p in train)}")

    if args.write_manifest:
        write_manifest(root, train, val)
        print(f"\nwrote {root / 'manifest.json'}")
    return 0


def write_manifest(root: Path, train: list[Path], val: list[Path]) -> None:
    """Aggregate ledgers + per-dataset reports into one auditable manifest.

    Regenerate after every conversion run; it is a snapshot, not a source of truth.
    """
    ledgers = {}
    for lp in sorted((root / "_ledgers").glob("*.json")):
        led = json.loads(lp.read_text())
        eps = led.get("episodes", {})
        ledgers[lp.stem] = {
            "episodes": len(eps),
            "splits": {
                s: sum(1 for r in eps.values() if r["split"] == s)
                for s in sorted({r["split"] for r in eps.values()})
            },
            "camera_map": led.get("camera_map"),
        }

    datasets = {}
    for ds in train + val:
        entry = summarize(ds)
        rp = ds / "_conversion_report.json"
        if rp.exists():
            rep = json.loads(rp.read_text())
            entry.update(
                {
                    "source_subset": rep.get("source_subset"),
                    "task_counts": rep.get("task_counts"),
                    "qc_dropped_last_run": len(rep.get("qc_dropped_this_run", [])),
                    "max_action_residual": rep.get("max_action_residual"),
                }
            )
        datasets[ds.name] = entry

    manifest = {
        "root": str(root),
        "train_datasets": [p.name for p in train],
        "val_datasets": [p.name for p in val],
        "totals": {
            "train_episodes": sum(datasets[p.name]["episodes"] for p in train),
            "train_frames": sum(datasets[p.name]["frames"] for p in train),
            "val_episodes": sum(datasets[p.name]["episodes"] for p in val),
            "val_frames": sum(datasets[p.name]["frames"] for p in val),
        },
        "datasets": datasets,
        "ledgers": ledgers,
        "dataset_path_train": os.pathsep.join(str(p) for p in train),
        "note": (
            "Append-only corpus. Re-run convert_semihumanoid.py per subset to add data; it "
            "converts only new episodes and deletes cached stats so they regenerate. "
            "Regenerate this manifest afterwards with --write-manifest."
        ),
    }
    tmp = root / ".manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=4))
    os.replace(tmp, root / "manifest.json")


if __name__ == "__main__":
    raise SystemExit(main())
