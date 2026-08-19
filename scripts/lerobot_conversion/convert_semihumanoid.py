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

"""Convert IndustrialNext ``zdata_hdf5`` semihumanoid episodes to GR00T LeRobot v2.

Run with h5py overlaid on the project environment (h5py is deliberately not a project
dependency, so ``uv.lock`` stays untouched)::

    uv run --no-sync --with h5py python scripts/lerobot_conversion/convert_semihumanoid.py \
        --source-subset ~/ml_data/data/training_data/semihumanoid/flexiv_matcha_v3 \
        --out-root ~/ml_data/data/training_data/gr00t/semihumanoid_260818

**The output is append-only, not sealed.** Re-run the same command after new episodes land
and only the new ones are converted; existing episodes keep their identity. Two mechanisms
make that safe:

* A per-subset **ledger** (``<out-root>/_ledgers/<subset>.json``) freezes each source
  episode's ``(dataset, episode_index, split, index_offset)``. Without it, episode indices
  would be positional in the discovery list, so a backfilled older recording date would
  renumber every later episode and invalidate already-written files.
* ``meta/stats.json`` / ``meta/relative_stats.json`` are **deleted whenever episodes are
  added**. GR00T's stats cache is fingerprinted over the ``info.json`` feature schema only
  (``gr00t/data/stats.py:183``), which does not change when episodes are appended -- so
  without this the next training run would silently normalize against the old, smaller
  episode set. They regenerate on the next ``stats.py`` run or at training start.

Three conversion details are load-bearing and easy to get wrong:

1. **rot6d convention.** The source encodes a rotation as the first two *columns* of R;
   GR00T reads the first two *rows*. Passing the source bytes through unchanged makes
   GR00T reconstruct R-transpose -- a silent ~170 degree error that still trains. See
   ``rot6d_source_to_groot``.
2. **Field selection by name.** The v3 subsets prepend 7-dim joint arrays per arm, which
   shifts every downstream slice. Slices are always resolved through
   ``state/field_names`` + ``state/field_slices``, never hardcoded.
3. **Multi-rate images.** State runs at 50 Hz and cameras at 30 Hz with per-camera
   counts. ``images/<cam>/frame_ref_index`` maps each state frame to an image index, so
   no resampling is needed -- we emit one video frame per state frame, repeating the
   referenced JPEG where the index repeats.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import h5py
import numpy as np
import pandas as pd


# --- canonical layout -------------------------------------------------------------

# canonical video key -> physical HDF5 group under images/
CAMERA_MAP: dict[str, str] = {
    "head": "head_rgb",
    "left_wrist": "eoat_left_bottom_rgb",
    "right_wrist": "eoat_right_bottom_rgb",
}

# Canonical state vector: per arm a contiguous 9-dim [xyz, rot6d] EEF block (required by
# ActionType.EEF + ActionFormat.XYZ_ROT6D), then gripper, then the FT wrench.
# (source field name, width) in output order.
STATE_FIELDS: list[tuple[str, int]] = [
    ("left_arm_pose_pos", 3),
    ("left_arm_pose_rot", 6),
    ("left_gripper", 1),
    ("left_ft", 6),
    ("right_arm_pose_pos", 3),
    ("right_arm_pose_rot", 6),
    ("right_gripper", 1),
    ("right_ft", 6),
]
ACTION_FIELDS: list[tuple[str, int]] = [
    ("left_arm_pose_pos", 3),
    ("left_arm_pose_rot", 6),
    ("left_gripper", 1),
    ("right_arm_pose_pos", 3),
    ("right_arm_pose_rot", 6),
    ("right_gripper", 1),
]
STATE_DIM = sum(w for _, w in STATE_FIELDS)
ACTION_DIM = sum(w for _, w in ACTION_FIELDS)

# modality.json slices, derived from the layouts above so they cannot drift.
STATE_SLICES = {
    "left_eef": (0, 9),
    "left_gripper": (9, 10),
    "left_ft": (10, 16),
    "right_eef": (16, 25),
    "right_gripper": (25, 26),
    "right_ft": (26, 32),
}
ACTION_SLICES = {
    "left_eef": (0, 9),
    "left_gripper": (9, 10),
    "right_eef": (10, 19),
    "right_gripper": (19, 20),
}

# Stable task indices, so tasks.jsonl is identical across datasets and stable as data
# grows. Task text is read from each episode's own HDF5 attrs; the fallback below covers
# tasks absent from a given subset (e.g. only ube_v3 contains bracket_handover, so without
# it matcha_v3's tasks.jsonl would carry a placeholder at index 2). Episode attrs win, and
# a mismatch is reported as catalog drift rather than silently accepted.
TASK_ORDER = ["generic_pick", "generic_place", "bracket_handover"]
TASK_TEXT_FALLBACK = {
    "generic_pick": "Pick the grounded target object and hold it securely in the gripper.",
    "generic_place": "Place the currently held object at the grounded destination and release it.",
    "bracket_handover": (
        "Hand over the bracket from the gripper holding it to the opposite gripper and "
        "secure it in the receiving gripper."
    ),
}

ACTION_HORIZON = 40
FPS = 50
CHUNKS_SIZE = 1000
IMAGE_HW = (256, 256)
LEDGER_VERSION = 1

# QC gate thresholds (see the plan's Scope & Data Selection section).
QC_COV_MIN = 0.45
QC_COV_MAX = 0.80
QC_AGE_P99_MAX_MS = 150.0
QC_MIN_FRAMES = ACTION_HORIZON + 1

STATS_FILES = ("stats.json", "relative_stats.json")


# --- rot6d ------------------------------------------------------------------------


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0.0, 1.0, n)


def rot6d_source_to_groot(v6: np.ndarray) -> np.ndarray:
    """Re-encode source rot6d (first two **columns** of R) as GR00T rot6d (first two **rows**).

    Mirrors ``industrialnext_ai.common.rotation_representations._rot6d_to_matrix`` for the
    decode and ``gr00t.data.state_action.pose.EndEffectorPose._matrix_to_rot6d`` for the
    encode. Vectorised over a leading axis.

    Args:
        v6: ``(..., 6)`` source-convention rot6d.

    Returns:
        ``(..., 6)`` GR00T-convention rot6d.
    """
    v = np.asarray(v6, dtype=np.float64)
    if v.shape[-1] != 6:
        raise ValueError(f"rot6d must have last dim 6, got {v.shape}")
    b1 = _safe_normalize(v[..., :3])
    second = v[..., 3:6]
    second = second - np.sum(b1 * second, axis=-1, keepdims=True) * b1
    b2 = _safe_normalize(second)
    b3 = np.cross(b1, b2)
    # R has b1,b2,b3 as columns; GR00T wants the first two rows of R, i.e. the first two
    # components of each basis vector.
    return np.concatenate(
        [
            np.stack([b1[..., 0], b2[..., 0], b3[..., 0]], axis=-1),
            np.stack([b1[..., 1], b2[..., 1], b3[..., 1]], axis=-1),
        ],
        axis=-1,
    )


# --- field selection --------------------------------------------------------------


def resolve_field_slices(group: h5py.Group) -> dict[str, tuple[int, int]]:
    """Map field name -> (start, end) from an HDF5 group's own metadata.

    Never assume positions: the v3 subsets carry extra ``*_arm_joints`` fields that shift
    every later slice.
    """
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in group["field_names"][:]]
    slices = group["field_slices"][:]
    if len(names) != len(slices):
        raise ValueError(f"field_names/field_slices length mismatch: {len(names)} vs {len(slices)}")
    return {n: (int(a), int(b)) for n, (a, b) in zip(names, slices)}


def gather_fields(
    flat: np.ndarray, resolved: dict[str, tuple[int, int]], spec: list[tuple[str, int]], what: str
) -> np.ndarray:
    """Concatenate the named fields in ``spec`` order, transposing rot6d fields."""
    cols = []
    for name, width in spec:
        if name not in resolved:
            raise ValueError(f"{what}: required field {name!r} absent; have {sorted(resolved)}")
        a, b = resolved[name]
        if b - a != width:
            raise ValueError(f"{what}: field {name!r} width {b - a}, expected {width}")
        block = flat[:, a:b]
        if name.endswith("_pose_rot"):
            block = rot6d_source_to_groot(block)
        cols.append(block.astype(np.float32, copy=False))
    out = np.concatenate(cols, axis=1)
    expected = sum(w for _, w in spec)
    if out.shape[1] != expected:
        raise ValueError(f"{what}: assembled dim {out.shape[1]}, expected {expected}")
    return out


# --- QC ---------------------------------------------------------------------------


@dataclass
class EpisodeQC:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    per_camera: dict[str, dict[str, float]] = field(default_factory=dict)
    frame_count: int = 0
    policy_type: str = ""


def evaluate_qc(h5: h5py.File, episode_dir: Path) -> EpisodeQC:
    """Camera-health / provenance gate. Runs before any conversion work for an episode."""
    n = int(h5.attrs["frame_count"])
    policy_type = episode_dir.name.rsplit("_", 1)[-1]
    qc = EpisodeQC(ok=True, frame_count=n, policy_type=policy_type)

    if n < QC_MIN_FRAMES:
        qc.reasons.append(
            f"frame_count {n} < {QC_MIN_FRAMES} (yields no {ACTION_HORIZON}-step chunk)"
        )
    if policy_type != "expert":
        qc.reasons.append(f"policy_type {policy_type!r} != 'expert'")

    for canon, physical in CAMERA_MAP.items():
        if physical not in h5["images"]:
            qc.reasons.append(f"{canon}: missing group images/{physical}")
            continue
        g = h5["images"][physical]
        img = int(g.attrs["image_count"])
        cov = img / n if n else 0.0
        p99 = float(np.percentile(g["frame_age_ms"][:], 99))
        qc.per_camera[canon] = {"image_count": img, "coverage": cov, "age_p99_ms": p99}
        if cov < QC_COV_MIN:
            qc.reasons.append(f"{canon}: coverage {cov:.3f} < {QC_COV_MIN}")
        if cov > QC_COV_MAX:
            qc.reasons.append(
                f"{canon}: coverage {cov:.3f} > {QC_COV_MAX} (truncated state stream?)"
            )
        if p99 > QC_AGE_P99_MAX_MS:
            qc.reasons.append(f"{canon}: frame_age_ms p99 {p99:.0f} > {QC_AGE_P99_MAX_MS}")

    qc.ok = not qc.reasons
    return qc


# --- discovery & split ------------------------------------------------------------


def discover_episodes(subset: Path) -> list[Path]:
    """Deterministically ordered ``episode.h5`` paths, excluding failed recordings."""
    hits = sorted(subset.glob("*/20*/*/*/*/episode.h5"))
    return [p for p in hits if "_failed_recordings" not in p.parts]


def episode_key(subset: Path, src: Path) -> str:
    """Stable identity for a source episode: its path relative to the subset root."""
    return src.relative_to(subset).as_posix()


def assign_split(key: str, val_every: int) -> str:
    """Deterministic, insertion-stable train/val assignment.

    Hashing the episode key (rather than using its position, ``i % N``) means a new or
    backfilled recording never moves an existing episode between splits. Once assigned the
    choice is frozen in the ledger anyway; this only decides for newly-seen episodes.
    """
    if val_every <= 0:
        return "train"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return "val" if int.from_bytes(digest[:8], "big") % val_every == 0 else "train"


# --- ledger -----------------------------------------------------------------------


def load_ledger(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {"version": LEDGER_VERSION, "camera_map": CAMERA_MAP, "episodes": {}}
    with open(path) as f:
        led = json.load(f)
    if led.get("version") != LEDGER_VERSION:
        raise ValueError(f"{path}: ledger version {led.get('version')} != {LEDGER_VERSION}")
    if led.get("camera_map") != CAMERA_MAP:
        raise ValueError(
            f"{path}: ledger camera_map {led.get('camera_map')} differs from current "
            f"{CAMERA_MAP}. Converting with a different camera mapping into the same "
            f"output would mix viewpoints under one key; use a fresh --out-root."
        )
    return led


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# --- writers ----------------------------------------------------------------------


def encode_video(h5: h5py.File, physical: str, n_frames: int, dest: Path, crf: int) -> int:
    """Pipe the referenced source JPEGs into an h264 MP4, one frame per state frame."""
    g = h5["images"][physical]
    offsets = g["offsets"][:]
    ref = g["frame_ref_index"][:]
    image_count = int(g.attrs["image_count"])
    if len(ref) != n_frames:
        raise ValueError(f"{physical}: frame_ref_index len {len(ref)} != frame_count {n_frames}")
    if len(offsets) != image_count + 1:
        raise ValueError(
            f"{physical}: offsets len {len(offsets)} != image_count+1 {image_count + 1}"
        )
    lo, hi = int(ref.min()), int(ref.max())
    if lo < 0 or hi >= image_count:
        raise ValueError(f"{physical}: frame_ref_index out of range [{lo}, {hi}] vs {image_count}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = g["blob"]
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "image2pipe",
            "-framerate",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(FPS),
            str(dest),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for i in range(n_frames):
            j = int(ref[i])
            proc.stdin.write(bytes(blob[offsets[j] : offsets[j + 1]]))
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {physical} -> {dest}")
    return dest.stat().st_size


def write_meta(out: Path, episodes: list[dict], task_text: dict[str, str]) -> None:
    """Rewrite ``meta/`` to describe *all* episodes currently in the dataset."""
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    resolved_text = {}
    for uuid in TASK_ORDER:
        observed = task_text.get(uuid)
        fallback = TASK_TEXT_FALLBACK[uuid]
        if observed is not None and observed != fallback:
            print(
                f"WARNING: task {uuid!r} text differs from the catalog fallback; using the "
                f"episode's own text.\n  episode: {observed!r}\n  fallback: {fallback!r}",
                file=sys.stderr,
            )
        resolved_text[uuid] = observed or fallback

    with open(meta / "tasks.jsonl", "w") as f:
        for idx, uuid in enumerate(TASK_ORDER):
            f.write(json.dumps({"task_index": idx, "task": resolved_text[uuid]}) + "\n")

    with open(meta / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(
                json.dumps(
                    {
                        "episode_index": ep["episode_index"],
                        "tasks": [resolved_text[ep["task_uuid"]]],
                        "length": ep["length"],
                    }
                )
                + "\n"
            )

    video_features = {
        f"observation.images.{canon}": {
            "dtype": "video",
            "shape": [IMAGE_HW[0], IMAGE_HW[1], 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": IMAGE_HW[0],
                "video.width": IMAGE_HW[1],
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }
        for canon in CAMERA_MAP
    }
    max_idx = max((ep["episode_index"] for ep in episodes), default=-1)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "semihumanoid_bimanual",
        "total_episodes": len(episodes),
        "total_frames": sum(ep["length"] for ep in episodes),
        "total_tasks": len(TASK_ORDER),
        "total_videos": len(episodes) * len(CAMERA_MAP),
        "total_chunks": (max_idx // CHUNKS_SIZE) + 1 if episodes else 0,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        # generate_stats() only computes statistics for features whose dtype contains
        # "float" -- action and observation.state MUST be declared float32 here or
        # normalization silently has no entries to work with.
        "features": {
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": None},
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            **video_features,
        },
    }
    dump_json(meta / "info.json", info)

    modality = {
        "state": {k: {"start": a, "end": b} for k, (a, b) in STATE_SLICES.items()},
        "action": {k: {"start": a, "end": b} for k, (a, b) in ACTION_SLICES.items()},
        "video": {canon: {"original_key": f"observation.images.{canon}"} for canon in CAMERA_MAP},
        # No annotation.* parquet column: the loader follows original_key to task_index.
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    dump_json(meta / "modality.json", modality)


def invalidate_stats(out: Path) -> list[str]:
    """Delete cached stats so they regenerate over the new episode set.

    GR00T fingerprints its stats cache over the ``info.json`` feature *schema* only
    (``gr00t/data/stats.py:183``), which is unchanged by appending episodes. Leaving the
    files in place would silently normalize against the old, smaller set.
    """
    removed = []
    for name in STATS_FILES:
        p = out / "meta" / name
        if p.exists():
            p.unlink()
            removed.append(name)
    return removed


# --- per-episode work -------------------------------------------------------------


def convert_episode(
    src: Path, out: Path, episode_index: int, global_index: int, crf: int, overwrite: bool
) -> dict:
    """Convert one episode. Returns a record; raises on any inconsistency."""
    chunk = episode_index // CHUNKS_SIZE
    parquet = out / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    videos = {
        canon: out
        / f"videos/chunk-{chunk:03d}/observation.images.{canon}/episode_{episode_index:06d}.mp4"
        for canon in CAMERA_MAP
    }

    with h5py.File(src, "r") as h:
        n = int(h.attrs["frame_count"])
        task_uuid = str(h.attrs["task_uuid"])
        task_text = str(h.attrs["task_text"])
        catalog_version = str(h.attrs.get("task_catalog_version", ""))

        if not overwrite and parquet.exists() and all(v.exists() for v in videos.values()):
            return {
                "episode_index": episode_index,
                "length": n,
                "task_uuid": task_uuid,
                "task_text": task_text,
                "task_catalog_version": catalog_version,
                "video_bytes": 0,
                "skipped": True,
            }

        state = gather_fields(
            h["state/flat"][:], resolve_field_slices(h["state"]), STATE_FIELDS, "state"
        )
        action = gather_fields(
            h["action/executed"][:], resolve_field_slices(h["action"]), ACTION_FIELDS, "action"
        )
        if len(state) != n or len(action) != n:
            raise ValueError(
                f"row count mismatch: state {len(state)} action {len(action)} attr {n}"
            )
        residual_max = float(np.abs(h["action/residual"][:]).max())
        elapsed_ms = h["frame/elapsed_ms"][:]
        done = h["frame/done"][:]

        video_bytes = 0
        for canon, physical in CAMERA_MAP.items():
            video_bytes += encode_video(h, physical, n, videos[canon], crf)

    task_index = TASK_ORDER.index(task_uuid)
    df = pd.DataFrame(
        {
            "observation.state": list(state),
            "action": list(action),
            "timestamp": (np.asarray(elapsed_ms, dtype=np.float32) / 1000.0),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, episode_index, dtype=np.int64),
            "index": np.arange(global_index, global_index + n, dtype=np.int64),
            "task_index": np.full(n, task_index, dtype=np.int64),
            "next.done": np.asarray(done, dtype=bool),
        }
    )
    parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet, index=False)

    return {
        "episode_index": episode_index,
        "length": n,
        "task_uuid": task_uuid,
        "task_text": task_text,
        "task_catalog_version": catalog_version,
        "residual_max": residual_max,
        "video_bytes": video_bytes,
        "skipped": False,
    }


def _worker(args) -> tuple[str, dict | None, str | None]:
    key, src, out, ep_idx, g_idx, crf, overwrite = args
    try:
        return key, convert_episode(Path(src), Path(out), ep_idx, g_idx, crf, overwrite), None
    except Exception:
        return key, None, traceback.format_exc(limit=4)


# --- driver -----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source-subset", required=True, type=Path, help="e.g. .../flexiv_matcha_v3")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument(
        "--out-name", default=None, help="dataset name (default: subset name minus 'flexiv_')"
    )
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="convert at most N new episodes")
    ap.add_argument("--overwrite", action="store_true", help="re-encode episodes already present")
    ap.add_argument(
        "--val-every",
        type=int,
        default=20,
        help="hold out ~1/N of episodes as <name>_val (0 = no val split)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="report what would change, write nothing"
    )
    args = ap.parse_args()

    subset = args.source_subset.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    name = args.out_name or subset.name.replace("flexiv_", "")
    ledger_path = out_root / "_ledgers" / f"{name}.json"
    started = time.time()

    sources = discover_episodes(subset)
    if not sources:
        print(f"ERROR: no episodes found under {subset}", file=sys.stderr)
        return 1
    ledger = load_ledger(ledger_path)
    known = ledger["episodes"]
    print(f"[{name}] discovered {len(sources)} source episodes; ledger knows {len(known)}")

    # --- QC gate + ledger assignment, before any conversion work
    task_text: dict[str, str] = {}
    new_keys: list[str] = []
    drops: list[dict] = []
    # next free episode_index / frame offset per dataset, from the ledger
    next_idx: dict[str, int] = {}
    next_off: dict[str, int] = {}
    for rec in known.values():
        ds = rec["dataset"]
        next_idx[ds] = max(next_idx.get(ds, 0), rec["episode_index"] + 1)
        next_off[ds] = max(next_off.get(ds, 0), rec["index_offset"] + rec["length"])

    for src in sources:
        key = episode_key(subset, src)
        if key in known:
            continue
        with h5py.File(src, "r") as h:
            qc = evaluate_qc(h, src.parent)
            uuid = str(h.attrs["task_uuid"])
            text = str(h.attrs["task_text"])
            length = int(h.attrs["frame_count"])
        if not qc.ok:
            drops.append(
                {
                    "key": key,
                    "frame_count": qc.frame_count,
                    "policy_type": qc.policy_type,
                    "reasons": qc.reasons,
                    "per_camera": qc.per_camera,
                }
            )
            continue
        task_text.setdefault(uuid, text)
        split = assign_split(key, args.val_every)
        ds = name if split == "train" else f"{name}_val"
        known[key] = {
            "dataset": ds,
            "split": split,
            "episode_index": next_idx.get(ds, 0),
            "index_offset": next_off.get(ds, 0),
            "length": length,
            "task_uuid": uuid,
            "source": str(src),
        }
        next_idx[ds] = known[key]["episode_index"] + 1
        next_off[ds] = known[key]["index_offset"] + length
        new_keys.append(key)
        if args.limit and len(new_keys) >= args.limit:
            break

    print(f"[{name}] QC: {len(new_keys)} new episodes accepted, {len(drops)} dropped")
    for d in drops:
        print(f"[{name}]   DROP {d['key']}: {'; '.join(d['reasons'])}")
    if not new_keys and not args.overwrite:
        print(f"[{name}] nothing new to convert (dataset is up to date)")

    if args.dry_run:
        by_ds: dict[str, int] = {}
        for k in new_keys:
            by_ds[known[k]["dataset"]] = by_ds.get(known[k]["dataset"], 0) + 1
        print(f"[{name}] DRY RUN: would add {by_ds or '{}'}; would drop {len(drops)}")
        return 0

    # --- convert, per dataset
    reports: dict[str, dict] = {}
    todo_keys = list(known) if args.overwrite else new_keys
    datasets = sorted({known[k]["dataset"] for k in known})
    for ds in datasets:
        out = out_root / ds
        ds_todo = [k for k in todo_keys if known[k]["dataset"] == ds]
        jobs = [
            (
                k,
                known[k]["source"],
                str(out),
                known[k]["episode_index"],
                known[k]["index_offset"],
                args.crf,
                args.overwrite,
            )
            for k in ds_todo
        ]
        records: dict[str, dict] = {}
        errors: list[tuple[str, str]] = []
        if jobs:
            out.mkdir(parents=True, exist_ok=True)
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futs = [pool.submit(_worker, j) for j in jobs]
                    for i, fut in enumerate(as_completed(futs), 1):
                        k, rec, err = fut.result()
                        (errors.append((k, err)) if err else records.__setitem__(k, rec))
                        if i % 50 == 0 or i == len(futs):
                            print(f"[{ds}] {i}/{len(futs)}", flush=True)
            else:
                for i, j in enumerate(jobs, 1):
                    k, rec, err = _worker(j)
                    (errors.append((k, err)) if err else records.__setitem__(k, rec))
                    if i % 50 == 0 or i == len(jobs):
                        print(f"[{ds}] {i}/{len(jobs)}", flush=True)
            if errors:
                print(f"[{ds}] {len(errors)} FAILED episodes", file=sys.stderr)
                for k, err in errors[:3]:
                    print(f"  {k}:\n{err}", file=sys.stderr)
                # Persist the ledger minus the failures so a re-run retries only them.
                for k, _ in errors:
                    known.pop(k, None)
                dump_json(ledger_path, ledger)
                return 1
            for k, rec in records.items():
                task_text.setdefault(rec["task_uuid"], rec["task_text"])

        # Rewrite meta over every episode this dataset now holds (old + new).
        ds_eps = sorted(
            (
                {
                    "episode_index": r["episode_index"],
                    "length": r["length"],
                    "task_uuid": r["task_uuid"],
                }
                for k, r in known.items()
                if r["dataset"] == ds
            ),
            key=lambda e: e["episode_index"],
        )
        if not ds_eps:
            continue
        # Task text for pre-existing episodes comes from the previous tasks.jsonl.
        prior = out / "meta" / "tasks.jsonl"
        if prior.exists():
            with open(prior) as f:
                for line in f:
                    row = json.loads(line)
                    uuid = TASK_ORDER[row["task_index"]]
                    # Ignore a previous run's placeholder (text == uuid), otherwise it
                    # would outrank TASK_TEXT_FALLBACK forever once written.
                    if row["task"] != uuid:
                        task_text.setdefault(uuid, row["task"])
        write_meta(out, ds_eps, task_text)

        removed = invalidate_stats(out) if jobs else []
        if removed:
            print(f"[{ds}] invalidated cached {', '.join(removed)} (episode set changed)")

        report = {
            "dataset": ds,
            "source_subset": str(subset),
            "camera_map": CAMERA_MAP,
            "episodes": len(ds_eps),
            "frames": sum(e["length"] for e in ds_eps),
            "added_this_run": len(records),
            "video_bytes_this_run": sum(r["video_bytes"] for r in records.values()),
            "task_counts": {u: sum(1 for e in ds_eps if e["task_uuid"] == u) for u in TASK_ORDER},
            "task_catalog_versions": sorted(
                {r.get("task_catalog_version", "") for r in records.values()}
            ),
            "max_action_residual": max(
                (r.get("residual_max", 0.0) for r in records.values()), default=0.0
            ),
            "qc_dropped_this_run": drops,
            "stats_invalidated": removed,
            "elapsed_s": round(time.time() - started, 1),
        }
        dump_json(out / "_conversion_report.json", report)
        reports[ds] = report
        print(
            f"[{ds}] now {report['episodes']} eps / {report['frames']} frames"
            f" (+{report['added_this_run']} this run) -> {out}"
        )

    dump_json(ledger_path, ledger)
    print(
        f"[{name}] DONE ledger={len(known)} episodes across {len(reports)} dataset(s)"
        f" in {time.time() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
