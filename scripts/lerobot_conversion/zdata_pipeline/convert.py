# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental zdata_hdf5 conversion, append ledger, and LeRobot metadata."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import h5py
import pandas as pd

from .common import STATS_FILES, assert_output_mutable, transaction_journals
from .config import (
    PipelineConfig,
    ResolvedLayout,
    action_processing_json,
    layout_json,
    modality_json,
    modality_module_path,
    render_modality_module,
    resolve_source_subsets,
    write_json_if_changed,
    write_text_if_changed,
)
from .source import (
    SourceDescription,
    StagedSource,
    assign_split,
    discover_episodes,
    inspect_source,
    stage_source_worker,
)


LEDGER_VERSION = 2
LAYOUT_VERSION = 2
REPO_ROOT = Path(__file__).parents[3]


@dataclass
class TaskTable:
    rows: dict[int, str]
    task_indices: dict[str, int]

    def index_for(self, config: PipelineConfig, task_id: str, observed_text: str) -> int:
        preferred = config.tasks.text_overrides.get(task_id)
        if task_id in self.task_indices:
            index = self.task_indices[task_id]
            existing = self.rows[index]
            if preferred is not None:
                self.rows[index] = preferred
            elif observed_text != existing:
                print(
                    f"WARNING: task {task_id!r} text changed; keeping existing text\n"
                    f"  existing: {existing!r}\n  observed: {observed_text!r}",
                    file=sys.stderr,
                )
            return index
        index = max(self.rows, default=-1) + 1
        self.task_indices[task_id] = index
        self.rows[index] = preferred or observed_text
        return index


def _read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


@contextmanager
def _writer_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_root, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _roll_forward(output_root: Path, journal_path: Path) -> None:
    journal = _read_json(journal_path)
    for replacement in journal["replacements"]:
        staging = output_root / replacement["staging"]
        final = output_root / replacement["final"]
        final.parent.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            os.replace(staging, final)
        elif not final.exists():
            raise RuntimeError(
                f"cannot recover {journal_path}: both staging and final target are missing for {final}"
            )
    for relative_path in journal.get("invalidate_stats", []):
        (output_root / relative_path).unlink(missing_ok=True)
    for relative_path in journal.get("cleanup_paths", []):
        cleanup = output_root / relative_path
        if cleanup.is_dir():
            shutil.rmtree(cleanup)
    transaction_directory = output_root / journal["transaction_directory"]
    journal_path.unlink()
    if transaction_directory.is_dir():
        shutil.rmtree(transaction_directory)


def recover_transactions(output_root: Path) -> None:
    for journal in transaction_journals(output_root):
        print(f"recovering incomplete transaction {journal}")
        _roll_forward(output_root, journal)


def _task_rows(dataset: Path) -> dict[int, str]:
    loaded = _read_jsonl(dataset / "meta/tasks.jsonl")
    rows = {int(row["task_index"]): str(row["task"]) for row in loaded}
    if len(rows) != len(loaded):
        raise ValueError(f"{dataset}: tasks.jsonl contains duplicate task indices")
    return rows


def _legacy_task_index(
    config: PipelineConfig,
    subset: Path,
    source_key: str,
    dataset: Path,
    task_id: str,
    cache: dict[Path, dict[int, str]],
) -> int:
    if dataset not in cache:
        cache[dataset] = _task_rows(dataset)
    rows = cache[dataset]
    expected_text = config.tasks.text_overrides.get(task_id)
    if expected_text is None:
        source = subset / source_key
        with h5py.File(source, "r") as h5:
            raw = h5.attrs[config.tasks.text_attr]
            expected_text = raw.decode() if isinstance(raw, bytes) else str(raw)
    matches = [index for index, text in rows.items() if text == expected_text]
    if len(matches) != 1:
        raise ValueError(
            f"cannot map legacy task {task_id!r} in {dataset}: expected one tasks.jsonl text "
            f"match for {expected_text!r}, found {matches}"
        )
    return matches[0]


def _load_ledger(config: PipelineConfig, subset: Path, ledger_path: Path) -> tuple[dict, bool]:
    if not ledger_path.exists():
        return {"version": LEDGER_VERSION, "sources": {}}, False
    ledger = _read_json(ledger_path)
    version = ledger.get("version")
    if version == LEDGER_VERSION:
        sources = ledger.get("sources")
        if not isinstance(sources, dict):
            raise ValueError(f"{ledger_path}: version-2 ledger is missing sources")
        return ledger, False
    if version != 1:
        raise ValueError(f"{ledger_path}: unsupported ledger version {version!r}")
    if ledger.get("camera_map") != config.cameras:
        raise ValueError(
            f"{ledger_path}: legacy camera_map differs from configured cameras; use a new output root"
        )
    migrated = {"version": LEDGER_VERSION, "sources": {}}
    task_cache: dict[Path, dict[int, str]] = {}
    for source_key, record in ledger.get("episodes", {}).items():
        dataset = config.output.root / record["dataset"]
        task_id = str(record["task_uuid"])
        task_index = _legacy_task_index(
            config,
            subset,
            source_key,
            dataset,
            task_id,
            task_cache,
        )
        length = int(record["length"])
        migrated["sources"][source_key] = {
            "status": "complete",
            "task_id": task_id,
            "segments": [
                {
                    "source_start": 0,
                    "source_end": length,
                    "dataset": record["dataset"],
                    "split": record["split"],
                    "episode_index": int(record["episode_index"]),
                    "index_offset": int(record["index_offset"]),
                    "length": length,
                    "task_index": task_index,
                }
            ],
        }
    return migrated, True


def _segments_for_dataset(ledger: dict, dataset_name: str) -> list[dict]:
    return sorted(
        (
            segment
            for record in ledger["sources"].values()
            if record.get("status") == "complete"
            for segment in record["segments"]
            if segment["dataset"] == dataset_name
        ),
        key=lambda segment: int(segment["episode_index"]),
    )


def _task_table(config: PipelineConfig, ledger: dict, dataset_name: str) -> TaskTable:
    dataset = config.output.root / dataset_name
    rows = _task_rows(dataset)
    task_indices: dict[str, int] = {}
    for record in ledger["sources"].values():
        if record.get("status") != "complete":
            continue
        for segment in record["segments"]:
            if segment["dataset"] != dataset_name:
                continue
            task_id = str(record["task_id"])
            index = int(segment["task_index"])
            previous = task_indices.setdefault(task_id, index)
            if previous != index:
                raise ValueError(
                    f"dataset {dataset_name}: task {task_id!r} has indices {previous} and {index}"
                )
            if index not in rows:
                rows[index] = config.tasks.text_overrides.get(task_id, task_id)
    return TaskTable(rows=rows, task_indices=task_indices)


def _task_metadata(ledger: dict, dataset_name: str, tasks: TaskTable) -> dict[str, str]:
    segments = _segments_for_dataset(ledger, dataset_name)
    episodes = [
        {
            "episode_index": int(segment["episode_index"]),
            "tasks": [tasks.rows[int(segment["task_index"])]],
            "length": int(segment["length"]),
        }
        for segment in segments
    ]
    task_rows = [{"task_index": index, "task": text} for index, text in sorted(tasks.rows.items())]
    return {
        "meta/episodes.jsonl": _jsonl(episodes),
        "meta/tasks.jsonl": _jsonl(task_rows),
    }


def _refresh_task_overrides(
    config: PipelineConfig, ledger: dict, dataset_name: str, *, dry_run: bool
) -> None:
    tasks = _task_table(config, ledger, dataset_name)
    changed = False
    for task_id, task_index in tasks.task_indices.items():
        preferred = config.tasks.text_overrides.get(task_id)
        if preferred is not None and tasks.rows[task_index] != preferred:
            tasks.rows[task_index] = preferred
            changed = True
    if not changed:
        return

    dataset = config.output.root / dataset_name
    desired = _task_metadata(ledger, dataset_name, tasks)
    changed_metadata = {
        relative_path: text
        for relative_path, text in desired.items()
        if not (dataset / relative_path).exists() or (dataset / relative_path).read_text() != text
    }
    if not changed_metadata:
        return
    if dry_run:
        print(f"[{dataset_name}] DRY RUN: would update configured task text")
        return

    transaction_directory = (
        config.output.root
        / "_staging"
        / "transactions"
        / f"{dataset_name}-task-text-{uuid.uuid4().hex}"
    )
    replacements: list[tuple[Path, Path]] = []
    for relative_path, text in changed_metadata.items():
        staging = transaction_directory / relative_path
        write_text_if_changed(staging, text)
        replacements.append((staging, dataset / relative_path))
    _execute_transaction(
        config,
        dataset_name,
        transaction_directory,
        replacements,
        cleanup_paths=[],
        invalidate_stats=False,
    )
    print(f"[{dataset_name}] updated configured task text")


def _layout_for_description(config: PipelineConfig, description: SourceDescription) -> dict:
    if description.layout is None:
        raise ValueError(f"{description.key}: no resolved layout")
    return layout_json(config, description.layout)


def _validate_dataset_layout(dataset: Path, expected: dict, expected_modality: dict) -> None:
    layout_path = dataset / "_layout.json"
    if layout_path.exists():
        observed = _read_json(layout_path)
        legacy_expected = deepcopy(expected)
        legacy_expected.pop("version")
        legacy_expected.pop("output")
        legacy_expected.pop("video")
        for entry in legacy_expected["action"]["keys"]:
            for key in ("rep", "type", "format", "state_key"):
                entry.pop(key)
        if observed not in (expected, legacy_expected):
            raise ValueError(f"{dataset}: _layout.json differs from configured/source layout")
    modality_path = dataset / "meta/modality.json"
    if modality_path.exists() and _read_json(modality_path) != expected_modality:
        raise ValueError(f"{dataset}: meta/modality.json differs from configured/source layout")


def _layout_entry_contract(entries: list[dict], *, action: bool) -> list[dict]:
    keys = ["key", "fields", "rot6d"]
    if action:
        keys.extend(["rep", "type", "format", "state_key"])
    return [{key: entry.get(key) for key in keys} for entry in entries]


def _configured_entry_contract(config: PipelineConfig, *, action: bool) -> list[dict]:
    entries = config.action.keys if action else config.state
    keys = ["key", "fields", "rot6d"]
    if action:
        keys.extend(["rep", "type", "format", "state_key"])
    return [
        {key: list(entry.fields) if key == "fields" else getattr(entry, key) for key in keys}
        for entry in entries
    ]


def _validate_existing_output(config: PipelineConfig, dataset: Path) -> None:
    layout = _read_json(dataset / "_layout.json")
    if layout.get("cameras") != config.cameras or int(layout.get("fps", -1)) != config.source.fps:
        raise ValueError(f"{dataset}: stored cameras/fps differ from config")
    if _layout_entry_contract(layout.get("state", []), action=False) != (
        _configured_entry_contract(config, action=False)
    ):
        raise ValueError(f"{dataset}: stored state layout differs from config")
    action = layout.get("action", {})
    if (
        action.get("source") != config.action.source
        or int(action.get("horizon", -1)) != config.action.horizon
    ):
        raise ValueError(f"{dataset}: stored action source/horizon differs from config")
    if config.action.source == "observation" and any(
        action.get(key) != value for key, value in action_processing_json(config).items()
    ):
        raise ValueError(f"{dataset}: stored observation target semantics differ from config")
    if layout.get("version") == LAYOUT_VERSION:
        if _layout_entry_contract(action.get("keys", []), action=True) != (
            _configured_entry_contract(config, action=True)
        ):
            raise ValueError(f"{dataset}: stored action semantics differ from config")
        expected_output = {
            "robot_type": config.output.robot_type,
            "chunks_size": config.output.chunks_size,
        }
        expected_video = {
            "codec": config.video.codec,
            "pixel_format": config.video.pixel_format,
        }
        if layout.get("output") != expected_output or layout.get("video") != expected_video:
            raise ValueError(f"{dataset}: stored output/video layout differs from config")
    else:
        expected_action_fields = [
            {"key": entry.key, "fields": list(entry.fields), "rot6d": entry.rot6d}
            for entry in config.action.keys
        ]
        if _layout_entry_contract(action.get("keys", []), action=False) != (expected_action_fields):
            raise ValueError(f"{dataset}: stored action fields differ from config")

    with (dataset / "meta/info.json").open() as stream:
        info = json.load(stream)
    state_dim = max((int(end) for _, end in layout.get("state_slices", {}).values()), default=0)
    action_dim = max((int(end) for _, end in layout.get("action_slices", {}).values()), default=0)
    expected_paths = {
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
    }
    if (
        info.get("robot_type") != config.output.robot_type
        or int(info.get("chunks_size", -1)) != config.output.chunks_size
        or int(info.get("fps", -1)) != config.source.fps
        or any(info.get(key) != value for key, value in expected_paths.items())
    ):
        raise ValueError(f"{dataset}: info.json output values/path geometry differ from config")
    expected_numeric_features = {
        "observation.state": ("float32", [state_dim]),
        "action": ("float32", [action_dim]),
    }
    for key, (dtype, shape) in expected_numeric_features.items():
        feature = info.get("features", {}).get(key, {})
        if feature.get("dtype") != dtype or feature.get("shape") != shape:
            raise ValueError(f"{dataset}: info.json feature {key!r} differs from stored layout")
    expected_codec = "h264" if config.video.codec == "libx264" else config.video.codec
    for canonical in config.cameras:
        feature = info.get("features", {}).get(f"observation.images.{canonical}")
        if not feature or feature.get("dtype") != "video":
            raise ValueError(f"{dataset}: info.json is missing video feature {canonical!r}")
        video_info = feature.get("info", {})
        if (
            video_info.get("video.codec") != expected_codec
            or video_info.get("video.pix_fmt") != config.video.pixel_format
            or int(video_info.get("video.fps", -1)) != config.source.fps
        ):
            raise ValueError(f"{dataset}: info.json video values differ from config")

    expected_modality = {
        "state": {
            key: {"start": value[0], "end": value[1]}
            for key, value in layout["state_slices"].items()
        },
        "action": {
            key: {"start": value[0], "end": value[1]}
            for key, value in layout["action_slices"].items()
        },
        "video": {key: {"original_key": f"observation.images.{key}"} for key in config.cameras},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    if _read_json(dataset / "meta/modality.json") != expected_modality:
        raise ValueError(f"{dataset}: meta/modality.json differs from stored/configured layout")


def _validate_existing_outputs(config: PipelineConfig) -> None:
    if not config.output.root.is_dir():
        return
    for layout_path in sorted(config.output.root.glob("*/_layout.json")):
        _validate_existing_output(config, layout_path.parent)


def _legacy_layouts(
    config: PipelineConfig, subset: Path, ledger: dict
) -> dict[str, tuple[dict, dict]]:
    layouts: dict[str, tuple[dict, dict]] = {}
    for source_key, record in ledger["sources"].items():
        if record.get("status") != "complete":
            continue
        missing_datasets = {
            segment["dataset"]
            for segment in record["segments"]
            if segment["dataset"] not in layouts
        }
        if not missing_datasets:
            continue
        description = inspect_source(config, subset, subset / source_key)
        if description.layout is None:
            raise ValueError(f"legacy source {source_key} no longer resolves to a usable layout")
        direct = _layout_for_description(config, description)
        modality = modality_json(config, description.layout)
        for dataset_name in missing_datasets:
            layouts[dataset_name] = (direct, modality)
            _validate_dataset_layout(config.output.root / dataset_name, direct, modality)
    return layouts


def _metadata(
    config: PipelineConfig,
    ledger: dict,
    dataset_name: str,
    layout: ResolvedLayout,
    tasks: TaskTable,
) -> dict[str, str]:
    segments = _segments_for_dataset(ledger, dataset_name)
    task_metadata = _task_metadata(ledger, dataset_name, tasks)
    height, width, channels = layout.image_shape
    video_features = {
        f"observation.images.{canonical}": {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264" if config.video.codec == "libx264" else config.video.codec,
                "video.pix_fmt": config.video.pixel_format,
                "video.is_depth_map": False,
                "video.fps": config.source.fps,
                "video.channels": channels,
                "has_audio": False,
            },
        }
        for canonical in config.cameras
    }
    max_index = max((int(segment["episode_index"]) for segment in segments), default=-1)
    info = {
        "codebase_version": "v2.1",
        "robot_type": config.output.robot_type,
        "total_episodes": len(segments),
        "total_frames": sum(int(segment["length"]) for segment in segments),
        "total_tasks": len(tasks.rows),
        "total_videos": len(segments) * len(config.cameras),
        "total_chunks": max_index // config.output.chunks_size + 1 if segments else 0,
        "chunks_size": config.output.chunks_size,
        "fps": config.source.fps,
        "splits": {"train": f"0:{len(segments)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "action": {"dtype": "float32", "shape": [layout.action_dim], "names": None},
            "observation.state": {
                "dtype": "float32",
                "shape": [layout.state_dim],
                "names": None,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            **video_features,
        },
    }
    return {
        "meta/info.json": json.dumps(info, indent=4) + "\n",
        "meta/modality.json": json.dumps(modality_json(config, layout), indent=4) + "\n",
        **task_metadata,
        "_layout.json": json.dumps(layout_json(config, layout), indent=4) + "\n",
    }


def _relative_to_root(path: Path, output_root: Path) -> str:
    return path.relative_to(output_root).as_posix()


def _execute_transaction(
    config: PipelineConfig,
    dataset_name: str,
    transaction_directory: Path,
    replacements: list[tuple[Path, Path]],
    cleanup_paths: list[Path],
    invalidate_stats: bool,
) -> None:
    output_root = config.output.root
    dataset = output_root / dataset_name
    dataset.mkdir(parents=True, exist_ok=True)
    journal_path = dataset / ".sync_transaction.json"
    journal = {
        "version": 1,
        "transaction_directory": _relative_to_root(transaction_directory, output_root),
        "replacements": [
            {
                "staging": _relative_to_root(staging, output_root),
                "final": _relative_to_root(final, output_root),
            }
            for staging, final in replacements
        ],
        "invalidate_stats": [
            _relative_to_root(dataset / "meta" / filename, output_root) for filename in STATS_FILES
        ]
        if invalidate_stats
        else [],
        "cleanup_paths": [
            _relative_to_root(path, output_root) for path in sorted(set(cleanup_paths))
        ],
    }
    write_json_if_changed(journal_path, journal)
    _roll_forward(output_root, journal_path)


def _prepare_dataset_transaction(
    config: PipelineConfig,
    ledger_path: Path,
    ledger: dict,
    dataset_name: str,
    staged: list[tuple[StagedSource, list[dict]]],
    layout: ResolvedLayout,
    tasks: TaskTable,
) -> None:
    output_root = config.output.root
    transaction_directory = (
        output_root / "_staging" / "transactions" / f"{dataset_name}-{uuid.uuid4().hex}"
    )
    replacements: list[tuple[Path, Path]] = []
    for staged_source, assignments in staged:
        for staged_segment, assignment in zip(staged_source.segments, assignments):
            episode_index = int(assignment["episode_index"])
            index_offset = int(assignment["index_offset"])
            task_index = int(assignment["task_index"])
            chunk = episode_index // config.output.chunks_size
            dataframe = pd.read_parquet(staged_segment.parquet)
            dataframe["episode_index"] = episode_index
            dataframe["index"] = range(index_offset, index_offset + len(dataframe))
            dataframe["task_index"] = task_index
            dataframe = dataframe[
                [
                    "observation.state",
                    "action",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                    "next.done",
                ]
            ]
            staged_parquet = transaction_directory / "data" / f"episode_{episode_index:06d}.parquet"
            staged_parquet.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_parquet(staged_parquet, index=False)
            final_parquet = (
                output_root
                / dataset_name
                / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
            )
            replacements.append((staged_parquet, final_parquet))
            for canonical, staged_video in staged_segment.videos.items():
                final_video = (
                    output_root
                    / dataset_name
                    / f"videos/chunk-{chunk:03d}/observation.images.{canonical}/"
                    f"episode_{episode_index:06d}.mp4"
                )
                replacements.append((staged_video, final_video))

    for relative_path, text in _metadata(config, ledger, dataset_name, layout, tasks).items():
        staging = transaction_directory / "metadata" / relative_path
        write_text_if_changed(staging, text)
        replacements.append((staging, output_root / dataset_name / relative_path))
    staged_ledger = transaction_directory / "ledger.json"
    write_json_if_changed(staged_ledger, ledger)
    replacements.append((staged_ledger, ledger_path))
    _execute_transaction(
        config,
        dataset_name,
        transaction_directory,
        replacements,
        [staged_source.segments[0].parquet.parents[1] for staged_source, _ in staged],
        invalidate_stats=bool(staged),
    )


def _selector_map(
    subsets: list[Path], selectors: list[str]
) -> tuple[dict[Path, set[str]], set[str]]:
    selected = {subset: set() for subset in subsets}
    unmatched = set(selectors)
    for selector in selectors:
        for subset in subsets:
            prefix = f"{subset.name}/"
            if selector.startswith(prefix):
                selected[subset].add(selector[len(prefix) :])
                unmatched.discard(selector)
                break
    return selected, unmatched


def _print_description(prefix: str, description: SourceDescription) -> None:
    for warning in description.warnings:
        print(f"[{prefix}] WARNING {description.key}: {warning}")
    if description.skip_reason:
        print(f"[{prefix}] SKIP {description.key}: {description.skip_reason}")


def _stage_descriptions(
    config: PipelineConfig,
    descriptions: list[SourceDescription],
    run_stage: Path,
    workers: int,
) -> tuple[dict[str, StagedSource], dict[str, str]]:
    staged: dict[str, StagedSource] = {}
    failures: dict[str, str] = {}
    jobs = [
        (config, description, run_stage / uuid.uuid5(uuid.NAMESPACE_URL, description.key).hex)
        for description in descriptions
    ]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(stage_source_worker, *job) for job in jobs]
            for future in as_completed(futures):
                key, result, error = future.result()
                if error is None and result is not None:
                    staged[key] = result
                else:
                    failures[key] = error or "unknown staging failure"
    else:
        for job in jobs:
            key, result, error = stage_source_worker(*job)
            if error is None and result is not None:
                staged[key] = result
            else:
                failures[key] = error or "unknown staging failure"
    return staged, failures


def _next_indices(ledger: dict, dataset_name: str) -> tuple[int, int]:
    segments = _segments_for_dataset(ledger, dataset_name)
    next_episode = max((int(segment["episode_index"]) + 1 for segment in segments), default=0)
    next_offset = max(
        (int(segment["index_offset"]) + int(segment["length"]) for segment in segments),
        default=0,
    )
    return next_episode, next_offset


def sync_config(
    config: PipelineConfig,
    *,
    workers: int = 4,
    limit: int | None = None,
    reconvert: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if not dry_run:
        assert_output_mutable(config.output.root)
    subsets = resolve_source_subsets(config)
    selector_map, unmatched = _selector_map(subsets, reconvert or [])
    if unmatched:
        raise ValueError(
            f"--reconvert selectors do not match configured subsets: {sorted(unmatched)}"
        )

    if dry_run:
        return _sync_locked(config, subsets, selector_map, workers, limit, dry_run=True)
    with _writer_lock(config.output.root):
        recover_transactions(config.output.root)
        return _sync_locked(config, subsets, selector_map, workers, limit, dry_run=False)


def _sync_locked(
    config: PipelineConfig,
    subsets: list[Path],
    selector_map: dict[Path, set[str]],
    workers: int,
    limit: int | None,
    *,
    dry_run: bool,
) -> int:
    _validate_existing_outputs(config)
    module_path = modality_module_path(config, REPO_ROOT)
    module_text = render_modality_module(config)
    if dry_run:
        if not module_path.exists() or module_path.read_text() != module_text:
            print(f"DRY RUN: would update generated modality module {module_path}")
    else:
        if write_text_if_changed(module_path, module_text):
            print(f"updated generated modality module {module_path}")

    overall_failures: dict[str, str] = {}
    remaining_limit = limit
    for subset in subsets:
        name = config.output_name(subset.name)
        ledger_path = config.output.root / "_ledgers" / f"{name}.json"
        ledger, migrated = _load_ledger(config, subset, ledger_path)
        original_ledger = deepcopy(ledger)
        existing_dataset_names = {
            segment["dataset"]
            for record in ledger["sources"].values()
            if record.get("status") == "complete"
            for segment in record["segments"]
        }
        needs_layout_seed = migrated or any(
            not (config.output.root / dataset_name / "_layout.json").exists()
            or _read_json(config.output.root / dataset_name / "_layout.json").get("version")
            != LAYOUT_VERSION
            for dataset_name in existing_dataset_names
        )
        legacy_layouts = _legacy_layouts(config, subset, ledger) if needs_layout_seed else {}
        for dataset_name in sorted(existing_dataset_names):
            _refresh_task_overrides(config, ledger, dataset_name, dry_run=dry_run)
        sources = discover_episodes(config, subset)
        if not sources:
            overall_failures[subset.name] = "no source episodes discovered"
            continue
        source_by_key = {source.relative_to(subset).as_posix(): source for source in sources}
        selectors = selector_map[subset]
        missing_selectors = selectors - set(source_by_key)
        if missing_selectors:
            overall_failures[subset.name] = (
                f"reconvert paths not found: {sorted(missing_selectors)}"
            )
            continue

        candidates: list[Path] = []
        for key, source in source_by_key.items():
            if key in selectors or key not in ledger["sources"]:
                if key not in selectors and remaining_limit is not None and remaining_limit <= 0:
                    continue
                candidates.append(source)
                if key not in selectors and remaining_limit is not None:
                    remaining_limit -= 1

        descriptions: list[SourceDescription] = []
        layouts_by_dataset: dict[str, tuple[dict, dict, ResolvedLayout]] = {}
        for source in candidates:
            key = source.relative_to(subset).as_posix()
            try:
                description = inspect_source(config, subset, source)
                _print_description(name, description)
                old = ledger["sources"].get(key)
                if old and old.get("status") == "complete" and description.skip_reason:
                    raise ValueError("reconversion would remove existing segment assignments")
                if description.skip_reason:
                    ledger["sources"][key] = {
                        "status": "skipped",
                        "reason": description.skip_reason,
                        "segments": [],
                    }
                    continue
                if description.layout is None:
                    raise ValueError("accepted source has no resolved layout")
                split = (
                    old["segments"][0]["split"]
                    if old and old.get("status") == "complete"
                    else assign_split(key, config.output.val_every)
                )
                dataset_name = name if split == "train" else f"{name}_val"
                direct = _layout_for_description(config, description)
                modality = modality_json(config, description.layout)
                dataset = config.output.root / dataset_name
                _validate_dataset_layout(dataset, direct, modality)
                previous = layouts_by_dataset.setdefault(
                    dataset_name, (direct, modality, description.layout)
                )
                if previous[:2] != (direct, modality):
                    raise ValueError(f"new sources for {dataset_name} resolve to different layouts")
                descriptions.append(description)
            except Exception as error:
                overall_failures[f"{subset.name}/{key}"] = f"{type(error).__name__}: {error}"

        print(
            f"[{name}] discovered={len(sources)} "
            f"complete={sum(r.get('status') == 'complete' for r in ledger['sources'].values())} "
            f"candidates={len(candidates)} accepted={len(descriptions)} migrated_v1={migrated}"
        )
        if dry_run:
            if migrated:
                print(f"[{name}] DRY RUN: would migrate {ledger_path} to version 2")
            if ledger != original_ledger:
                print(f"[{name}] DRY RUN: ledger bookkeeping would change")
            continue

        run_stage = config.output.root / "_staging" / "runs" / uuid.uuid4().hex
        staged, failures = _stage_descriptions(config, descriptions, run_stage, workers)
        overall_failures.update({f"{subset.name}/{key}": error for key, error in failures.items()})

        changes_by_dataset: dict[str, list[tuple[str, StagedSource, list[dict], dict]]] = {}
        task_tables: dict[str, TaskTable] = {}
        next_values: dict[str, tuple[int, int]] = {}
        for description in sorted(descriptions, key=lambda item: item.key):
            if description.key not in staged:
                continue
            staged_source = staged[description.key]
            old = ledger["sources"].get(description.key)
            if old and old.get("status") == "complete":
                old_segments = old["segments"]
                new_lengths = [segment.length for segment in staged_source.segments]
                old_lengths = [int(segment["length"]) for segment in old_segments]
                if new_lengths != old_lengths:
                    overall_failures[f"{subset.name}/{description.key}"] = (
                        "reconversion changed segment count/length; use a new output root or rebuild"
                    )
                    continue
                dataset_name = str(old_segments[0]["dataset"])
                split = str(old_segments[0]["split"])
                assignments = deepcopy(old_segments)
            else:
                split = assign_split(description.key, config.output.val_every)
                dataset_name = name if split == "train" else f"{name}_val"
                if dataset_name not in next_values:
                    next_values[dataset_name] = _next_indices(ledger, dataset_name)
                next_episode, next_offset = next_values[dataset_name]
                assignments = []
                for segment in staged_source.segments:
                    assignments.append(
                        {
                            "source_start": segment.source_start,
                            "source_end": segment.source_end,
                            "dataset": dataset_name,
                            "split": split,
                            "episode_index": next_episode,
                            "index_offset": next_offset,
                            "length": segment.length,
                        }
                    )
                    next_episode += 1
                    next_offset += segment.length
                next_values[dataset_name] = (next_episode, next_offset)

            if dataset_name not in task_tables:
                task_tables[dataset_name] = _task_table(config, ledger, dataset_name)
            tasks = task_tables[dataset_name]
            if description.task_id is None or description.task_text is None:
                raise ValueError(f"{description.key}: accepted source is missing task metadata")
            task_index = tasks.index_for(config, description.task_id, description.task_text)
            for assignment, staged_segment in zip(assignments, staged_source.segments, strict=True):
                assignment["task_index"] = task_index
                if staged_segment.preprocessing is not None:
                    assignment["target_preprocessing"] = staged_segment.preprocessing
            current_stat = description.path.stat()
            if (
                description.source_size_bytes is None
                or description.source_mtime_ns is None
                or current_stat.st_size != description.source_size_bytes
                or current_stat.st_mtime_ns != description.source_mtime_ns
            ):
                overall_failures[f"{subset.name}/{description.key}"] = (
                    "source changed between inspection and conversion commit"
                )
                continue
            pending_record = {
                "status": "complete",
                "task_id": description.task_id,
                "source_size_bytes": description.source_size_bytes,
                "source_mtime_ns": description.source_mtime_ns,
                "segments": assignments,
            }
            changes_by_dataset.setdefault(dataset_name, []).append(
                (description.key, staged_source, assignments, pending_record)
            )

        for dataset_name in sorted(changes_by_dataset):
            for source_key, _, _, pending_record in changes_by_dataset[dataset_name]:
                ledger["sources"][source_key] = pending_record
            direct, modality, resolved_layout = layouts_by_dataset[dataset_name]
            _validate_dataset_layout(config.output.root / dataset_name, direct, modality)
            _prepare_dataset_transaction(
                config,
                ledger_path,
                ledger,
                dataset_name,
                [
                    (staged_source, assignments)
                    for _, staged_source, assignments, _ in changes_by_dataset[dataset_name]
                ],
                resolved_layout,
                task_tables[dataset_name],
            )
            print(
                f"[{dataset_name}] committed "
                f"{sum(len(item[2]) for item in changes_by_dataset[dataset_name])} segment(s)"
            )

        for dataset_name, (direct, _) in legacy_layouts.items():
            write_json_if_changed(config.output.root / dataset_name / "_layout.json", direct)
        if migrated and not changes_by_dataset:
            write_json_if_changed(ledger_path, ledger)
            print(f"[{name}] migrated legacy ledger/layout metadata without touching episode data")
        elif ledger != original_ledger and not changes_by_dataset:
            write_json_if_changed(ledger_path, ledger)

        if run_stage.is_dir():
            shutil.rmtree(run_stage)

    if overall_failures:
        print(f"sync completed with {len(overall_failures)} failure(s):", file=sys.stderr)
        for key, error in sorted(overall_failures.items()):
            print(f"  {key}: {error}", file=sys.stderr)
        return 1
    return 0
