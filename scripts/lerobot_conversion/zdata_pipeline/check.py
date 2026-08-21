# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset enumeration, statistics, structural checks, and training launch."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

import numpy as np
import pandas as pd

from .common import STATS_FILES, assert_no_incomplete_transactions, frozen_corpus_manifest_path
from .config import (
    PipelineConfig,
    modality_module_path,
    render_modality_module,
    resolve_source_subsets,
)


REPO_ROOT = Path(__file__).parents[3]
VAL_SUFFIX = "_val"


def find_datasets(output_root: Path) -> tuple[list[Path], list[Path]]:
    if not output_root.is_dir():
        return [], []
    datasets = sorted(path.parent.parent for path in output_root.glob("*/meta/info.json"))
    return (
        [path for path in datasets if not path.name.endswith(VAL_SUFFIX)],
        [path for path in datasets if path.name.endswith(VAL_SUFFIX)],
    )


def _modality_path(config: PipelineConfig) -> Path:
    path = modality_module_path(config, REPO_ROOT)
    expected = render_modality_module(config)
    if not path.exists():
        raise FileNotFoundError(f"generated modality module is missing: {path}; run sync first")
    if path.read_text() != expected:
        raise ValueError(f"generated modality module is stale: {path}; run sync first")
    return path


def stats_command(config: PipelineConfig, dataset: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "gr00t/data/stats.py"),
        "--dataset-path",
        str(dataset),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(_modality_path(config)),
    ]


def _run_stats(config: PipelineConfig, dataset: Path) -> tuple[Path, int]:
    environment = os.environ.copy()
    environment.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    result = subprocess.run(stats_command(config, dataset), env=environment, check=False)
    return dataset, result.returncode


def _generate_missing_stats(config: PipelineConfig, datasets: list[Path], jobs: int | None) -> int:
    missing = [
        dataset
        for dataset in datasets
        if any(not (dataset / "meta" / filename).exists() for filename in STATS_FILES)
    ]
    if not missing:
        print(f"statistics are current for {len(datasets)} dataset(s)")
        return 0
    frozen_manifest = frozen_corpus_manifest_path(config.output.root)
    if frozen_manifest.exists():
        raise RuntimeError(
            f"frozen corpus is missing statistics for {missing}; use a new output root"
        )
    worker_count = jobs if jobs is not None else min(4, len(missing))
    if worker_count <= 0:
        raise ValueError("--jobs must be positive")
    failures: list[Path] = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_run_stats, config, dataset) for dataset in missing]
        for future in as_completed(futures):
            dataset, return_code = future.result()
            if return_code:
                failures.append(dataset)
                print(f"stats FAILED for {dataset} (exit {return_code})", file=sys.stderr)
            else:
                print(f"stats complete for {dataset}")
    if failures:
        print(
            f"statistics failed for {len(failures)} dataset(s): "
            + ", ".join(path.name for path in failures),
            file=sys.stderr,
        )
        return 1
    return 0


def generate_missing_stats(config: PipelineConfig, jobs: int | None = None) -> int:
    assert_no_incomplete_transactions(config.output.root)
    train_datasets, validation_datasets = find_datasets(config.output.root)
    datasets = train_datasets + validation_datasets
    if not datasets:
        raise ValueError(f"no datasets with meta/info.json under {config.output.root}")
    return _generate_missing_stats(config, datasets, jobs)


def _ledger_segments(output_root: Path) -> dict[str, list[dict]]:
    by_dataset: dict[str, list[dict]] = {}
    for ledger_path in sorted((output_root / "_ledgers").glob("*.json")):
        with ledger_path.open() as stream:
            ledger = json.load(stream)
        if ledger.get("version") != 2:
            raise ValueError(f"{ledger_path}: run sync to migrate the version-1 ledger")
        for record in ledger.get("sources", {}).values():
            if record.get("status") != "complete":
                continue
            for segment in record["segments"]:
                by_dataset.setdefault(segment["dataset"], []).append(segment)
    for segments in by_dataset.values():
        segments.sort(key=lambda segment: int(segment["episode_index"]))
    return by_dataset


def _episode_paths(dataset: Path, info: dict, episode_index: int) -> tuple[Path, dict[str, Path]]:
    chunk = episode_index // int(info["chunks_size"])
    parquet = dataset / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    videos = {
        feature.removeprefix("observation.images."): dataset
        / info["video_path"].format(
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key=feature,
        )
        for feature, metadata in info["features"].items()
        if metadata.get("dtype") == "video"
    }
    return parquet, videos


def _decode_one_frame(video: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"cannot decode a frame from {video}")


def _video_frame_count(video: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not result.stdout.strip().isdigit():
        raise ValueError(f"cannot count frames in {video}: {result.stderr.strip()}")
    return int(result.stdout.strip())


def _check_parquet(
    parquet: Path,
    *,
    state_dim: int,
    action_dim: int,
    episode_index: int,
    task_index: int,
    length: int,
    index_offset: int | None = None,
) -> pd.DataFrame:
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    frame = pd.read_parquet(parquet)
    required = {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "next.done",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{parquet}: missing columns {sorted(missing)}")
    if len(frame) != length:
        raise ValueError(f"{parquet}: length {len(frame)} != metadata length {length}")
    if np.asarray(frame["observation.state"].iloc[0]).shape != (state_dim,):
        raise ValueError(f"{parquet}: observation.state shape mismatch")
    if np.asarray(frame["action"].iloc[0]).shape != (action_dim,):
        raise ValueError(f"{parquet}: action shape mismatch")
    if not np.issubdtype(np.asarray(frame["observation.state"].iloc[0]).dtype, np.floating):
        raise ValueError(f"{parquet}: observation.state is not floating point")
    if not np.issubdtype(np.asarray(frame["action"].iloc[0]).dtype, np.floating):
        raise ValueError(f"{parquet}: action is not floating point")
    if not (frame["episode_index"] == episode_index).all():
        raise ValueError(f"{parquet}: episode_index column mismatch")
    if not (frame["task_index"] == task_index).all():
        raise ValueError(f"{parquet}: task_index column mismatch")
    if index_offset is not None:
        expected = np.arange(index_offset, index_offset + length)
        if not np.array_equal(frame["index"].to_numpy(), expected):
            raise ValueError(f"{parquet}: global index is not contiguous")
    return frame


def _load_modality_config(config: PipelineConfig) -> dict:
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag

    tag = EmbodimentTag.NEW_EMBODIMENT.value
    if tag not in MODALITY_CONFIGS:
        path = _modality_path(config)
        spec = importlib.util.spec_from_file_location(f"{config.name}_generated_modality", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import generated modality module {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return MODALITY_CONFIGS[tag]


def _loader_sample(config: PipelineConfig, dataset: Path) -> None:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    loader = LeRobotEpisodeLoader(dataset, _load_modality_config(config))
    if not len(loader):
        raise ValueError(f"{dataset}: loader has no episodes")
    sample = loader[0]
    required = {
        *(f"video.{key}" for key in config.cameras),
        *(f"state.{entry.key}" for entry in config.state),
        *(f"action.{entry.key}" for entry in config.action.keys),
        "language.annotation.human.task_description",
    }
    missing = required - set(sample.columns)
    if missing:
        raise ValueError(f"{dataset}: loader sample misses columns {sorted(missing)}")


def _check_dataset(
    config: PipelineConfig, dataset: Path, ledger_segments: list[dict], full: bool
) -> None:
    meta = dataset / "meta"
    with (meta / "info.json").open() as stream:
        info = json.load(stream)
    episodes = _read_jsonl(meta / "episodes.jsonl")
    tasks = _read_jsonl(meta / "tasks.jsonl")
    with (meta / "modality.json").open() as stream:
        modality = json.load(stream)
    if info["total_episodes"] != len(episodes) or len(episodes) != len(ledger_segments):
        raise ValueError(f"{dataset}: ledger/episodes/info episode counts differ")
    if info["total_frames"] != sum(int(row["length"]) for row in episodes):
        raise ValueError(f"{dataset}: total_frames differs from episodes.jsonl")
    if info["total_tasks"] != len(tasks):
        raise ValueError(f"{dataset}: total_tasks differs from tasks.jsonl")
    video_count = sum(
        metadata.get("dtype") == "video" for metadata in info.get("features", {}).values()
    )
    if info["total_videos"] != len(episodes) * video_count:
        raise ValueError(f"{dataset}: total_videos differs from episodes/video features")
    max_episode_index = max((int(row["episode_index"]) for row in episodes), default=-1)
    expected_chunks = max_episode_index // int(info["chunks_size"]) + 1 if episodes else 0
    if info["total_chunks"] != expected_chunks:
        raise ValueError(f"{dataset}: total_chunks differs from episode indices/chunk size")
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    action_dim = int(info["features"]["action"]["shape"][0])
    if not episodes:
        raise ValueError(f"{dataset}: dataset has no episodes")
    task_by_index: dict[int, str] = {}
    for row in tasks:
        task_index = int(row["task_index"])
        if task_index in task_by_index:
            raise ValueError(f"{dataset}: duplicate task_index {task_index}")
        task_by_index[task_index] = str(row["task"])
    if set(task_by_index) != set(range(len(tasks))):
        raise ValueError(f"{dataset}: task indices are not contiguous")
    segment_by_index = {int(segment["episode_index"]): segment for segment in ledger_segments}
    if len(segment_by_index) != len(ledger_segments):
        raise ValueError(f"{dataset}: ledger contains duplicate episode indices")
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if episode_index not in segment_by_index:
            raise ValueError(f"{dataset}: episode {episode_index} is absent from the ledger")
        segment = segment_by_index[episode_index]
        task_index = int(segment["task_index"])
        if task_index not in task_by_index:
            raise ValueError(f"{dataset}: ledger references unknown task_index {task_index}")
        if episode.get("tasks") != [task_by_index[task_index]]:
            raise ValueError(
                f"{dataset}: episode {episode_index} task text differs from tasks.jsonl"
            )
        if int(episode["length"]) != int(segment["length"]):
            raise ValueError(f"{dataset}: episode {episode_index} length differs from the ledger")
    selected = min(episodes, key=lambda row: int(row["episode_index"]))
    selected_index = int(selected["episode_index"])
    selected_segment = segment_by_index[selected_index]
    parquet, videos = _episode_paths(dataset, info, selected_index)
    _check_parquet(
        parquet,
        state_dim=state_dim,
        action_dim=action_dim,
        episode_index=selected_index,
        task_index=int(selected_segment["task_index"]),
        length=int(selected["length"]),
        index_offset=int(selected_segment["index_offset"]),
    )
    expected_video_keys = {
        value["original_key"].removeprefix("observation.images.")
        for value in modality["video"].values()
    }
    if set(videos) != expected_video_keys:
        raise ValueError(f"{dataset}: info.json and modality.json video keys differ")
    for video in videos.values():
        if not video.is_file():
            raise FileNotFoundError(video)
        _decode_one_frame(video)

    if full:
        expected_global = 0
        for expected_episode, (episode, segment) in enumerate(zip(episodes, ledger_segments)):
            episode_index = int(episode["episode_index"])
            if (
                episode_index != expected_episode
                or int(segment["episode_index"]) != expected_episode
            ):
                raise ValueError(f"{dataset}: episode indices are not contiguous")
            length = int(episode["length"])
            if int(segment["length"]) != length or int(segment["index_offset"]) != expected_global:
                raise ValueError(f"{dataset}: ledger episode length/global offset mismatch")
            parquet, videos = _episode_paths(dataset, info, episode_index)
            frame = _check_parquet(
                parquet,
                state_dim=state_dim,
                action_dim=action_dim,
                episode_index=episode_index,
                task_index=int(segment["task_index"]),
                length=length,
                index_offset=expected_global,
            )
            if not np.array_equal(frame["frame_index"].to_numpy(), np.arange(length)):
                raise ValueError(f"{parquet}: frame_index is not contiguous")
            for video in videos.values():
                if _video_frame_count(video) != length:
                    raise ValueError(f"{video}: video frame count differs from episode length")
            expected_global += length

    if all((meta / filename).exists() for filename in STATS_FILES):
        _loader_sample(config, dataset)
    else:
        print(f"[{dataset.name}] loader sample deferred: statistics are missing")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def check_outputs(config: PipelineConfig, full: bool = False) -> int:
    assert_no_incomplete_transactions(config.output.root)
    train_datasets, validation_datasets = find_datasets(config.output.root)
    datasets = train_datasets + validation_datasets
    if not datasets:
        raise ValueError(f"no datasets with meta/info.json under {config.output.root}")
    ledger_segments = _ledger_segments(config.output.root)
    failures: dict[str, str] = {}
    for dataset in datasets:
        try:
            _check_dataset(config, dataset, ledger_segments.get(dataset.name, []), full)
            print(f"[{dataset.name}] check passed")
        except Exception as error:
            failures[dataset.name] = f"{type(error).__name__}: {error}"
    if failures:
        for dataset, error in failures.items():
            print(f"[{dataset}] check FAILED: {error}", file=sys.stderr)
        return 1
    return 0


def trainable_starts(datasets: list[Path], horizon: int) -> int:
    return sum(
        max(0, int(episode["length"]) - horizon + 1)
        for dataset in datasets
        for episode in _read_jsonl(dataset / "meta/episodes.jsonl")
    )


def _dataset_dimensions(datasets: list[Path]) -> tuple[int, int]:
    dimensions: set[tuple[int, int]] = set()
    for dataset in datasets:
        with (dataset / "meta/info.json").open() as stream:
            info = json.load(stream)
        dimensions.add(
            (
                int(info["features"]["observation.state"]["shape"][0]),
                int(info["features"]["action"]["shape"][0]),
            )
        )
    if len(dimensions) != 1:
        raise ValueError(
            f"training datasets have inconsistent state/action dimensions: {dimensions}"
        )
    return next(iter(dimensions))


def build_train_command(
    config: PipelineConfig,
    datasets: list[Path],
    *,
    resume_from: Path | None = None,
    now: Callable[[], datetime] | None = None,
    gpus_override: int | None = None,
    batch_override: int | None = None,
    max_steps_override: int | None = None,
    logging_steps_override: int | None = None,
    output_name_suffix: str = "",
) -> tuple[list[str], Path, int, int]:
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

    if not datasets:
        raise ValueError("no train datasets")
    if any(dataset.name.endswith(VAL_SUFFIX) for dataset in datasets):
        raise ValueError("validation dataset included in training paths")
    gpus = config.train.gpus if gpus_override is None else gpus_override
    batch = config.train.batch if batch_override is None else batch_override
    if gpus <= 0 or batch <= 0:
        raise ValueError("training GPU count and batch must be positive")
    if batch % gpus:
        raise ValueError(f"train.batch {batch} must be divisible by train.gpus {gpus}")
    state_dim, action_dim = _dataset_dimensions(datasets)
    model_limits = Gr00tN1d7Config()
    if state_dim > model_limits.max_state_dim or action_dim > model_limits.max_action_dim:
        raise ValueError(
            f"dataset dimensions state={state_dim}, action={action_dim} exceed N1.7 limits "
            f"{model_limits.max_state_dim}/{model_limits.max_action_dim}"
        )
    if config.action.horizon > model_limits.action_horizon:
        raise ValueError(
            f"action horizon {config.action.horizon} exceeds N1.7 limit {model_limits.action_horizon}"
        )
    if config.train.rtc_training_max_prefix_steps > config.action.horizon - 16:
        raise ValueError(
            "rtc_training_max_prefix_steps must leave at least 16 postfix action steps"
        )
    starts = trainable_starts(datasets, config.action.horizon)
    if starts <= 0:
        raise ValueError("training datasets contain no trainable action windows")
    steps = max_steps_override
    if steps is None:
        steps = (
            config.train.max_steps
            if config.train.max_steps is not None
            else math.ceil(float(config.train.epochs) * starts / batch)
        )
    if steps <= 0:
        raise ValueError(f"derived max_steps must be positive, got {steps}")

    if resume_from is not None:
        output_directory = resume_from.expanduser().resolve()
        if not output_directory.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {output_directory}")
    else:
        clock = now or (lambda: datetime.now(timezone.utc))
        timestamp = clock().astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_directory = config.train.out_base / f"{config.name}{output_name_suffix}_{timestamp}"
        if output_directory.exists():
            raise FileExistsError(f"fresh training directory already exists: {output_directory}")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(gpus),
        str(REPO_ROOT / "gr00t/experiment/launch_finetune.py"),
        "--base-model-path",
        config.train.base_model,
        "--dataset-path",
        os.pathsep.join(str(dataset) for dataset in datasets),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(_modality_path(config)),
        "--num-gpus",
        str(gpus),
        "--output-dir",
        str(output_directory),
        "--max-steps",
        str(steps),
        "--save-steps",
        str(config.train.save_steps),
        "--save-total-limit",
        str(config.train.save_total_limit),
        "--global-batch-size",
        str(batch),
        "--dataloader-num-workers",
        str(config.train.workers),
        "--learning-rate",
        str(config.train.lr),
        "--warmup-ratio",
        str(config.train.warmup_ratio),
        "--weight-decay",
        str(config.train.weight_decay),
        "--state-dropout-prob",
        str(config.train.state_dropout_prob),
        "--rtc-training-max-prefix-steps",
        str(config.train.rtc_training_max_prefix_steps),
        "--shortest-image-edge",
        str(config.train.shortest_image_edge),
        "--crop-fraction",
        str(config.train.crop_fraction),
    ]
    if config.train.color_jitter:
        command.append("--color-jitter-params")
        for key, value in config.train.color_jitter.items():
            command.extend([key, str(value)])
    if config.train.use_wandb:
        command.extend(["--use-wandb", "--wandb-project", config.train.wandb_project])
    if logging_steps_override is not None:
        if logging_steps_override <= 0:
            raise ValueError("logging_steps_override must be positive")
        command.extend(["--logging-steps", str(logging_steps_override)])
    if resume_from is not None:
        command.append("--resume-from-checkpoint")
    return command, output_directory, int(steps), starts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=False,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def _base_model_revision(model: str) -> str:
    local = Path(model).expanduser()
    if local.is_dir():
        config_path = local / "config.json"
        if not config_path.is_file():
            raise ValueError(f"local base model lacks config.json: {local}")
        return f"local-config-sha256:{_sha256(config_path)}"
    from huggingface_hub import HfApi, try_to_load_from_cache

    cached = try_to_load_from_cache(model, "config.json")
    if isinstance(cached, str):
        parts = Path(cached).parts
        if "snapshots" in parts:
            return parts[parts.index("snapshots") + 1]
    try:
        return str(HfApi().model_info(model).sha)
    except Exception as exc:
        raise RuntimeError(f"could not resolve immutable base-model revision for {model}") from exc


def _repository_state() -> tuple[str, str]:
    repository_diff = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    return _git_output("rev-parse", "HEAD"), hashlib.sha256(repository_diff).hexdigest()


def _artifact_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _source_stat_inventory(config: PipelineConfig) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for subset in resolve_source_subsets(config):
        ledger_path = config.output.root / "_ledgers" / f"{config.output_name(subset.name)}.json"
        if not ledger_path.is_file():
            raise FileNotFoundError(f"missing conversion ledger: {ledger_path}")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("version") != 2:
            raise ValueError(f"unsupported conversion ledger version: {ledger_path}")
        for source_key, record in sorted(ledger.get("sources", {}).items()):
            if record.get("status") != "complete":
                continue
            source = subset / source_key
            if not source.is_file():
                raise FileNotFoundError(f"admitted source is missing: {source}")
            stat = source.stat()
            expected_size = record.get("source_size_bytes")
            expected_mtime = record.get("source_mtime_ns")
            if expected_size is None or expected_mtime is None:
                raise RuntimeError(f"conversion ledger lacks a source stat guard: {source}")
            if stat.st_size != expected_size or stat.st_mtime_ns != expected_mtime:
                raise RuntimeError(f"admitted source changed after conversion: {source}")
            inventory.append(
                {
                    "path": str(source.resolve()),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return inventory


def _bind_missing_source_stat_guards(config: PipelineConfig) -> None:
    """Bind legacy guard-less ledger rows immediately before the one-time corpus freeze."""
    for subset in resolve_source_subsets(config):
        ledger_path = config.output.root / "_ledgers" / f"{config.output_name(subset.name)}.json"
        if not ledger_path.is_file():
            raise FileNotFoundError(f"missing conversion ledger: {ledger_path}")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        changed = False
        for source_key, record in sorted(ledger.get("sources", {}).items()):
            if record.get("status") != "complete":
                continue
            source = subset / source_key
            stat = source.stat()
            expected_size = record.get("source_size_bytes")
            expected_mtime = record.get("source_mtime_ns")
            if expected_size is None and expected_mtime is None:
                record["source_size_bytes"] = stat.st_size
                record["source_mtime_ns"] = stat.st_mtime_ns
                changed = True
            elif expected_size != stat.st_size or expected_mtime != stat.st_mtime_ns:
                raise RuntimeError(f"admitted source changed after conversion: {source}")
        if changed:
            temporary_path = ledger_path.with_name(f".{ledger_path.name}.tmp")
            temporary_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
            temporary_path.replace(ledger_path)


def _converted_artifact_inventory(
    config: PipelineConfig, datasets: list[Path]
) -> list[dict[str, object]]:
    converted_paths = {
        path.resolve() for dataset in datasets for path in dataset.rglob("*") if path.is_file()
    }
    converted_paths.update(
        path.resolve() for path in (config.output.root / "_ledgers").glob("*.json")
    )
    root_manifest = config.output.root / "manifest.json"
    if root_manifest.is_file():
        converted_paths.add(root_manifest.resolve())
    return [_artifact_record(path) for path in sorted(converted_paths)]


def verify_frozen_corpus(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported frozen corpus manifest: {manifest_path}")
    for section in ("pipeline_config", "modality_module"):
        record = manifest[section]
        path = Path(record["path"])
        if path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen corpus verification failed for {path}")
    for record in manifest["converted_artifact_inventory"]:
        path = Path(record["path"])
        if path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen corpus verification failed for {path}")
    for record in manifest["source_stat_inventory"]:
        stat = Path(record["path"]).stat()
        if stat.st_size != record["size_bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise RuntimeError(f"frozen corpus source guard changed: {record['path']}")


def freeze_corpus(config: PipelineConfig) -> int:
    """Validate and freeze one converted corpus against further pipeline writes."""
    assert_no_incomplete_transactions(config.output.root)
    manifest_path = frozen_corpus_manifest_path(config.output.root)
    if manifest_path.exists():
        verify_frozen_corpus(manifest_path)
        print(f"frozen corpus manifest is current: {manifest_path}")
        return 0
    _bind_missing_source_stat_guards(config)
    if check_outputs(config, full=True):
        return 1
    train_datasets, validation_datasets = find_datasets(config.output.root)
    datasets = train_datasets + validation_datasets
    missing = [
        dataset
        for dataset in datasets
        if any(not (dataset / "meta" / filename).exists() for filename in STATS_FILES)
    ]
    if missing:
        raise RuntimeError(f"cannot freeze corpus with missing statistics: {missing}")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_name": config.name,
        "pipeline_config": _artifact_record(config.config_path),
        "modality_module": _artifact_record(_modality_path(config).resolve()),
        "source_stat_inventory": _source_stat_inventory(config),
        "converted_artifact_inventory": _converted_artifact_inventory(config, datasets),
        "datasets": [str(path.resolve()) for path in datasets],
        "action_horizon": config.action.horizon,
        "fps": config.source.fps,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(manifest_path)
    verify_frozen_corpus(manifest_path)
    print(f"froze converted corpus: {manifest_path}")
    return 0


def create_training_manifest(
    config: PipelineConfig,
    datasets: list[Path],
    command: list[str],
    output_directory: Path,
    *,
    steps: int,
    starts: int,
    batch: int,
) -> Path:
    """Write a content-bound fresh-run manifest atomically and verify it."""
    frozen_manifest = frozen_corpus_manifest_path(config.output.root)
    if not frozen_manifest.is_file():
        raise FileNotFoundError(f"training requires a frozen corpus: {frozen_manifest}")
    verify_frozen_corpus(frozen_manifest)
    modality_path = _modality_path(config).resolve()
    repository_revision, repository_diff_sha256 = _repository_state()
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_name": config.name,
        "pipeline_config": _artifact_record(config.config_path),
        "modality_module": _artifact_record(modality_path),
        "source_stat_inventory": _source_stat_inventory(config),
        "frozen_corpus_manifest": _artifact_record(frozen_manifest),
        "converted_artifact_inventory": _converted_artifact_inventory(config, datasets),
        "datasets": [str(path.resolve()) for path in datasets],
        "action_horizon": config.action.horizon,
        "fps": config.source.fps,
        "trainable_starts": starts,
        "max_steps": steps,
        "global_batch_size": batch,
        "effective_epochs": steps * batch / starts,
        "rtc_prefix_distribution": {
            "kind": "uniform_integer_inclusive",
            "minimum": 0,
            "maximum": config.train.rtc_training_max_prefix_steps,
        },
        "base_model": config.train.base_model,
        "base_model_revision": _base_model_revision(config.train.base_model),
        "repository_revision": repository_revision,
        "repository_dirty_diff_sha256": repository_diff_sha256,
        "command": command,
    }
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir()
    manifest_path = output_directory / "run_manifest.json"
    temporary_path = output_directory / ".run_manifest.json.tmp"
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(manifest_path)
    verify_training_manifest(manifest_path)
    return manifest_path


def verify_training_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for section in ("pipeline_config", "modality_module", "frozen_corpus_manifest"):
        record = manifest[section]
        path = Path(record["path"])
        if path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"training manifest verification failed for {path}")
    for record in manifest["converted_artifact_inventory"]:
        path = Path(record["path"])
        if path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"training manifest verification failed for {path}")
    for record in manifest["source_stat_inventory"]:
        stat = Path(record["path"]).stat()
        if stat.st_size != record["size_bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise RuntimeError(f"source stat guard changed before training: {record['path']}")


def validate_training_smoke(output_directory: Path, expected_prefix_max: int) -> None:
    """Reject a nominally successful smoke that did not exercise the requested objective."""
    saved_config_path = output_directory / "config.json"
    instantiated_config_path = output_directory / "experiment_cfg/final_model_config.json"
    checkpoints = sorted(
        output_directory.glob("checkpoint-*"),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    if not checkpoints:
        raise RuntimeError("training smoke did not save a checkpoint")
    trainer_state_path = checkpoints[-1] / "trainer_state.json"
    for path in (saved_config_path, instantiated_config_path, trainer_state_path):
        if not path.is_file():
            raise RuntimeError(f"training smoke artifact is missing: {path}")
    for path in (saved_config_path, instantiated_config_path):
        saved_config = json.loads(path.read_text(encoding="utf-8"))
        if saved_config.get("rtc_training_max_prefix_steps", 0) != expected_prefix_max:
            raise RuntimeError(
                f"training smoke instantiated RTC prefix maximum "
                f"{saved_config.get('rtc_training_max_prefix_steps', 0)} instead of "
                f"{expected_prefix_max}: {path}"
            )

    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    log_history = trainer_state.get("log_history", [])
    losses = [float(row["loss"]) for row in log_history if "loss" in row]
    postfix_counts = [
        float(row["rtc_postfix_valid_elements"])
        for row in log_history
        if "rtc_postfix_valid_elements" in row
    ]
    prefix_counts = {
        index: sum(float(row.get(f"rtc_prefix_count_{index}", 0.0)) for row in log_history)
        for index in range(expected_prefix_max + 1)
    }
    if not losses or not all(math.isfinite(value) for value in losses):
        raise RuntimeError("training smoke did not record finite loss")
    if not postfix_counts or not all(
        math.isfinite(value) and value > 0 for value in postfix_counts
    ):
        raise RuntimeError("training smoke did not record positive postfix coverage")
    if expected_prefix_max > 0:
        if prefix_counts[0] <= 0:
            raise RuntimeError("training smoke did not sample the zero-prefix objective")
        if sum(count > 0 for count in prefix_counts.values()) < 2:
            raise RuntimeError("training smoke did not sample varying prefix lengths")


def _report_resources(output_base: Path) -> None:
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            print(f"WARNING: could not query GPU occupancy: {error}", file=sys.stderr)
        else:
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    try:
                        index, used = (part.strip() for part in line.split(",", 1))
                        used_mib = int(used)
                    except ValueError:
                        print(
                            f"WARNING: ignoring malformed nvidia-smi output: {line!r}",
                            file=sys.stderr,
                        )
                        continue
                    if used_mib > 1000:
                        print(f"WARNING: GPU {index} already uses {used} MiB", file=sys.stderr)
    probe = output_base.expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as error:
        print(f"WARNING: could not query free disk at {probe}: {error}", file=sys.stderr)
    else:
        print(f"training output free disk: {usage.free / 1024**3:.1f} GiB at {probe}")


def train(
    config: PipelineConfig,
    jobs: int | None = None,
    resume_from: Path | None = None,
    *,
    smoke_max_steps: int | None = None,
    smoke_batch: int = 1,
) -> int:
    if smoke_max_steps is not None:
        if resume_from is not None:
            raise ValueError("smoke launch cannot resume an existing run")
        if smoke_max_steps <= 0 or smoke_batch <= 0:
            raise ValueError("smoke_max_steps and smoke_batch must be positive")
    assert_no_incomplete_transactions(config.output.root)
    frozen_manifest = frozen_corpus_manifest_path(config.output.root)
    if not frozen_manifest.is_file():
        raise FileNotFoundError(f"training requires a frozen corpus: {frozen_manifest}")
    verify_frozen_corpus(frozen_manifest)
    train_datasets, _ = find_datasets(config.output.root)
    if not train_datasets:
        raise ValueError(f"no train datasets under {config.output.root}")
    if _generate_missing_stats(config, train_datasets, jobs):
        return 1
    missing = [
        dataset
        for dataset in train_datasets
        if any(not (dataset / "meta" / filename).exists() for filename in STATS_FILES)
    ]
    if missing:
        raise RuntimeError(f"statistics remain missing after generation: {missing}")
    _load_modality_config(config)
    for dataset in train_datasets:
        _loader_sample(config, dataset)
    command, output_directory, steps, starts = build_train_command(
        config,
        train_datasets,
        resume_from=resume_from,
        gpus_override=1 if smoke_max_steps is not None else None,
        batch_override=smoke_batch if smoke_max_steps is not None else None,
        max_steps_override=smoke_max_steps,
        logging_steps_override=1 if smoke_max_steps is not None else None,
        output_name_suffix="_smoke" if smoke_max_steps is not None else "",
    )
    effective_batch = smoke_batch if smoke_max_steps is not None else config.train.batch
    effective_epochs = steps * effective_batch / starts
    print(
        f"training datasets={len(train_datasets)} starts={starts} max_steps={steps} "
        f"effective_epochs={effective_epochs:.3f} output={output_directory}"
    )
    print("launch:", " ".join(command))
    _report_resources(config.train.out_base)
    if resume_from is None:
        manifest_path = create_training_manifest(
            config,
            train_datasets,
            command,
            output_directory,
            steps=steps,
            starts=starts,
            batch=effective_batch,
        )
        print(f"verified training manifest: {manifest_path}")
    else:
        manifest_path = output_directory / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"resume run lacks run_manifest.json: {output_directory}")
        verify_training_manifest(manifest_path)
    environment = os.environ.copy()
    environment.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    return_code = subprocess.run(command, env=environment, check=False).returncode
    if return_code == 0 and smoke_max_steps is not None:
        validate_training_smoke(
            output_directory,
            config.train.rtc_training_max_prefix_steps,
        )
        print(f"validated training smoke: {output_directory}")
    return return_code
