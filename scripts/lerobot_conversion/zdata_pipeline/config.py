# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and generated artifacts for the zdata_hdf5 conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import warnings

import yaml


ROT6D_TRANSFORM = "source_columns_to_groot_rows"
ACTION_REPRESENTATIONS = {"RELATIVE", "DELTA", "ABSOLUTE"}
ACTION_TYPES = {"EEF", "NON_EEF"}
ACTION_FORMATS = {"DEFAULT", "XYZ_ROT6D", "XYZ_ROTVEC"}


@dataclass(frozen=True)
class SourceConfig:
    root: Path
    subsets: tuple[str, ...]
    episode_glob: str = "*/20*/*/*/*/episode.h5"
    exclude_path_contains: tuple[str, ...] = ("_failed_recordings",)
    fps: int = 50


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    robot_type: str
    strip_subset_prefix: str = ""
    val_every: int = 0
    chunks_size: int = 1000


@dataclass(frozen=True)
class VideoConfig:
    codec: str = "libx264"
    crf: int = 23
    preset: str = "veryfast"
    pixel_format: str = "yuv420p"


@dataclass(frozen=True)
class LayoutEntry:
    key: str
    fields: tuple[str, ...]
    rot6d: str | None = None
    rep: str | None = None
    type: str | None = None
    format: str | None = None
    state_key: str | None = None


@dataclass(frozen=True)
class ActionLayout:
    source: str
    horizon: int
    keys: tuple[LayoutEntry, ...]


@dataclass(frozen=True)
class TasksConfig:
    id_attr: str = "task_uuid"
    text_attr: str = "task_text"
    text_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionConfig:
    require_valid_for_training: bool = False
    policy_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class WarningConfig:
    camera_coverage_below: float | None = None
    camera_coverage_above: float | None = None
    camera_age_p99_ms_above: float | None = None
    frame_gap_ms_above: float | None = None
    nonzero_action_residual: bool = False


@dataclass(frozen=True)
class ContinuityConfig:
    split_on_gap_ms: float | None = None


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str = "nvidia/GR00T-N1.7-3B"
    out_base: Path = Path("outputs")
    gpus: int = 1
    batch: int = 32
    epochs: float | None = 1.0
    max_steps: int | None = None
    lr: float = 1e-4
    workers: int = 2
    save_steps: int = 1000
    save_total_limit: int = 5
    state_dropout_prob: float = 0.2
    shortest_image_edge: int = 256
    crop_fraction: float = 0.95
    color_jitter: dict[str, float] = field(default_factory=dict)
    use_wandb: bool = False
    wandb_project: str = "finetune-gr00t-n1d7"
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    source: SourceConfig
    output: OutputConfig
    cameras: dict[str, str]
    video: VideoConfig
    state: tuple[LayoutEntry, ...]
    action: ActionLayout
    tasks: TasksConfig
    select: SelectionConfig
    warn: WarningConfig
    continuity: ContinuityConfig
    train: TrainingConfig
    config_path: Path

    def output_name(self, subset_name: str) -> str:
        prefix = self.output.strip_subset_prefix
        name = (
            subset_name[len(prefix) :] if prefix and subset_name.startswith(prefix) else subset_name
        )
        if not name:
            raise ValueError(f"subset {subset_name!r} becomes an empty output name")
        return name


@dataclass(frozen=True)
class ResolvedEntry:
    entry: LayoutEntry
    fields: tuple[tuple[str, int], ...]
    start: int
    end: int
    rot6d_field: str | None


@dataclass(frozen=True)
class ResolvedLayout:
    state: tuple[ResolvedEntry, ...]
    action: tuple[ResolvedEntry, ...]
    state_slices: dict[str, tuple[int, int]]
    action_slices: dict[str, tuple[int, int]]
    state_dim: int
    action_dim: int
    image_shape: tuple[int, int, int]


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _warn_unknown(data: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        warnings.warn(f"{context}: ignoring unknown keys {unknown}", stacklevel=3)


def _require(data: dict[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise ValueError(f"{context}: missing required key {key!r}")
    return data[key]


def _path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty path string")
    return Path(value).expanduser()


def _string(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{context} must be {qualifier}")
    return value


def _string_tuple(value: Any, context: str, *, allow_empty_list: bool = False) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty_list)
        or not all(isinstance(item, str) and item for item in value)
    ):
        qualifier = "a list" if allow_empty_list else "a non-empty list"
        raise ValueError(f"{context} must be {qualifier} of non-empty strings")
    return tuple(value)


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def _optional_float(value: Any, context: str, *, maximum: float | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a finite non-negative number or null")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a finite non-negative number or null") from error
    if not math.isfinite(parsed) or parsed < 0 or (maximum is not None and parsed > maximum):
        range_text = f"between 0 and {maximum}" if maximum is not None else "non-negative"
        raise ValueError(f"{context} must be a finite {range_text} number or null")
    return parsed


def _layout_entries(value: Any, context: str, *, action: bool) -> tuple[LayoutEntry, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list")
    allowed = {"key", "fields", "rot6d"}
    if action:
        allowed |= {"rep", "type", "format", "state_key"}
    entries: list[LayoutEntry] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        item = _mapping(raw, f"{context}[{index}]")
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise ValueError(f"{context}[{index}]: unknown layout keys {unknown}")
        key = _require(item, "key", f"{context}[{index}]")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{context}[{index}].key must be a non-empty string")
        if key in seen:
            raise ValueError(f"{context}: duplicate key {key!r}")
        seen.add(key)
        fields = _string_tuple(
            _require(item, "fields", f"{context}[{index}]"), f"{context}[{index}].fields"
        )
        rot6d = item.get("rot6d")
        if rot6d is not None and rot6d != ROT6D_TRANSFORM:
            raise ValueError(f"{context}[{index}].rot6d must be {ROT6D_TRANSFORM!r}, got {rot6d!r}")
        rep = item.get("rep")
        action_type = item.get("type")
        action_format = item.get("format")
        state_key = item.get("state_key")
        if action:
            if rep not in ACTION_REPRESENTATIONS:
                raise ValueError(
                    f"{context}[{index}].rep must be one of {sorted(ACTION_REPRESENTATIONS)}"
                )
            if action_type not in ACTION_TYPES:
                raise ValueError(f"{context}[{index}].type must be one of {sorted(ACTION_TYPES)}")
            if action_format not in ACTION_FORMATS:
                raise ValueError(
                    f"{context}[{index}].format must be one of {sorted(ACTION_FORMATS)}"
                )
            if action_type == "EEF" and action_format != "XYZ_ROT6D":
                raise ValueError(f"{context}[{index}]: EEF actions must use XYZ_ROT6D")
            if state_key is not None and (not isinstance(state_key, str) or not state_key):
                raise ValueError(f"{context}[{index}].state_key must be a non-empty string")
            if action_type == "EEF" and rep == "RELATIVE" and not state_key:
                raise ValueError(f"{context}[{index}]: relative EEF actions require state_key")
        entries.append(
            LayoutEntry(
                key=key,
                fields=fields,
                rot6d=rot6d,
                rep=rep,
                type=action_type,
                format=action_format,
                state_key=state_key,
            )
        )
    return tuple(entries)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open() as stream:
        raw = yaml.safe_load(stream)
    data = _mapping(raw, str(config_path))
    root_allowed = {
        "name",
        "source",
        "output",
        "cameras",
        "video",
        "state",
        "action",
        "tasks",
        "select",
        "warn",
        "continuity",
        "train",
    }
    _warn_unknown(data, root_allowed, str(config_path))

    name = _require(data, "name", str(config_path))
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"name must be a valid Python identifier, got {name!r}")

    source_raw = _mapping(_require(data, "source", "config"), "source")
    _warn_unknown(
        source_raw,
        {"root", "subsets", "episode_glob", "exclude_path_contains", "fps"},
        "source",
    )
    source = SourceConfig(
        root=_path(_require(source_raw, "root", "source"), "source.root"),
        subsets=_string_tuple(_require(source_raw, "subsets", "source"), "source.subsets"),
        episode_glob=_string(
            source_raw.get("episode_glob", "*/20*/*/*/*/episode.h5"), "source.episode_glob"
        ),
        exclude_path_contains=_string_tuple(
            source_raw.get("exclude_path_contains", ["_failed_recordings"]),
            "source.exclude_path_contains",
            allow_empty_list=True,
        ),
        fps=int(source_raw.get("fps", 50)),
    )
    if source.fps <= 0:
        raise ValueError("source.fps must be positive")

    output_raw = _mapping(_require(data, "output", "config"), "output")
    _warn_unknown(
        output_raw,
        {"root", "robot_type", "strip_subset_prefix", "val_every", "chunks_size"},
        "output",
    )
    output = OutputConfig(
        root=_path(_require(output_raw, "root", "output"), "output.root"),
        robot_type=_string(_require(output_raw, "robot_type", "output"), "output.robot_type"),
        strip_subset_prefix=_string(
            output_raw.get("strip_subset_prefix", ""),
            "output.strip_subset_prefix",
            allow_empty=True,
        ),
        val_every=int(output_raw.get("val_every", 0)),
        chunks_size=int(output_raw.get("chunks_size", 1000)),
    )
    if output.val_every < 0 or output.chunks_size <= 0:
        raise ValueError("output.val_every must be non-negative and chunks_size must be positive")

    cameras_raw = _mapping(_require(data, "cameras", "config"), "cameras")
    if not cameras_raw or not all(
        isinstance(key, str) and key and isinstance(value, str) and value
        for key, value in cameras_raw.items()
    ):
        raise ValueError("cameras must map non-empty canonical names to source group names")
    cameras = dict(cameras_raw)

    video_raw = _mapping(data.get("video", {}), "video")
    _warn_unknown(video_raw, {"codec", "crf", "preset", "pixel_format"}, "video")
    video = VideoConfig(
        codec=_string(video_raw.get("codec", "libx264"), "video.codec"),
        crf=int(video_raw.get("crf", 23)),
        preset=_string(video_raw.get("preset", "veryfast"), "video.preset"),
        pixel_format=_string(video_raw.get("pixel_format", "yuv420p"), "video.pixel_format"),
    )

    state = _layout_entries(_require(data, "state", "config"), "state", action=False)
    action_raw = _mapping(_require(data, "action", "config"), "action")
    _warn_unknown(action_raw, {"source", "horizon", "keys"}, "action")
    action_layout = ActionLayout(
        source=_string(_require(action_raw, "source", "action"), "action.source"),
        horizon=int(_require(action_raw, "horizon", "action")),
        keys=_layout_entries(_require(action_raw, "keys", "action"), "action.keys", action=True),
    )
    if action_layout.horizon <= 0:
        raise ValueError("action.horizon must be positive")
    state_keys = {entry.key for entry in state}
    for entry in action_layout.keys:
        if entry.rep != "RELATIVE" and entry.state_key is not None:
            raise ValueError(
                f"action key {entry.key!r} sets state_key, but only RELATIVE actions use it"
            )
        reference_key = entry.state_key or (entry.key if entry.rep == "RELATIVE" else None)
        if reference_key is not None and reference_key not in state_keys:
            raise ValueError(
                f"action key {entry.key!r} references unknown state key {reference_key!r}"
            )

    tasks_raw = _mapping(data.get("tasks", {}), "tasks")
    _warn_unknown(tasks_raw, {"id_attr", "text_attr", "text_overrides"}, "tasks")
    overrides = _mapping(tasks_raw.get("text_overrides", {}), "tasks.text_overrides")
    if not all(
        isinstance(key, str) and key and isinstance(value, str) and value
        for key, value in overrides.items()
    ):
        raise ValueError("tasks.text_overrides must map non-empty strings to non-empty strings")
    tasks = TasksConfig(
        id_attr=_string(tasks_raw.get("id_attr", "task_uuid"), "tasks.id_attr"),
        text_attr=_string(tasks_raw.get("text_attr", "task_text"), "tasks.text_attr"),
        text_overrides=dict(overrides),
    )

    select_raw = _mapping(data.get("select", {}), "select")
    _warn_unknown(select_raw, {"require_valid_for_training", "policy_types"}, "select")
    select = SelectionConfig(
        require_valid_for_training=_boolean(
            select_raw.get("require_valid_for_training", False),
            "select.require_valid_for_training",
        ),
        policy_types=_string_tuple(
            select_raw.get("policy_types", []),
            "select.policy_types",
            allow_empty_list=True,
        ),
    )

    warn_raw = _mapping(data.get("warn", {}), "warn")
    _warn_unknown(
        warn_raw,
        {
            "camera_coverage_below",
            "camera_coverage_above",
            "camera_age_p99_ms_above",
            "frame_gap_ms_above",
            "nonzero_action_residual",
        },
        "warn",
    )
    warn = WarningConfig(
        camera_coverage_below=_optional_float(
            warn_raw.get("camera_coverage_below"), "warn.camera_coverage_below", maximum=1
        ),
        camera_coverage_above=_optional_float(
            warn_raw.get("camera_coverage_above"), "warn.camera_coverage_above", maximum=1
        ),
        camera_age_p99_ms_above=_optional_float(
            warn_raw.get("camera_age_p99_ms_above"), "warn.camera_age_p99_ms_above"
        ),
        frame_gap_ms_above=_optional_float(
            warn_raw.get("frame_gap_ms_above"), "warn.frame_gap_ms_above"
        ),
        nonzero_action_residual=_boolean(
            warn_raw.get("nonzero_action_residual", False), "warn.nonzero_action_residual"
        ),
    )

    continuity_raw = _mapping(data.get("continuity", {}), "continuity")
    _warn_unknown(continuity_raw, {"split_on_gap_ms"}, "continuity")
    continuity = ContinuityConfig(
        split_on_gap_ms=_optional_float(
            continuity_raw.get("split_on_gap_ms"), "continuity.split_on_gap_ms"
        )
    )

    train_raw = _mapping(data.get("train", {}), "train")
    _warn_unknown(
        train_raw,
        {
            "base_model",
            "out_base",
            "gpus",
            "batch",
            "epochs",
            "max_steps",
            "lr",
            "workers",
            "save_steps",
            "save_total_limit",
            "state_dropout_prob",
            "shortest_image_edge",
            "crop_fraction",
            "color_jitter",
            "use_wandb",
            "wandb_project",
            "weight_decay",
            "warmup_ratio",
        },
        "train",
    )
    color_jitter = _mapping(train_raw.get("color_jitter", {}), "train.color_jitter")
    epochs_raw = train_raw.get("epochs", 1.0)
    train = TrainingConfig(
        base_model=_string(train_raw.get("base_model", "nvidia/GR00T-N1.7-3B"), "train.base_model"),
        out_base=_path(train_raw.get("out_base", "outputs"), "train.out_base"),
        gpus=int(train_raw.get("gpus", 1)),
        batch=int(train_raw.get("batch", 32)),
        epochs=float(epochs_raw) if epochs_raw is not None else None,
        max_steps=int(train_raw["max_steps"]) if train_raw.get("max_steps") is not None else None,
        lr=float(train_raw.get("lr", 1e-4)),
        workers=int(train_raw.get("workers", 2)),
        save_steps=int(train_raw.get("save_steps", 1000)),
        save_total_limit=int(train_raw.get("save_total_limit", 5)),
        state_dropout_prob=float(train_raw.get("state_dropout_prob", 0.2)),
        shortest_image_edge=int(train_raw.get("shortest_image_edge", 256)),
        crop_fraction=float(train_raw.get("crop_fraction", 0.95)),
        color_jitter={str(key): float(value) for key, value in color_jitter.items()},
        use_wandb=_boolean(train_raw.get("use_wandb", False), "train.use_wandb"),
        wandb_project=_string(
            train_raw.get("wandb_project", "finetune-gr00t-n1d7"), "train.wandb_project"
        ),
        weight_decay=float(train_raw.get("weight_decay", 1e-5)),
        warmup_ratio=float(train_raw.get("warmup_ratio", 0.05)),
    )
    if train.gpus <= 0 or train.batch <= 0 or train.workers < 0:
        raise ValueError(
            "train.gpus and train.batch must be positive; train.workers must be non-negative"
        )
    if train.max_steps is None and (train.epochs is None or train.epochs <= 0):
        raise ValueError("train requires positive epochs or max_steps")

    return PipelineConfig(
        name=name,
        source=source,
        output=output,
        cameras=cameras,
        video=video,
        state=state,
        action=action_layout,
        tasks=tasks,
        select=select,
        warn=warn,
        continuity=continuity,
        train=train,
        config_path=config_path,
    )


def resolve_source_subsets(config: PipelineConfig) -> list[Path]:
    found: dict[Path, None] = {}
    for pattern in config.source.subsets:
        for path in sorted(config.source.root.glob(pattern)):
            if path.is_dir():
                found[path.resolve()] = None
    subsets = list(found)
    if not subsets:
        raise ValueError(
            f"no source subsets match {list(config.source.subsets)} under {config.source.root}"
        )
    names: dict[str, Path] = {}
    for subset in subsets:
        output_name = config.output_name(subset.name)
        if output_name in names:
            raise ValueError(
                f"source subsets {names[output_name]} and {subset} both map to output {output_name!r}"
            )
        names[output_name] = subset
    return subsets


def _resolve_entries(
    entries: tuple[LayoutEntry, ...], widths: dict[str, int], context: str
) -> tuple[tuple[ResolvedEntry, ...], dict[str, tuple[int, int]], int]:
    resolved: list[ResolvedEntry] = []
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for entry in entries:
        fields: list[tuple[str, int]] = []
        for field_name in entry.fields:
            if field_name not in widths:
                raise ValueError(f"{context}.{entry.key}: source field {field_name!r} is absent")
            width = int(widths[field_name])
            if width <= 0:
                raise ValueError(f"{context}.{entry.key}: field {field_name!r} has width {width}")
            fields.append((field_name, width))
        rot6d_field = None
        if entry.rot6d is not None:
            six_dimensional = [name for name, width in fields if width == 6]
            if len(six_dimensional) != 1:
                raise ValueError(
                    f"{context}.{entry.key}: rot6d needs exactly one 6D field, found {six_dimensional}"
                )
            rot6d_field = six_dimensional[0]
        width = sum(field_width for _, field_width in fields)
        if context == "action" and entry.type == "EEF" and width != 9:
            raise ValueError(f"action.{entry.key}: EEF/XYZ_ROT6D must be 9D, got {width}")
        slices[entry.key] = (offset, offset + width)
        resolved.append(
            ResolvedEntry(
                entry=entry,
                fields=tuple(fields),
                start=offset,
                end=offset + width,
                rot6d_field=rot6d_field,
            )
        )
        offset += width
    return tuple(resolved), slices, offset


def derive_layout(
    config: PipelineConfig,
    state_widths: dict[str, int],
    action_widths: dict[str, int],
    image_shape: tuple[int, int, int],
) -> ResolvedLayout:
    if len(image_shape) != 3 or image_shape[2] != 3 or min(image_shape) <= 0:
        raise ValueError(f"expected an HxWx3 image shape, got {image_shape}")
    state, state_slices, state_dim = _resolve_entries(config.state, state_widths, "state")
    action, action_slices, action_dim = _resolve_entries(
        config.action.keys, action_widths, "action"
    )
    state_by_key = {entry.entry.key: entry for entry in state}
    for action_entry in action:
        if action_entry.entry.rep != "RELATIVE":
            continue
        reference_key = action_entry.entry.state_key or action_entry.entry.key
        state_entry = state_by_key[reference_key]
        state_width = state_entry.end - state_entry.start
        action_width = action_entry.end - action_entry.start
        if state_width != action_width:
            raise ValueError(
                f"action.{action_entry.entry.key}: relative action width {action_width} differs "
                f"from state.{reference_key} width {state_width}"
            )
    return ResolvedLayout(
        state=state,
        action=action,
        state_slices=state_slices,
        action_slices=action_slices,
        state_dim=state_dim,
        action_dim=action_dim,
        image_shape=image_shape,
    )


def modality_json(config: PipelineConfig, layout: ResolvedLayout) -> dict[str, Any]:
    return {
        "state": {
            key: {"start": start, "end": end} for key, (start, end) in layout.state_slices.items()
        },
        "action": {
            key: {"start": start, "end": end} for key, (start, end) in layout.action_slices.items()
        },
        "video": {key: {"original_key": f"observation.images.{key}"} for key in config.cameras},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def layout_json(config: PipelineConfig, layout: ResolvedLayout) -> dict[str, Any]:
    def entries(items: tuple[ResolvedEntry, ...], *, action: bool) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for item in items:
            value = {
                "key": item.entry.key,
                "fields": [name for name, _ in item.fields],
                "widths": [width for _, width in item.fields],
                "rot6d": item.entry.rot6d,
                "start": item.start,
                "end": item.end,
            }
            if action:
                value.update(
                    {
                        "rep": item.entry.rep,
                        "type": item.entry.type,
                        "format": item.entry.format,
                        "state_key": item.entry.state_key,
                    }
                )
            rendered.append(value)
        return rendered

    return {
        "version": 2,
        "output": {
            "robot_type": config.output.robot_type,
            "chunks_size": config.output.chunks_size,
        },
        "cameras": config.cameras,
        "video": {
            "codec": config.video.codec,
            "pixel_format": config.video.pixel_format,
        },
        "state": entries(layout.state, action=False),
        "action": {
            "source": config.action.source,
            "keys": entries(layout.action, action=True),
            "horizon": config.action.horizon,
        },
        "fps": config.source.fps,
        "image_shape": list(layout.image_shape),
        "state_slices": {key: list(value) for key, value in layout.state_slices.items()},
        "action_slices": {key: list(value) for key, value in layout.action_slices.items()},
    }


def render_modality_module(config: PipelineConfig) -> str:
    def rendered_keys(values: list[str], indent: int) -> str:
        padding = " " * indent
        return (
            "[\n"
            + "".join(f"{padding}{json.dumps(value)},\n" for value in values)
            + " " * (indent - 4)
            + "]"
        )

    action_configs = []
    for entry in config.action.keys:
        state_key = (
            f",\n                state_key={json.dumps(entry.state_key)}" if entry.state_key else ""
        )
        action_configs.append(
            "            ActionConfig(\n"
            f"                rep=ActionRepresentation.{entry.rep},\n"
            f"                type=ActionType.{entry.type},\n"
            f"                format=ActionFormat.{entry.format}{state_key},\n"
            "            ),"
        )
    action_text = "\n".join(action_configs)
    video_keys = "[" + ", ".join(json.dumps(key) for key in config.cameras) + "]"
    state_keys = rendered_keys([entry.key for entry in config.state], 12)
    action_keys = rendered_keys([entry.key for entry in config.action.keys], 12)
    return f'''# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Generated from ``{config.config_path.name}``; edit the YAML, not this file."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


{config.name}_config = {{
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys={video_keys},
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys={state_keys},
    ),
    "action": ModalityConfig(
        delta_indices=list(range({config.action.horizon})),
        modality_keys={action_keys},
        action_configs=[
{action_text}
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}}

register_modality_config({config.name}_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''


def modality_module_path(config: PipelineConfig, repo_root: Path) -> Path:
    return repo_root / "examples" / config.name / f"{config.name}_config.py"


def write_text_if_changed(path: Path, text: str) -> bool:
    encoded = text.encode()
    if path.exists() and path.read_bytes() == encoded:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return True


def write_json_if_changed(path: Path, value: Any) -> bool:
    return write_text_if_changed(path, json.dumps(value, indent=4, sort_keys=False) + "\n")
