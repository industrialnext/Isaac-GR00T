# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""zdata_hdf5 discovery, validation, field assembly, and video encoding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import subprocess

from gr00t.data.state_action.rot6d import rot6d_source_to_groot
import h5py
import numpy as np
import pandas as pd
from PIL import Image

from .config import PipelineConfig, ResolvedEntry, ResolvedLayout, derive_layout


@dataclass(frozen=True)
class SourceDescription:
    path: Path
    key: str
    task_id: str | None
    task_text: str | None
    frame_count: int
    policy_type: str
    layout: ResolvedLayout | None
    segments: tuple[tuple[int, int], ...]
    warnings: tuple[str, ...]
    skip_reason: str | None = None
    source_size_bytes: int | None = None
    source_mtime_ns: int | None = None


@dataclass(frozen=True)
class StagedSegment:
    source_start: int
    source_end: int
    length: int
    parquet: Path
    videos: dict[str, Path]


@dataclass(frozen=True)
class StagedSource:
    description: SourceDescription
    segments: tuple[StagedSegment, ...]


def resolve_field_slices(group: h5py.Group) -> dict[str, tuple[int, int]]:
    names = [
        name.decode() if isinstance(name, bytes) else str(name) for name in group["field_names"][:]
    ]
    slices = group["field_slices"][:]
    if not names:
        raise ValueError("field_names must not be empty")
    if slices.ndim != 2 or slices.shape[1:] != (2,):
        raise ValueError(f"field_slices must have shape [N, 2], got {slices.shape}")
    if len(names) != len(slices):
        raise ValueError(f"field_names/field_slices length mismatch: {len(names)} vs {len(slices)}")
    resolved = {name: (int(start), int(end)) for name, (start, end) in zip(names, slices)}
    if len(resolved) != len(names):
        raise ValueError("field_names contains duplicate names")
    for name, (start, end) in resolved.items():
        if start < 0 or end <= start:
            raise ValueError(f"field {name!r} has invalid slice [{start}, {end})")
    ordered = sorted((start, end, name) for name, (start, end) in resolved.items())
    for (_, previous_end, previous_name), (start, _, name) in zip(ordered, ordered[1:]):
        if start < previous_end:
            raise ValueError(f"fields {previous_name!r} and {name!r} have overlapping slices")
    return resolved


def gather_fields(
    flat: np.ndarray, resolved: dict[str, tuple[int, int]], entries: tuple[ResolvedEntry, ...]
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for entry in entries:
        for field_name, expected_width in entry.fields:
            if field_name not in resolved:
                raise ValueError(f"required field {field_name!r} is absent")
            start, end = resolved[field_name]
            if end - start != expected_width:
                raise ValueError(
                    f"field {field_name!r} width {end - start}, expected {expected_width}"
                )
            block = flat[:, start:end]
            if field_name == entry.rot6d_field:
                block = rot6d_source_to_groot(block)
            columns.append(block.astype(np.float32, copy=False))
    if not columns:
        raise ValueError("canonical layout contains no fields")
    result = np.concatenate(columns, axis=1)
    expected_width = sum(entry.end - entry.start for entry in entries)
    if result.shape[1] != expected_width:
        raise ValueError(
            f"canonical tensor width {result.shape[1]} differs from layout width {expected_width}"
        )
    if not np.isfinite(result).all():
        raise ValueError("canonical tensor contains non-finite values")
    return result


def discover_episodes(config: PipelineConfig, subset: Path) -> list[Path]:
    discovered = sorted(subset.glob(config.source.episode_glob))
    return [
        path
        for path in discovered
        if not any(
            token in path.relative_to(subset).as_posix()
            for token in config.source.exclude_path_contains
        )
    ]


def episode_key(subset: Path, source: Path) -> str:
    return source.relative_to(subset).as_posix()


def assign_split(key: str, val_every: int) -> str:
    if val_every <= 0:
        return "train"
    digest = hashlib.sha256(key.encode()).digest()
    return "val" if int.from_bytes(digest[:8], "big") % val_every == 0 else "train"


def _policy_type(source: Path) -> str:
    return source.parent.name.rsplit("_", 1)[-1]


def _attr_text(attributes: h5py.AttributeManager, key: str) -> str:
    if key not in attributes:
        raise ValueError(f"missing required HDF5 attribute {key!r}")
    value = attributes[key]
    return value.decode() if isinstance(value, bytes) else str(value)


def _segments(elapsed_ms: np.ndarray, gap_ms: float | None) -> tuple[tuple[int, int], ...]:
    count = len(elapsed_ms)
    if count == 0:
        return ()
    if gap_ms is None:
        return ((0, count),)
    boundaries = [0]
    boundaries.extend((np.flatnonzero(np.diff(elapsed_ms) > gap_ms) + 1).tolist())
    boundaries.append(count)
    return tuple((start, end) for start, end in zip(boundaries, boundaries[1:]))


def _validate_camera(
    config: PipelineConfig,
    h5: h5py.File,
    canonical: str,
    physical: str,
    frame_count: int,
    image_shape: tuple[int, int, int],
    warning_messages: list[str],
) -> None:
    if "images" not in h5 or physical not in h5["images"]:
        raise ValueError(f"{canonical}: missing source camera images/{physical}")
    group = h5["images"][physical]
    for required in ("blob", "offsets", "frame_ref_index"):
        if required not in group:
            raise ValueError(f"{canonical}: missing images/{physical}/{required}")
    if "image_count" not in group.attrs:
        raise ValueError(f"{canonical}: missing image_count")
    image_count = int(group.attrs["image_count"])
    offsets = group["offsets"][:]
    references = group["frame_ref_index"][:]
    if len(references) != frame_count:
        raise ValueError(
            f"{canonical}: frame_ref_index length {len(references)} != frame_count {frame_count}"
        )
    if len(offsets) != image_count + 1:
        raise ValueError(f"{canonical}: offsets length {len(offsets)} != {image_count + 1}")
    if image_count <= 0:
        raise ValueError(f"{canonical}: image_count must be positive")
    low, high = int(references.min()), int(references.max())
    if low < 0 or high >= image_count:
        raise ValueError(
            f"{canonical}: frame_ref_index range [{low}, {high}] outside [0, {image_count})"
        )
    first = bytes(group["blob"][int(offsets[0]) : int(offsets[1])])
    with Image.open(BytesIO(first)) as image:
        observed = (image.height, image.width, len(image.getbands()))
    if observed != image_shape:
        raise ValueError(
            f"{canonical}: JPEG shape {observed} != configured source shape {image_shape}"
        )

    coverage = image_count / frame_count if frame_count else 0.0
    if (
        config.warn.camera_coverage_below is not None
        and coverage < config.warn.camera_coverage_below
    ):
        warning_messages.append(
            f"{canonical}: camera coverage {coverage:.3f} < {config.warn.camera_coverage_below}"
        )
    if (
        config.warn.camera_coverage_above is not None
        and coverage > config.warn.camera_coverage_above
    ):
        warning_messages.append(
            f"{canonical}: camera coverage {coverage:.3f} > {config.warn.camera_coverage_above}"
        )
    if config.warn.camera_age_p99_ms_above is not None:
        if "frame_age_ms" not in group:
            warning_messages.append(f"{canonical}: frame_age_ms measurement unavailable")
        else:
            p99 = float(np.percentile(group["frame_age_ms"][:], 99))
            if p99 > config.warn.camera_age_p99_ms_above:
                warning_messages.append(
                    f"{canonical}: frame_age_ms p99 {p99:.1f} > "
                    f"{config.warn.camera_age_p99_ms_above}"
                )


def inspect_source(config: PipelineConfig, subset: Path, source: Path) -> SourceDescription:
    key = episode_key(subset, source)
    policy_type = _policy_type(source)
    warning_messages: list[str] = []
    source_stat = source.stat()
    with h5py.File(source, "r") as h5:
        frame_count = int(h5.attrs.get("frame_count", 0))
        if frame_count <= 0:
            raise ValueError(f"frame_count must be positive, got {frame_count}")
        if config.select.require_valid_for_training:
            if "valid_for_training" not in h5.attrs:
                return SourceDescription(
                    path=source,
                    key=key,
                    task_id=None,
                    task_text=None,
                    frame_count=frame_count,
                    policy_type=policy_type,
                    layout=None,
                    segments=(),
                    warnings=(),
                    skip_reason="valid_for_training is missing",
                )
            elif not bool(h5.attrs["valid_for_training"]):
                return SourceDescription(
                    path=source,
                    key=key,
                    task_id=None,
                    task_text=None,
                    frame_count=frame_count,
                    policy_type=policy_type,
                    layout=None,
                    segments=(),
                    warnings=tuple(warning_messages),
                    skip_reason="valid_for_training=false",
                )
        if config.select.policy_types and policy_type not in config.select.policy_types:
            return SourceDescription(
                path=source,
                key=key,
                task_id=None,
                task_text=None,
                frame_count=frame_count,
                policy_type=policy_type,
                layout=None,
                segments=(),
                warnings=tuple(warning_messages),
                skip_reason=f"policy_type {policy_type!r} is not selected",
            )

        sampling_hz = float(h5.attrs.get("sampling_hz", h5.attrs.get("state_sampling_hz", 0)))
        if not np.isclose(sampling_hz, config.source.fps):
            raise ValueError(f"sampling_hz {sampling_hz} != configured fps {config.source.fps}")
        height = int(h5.attrs.get("image_height", 0))
        width = int(h5.attrs.get("image_width", 0))
        image_shape = (height, width, 3)
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid source image shape {image_shape}")

        if "state" not in h5 or "flat" not in h5["state"]:
            raise ValueError("missing state/flat")
        if "action" not in h5 or config.action.source not in h5["action"]:
            raise ValueError(f"missing action/{config.action.source}")
        state_slices = resolve_field_slices(h5["state"])
        action_slices = resolve_field_slices(h5["action"])
        state_flat = h5["state/flat"]
        action_flat = h5[f"action/{config.action.source}"]
        if (
            state_flat.ndim != 2
            or max(end for _, end in state_slices.values()) > state_flat.shape[1]
        ):
            raise ValueError("state field_slices exceed state/flat width")
        if (
            action_flat.ndim != 2
            or max(end for _, end in action_slices.values()) > action_flat.shape[1]
        ):
            raise ValueError(f"action field_slices exceed action/{config.action.source} width")
        state_widths = {name: end - start for name, (start, end) in state_slices.items()}
        action_widths = {name: end - start for name, (start, end) in action_slices.items()}
        layout = derive_layout(config, state_widths, action_widths, image_shape)
        if len(state_flat) != frame_count or len(action_flat) != frame_count:
            raise ValueError("state/action row count differs from frame_count")
        if "frame" not in h5 or "elapsed_ms" not in h5["frame"] or "done" not in h5["frame"]:
            raise ValueError("missing frame/elapsed_ms or frame/done")
        elapsed_ms = np.asarray(h5["frame/elapsed_ms"][:], dtype=np.float64)
        if len(elapsed_ms) != frame_count or len(h5["frame/done"]) != frame_count:
            raise ValueError("frame metadata length differs from frame_count")
        if not np.isfinite(elapsed_ms).all():
            raise ValueError("frame/elapsed_ms contains non-finite values")
        if len(elapsed_ms) > 1 and np.any(np.diff(elapsed_ms) < 0):
            raise ValueError("frame/elapsed_ms is not monotonic")

        segments = _segments(elapsed_ms, config.continuity.split_on_gap_ms)
        if config.warn.frame_gap_ms_above is not None and len(elapsed_ms) > 1:
            gap = float(np.max(np.diff(elapsed_ms)))
            if gap > config.warn.frame_gap_ms_above:
                warning_messages.append(
                    f"frame gap {gap:.1f} ms > {config.warn.frame_gap_ms_above} ms"
                )
        usable_segments = tuple(
            (start, end) for start, end in segments if end - start >= config.action.horizon
        )
        for start, end in segments:
            if end - start < config.action.horizon:
                warning_messages.append(
                    f"dropping segment [{start}, {end}) shorter than action horizon "
                    f"{config.action.horizon}"
                )
        if not usable_segments:
            return SourceDescription(
                path=source,
                key=key,
                task_id=None,
                task_text=None,
                frame_count=frame_count,
                policy_type=policy_type,
                layout=layout,
                segments=(),
                warnings=tuple(warning_messages),
                skip_reason="all segments are shorter than the action horizon",
            )

        for canonical, physical in config.cameras.items():
            _validate_camera(
                config,
                h5,
                canonical,
                physical,
                frame_count,
                image_shape,
                warning_messages,
            )
        if config.warn.nonzero_action_residual:
            if "residual" not in h5["action"]:
                warning_messages.append("action/residual measurement unavailable")
            else:
                residual = np.asarray(h5["action/residual"][:])
                if residual.shape[0] != frame_count:
                    warning_messages.append("action/residual length measurement unavailable")
                elif np.any(residual != 0):
                    warning_messages.append(
                        f"nonzero action/residual max_abs={float(np.max(np.abs(residual))):.6g}"
                    )

        task_id = _attr_text(h5.attrs, config.tasks.id_attr)
        task_text = _attr_text(h5.attrs, config.tasks.text_attr)
        current_stat = source.stat()
        if (
            current_stat.st_size != source_stat.st_size
            or current_stat.st_mtime_ns != source_stat.st_mtime_ns
        ):
            raise RuntimeError("source changed while it was being inspected")
        return SourceDescription(
            path=source,
            key=key,
            task_id=task_id,
            task_text=task_text,
            frame_count=frame_count,
            policy_type=policy_type,
            layout=layout,
            segments=usable_segments,
            warnings=tuple(warning_messages),
            source_size_bytes=source_stat.st_size,
            source_mtime_ns=source_stat.st_mtime_ns,
        )


def _encode_video(
    config: PipelineConfig,
    h5: h5py.File,
    physical: str,
    start: int,
    end: int,
    destination: Path,
) -> None:
    group = h5["images"][physical]
    offsets = group["offsets"][:]
    references = group["frame_ref_index"][start:end]
    blob = group["blob"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "image2pipe",
            "-framerate",
            str(config.source.fps),
            "-i",
            "-",
            "-c:v",
            config.video.codec,
            "-preset",
            config.video.preset,
            "-crf",
            str(config.video.crf),
            "-pix_fmt",
            config.video.pixel_format,
            "-g",
            str(config.source.fps),
            str(destination),
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin pipe was not created")
    broken_pipe: BrokenPipeError | None = None
    try:
        try:
            for reference in references:
                image_index = int(reference)
                process.stdin.write(
                    bytes(blob[int(offsets[image_index]) : int(offsets[image_index + 1])])
                )
        finally:
            process.stdin.close()
    except BrokenPipeError as error:
        broken_pipe = error
    finally:
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for images/{physical}") from broken_pipe
    if broken_pipe is not None:
        raise broken_pipe


def stage_source(
    config: PipelineConfig, description: SourceDescription, stage_directory: Path
) -> StagedSource:
    if description.layout is None or description.skip_reason is not None:
        raise ValueError(f"cannot stage skipped source {description.key}")
    layout = description.layout
    staged_segments: list[StagedSegment] = []
    with h5py.File(description.path, "r") as h5:
        state = gather_fields(h5["state/flat"][:], resolve_field_slices(h5["state"]), layout.state)
        action = gather_fields(
            h5[f"action/{config.action.source}"][:],
            resolve_field_slices(h5["action"]),
            layout.action,
        )
        elapsed_ms = np.asarray(h5["frame/elapsed_ms"][:], dtype=np.float64)
        done = np.asarray(h5["frame/done"][:], dtype=bool)
        for segment_index, (start, end) in enumerate(description.segments):
            segment_directory = stage_directory / f"segment_{segment_index:03d}"
            parquet = segment_directory / "episode.parquet"
            segment_done = done[start:end].copy()
            segment_done[-1] = True
            frame_count = end - start
            dataframe = pd.DataFrame(
                {
                    "observation.state": list(state[start:end]),
                    "action": list(action[start:end]),
                    "timestamp": np.asarray(
                        (elapsed_ms[start:end] - elapsed_ms[start]) / 1000.0,
                        dtype=np.float32,
                    ),
                    "frame_index": np.arange(frame_count, dtype=np.int64),
                    "next.done": segment_done,
                }
            )
            parquet.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_parquet(parquet, index=False)
            videos: dict[str, Path] = {}
            for canonical, physical in config.cameras.items():
                video = segment_directory / f"observation.images.{canonical}.mp4"
                _encode_video(config, h5, physical, start, end, video)
                videos[canonical] = video
            staged_segments.append(
                StagedSegment(
                    source_start=start,
                    source_end=end,
                    length=frame_count,
                    parquet=parquet,
                    videos=videos,
                )
            )
    return StagedSource(description=description, segments=tuple(staged_segments))


def stage_source_worker(
    config: PipelineConfig, description: SourceDescription, stage_directory: Path
) -> tuple[str, StagedSource | None, str | None]:
    try:
        return description.key, stage_source(config, description, stage_directory), None
    except Exception as error:
        return description.key, None, f"{type(error).__name__}: {error}"
