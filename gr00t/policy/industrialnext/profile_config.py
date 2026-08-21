# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config-driven wire/model profile for Industrial Next GR00T serving."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Protocol

import cv2
import numpy as np
import yaml

from gr00t.data.state_action.rot6d import rot6d_groot_to_source, rot6d_source_to_groot
from gr00t.data.types import ActionFormat, ActionRepresentation, ActionType, ModalityConfig

from .adapter import CachedImage, ObservationAdmission, ObservationSnapshot
from .task_catalog import TaskCatalog, task_catalog_from_mapping


LANGUAGE_KEY = "annotation.human.task_description"
ROT6D_TRANSFORM = "source_columns_to_groot_rows"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class PolicyWithModalityConfig(Protocol):
    def get_modality_config(self) -> dict[str, ModalityConfig]: ...


@dataclass(frozen=True)
class ProfileLayout:
    key: str
    fields: tuple[str, ...]
    widths: tuple[int, ...]
    rot6d: str | None
    rot6d_index: int | None
    rep: str | None = None
    action_type: str | None = None
    action_format: str | None = None
    state_key: str | None = None

    @property
    def width(self) -> int:
        return sum(self.widths)


@dataclass(frozen=True)
class ConfigDrivenIndustrialNextProfile:
    name: str
    profile_name: str
    config_path: Path
    model_path: Path
    embodiment_tag: str
    device: str
    host: str
    port: int
    control_hz: float
    image_height: int
    image_width: int
    rtc_mode: str
    supported_rtc_modes: tuple[str, ...]
    action_horizon: int
    action_start_offset_steps: int
    wire_image_to_model: Mapping[str, str]
    state_layouts: tuple[ProfileLayout, ...]
    action_layouts: tuple[ProfileLayout, ...]
    field_lengths: Mapping[str, int]
    ignored_observation_keys: frozenset[str]
    field_units: Mapping[str, str]
    eef_frame: str
    gripper_action_keys: tuple[str, ...]
    task_catalog: TaskCatalog

    @property
    def state_fields(self) -> tuple[str, ...]:
        return tuple(field for layout in self.state_layouts for field in layout.fields)

    @property
    def action_fields(self) -> tuple[str, ...]:
        return tuple(field for layout in self.action_layouts for field in layout.fields)

    @property
    def rotation_state_fields(self) -> tuple[str, ...]:
        return tuple(
            layout.fields[layout.rot6d_index]
            for layout in self.state_layouts
            if layout.rot6d_index is not None
        )

    @property
    def position_action_fields(self) -> tuple[str, ...]:
        return tuple(
            layout.fields[0] for layout in self.action_layouts if layout.action_type == "EEF"
        )

    @property
    def rotation_action_fields(self) -> tuple[str, ...]:
        return tuple(
            layout.fields[layout.rot6d_index]
            for layout in self.action_layouts
            if layout.rot6d_index is not None
        )

    @property
    def auxiliary_action_fields(self) -> tuple[str, ...]:
        pose_fields = set(self.position_action_fields) | set(self.rotation_action_fields)
        return tuple(field for field in self.action_fields if field not in pose_fields)

    def assert_policy_contract(self, policy: PolicyWithModalityConfig) -> None:
        configs = policy.get_modality_config()
        expected_modalities = {"video", "state", "action", "language"}
        if set(configs) != expected_modalities:
            raise ValueError(f"saved modality names differ: {sorted(configs)}")
        _require_modality(configs["video"], [0], list(self.wire_image_to_model.values()), "video")
        _require_modality(configs["state"], [0], [item.key for item in self.state_layouts], "state")
        _require_modality(configs["language"], [0], [LANGUAGE_KEY], "language")
        action = configs["action"]
        _require_modality(
            action,
            list(range(self.action_horizon)),
            [item.key for item in self.action_layouts],
            "action",
        )
        expected = tuple(
            (
                ActionRepresentation[item.rep or "ABSOLUTE"],
                ActionType[item.action_type or "NON_EEF"],
                ActionFormat[item.action_format or "DEFAULT"],
                item.state_key,
            )
            for item in self.action_layouts
        )
        actual = tuple(
            (item.rep, item.type, item.format, item.state_key)
            for item in (action.action_configs or [])
        )
        if actual != expected:
            raise ValueError(f"saved action configs differ: expected {expected}, got {actual}")

    def admit_observation(
        self,
        observation: Mapping[str, Any],
        *,
        image_cache: MutableMapping[str, CachedImage],
        timestep: int,
        task_uuid: str,
        task_text: str,
        generation: int,
        max_image_staleness_steps: int,
    ) -> ObservationAdmission:
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be a mapping")
        state = {
            field: _finite_tuple(observation.get(field), self.field_lengths[field], field)
            for field in self.state_fields
        }
        if observation.get("task_uuid", task_uuid) != task_uuid:
            raise ValueError("observation task_uuid does not match the registered session")
        if observation.get("task_text", task_text) != task_text:
            raise ValueError("observation task_text does not match the registered session")
        raw_metadata = observation.get("images_meta", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("images_meta must be a mapping")
        allowed = set(self.state_fields) | {"task_uuid", "task_text", "images_meta"}
        updates: dict[str, CachedImage] = {}
        ignored = 0
        for name, value in observation.items():
            if name in allowed:
                continue
            if name in self.ignored_observation_keys:
                ignored += 1
                continue
            if name not in self.wire_image_to_model:
                raise ValueError(f"unexpected image or observation field {name!r}")
            if not isinstance(value, bytes | bytearray | memoryview) or not value:
                raise ValueError(f"{name} must contain encoded image bytes")
            metadata = raw_metadata.get(name)
            if not isinstance(metadata, Mapping):
                raise ValueError(f"images_meta.{name} must be provided as a mapping")
            updates[name] = CachedImage(
                payload=bytes(value),
                metadata=_validate_rgb_metadata(
                    metadata, name, self.image_width, self.image_height
                ),
                updated_timestep=timestep,
            )
        for name in raw_metadata:
            if name not in updates and name not in self.ignored_observation_keys:
                raise ValueError(f"orphan or unexpected image metadata {name!r}")
        image_cache.update(updates)
        ages = {
            name: None if name not in image_cache else timestep - image_cache[name].updated_timestep
            for name in self.wire_image_to_model
        }
        missing = tuple(name for name, age in ages.items() if age is None)
        stale = tuple(
            name
            for name, age in ages.items()
            if age is not None and age > max_image_staleness_steps
        )
        snapshot = None
        if not missing and not stale:
            snapshot = ObservationSnapshot(
                state=MappingProxyType(state),
                images=MappingProxyType(dict(image_cache)),
                task_uuid=task_uuid,
                task_text=task_text,
                source_timestep=timestep,
                generation=generation,
            )
        return ObservationAdmission(
            snapshot=snapshot,
            image_ages=MappingProxyType(ages),
            missing_images=missing,
            stale_images=stale,
            ignored_depth_fields=ignored,
        )

    def build_model_observation(self, snapshot: ObservationSnapshot) -> dict[str, Any]:
        video: dict[str, np.ndarray] = {}
        for wire_name, model_name in self.wire_image_to_model.items():
            decoded = cv2.imdecode(
                np.frombuffer(snapshot.images[wire_name].payload, np.uint8), cv2.IMREAD_COLOR
            )
            expected = (self.image_height, self.image_width, 3)
            if decoded is None or decoded.shape != expected:
                shape = None if decoded is None else decoded.shape
                raise ValueError(f"decoded {wire_name!r} shape {shape} differs from {expected}")
            video[model_name] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)[None, None, ...]
        return {
            "video": video,
            "state": {
                layout.key: self._assemble(layout, snapshot.state) for layout in self.state_layouts
            },
            "language": {LANGUAGE_KEY: [[snapshot.task_text]]},
        }

    def build_synthetic_model_observation(self, task_text: str) -> dict[str, Any]:
        fields: dict[str, tuple[float, ...]] = {}
        for field in self.state_fields:
            width = self.field_lengths[field]
            fields[field] = tuple(0.0 for _ in range(width))
        for layout in self.state_layouts:
            if layout.rot6d_index is not None:
                fields[layout.fields[layout.rot6d_index]] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        return {
            "video": {
                model: np.zeros((1, 1, self.image_height, self.image_width, 3), dtype=np.uint8)
                for model in self.wire_image_to_model.values()
            },
            "state": {layout.key: self._assemble(layout, fields) for layout in self.state_layouts},
            "language": {LANGUAGE_KEY: [[task_text]]},
        }

    def map_action_chunk(self, action: Mapping[str, Any]) -> tuple[dict[str, list[float]], ...]:
        expected = {
            layout.key: (1, self.action_horizon, layout.width) for layout in self.action_layouts
        }
        if set(action) != set(expected):
            raise ValueError(
                f"decoded action keys differ: expected {sorted(expected)}, got {sorted(action)}"
            )
        arrays: dict[str, np.ndarray] = {}
        for key, shape in expected.items():
            value = action[key]
            if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"decoded action {key!r} must be a numeric numpy array")
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"decoded action {key!r} must be finite with shape {shape}")
            arrays[key] = value
        rows: list[dict[str, list[float]]] = []
        for index in range(self.action_horizon):
            row: dict[str, list[float]] = {}
            for layout in self.action_layouts:
                values = arrays[layout.key][0, index]
                cursor = 0
                for field_index, (field, width) in enumerate(zip(layout.fields, layout.widths)):
                    part = values[cursor : cursor + width]
                    if field_index == layout.rot6d_index:
                        part = rot6d_groot_to_source(part)
                    row[field] = part.astype(float).tolist()
                    cursor += width
            rows.append(row)
        return tuple(rows)

    def map_wire_action_prefix(self, rows: Any) -> dict[str, np.ndarray]:
        if not isinstance(rows, list | tuple) or not 0 < len(rows) <= self.action_horizon:
            raise ValueError("action prefix must be a non-empty sequence within the action horizon")
        expected_fields = set(self.action_fields)
        output: dict[str, np.ndarray] = {}
        for layout in self.action_layouts:
            assembled: list[np.ndarray] = []
            for row_index, row in enumerate(rows):
                if not isinstance(row, Mapping) or set(row) != expected_fields:
                    raise ValueError(f"action prefix row {row_index} fields differ")
                parts = []
                for field_index, (field, width) in enumerate(zip(layout.fields, layout.widths)):
                    part = np.asarray(_finite_tuple(row[field], width, field), dtype=np.float32)
                    if field_index == layout.rot6d_index:
                        part = rot6d_source_to_groot(part)
                    parts.append(part)
                assembled.append(np.concatenate(parts).astype(np.float32, copy=False))
            output[layout.key] = np.stack(assembled)[None, ...]
        return output

    def service_metadata(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "expert_camera_height": self.image_height,
            "expert_camera_width": self.image_width,
            "video_keys": list(self.wire_image_to_model.values()),
            "wire_image_keys": list(self.wire_image_to_model),
            "state_keys": [layout.key for layout in self.state_layouts],
            "action_keys": [layout.key for layout in self.action_layouts],
            "wire_state_fields": list(self.state_fields),
            "wire_action_fields": list(self.action_fields),
            "position_action_fields": list(self.position_action_fields),
            "rotation_action_fields": list(self.rotation_action_fields),
            "auxiliary_action_fields": list(self.auxiliary_action_fields),
            "action_horizon": self.action_horizon,
            "action_start_offset_steps": self.action_start_offset_steps,
            "field_lengths": dict(self.field_lengths),
            "field_units": dict(self.field_units),
            "eef_frame": self.eef_frame,
        }

    def monitoring_gripper_values(
        self, action: Mapping[str, list[float]] | None
    ) -> dict[str, list[float]]:
        if action is None:
            return {}
        return {key: list(action[key]) for key in self.gripper_action_keys}

    def _assemble(self, layout: ProfileLayout, fields: Mapping[str, Any]) -> np.ndarray:
        parts = []
        for index, field in enumerate(layout.fields):
            part = np.asarray(fields[field], dtype=np.float32)
            if index == layout.rot6d_index:
                part = rot6d_source_to_groot(part)
            parts.append(part)
        return np.concatenate(parts).astype(np.float32, copy=False).reshape(1, 1, layout.width)


def load_industrialnext_profile(path: str | Path) -> ConfigDrivenIndustrialNextProfile:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("embodiment config must be a mapping")
    serving = _mapping(raw.get("serving"), "serving")
    serving_keys = {
        "profile",
        "embodiment_tag",
        "model_path",
        "device",
        "host",
        "port",
        "control_hz",
        "image_size",
        "rtc_mode",
        "field_lengths",
        "ignored_observation_keys",
        "field_units",
        "eef_frame",
        "supported_rtc_modes",
        "gripper_action_keys",
    }
    unknown_serving_keys = sorted(set(serving) - serving_keys)
    if unknown_serving_keys:
        raise ValueError(f"serving contains unknown keys: {unknown_serving_keys}")
    field_lengths_raw = _mapping(serving.get("field_lengths"), "serving.field_lengths")
    field_lengths = {
        _string(key, "field name"): _positive_int(value, f"field_lengths.{key}")
        for key, value in field_lengths_raw.items()
    }
    state_layouts = _parse_layouts(raw.get("state"), field_lengths, action=False)
    action_raw = _mapping(raw.get("action"), "action")
    action_layouts = _parse_layouts(action_raw.get("keys"), field_lengths, action=True)
    state_by_key = {layout.key: layout for layout in state_layouts}
    for layout in action_layouts:
        if layout.action_type == "EEF" and layout.action_format != "XYZ_ROT6D":
            raise ValueError(f"EEF action {layout.key!r} must use XYZ_ROT6D")
        if layout.action_type == "EEF" and layout.rot6d != ROT6D_TRANSFORM:
            raise ValueError(f"EEF action {layout.key!r} must declare the rot6d transform")
        if layout.action_type == "EEF" and (layout.widths != (3, 6) or layout.rot6d_index != 1):
            raise ValueError(
                f"EEF action {layout.key!r} must be ordered as one XYZ field then one rot6d field"
            )
        if layout.rep == "RELATIVE" and layout.state_key not in state_by_key:
            raise ValueError(
                f"relative action {layout.key!r} must reference a configured state key"
            )
        if layout.action_type == "EEF":
            state_layout = state_by_key.get(layout.state_key or "")
            if state_layout is None or (
                state_layout.widths != (3, 6) or state_layout.rot6d_index != 1
            ):
                raise ValueError(
                    f"EEF action {layout.key!r} requires a matching XYZ-then-rot6d state layout"
                )
    state_fields = tuple(field for layout in state_layouts for field in layout.fields)
    action_fields = tuple(field for layout in action_layouts for field in layout.fields)
    if len(state_fields) != len(set(state_fields)):
        raise ValueError("state wire fields must not be repeated across layouts")
    if len(action_fields) != len(set(action_fields)):
        raise ValueError("action wire fields must not be repeated across layouts")
    used_fields = set(state_fields) | set(action_fields)
    if used_fields != set(field_lengths):
        raise ValueError(
            "serving.field_lengths must exactly cover state/action wire fields: "
            f"missing={sorted(used_fields - set(field_lengths))}, "
            f"unused={sorted(set(field_lengths) - used_fields)}"
        )
    cameras = _mapping(raw.get("cameras"), "cameras")
    if not cameras:
        raise ValueError("cameras must not be empty")
    wire_image_to_model = {
        _string(wire, "camera wire key"): _string(model, "camera key")
        for model, wire in cameras.items()
    }
    if len(wire_image_to_model) != len(cameras):
        raise ValueError("camera wire keys must be unique")
    image_size = serving.get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError("serving.image_size must be [height, width]")
    tasks_raw = _mapping(raw.get("tasks"), "tasks")
    task_overrides = _mapping(tasks_raw.get("text_overrides"), "tasks.text_overrides")
    rtc_mode = _string(serving.get("rtc_mode"), "serving.rtc_mode")
    supported = _string_tuple(serving.get("supported_rtc_modes"), "serving.supported_rtc_modes")
    if not supported or len(supported) != len(set(supported)):
        raise ValueError("supported_rtc_modes must be a non-empty unique list")
    if not set(supported) <= {"off", "native", "trained_prefix"}:
        raise ValueError("supported_rtc_modes contains an unsupported mode")
    if rtc_mode not in supported:
        raise ValueError("serving.rtc_mode must be listed in supported_rtc_modes")
    model_path = Path(_string(serving.get("model_path"), "serving.model_path")).expanduser()
    if not model_path.is_absolute():
        model_path = (REPOSITORY_ROOT / model_path).resolve()
    control_hz = float(serving.get("control_hz"))
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("serving.control_hz must be finite and positive")
    action_offset = action_raw.get("observation_offset", 0)
    if isinstance(action_offset, bool) or not isinstance(action_offset, int) or action_offset < 0:
        raise ValueError("action.observation_offset must be a non-negative integer")
    ignored_keys = _string_tuple(
        serving.get("ignored_observation_keys", []),
        "serving.ignored_observation_keys",
        allow_empty=True,
    )
    if len(ignored_keys) != len(set(ignored_keys)):
        raise ValueError("ignored_observation_keys must be unique")
    if set(ignored_keys) & (set(state_fields) | set(wire_image_to_model)):
        raise ValueError("ignored_observation_keys overlap required observation fields")
    field_units = dict(_mapping(serving.get("field_units", {}), "serving.field_units"))
    if set(field_units) != set(field_lengths) or any(
        not isinstance(value, str) or not value for value in field_units.values()
    ):
        raise ValueError("serving.field_units must describe every configured field")
    gripper_keys = _string_tuple(
        serving.get("gripper_action_keys", []),
        "serving.gripper_action_keys",
        allow_empty=True,
    )
    if len(gripper_keys) != len(set(gripper_keys)) or any(
        key not in action_fields for key in gripper_keys
    ):
        raise ValueError("gripper_action_keys must be unique configured action wire fields")
    return ConfigDrivenIndustrialNextProfile(
        name=_string(raw.get("name"), "name"),
        profile_name=_string(serving.get("profile"), "serving.profile"),
        config_path=config_path,
        model_path=model_path,
        embodiment_tag=_string(serving.get("embodiment_tag"), "serving.embodiment_tag"),
        device=_string(serving.get("device"), "serving.device"),
        host=_string(serving.get("host"), "serving.host"),
        port=_positive_int(serving.get("port"), "serving.port"),
        control_hz=control_hz,
        image_height=_positive_int(image_size[0], "serving.image_size[0]"),
        image_width=_positive_int(image_size[1], "serving.image_size[1]"),
        rtc_mode=rtc_mode,
        supported_rtc_modes=supported,
        action_horizon=_positive_int(action_raw.get("horizon"), "action.horizon"),
        action_start_offset_steps=action_offset,
        wire_image_to_model=MappingProxyType(wire_image_to_model),
        state_layouts=state_layouts,
        action_layouts=action_layouts,
        field_lengths=MappingProxyType(field_lengths),
        ignored_observation_keys=frozenset(ignored_keys),
        field_units=MappingProxyType(field_units),
        eef_frame=_string(serving.get("eef_frame"), "serving.eef_frame"),
        gripper_action_keys=gripper_keys,
        task_catalog=task_catalog_from_mapping(_string(raw.get("name"), "name"), task_overrides),
    )


def _parse_layouts(
    value: Any, field_lengths: Mapping[str, int], *, action: bool
) -> tuple[ProfileLayout, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("layout must be a non-empty list")
    layouts = []
    seen_keys: set[str] = set()
    for raw in value:
        item = _mapping(raw, "layout entry")
        allowed_keys = {"key", "fields", "rot6d"}
        if action:
            allowed_keys |= {"rep", "type", "format", "state_key"}
        unknown_keys = sorted(set(item) - allowed_keys)
        if unknown_keys:
            raise ValueError(f"layout contains unknown keys: {unknown_keys}")
        raw_fields = item.get("fields")
        if not isinstance(raw_fields, list) or not all(
            isinstance(field_name, str) and field_name for field_name in raw_fields
        ):
            raise ValueError("layout.fields must be a non-empty list of strings")
        fields = tuple(raw_fields)
        if not fields or any(field not in field_lengths for field in fields):
            raise ValueError(f"layout fields must all have serving.field_lengths: {fields}")
        widths = tuple(field_lengths[field] for field in fields)
        rot6d = item.get("rot6d")
        rot_index = None
        if rot6d is not None:
            if rot6d != ROT6D_TRANSFORM:
                raise ValueError(f"unsupported rot6d transform {rot6d!r}")
            candidates = [index for index, width in enumerate(widths) if width == 6]
            if len(candidates) != 1:
                raise ValueError(
                    f"rot6d layout needs exactly one six-value rotation field: {fields}"
                )
            rot_index = candidates[0]
        key = _string(item.get("key"), "layout.key")
        if key in seen_keys:
            raise ValueError(f"duplicate layout key {key!r}")
        seen_keys.add(key)
        if action:
            if item.get("rep") not in {"RELATIVE", "DELTA", "ABSOLUTE"}:
                raise ValueError(f"invalid action representation for {key!r}")
            if item.get("type") not in {"EEF", "NON_EEF"}:
                raise ValueError(f"invalid action type for {key!r}")
            if item.get("format") not in {"DEFAULT", "XYZ_ROT6D", "XYZ_ROTVEC"}:
                raise ValueError(f"invalid action format for {key!r}")
            state_key = item.get("state_key")
            if state_key is not None and (not isinstance(state_key, str) or not state_key):
                raise ValueError(f"invalid action state_key for {key!r}")
        layouts.append(
            ProfileLayout(
                key=key,
                fields=fields,
                widths=widths,
                rot6d=rot6d,
                rot6d_index=rot_index,
                rep=item.get("rep") if action else None,
                action_type=item.get("type") if action else None,
                action_format=item.get("format") if action else None,
                state_key=item.get("state_key") if action else None,
            )
        )
    return tuple(layouts)


def _require_modality(
    config: ModalityConfig, delta_indices: list[int], keys: list[str], name: str
) -> None:
    if config.delta_indices != delta_indices or config.modality_keys != keys:
        raise ValueError(
            f"saved {name} contract differs: deltas={config.delta_indices}, keys={config.modality_keys}"
        )


def _validate_rgb_metadata(
    metadata: Mapping[str, Any], name: str, width: int, height: int
) -> Mapping[str, Any]:
    expected = {
        "format": "jpeg",
        "dtype": "uint8",
        "channels": 3,
        "height": height,
        "width": width,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"images_meta.{name}.{key} must be {expected_value!r}, got {metadata.get(key)!r}"
            )
    quality = metadata.get("quality")
    if quality is not None and (
        not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 100
    ):
        raise ValueError(f"images_meta.{name}.quality must be an integer in [1, 100]")
    return MappingProxyType(dict(metadata))


def _finite_tuple(value: Any, width: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple | np.ndarray):
        raise ValueError(f"{name} must be a numeric sequence")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {width} finite values")
    return tuple(float(item) for item in array)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    return value


def _string_tuple(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{name} must be a {'possibly empty ' if allow_empty else ''}list")
    return tuple(_string(item, f"{name} item") for item in value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
