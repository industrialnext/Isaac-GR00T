# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Industrial Next wire-to-GR00T semihumanoid adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Protocol

import cv2
import numpy as np

from gr00t.data.state_action.rot6d import rot6d_groot_to_source, rot6d_source_to_groot
from gr00t.data.types import ActionFormat, ActionRepresentation, ActionType, ModalityConfig


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
ACTION_HORIZON = 40
LANGUAGE_KEY = "annotation.human.task_description"

STATE_FIELD_WIDTHS: tuple[tuple[str, int], ...] = (
    ("left_arm_pose_pos", 3),
    ("left_arm_pose_rot", 6),
    ("left_gripper", 1),
    ("left_ft", 6),
    ("right_arm_pose_pos", 3),
    ("right_arm_pose_rot", 6),
    ("right_gripper", 1),
    ("right_ft", 6),
)
IMAGE_KEY_TO_MODEL_KEY: Mapping[str, str] = MappingProxyType(
    {
        "head_rgb": "head",
        "eoat_left_bottom_rgb": "left_wrist",
        "eoat_right_bottom_rgb": "right_wrist",
    }
)


class PolicyWithModalityConfig(Protocol):
    def get_modality_config(self) -> dict[str, ModalityConfig]: ...


@dataclass(frozen=True)
class CachedImage:
    payload: bytes
    metadata: Mapping[str, Any]
    updated_timestep: int


@dataclass(frozen=True)
class ObservationSnapshot:
    state: Mapping[str, tuple[float, ...]]
    images: Mapping[str, CachedImage]
    task_uuid: str
    task_text: str
    source_timestep: int
    generation: int


@dataclass(frozen=True)
class ObservationAdmission:
    snapshot: ObservationSnapshot | None
    image_ages: Mapping[str, int | None]
    missing_images: tuple[str, ...]
    stale_images: tuple[str, ...]
    ignored_depth_fields: int

    @property
    def ready(self) -> bool:
        return self.snapshot is not None


def assert_semihumanoid_policy_contract(policy: PolicyWithModalityConfig) -> None:
    """Reject checkpoints whose saved processor contract differs from this adapter."""
    configs = policy.get_modality_config()
    expected_modalities = {"video", "state", "action", "language"}
    if set(configs) != expected_modalities:
        raise ValueError(
            f"saved modality names differ: expected {sorted(expected_modalities)}, "
            f"got {sorted(configs)}"
        )
    _require_modality(configs["video"], [0], ["head", "left_wrist", "right_wrist"], "video")
    _require_modality(
        configs["state"],
        [0],
        ["left_eef", "left_gripper", "left_ft", "right_eef", "right_gripper", "right_ft"],
        "state",
    )
    _require_modality(configs["language"], [0], [LANGUAGE_KEY], "language")
    action = configs["action"]
    _require_modality(
        action,
        list(range(ACTION_HORIZON)),
        ["left_eef", "left_gripper", "right_eef", "right_gripper"],
        "action",
    )
    expected_action_configs = (
        (ActionRepresentation.RELATIVE, ActionType.EEF, ActionFormat.XYZ_ROT6D, "left_eef"),
        (ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT, None),
        (ActionRepresentation.RELATIVE, ActionType.EEF, ActionFormat.XYZ_ROT6D, "right_eef"),
        (ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT, None),
    )
    if action.action_configs is None or len(action.action_configs) != len(expected_action_configs):
        raise ValueError("saved action configs are missing or have the wrong length")
    actual_action_configs = tuple(
        (config.rep, config.type, config.format, config.state_key)
        for config in action.action_configs
    )
    if actual_action_configs != expected_action_configs:
        raise ValueError(
            f"saved action configs differ: expected {expected_action_configs}, "
            f"got {actual_action_configs}"
        )


def admit_observation(
    observation: Mapping[str, Any],
    *,
    image_cache: MutableMapping[str, CachedImage],
    timestep: int,
    task_uuid: str,
    task_text: str,
    generation: int,
    max_image_staleness_steps: int,
) -> ObservationAdmission:
    """Validate one sparse wire observation and update its RGB cache transactionally."""
    if not isinstance(observation, Mapping):
        raise ValueError("observation must be a mapping")
    if max_image_staleness_steps < 0:
        raise ValueError("max_image_staleness_steps must be non-negative")
    state = {
        field_name: _finite_tuple(observation.get(field_name), width, field_name)
        for field_name, width in STATE_FIELD_WIDTHS
    }
    if "task_uuid" in observation and observation["task_uuid"] != task_uuid:
        raise ValueError("observation task_uuid does not match the registered session")
    if "task_text" in observation and observation["task_text"] != task_text:
        raise ValueError("observation task_text does not match the registered session")

    raw_metadata = observation.get("images_meta", {})
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("images_meta must be a mapping when provided")

    allowed_fields = {name for name, _ in STATE_FIELD_WIDTHS} | {
        "task_uuid",
        "task_text",
        "images_meta",
    }
    pending_updates: dict[str, CachedImage] = {}
    depth_fields: set[str] = set()
    for field_name, payload in observation.items():
        if field_name in allowed_fields:
            continue
        if field_name.endswith("_depth"):
            depth_fields.add(field_name)
            continue
        if field_name not in IMAGE_KEY_TO_MODEL_KEY:
            raise ValueError(f"unexpected image or observation field {field_name!r}")
        if payload is None:
            raise ValueError(f"{field_name} must not be null")
        if not isinstance(payload, bytes | bytearray | memoryview):
            raise ValueError(f"{field_name} must contain encoded image bytes")
        immutable_payload = bytes(payload)
        if not immutable_payload:
            raise ValueError(f"{field_name} image payload must not be empty")
        metadata = raw_metadata.get(field_name)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"images_meta.{field_name} must be provided as a mapping")
        pending_updates[field_name] = CachedImage(
            payload=immutable_payload,
            metadata=_validate_rgb_metadata(metadata, field_name),
            updated_timestep=timestep,
        )

    for metadata_name in raw_metadata:
        if metadata_name in pending_updates:
            continue
        if metadata_name in depth_fields and metadata_name.endswith("_depth"):
            continue
        raise ValueError(f"orphan or unexpected image metadata {metadata_name!r}")

    image_cache.update(pending_updates)
    image_ages = {
        image_name: (
            None
            if image_name not in image_cache
            else timestep - image_cache[image_name].updated_timestep
        )
        for image_name in IMAGE_KEY_TO_MODEL_KEY
    }
    missing = tuple(name for name, age in image_ages.items() if age is None)
    stale = tuple(
        name
        for name, age in image_ages.items()
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
        image_ages=MappingProxyType(image_ages),
        missing_images=missing,
        stale_images=stale,
        ignored_depth_fields=len(depth_fields),
    )


def snapshot_is_fresh(
    snapshot: ObservationSnapshot,
    *,
    current_timestep: int,
    active_generation: int,
    max_staleness_steps: int,
) -> bool:
    """Return whether pending work is still current enough to launch."""
    if snapshot.generation != active_generation:
        return False
    if current_timestep - snapshot.source_timestep > max_staleness_steps:
        return False
    return all(
        current_timestep - image.updated_timestep <= max_staleness_steps
        for image in snapshot.images.values()
    )


def build_model_observation(snapshot: ObservationSnapshot) -> dict[str, Any]:
    """Decode one immutable snapshot into the exact strict GR00T observation."""
    video: dict[str, np.ndarray] = {}
    for wire_name, model_name in IMAGE_KEY_TO_MODEL_KEY.items():
        cached = snapshot.images[wire_name]
        encoded = np.frombuffer(cached.payload, dtype=np.uint8)
        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            raise ValueError(f"failed to decode JPEG view {wire_name!r}")
        if decoded_bgr.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            raise ValueError(
                f"decoded {wire_name!r} shape {decoded_bgr.shape} differs from "
                f"{(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}"
            )
        decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        video[model_name] = decoded_rgb[np.newaxis, np.newaxis, ...]

    state = snapshot.state
    model_state = {
        "left_eef": _model_eef(state, "left"),
        "left_gripper": _model_array(state["left_gripper"]),
        "left_ft": _model_array(state["left_ft"]),
        "right_eef": _model_eef(state, "right"),
        "right_gripper": _model_array(state["right_gripper"]),
        "right_ft": _model_array(state["right_ft"]),
    }
    return {
        "video": video,
        "state": model_state,
        "language": {LANGUAGE_KEY: [[snapshot.task_text]]},
    }


def build_synthetic_model_observation(task_text: str) -> dict[str, Any]:
    """Build the finite identity-pose observation used for startup warmup."""
    if not isinstance(task_text, str) or not task_text.strip() or task_text != task_text.strip():
        raise ValueError("task_text must be a non-empty string without surrounding whitespace")
    identity_eef = np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32
    ).reshape(1, 1, 9)
    zeros_ft = np.zeros((1, 1, 6), dtype=np.float32)
    return {
        "video": {
            model_name: np.zeros((1, 1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
            for model_name in IMAGE_KEY_TO_MODEL_KEY.values()
        },
        "state": {
            "left_eef": identity_eef.copy(),
            "left_gripper": np.full((1, 1, 1), 0.5, dtype=np.float32),
            "left_ft": zeros_ft.copy(),
            "right_eef": identity_eef.copy(),
            "right_gripper": np.full((1, 1, 1), 0.5, dtype=np.float32),
            "right_ft": zeros_ft.copy(),
        },
        "language": {LANGUAGE_KEY: [[task_text]]},
    }


def map_action_chunk(action: Mapping[str, Any]) -> tuple[dict[str, list[float]], ...]:
    """Validate a decoded absolute GR00T chunk and map every row to ROS field names."""
    expected_shapes = {
        "left_eef": (1, ACTION_HORIZON, 9),
        "left_gripper": (1, ACTION_HORIZON, 1),
        "right_eef": (1, ACTION_HORIZON, 9),
        "right_gripper": (1, ACTION_HORIZON, 1),
    }
    if set(action) != set(expected_shapes):
        raise ValueError(
            f"decoded action keys differ: expected {sorted(expected_shapes)}, got {sorted(action)}"
        )
    arrays: dict[str, np.ndarray] = {}
    for key, expected_shape in expected_shapes.items():
        value = action[key]
        if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"decoded action {key!r} must be a numeric numpy array")
        if value.shape != expected_shape:
            raise ValueError(
                f"decoded action {key!r} shape {value.shape} differs from {expected_shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"decoded action {key!r} contains non-finite values")
        arrays[key] = value

    rows: list[dict[str, list[float]]] = []
    for index in range(ACTION_HORIZON):
        left_eef = arrays["left_eef"][0, index]
        right_eef = arrays["right_eef"][0, index]
        rows.append(
            {
                "left_arm_pose_pos": left_eef[:3].astype(float).tolist(),
                "left_arm_pose_rot": rot6d_groot_to_source(left_eef[3:]).tolist(),
                "left_gripper": arrays["left_gripper"][0, index].astype(float).tolist(),
                "right_arm_pose_pos": right_eef[:3].astype(float).tolist(),
                "right_arm_pose_rot": rot6d_groot_to_source(right_eef[3:]).tolist(),
                "right_gripper": arrays["right_gripper"][0, index].astype(float).tolist(),
            }
        )
    return tuple(rows)


def _require_modality(
    config: ModalityConfig,
    delta_indices: list[int],
    modality_keys: list[str],
    modality_name: str,
) -> None:
    if config.delta_indices != delta_indices or config.modality_keys != modality_keys:
        raise ValueError(
            f"saved {modality_name} contract differs: expected deltas={delta_indices}, "
            f"keys={modality_keys}; got deltas={config.delta_indices}, "
            f"keys={config.modality_keys}"
        )


def _finite_tuple(value: Any, width: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple | np.ndarray):
        raise ValueError(f"{field_name} must be a numeric sequence of length {width}")
    if len(value) != width:
        raise ValueError(f"{field_name} must have length {width}, got {len(value)}")
    try:
        result = tuple(float(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numeric values") from exc
    if not all(math.isfinite(part) for part in result):
        raise ValueError(f"{field_name} contains non-finite values")
    return result


def _validate_rgb_metadata(metadata: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    expected = {
        "format": "jpeg",
        "dtype": "uint8",
        "channels": 3,
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"images_meta.{field_name}.{key} must be {expected_value!r}, "
                f"got {metadata.get(key)!r}"
            )
    quality = metadata.get("quality")
    if quality is not None and (
        not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 100
    ):
        raise ValueError(f"images_meta.{field_name}.quality must be an integer in [1, 100]")
    return MappingProxyType(dict(metadata))


def _model_eef(state: Mapping[str, tuple[float, ...]], side: str) -> np.ndarray:
    position = np.asarray(state[f"{side}_arm_pose_pos"], dtype=np.float32)
    rotation = rot6d_source_to_groot(np.asarray(state[f"{side}_arm_pose_rot"])).astype(np.float32)
    return np.concatenate([position, rotation])[np.newaxis, np.newaxis, :]


def _model_array(value: tuple[float, ...]) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)[np.newaxis, np.newaxis, :]
