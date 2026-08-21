# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config-driven Industrial Next profile tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
from gr00t.policy.industrialnext import load_industrialnext_profile, task_catalog_from_mapping
import numpy as np
import pytest
import yaml


IDENTITY_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("path", "profile_name", "offset", "state_keys", "action_keys"),
    [
        (
            "configs/embodiments/xarm_psyonic_mori_bracket.yaml",
            "xarm_psyonic",
            1,
            ["right_eef", "right_hand"],
            ["right_eef", "right_hand"],
        ),
        (
            "configs/embodiments/semihumanoid.yaml",
            "semihumanoid",
            0,
            ["left_eef", "left_gripper", "left_ft", "right_eef", "right_gripper", "right_ft"],
            ["left_eef", "left_gripper", "right_eef", "right_gripper"],
        ),
    ],
)
def test_checked_in_profiles_load(
    path: str,
    profile_name: str,
    offset: int,
    state_keys: list[str],
    action_keys: list[str],
) -> None:
    profile = load_industrialnext_profile(path)
    assert profile.profile_name == profile_name
    assert profile.action_horizon == 40
    assert profile.action_start_offset_steps == offset
    assert [layout.key for layout in profile.state_layouts] == state_keys
    assert [layout.key for layout in profile.action_layouts] == action_keys
    assert profile.port == 10012
    assert profile.task_catalog.tasks
    assert profile.position_action_fields
    assert profile.rotation_action_fields


def test_xarm_observation_action_and_prefix_mapping() -> None:
    profile = load_industrialnext_profile("configs/embodiments/xarm_psyonic_mori_bracket.yaml")
    ok, encoded = cv2.imencode(
        ".jpg", np.zeros((profile.image_height, profile.image_width, 3), dtype=np.uint8)
    )
    assert ok
    task = profile.task_catalog.tasks[0]
    observation = {
        "right_arm_pose_pos": [0.1, 0.2, 0.3],
        "right_arm_pose_rot": IDENTITY_ROT6D,
        "right_hand": [0.0] * 6,
        "task_uuid": task.task_uuid,
        "task_text": task.task_text,
        "images_meta": {},
        "static_center_depth": b"ignored",
    }
    metadata = {
        "format": "jpeg",
        "quality": 90,
        "dtype": "uint8",
        "channels": 3,
        "height": profile.image_height,
        "width": profile.image_width,
    }
    for key in profile.wire_image_to_model:
        observation[key] = encoded.tobytes()
        observation["images_meta"][key] = dict(metadata)
    admission = profile.admit_observation(
        observation,
        image_cache={},
        timestep=0,
        task_uuid=task.task_uuid,
        task_text=task.task_text,
        generation=1,
        max_image_staleness_steps=5,
    )
    assert admission.ready
    assert admission.ignored_depth_fields == 1
    model_observation = profile.build_model_observation(admission.snapshot)
    assert model_observation["state"]["right_eef"].shape == (1, 1, 9)
    assert model_observation["state"]["right_hand"].shape == (1, 1, 6)
    assert model_observation["state"]["right_eef"].dtype == np.float32

    eef = np.zeros((1, 40, 9), dtype=np.float32)
    eef[..., 3:] = IDENTITY_ROT6D
    hand = np.arange(240, dtype=np.float32).reshape(1, 40, 6)
    rows = profile.map_action_chunk({"right_eef": eef, "right_hand": hand})
    assert set(rows[0]) == {"right_arm_pose_pos", "right_arm_pose_rot", "right_hand"}
    np.testing.assert_allclose(rows[0]["right_arm_pose_rot"], IDENTITY_ROT6D)
    prefix = profile.map_wire_action_prefix(rows[:3])
    np.testing.assert_allclose(prefix["right_eef"], eef[:, :3])
    np.testing.assert_allclose(prefix["right_hand"], hand[:, :3])
    assert prefix["right_eef"].dtype == np.float32
    with pytest.raises(ValueError, match="numeric numpy array"):
        profile.map_action_chunk({"right_eef": eef.astype(str), "right_hand": hand})

    unexpected = dict(observation)
    unexpected["unconfigured_depth"] = b"not accepted"
    with pytest.raises(ValueError, match="unexpected image or observation"):
        profile.admit_observation(
            unexpected,
            image_cache={},
            timestep=0,
            task_uuid=task.task_uuid,
            task_text=task.task_text,
            generation=1,
            max_image_staleness_steps=5,
        )


def test_profile_rejects_unknown_layout_keys(tmp_path: Path) -> None:
    source = Path("configs/embodiments/xarm_psyonic_mori_bracket.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["action"]["keys"][0]["typo"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_industrialnext_profile(path)


def test_task_catalog_mapping_rejects_non_mapping_display_names() -> None:
    with pytest.raises(ValueError, match="display_names must be a mapping"):
        task_catalog_from_mapping("test", {"pick": "Pick it."}, display_names=cast(Any, "Pick"))
