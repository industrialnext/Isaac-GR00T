# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe tests for the Industrial Next semihumanoid adapter."""

from __future__ import annotations

from pathlib import Path

import cv2
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.policy.industrialnext.adapter import (
    ACTION_HORIZON,
    IMAGE_KEY_TO_MODEL_KEY,
    STATE_FIELD_WIDTHS,
    CachedImage,
    admit_observation,
    assert_semihumanoid_policy_contract,
    build_model_observation,
    build_synthetic_model_observation,
    map_action_chunk,
    snapshot_is_fresh,
)
from gr00t.policy.industrialnext.task_catalog import load_task_catalog
import numpy as np
import pytest


TASK_UUID = "generic_pick"
TASK_TEXT = "Pick the grounded target object."
IDENTITY_SOURCE_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
IDENTITY_GROOT_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class _PolicyContract:
    def __init__(self, configs: dict[str, ModalityConfig]):
        self._configs = configs

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self._configs


def _modality_configs() -> dict[str, ModalityConfig]:
    return {
        "video": ModalityConfig([0], ["head", "left_wrist", "right_wrist"]),
        "state": ModalityConfig(
            [0],
            ["left_eef", "left_gripper", "left_ft", "right_eef", "right_gripper", "right_ft"],
        ),
        "action": ModalityConfig(
            list(range(ACTION_HORIZON)),
            ["left_eef", "left_gripper", "right_eef", "right_gripper"],
            action_configs=[
                ActionConfig(
                    ActionRepresentation.RELATIVE,
                    ActionType.EEF,
                    ActionFormat.XYZ_ROT6D,
                    "left_eef",
                ),
                ActionConfig(
                    ActionRepresentation.ABSOLUTE,
                    ActionType.NON_EEF,
                    ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    ActionRepresentation.RELATIVE,
                    ActionType.EEF,
                    ActionFormat.XYZ_ROT6D,
                    "right_eef",
                ),
                ActionConfig(
                    ActionRepresentation.ABSOLUTE,
                    ActionType.NON_EEF,
                    ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig([0], ["annotation.human.task_description"]),
    }


def _jpeg_payload(rgb: tuple[int, int, int] = (220, 30, 10)) -> bytes:
    image_rgb = np.empty((256, 256, 3), dtype=np.uint8)
    image_rgb[:] = rgb
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return encoded.tobytes()


def _image_metadata() -> dict[str, object]:
    return {
        "format": "jpeg",
        "quality": 90,
        "dtype": "uint8",
        "channels": 3,
        "height": 256,
        "width": 256,
    }


def _wire_observation(*, include_images: bool = True) -> dict[str, object]:
    observation: dict[str, object] = {
        "left_arm_pose_pos": [0.1, 0.2, 0.3],
        "left_arm_pose_rot": IDENTITY_SOURCE_ROT6D,
        "left_gripper": [0.25],
        "left_ft": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
        "right_arm_pose_pos": [-0.1, -0.2, -0.3],
        "right_arm_pose_rot": IDENTITY_SOURCE_ROT6D,
        "right_gripper": [0.75],
        "right_ft": [-1.0, -2.0, -3.0, -0.1, -0.2, -0.3],
        "task_uuid": TASK_UUID,
        "task_text": TASK_TEXT,
    }
    if include_images:
        metadata = {}
        for image_name in IMAGE_KEY_TO_MODEL_KEY:
            observation[image_name] = _jpeg_payload()
            metadata[image_name] = _image_metadata()
        observation["images_meta"] = metadata
    return observation


def _admit(
    observation: dict[str, object],
    cache: dict[str, CachedImage],
    *,
    timestep: int = 0,
    max_staleness: int = 5,
):
    return admit_observation(
        observation,
        image_cache=cache,
        timestep=timestep,
        task_uuid=TASK_UUID,
        task_text=TASK_TEXT,
        generation=3,
        max_image_staleness_steps=max_staleness,
    )


def test_real_catalog_schema_and_order(tmp_path: Path) -> None:
    catalog_path = tmp_path / "task_catalog.yaml"
    catalog_path.write_text(
        """\
schema_version: 1
task_family: generic_pick_and_place
catalog_version: "2026-08-16"
tasks:
  - task_uuid: generic_pick
    task_text: "Pick the grounded target object."
    display_name: Pick
  - task_uuid: generic_place
    task_text: "Place the currently held object."
    display_name: Place
  - task_uuid: bracket_handover
    task_text: "Hand over the bracket."
    display_name: Bracket Handover
""",
        encoding="utf-8",
    )
    catalog = load_task_catalog(catalog_path)
    assert list(catalog.task_uuid_to_text) == [
        "generic_pick",
        "generic_place",
        "bracket_handover",
    ]
    assert catalog.resolve("generic_pick", TASK_TEXT) == TASK_TEXT
    assert catalog.to_metadata()[0]["display_name"] == "Pick"
    with pytest.raises(ValueError, match="surrounding whitespace"):
        catalog.resolve(" generic_pick", TASK_TEXT)
    with pytest.raises(ValueError, match="surrounding whitespace"):
        catalog.resolve("generic_pick", f"{TASK_TEXT} ")


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        ("schema_version: 1\ntask_family: test\ntasks: []\n", "non-empty"),
        (
            "schema_version: 1\ntask_family: test\nunknown: true\ntasks: []\n",
            "unknown keys",
        ),
        (
            """\
schema_version: 1
task_family: test
tasks:
  - {task_uuid: duplicate, task_text: first, display_name: First}
  - {task_uuid: duplicate, task_text: second, display_name: Second}
""",
            "duplicate",
        ),
        (
            "schema_version: 2\ntask_family: test\ntasks: [{}]\n",
            "unsupported schema_version",
        ),
        (
            """\
schema_version: 1
task_family: test
tasks:
  - {task_uuid: " spaced", task_text: text, display_name: Name}
""",
            "surrounding whitespace",
        ),
        ("schema_version: [\n", "failed to load task catalog"),
    ],
)
def test_malformed_catalog_is_rejected(tmp_path: Path, yaml_text: str, message: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_task_catalog(path)


def test_saved_contract_is_exact() -> None:
    assert_semihumanoid_policy_contract(_PolicyContract(_modality_configs()))
    bad = _modality_configs()
    bad["video"] = ModalityConfig([0], ["head", "right_wrist", "left_wrist"])
    with pytest.raises(ValueError, match="video contract"):
        assert_semihumanoid_policy_contract(_PolicyContract(bad))


def test_synthetic_warmup_observation_matches_the_saved_contract() -> None:
    observation = build_synthetic_model_observation(TASK_TEXT)
    assert set(observation["video"]) == {"head", "left_wrist", "right_wrist"}
    assert all(value.shape == (1, 1, 256, 256, 3) for value in observation["video"].values())
    assert observation["state"]["left_eef"].shape == (1, 1, 9)
    assert observation["state"]["left_ft"].shape == (1, 1, 6)
    assert observation["language"] == {"annotation.human.task_description": [[TASK_TEXT]]}
    with pytest.raises(ValueError, match="task_text"):
        build_synthetic_model_observation(" ")


def test_sparse_images_are_cached_and_staleness_is_enforced() -> None:
    cache: dict[str, CachedImage] = {}
    initial = _admit(_wire_observation(), cache)
    assert initial.ready
    assert initial.image_ages == {name: 0 for name in IMAGE_KEY_TO_MODEL_KEY}

    reused = _admit(_wire_observation(include_images=False), cache, timestep=5)
    assert reused.ready
    assert reused.image_ages == {name: 5 for name in IMAGE_KEY_TO_MODEL_KEY}
    assert reused.snapshot is not None
    assert snapshot_is_fresh(
        reused.snapshot,
        current_timestep=5,
        active_generation=3,
        max_staleness_steps=5,
    )

    stale = _admit(_wire_observation(include_images=False), cache, timestep=6)
    assert not stale.ready
    assert stale.stale_images == tuple(IMAGE_KEY_TO_MODEL_KEY)


def test_invalid_observation_does_not_mutate_image_cache() -> None:
    cache: dict[str, CachedImage] = {}
    observation = _wire_observation()
    observation["left_ft"] = [float("nan")] * 6
    with pytest.raises(ValueError, match="non-finite"):
        _admit(observation, cache)
    assert cache == {}


def test_depth_is_ignored_and_unknown_or_orphan_images_are_rejected() -> None:
    cache: dict[str, CachedImage] = {}
    observation = _wire_observation()
    observation["head_depth"] = b"ignored"
    metadata = observation["images_meta"]
    assert isinstance(metadata, dict)
    metadata["head_depth"] = {"format": "png"}
    admitted = _admit(observation, cache)
    assert admitted.ignored_depth_fields == 1

    unknown = _wire_observation()
    unknown["side_rgb"] = _jpeg_payload()
    with pytest.raises(ValueError, match="unexpected"):
        _admit(unknown, {})

    orphan = _wire_observation()
    orphan_metadata = orphan["images_meta"]
    assert isinstance(orphan_metadata, dict)
    orphan_metadata["side_rgb"] = _image_metadata()
    with pytest.raises(ValueError, match="orphan"):
        _admit(orphan, {})


def test_snapshot_decodes_to_strict_model_observation() -> None:
    admitted = _admit(_wire_observation(), {})
    assert admitted.snapshot is not None
    model_observation = build_model_observation(admitted.snapshot)
    assert list(model_observation["video"]) == ["head", "left_wrist", "right_wrist"]
    for image in model_observation["video"].values():
        assert image.shape == (1, 1, 256, 256, 3)
        assert image.dtype == np.uint8
        assert float(image[..., 0].mean()) > float(image[..., 1].mean())
    assert model_observation["state"]["left_eef"].shape == (1, 1, 9)
    assert model_observation["state"]["left_eef"].dtype == np.float32
    np.testing.assert_allclose(
        model_observation["state"]["left_eef"][0, 0, 3:],
        IDENTITY_GROOT_ROT6D,
    )
    assert model_observation["language"] == {"annotation.human.task_description": [[TASK_TEXT]]}


def test_action_chunk_maps_absolute_pose_and_passes_grippers_through() -> None:
    left_eef = np.zeros((1, ACTION_HORIZON, 9), dtype=np.float32)
    right_eef = np.zeros((1, ACTION_HORIZON, 9), dtype=np.float32)
    left_eef[..., :3] = [0.1, 0.2, 0.3]
    right_eef[..., :3] = [-0.1, -0.2, -0.3]
    left_eef[..., 3:] = IDENTITY_GROOT_ROT6D
    right_eef[..., 3:] = IDENTITY_GROOT_ROT6D
    left_gripper = np.linspace(-0.2, 1.2, ACTION_HORIZON, dtype=np.float32)[None, :, None]
    right_gripper = np.linspace(1.2, -0.2, ACTION_HORIZON, dtype=np.float32)[None, :, None]

    rows = map_action_chunk(
        {
            "left_eef": left_eef,
            "left_gripper": left_gripper,
            "right_eef": right_eef,
            "right_gripper": right_gripper,
        }
    )
    assert len(rows) == ACTION_HORIZON
    assert rows[0]["left_arm_pose_pos"] == pytest.approx([0.1, 0.2, 0.3])
    assert rows[0]["left_arm_pose_rot"] == pytest.approx(IDENTITY_SOURCE_ROT6D)
    assert rows[0]["left_gripper"] == pytest.approx([-0.2])
    assert rows[-1]["left_gripper"] == pytest.approx([1.2])


def test_action_chunk_rejects_wrong_shape_nonfinite_and_extra_key() -> None:
    valid = {
        "left_eef": np.zeros((1, ACTION_HORIZON, 9), dtype=np.float32),
        "left_gripper": np.zeros((1, ACTION_HORIZON, 1), dtype=np.float32),
        "right_eef": np.zeros((1, ACTION_HORIZON, 9), dtype=np.float32),
        "right_gripper": np.zeros((1, ACTION_HORIZON, 1), dtype=np.float32),
    }
    valid["left_eef"][..., 3:] = IDENTITY_GROOT_ROT6D
    valid["right_eef"][..., 3:] = IDENTITY_GROOT_ROT6D

    wrong_shape = dict(valid)
    wrong_shape["left_gripper"] = np.zeros((1, ACTION_HORIZON), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        map_action_chunk(wrong_shape)

    nonfinite = {key: value.copy() for key, value in valid.items()}
    nonfinite["right_eef"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        map_action_chunk(nonfinite)

    with pytest.raises(ValueError, match="keys differ"):
        map_action_chunk(valid | {"extra": np.zeros(1)})


@pytest.mark.parametrize("field_name, width", STATE_FIELD_WIDTHS)
def test_every_required_state_field_is_enforced(field_name: str, width: int) -> None:
    observation = _wire_observation()
    del observation[field_name]
    with pytest.raises(ValueError, match=field_name):
        _admit(observation, {})
    assert width > 0
