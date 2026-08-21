# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send recorded xArm/PSYONIC samples through the Industrial Next async protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.rot6d import rot6d_groot_to_source
from gr00t.policy.industrialnext import load_industrialnext_profile
from industrialnext_rpc.direct.client import DirectClient
import numpy as np
import tyro
from xarm_psyonic_mori_bracket_config import xarm_psyonic_mori_bracket_config


@dataclass(frozen=True)
class ClientConfig:
    config: str = "configs/embodiments/xarm_psyonic_mori_bracket.yaml"
    dataset_path: str = (
        "data/training_data/gr00t/xarm_psyonic_mori_bracket_v1_20260821/xarm_psyonic_val"
    )
    episode_index: int = 0
    steps: int = 60
    image_refresh_steps: int = 4
    host: str | None = None
    port: int | None = None

    def __post_init__(self) -> None:
        for name, value in {
            "episode_index": self.episode_index,
            "steps": self.steps,
            "image_refresh_steps": self.image_refresh_steps,
        }.items():
            minimum = 0 if name == "episode_index" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


def _jpeg(image: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    image = np.asarray(image, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode recorded RGB image")
    height, width = image.shape[:2]
    return encoded.tobytes(), {
        "format": "jpeg",
        "quality": 90,
        "dtype": "uint8",
        "channels": 3,
        "height": height,
        "width": width,
    }


def main(config: ClientConfig) -> None:
    profile = load_industrialnext_profile(config.config)
    task = profile.task_catalog.tasks[0]
    dataset_path = Path(config.dataset_path).expanduser().resolve()
    loader = LeRobotEpisodeLoader(dataset_path, xarm_psyonic_mori_bracket_config)
    episode = loader[config.episode_index]
    client = DirectClient(
        profile.host if config.host is None else config.host,
        profile.port if config.port is None else config.port,
    )
    client.connect()
    try:
        registration = client.request(
            {
                "type": "register_session",
                "control_hz": profile.control_hz,
                "task_uuid": task.task_uuid,
                "task_text": task.task_text,
            }
        )
        if registration.get("error"):
            raise RuntimeError(registration["error"])
        session_id = registration["session_id"]
        deadline = time.monotonic()
        action_count = 0
        for step_index in range(min(config.steps, len(episode))):
            point = extract_step_data(
                episode,
                step_index,
                {
                    name: value
                    for name, value in xarm_psyonic_mori_bracket_config.items()
                    if name != "action"
                },
                EmbodimentTag.NEW_EMBODIMENT,
            )
            eef = np.asarray(point.states["right_eef"])[-1]
            observation: dict[str, Any] = {
                "right_arm_pose_pos": eef[:3].astype(float).tolist(),
                "right_arm_pose_rot": rot6d_groot_to_source(eef[3:]).astype(float).tolist(),
                "right_hand": np.asarray(point.states["right_hand"])[-1].astype(float).tolist(),
                "task_uuid": task.task_uuid,
                "task_text": task.task_text,
            }
            if step_index % config.image_refresh_steps == 0:
                observation["images_meta"] = {}
                for model_key, wire_key in (
                    ("front", "static_center_rgb"),
                    ("wrist", "eoat_right_bottom_rgb"),
                ):
                    payload, metadata = _jpeg(np.asarray(point.images[model_key])[-1])
                    observation[wire_key] = payload
                    observation["images_meta"][wire_key] = metadata
            response = client.request(
                {"type": "step", "session_id": session_id, "observation": observation}
            )
            if response.get("error"):
                raise RuntimeError(response["error"])
            action = response.get("action")
            if action is not None:
                action_count += 1
                if set(action) != set(profile.action_fields):
                    raise ValueError(f"unexpected action fields: {sorted(action)}")
                if not all(np.isfinite(value).all() for value in map(np.asarray, action.values())):
                    raise ValueError("server returned a non-finite action")
            deadline += 1.0 / profile.control_hz
            time.sleep(max(0.0, deadline - time.monotonic()))
        client.request({"type": "close_session", "session_id": session_id})
        if action_count == 0:
            raise RuntimeError("recorded run completed without receiving an action")
        print(f"recorded Industrial Next smoke passed: actions={action_count}")
    finally:
        client.close()


if __name__ == "__main__":
    main(tyro.cli(ClientConfig))
