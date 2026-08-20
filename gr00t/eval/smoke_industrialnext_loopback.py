# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paced, synthetic, no-motion smoke client for an Industrial Next policy server."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
from gr00t.policy.industrialnext.adapter import IMAGE_KEY_TO_MODEL_KEY
from industrialnext_rpc.direct.client import DirectClient
import numpy as np
import tyro


IDENTITY_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


@dataclass(frozen=True)
class LoopbackSmokeConfig:
    output_json_path: str
    host: str = "127.0.0.1"
    port: int = 10012
    task_uuid: str = "generic_pick"
    task_text: str = "Pick the grounded target object and hold it securely in the gripper."
    control_hz: float = 50.0
    steps: int = 60
    image_refresh_steps: int = 4

    def __post_init__(self) -> None:
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.image_refresh_steps <= 0:
            raise ValueError("image_refresh_steps must be positive")


def _observation(*, include_images: bool, task_uuid: str, task_text: str) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "left_arm_pose_pos": [0.1, 0.2, 0.3],
        "left_arm_pose_rot": IDENTITY_ROT6D,
        "left_gripper": [0.25],
        "left_ft": [0.0] * 6,
        "right_arm_pose_pos": [-0.1, -0.2, -0.3],
        "right_arm_pose_rot": IDENTITY_ROT6D,
        "right_gripper": [0.75],
        "right_ft": [0.0] * 6,
        "task_uuid": task_uuid,
        "task_text": task_text,
    }
    if include_images:
        ok, encoded = cv2.imencode(".jpg", np.zeros((256, 256, 3), dtype=np.uint8))
        if not ok:
            raise RuntimeError("failed to encode synthetic loopback image")
        metadata = {
            "format": "jpeg",
            "quality": 90,
            "dtype": "uint8",
            "channels": 3,
            "height": 256,
            "width": 256,
        }
        observation["images_meta"] = {}
        for key in IMAGE_KEY_TO_MODEL_KEY:
            observation[key] = encoded.tobytes()
            observation["images_meta"][key] = dict(metadata)
    return observation


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def main(config: LoopbackSmokeConfig) -> None:
    output_path = Path(config.output_json_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"loopback report already exists: {output_path}")

    client = DirectClient(config.host, config.port)
    responses: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    registration: dict[str, Any] | None = None
    failure: str | None = None
    try:
        client.connect()
        metadata = client.get_metadata()
        registration = client.request(
            {
                "type": "register_session",
                "control_hz": config.control_hz,
                "task_uuid": config.task_uuid,
                "task_text": config.task_text,
            }
        )
        session_id = registration["session_id"]
        next_deadline = time.monotonic()
        for step in range(config.steps):
            response = client.request(
                {
                    "type": "step",
                    "session_id": session_id,
                    "observation": _observation(
                        include_images=step % config.image_refresh_steps == 0,
                        task_uuid=config.task_uuid,
                        task_text=config.task_text,
                    ),
                }
            )
            action = response.get("action")
            responses.append(
                {
                    "step": step,
                    "error": response.get("error"),
                    "has_action": action is not None,
                    "finite_action": action is None or _all_finite(action),
                    "monitoring_timestep": response.get("monitoring_timestep"),
                    "monitoring": response.get("monitoring"),
                }
            )
            next_deadline += 1.0 / config.control_hz
            time.sleep(max(0.0, next_deadline - time.monotonic()))
        client.request({"type": "close_session", "session_id": session_id})
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        client.close()

    action_responses = sum(item["has_action"] for item in responses)
    errors = [item for item in responses if item["error"] is not None]
    nonfinite = [item for item in responses if not item["finite_action"]]
    report = {
        "schema_version": 1,
        "config": config.__dict__,
        "service_metadata": None if metadata is None else metadata.get("service_metadata"),
        "registration": registration,
        "summary": {
            "responses": len(responses),
            "action_responses": action_responses,
            "null_action_responses": len(responses) - action_responses,
            "protocol_errors": len(errors),
            "nonfinite_actions": len(nonfinite),
            "exception": failure,
        },
        "steps": responses,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failure is not None or errors or nonfinite or action_responses == 0:
        raise RuntimeError(f"loopback smoke failed; see {output_path}")


if __name__ == "__main__":
    main(tyro.cli(LoopbackSmokeConfig))
