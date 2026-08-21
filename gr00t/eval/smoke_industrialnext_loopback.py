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
from gr00t.policy.industrialnext import (
    ConfigDrivenIndustrialNextProfile,
    load_industrialnext_profile,
)
from industrialnext_rpc.direct.client import DirectClient
import numpy as np
import tyro


IDENTITY_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


@dataclass(frozen=True)
class LoopbackSmokeConfig:
    config: str
    output_json_path: str | None = None
    host: str | None = None
    port: int | None = None
    task_uuid: str | None = None
    task_text: str | None = None
    steps: int = 60
    image_refresh_steps: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps <= 0:
            raise ValueError("steps must be positive")
        if (
            isinstance(self.image_refresh_steps, bool)
            or not isinstance(self.image_refresh_steps, int)
            or self.image_refresh_steps <= 0
        ):
            raise ValueError("image_refresh_steps must be positive")


def _observation(
    profile: ConfigDrivenIndustrialNextProfile,
    *,
    include_images: bool,
    task_uuid: str,
    task_text: str,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "task_uuid": task_uuid,
        "task_text": task_text,
    }
    rotation_fields = set(profile.rotation_state_fields)
    for field in profile.state_fields:
        width = profile.field_lengths[field]
        observation[field] = IDENTITY_ROT6D if field in rotation_fields else [0.0] * width
    if include_images:
        ok, encoded = cv2.imencode(
            ".jpg", np.zeros((profile.image_height, profile.image_width, 3), dtype=np.uint8)
        )
        if not ok:
            raise RuntimeError("failed to encode synthetic loopback image")
        metadata = {
            "format": "jpeg",
            "quality": 90,
            "dtype": "uint8",
            "channels": 3,
            "height": profile.image_height,
            "width": profile.image_width,
        }
        observation["images_meta"] = {}
        for key in profile.wire_image_to_model:
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
    profile = load_industrialnext_profile(config.config)
    task = profile.task_catalog.tasks[0]
    task_uuid = task.task_uuid if config.task_uuid is None else config.task_uuid
    task_text = task.task_text if config.task_text is None else config.task_text
    host = profile.host if config.host is None else config.host
    port = profile.port if config.port is None else config.port
    control_hz = profile.control_hz
    output_path = (
        Path("/tmp") / f"gr00t_industrialnext_loopback_{profile.profile_name}_{time.time_ns()}.json"
        if config.output_json_path is None
        else Path(config.output_json_path).expanduser().resolve()
    )
    if config.output_json_path is not None and output_path.exists():
        raise FileExistsError(f"loopback report already exists: {output_path}")

    client = DirectClient(host, port)
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
                "control_hz": control_hz,
                "task_uuid": task_uuid,
                "task_text": task_text,
            }
        )
        if registration.get("error"):
            raise RuntimeError(registration["error"])
        session_id = registration["session_id"]
        next_deadline = time.monotonic()
        for step in range(config.steps):
            response = client.request(
                {
                    "type": "step",
                    "session_id": session_id,
                    "observation": _observation(
                        profile,
                        include_images=step % config.image_refresh_steps == 0,
                        task_uuid=task_uuid,
                        task_text=task_text,
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
            next_deadline += 1.0 / control_hz
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
    print(f"Industrial Next loopback smoke passed: {output_path}")


if __name__ == "__main__":
    main(tyro.cli(LoopbackSmokeConfig))
