# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loopback conformance tests using the real direct WebSocket transport."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
from gr00t.eval.run_gr00t_industrialnext_server import (
    ServerConfig,
    build_service_provenance,
    resolve_server_config,
)
from gr00t.policy.industrialnext import (
    ACTION_HORIZON,
    IndustrialNextAsyncServer,
    IndustrialNextServingConfig,
    TaskCatalog,
    TaskCatalogEntry,
    load_industrialnext_profile,
)
from gr00t.policy.industrialnext.adapter import IMAGE_KEY_TO_MODEL_KEY
from industrialnext_rpc.direct.client import DirectClient
from industrialnext_rpc.direct.server import DirectServer
import numpy as np
import pytest


TASK_UUID = "generic_pick"
TASK_TEXT = "Pick the grounded target object."
IDENTITY_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
PROFILE = load_industrialnext_profile("configs/embodiments/semihumanoid.yaml")


class _BlockingPolicy:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        assert observation["language"]["annotation.human.task_description"] == [[TASK_TEXT]]
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("test did not release fake inference")
        eef = np.zeros((1, ACTION_HORIZON, 9), dtype=np.float32)
        eef[0, :, 0] = np.arange(ACTION_HORIZON)
        eef[0, :, 3:] = IDENTITY_ROT6D
        return {
            "left_eef": eef.copy(),
            "left_gripper": np.full((1, ACTION_HORIZON, 1), 0.2, dtype=np.float32),
            "right_eef": eef.copy(),
            "right_gripper": np.full((1, ACTION_HORIZON, 1), 0.8, dtype=np.float32),
        }, {}


def _handler(policy: _BlockingPolicy) -> IndustrialNextAsyncServer:
    task_catalog = TaskCatalog(
        schema_version=1,
        task_family="test",
        catalog_version="test",
        tasks=(TaskCatalogEntry(TASK_UUID, TASK_TEXT, "Pick"),),
    )
    return IndustrialNextAsyncServer(
        policy=policy,
        executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="groot-protocol-test"),
        config=IndustrialNextServingConfig(stats_log_interval_steps=0),
        service_provenance={"model_path": "/test/model"},
        embodiment_tag="new_embodiment",
        profile=replace(PROFILE, task_catalog=task_catalog),
    )


def _observation(*, include_images: bool) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "left_arm_pose_pos": [0.1, 0.2, 0.3],
        "left_arm_pose_rot": IDENTITY_ROT6D,
        "left_gripper": [0.25],
        "left_ft": [0.0] * 6,
        "right_arm_pose_pos": [-0.1, -0.2, -0.3],
        "right_arm_pose_rot": IDENTITY_ROT6D,
        "right_gripper": [0.75],
        "right_ft": [0.0] * 6,
        "task_uuid": TASK_UUID,
        "task_text": TASK_TEXT,
    }
    if include_images:
        ok, encoded = cv2.imencode(".jpg", np.zeros((256, 256, 3), dtype=np.uint8))
        assert ok
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


async def _wait_for_inference(handler: IndustrialNextAsyncServer) -> None:
    deadline = time.monotonic() + 2.0
    while handler._inference_future is not None:
        if time.monotonic() >= deadline:
            raise TimeoutError("inference completion callback did not run")
        await asyncio.sleep(0.005)


def test_real_direct_server_client_loopback_contract() -> None:
    async def scenario() -> None:
        policy = _BlockingPolicy()
        handler = _handler(policy)
        client_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="direct-client")
        client: DirectClient | None = None
        try:
            async with DirectServer("127.0.0.1", 0, handler) as transport:
                port = transport.server.sockets[0].getsockname()[1]
                client = DirectClient("127.0.0.1", port)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(client_executor, client.connect)

                metadata = await loop.run_in_executor(client_executor, client.get_metadata)
                service = metadata["service_metadata"]
                assert service["async_protocol_version"] == 2
                assert service["task_conditioning"]["task_uuid_to_text"] == {TASK_UUID: TASK_TEXT}

                registration = await loop.run_in_executor(
                    client_executor,
                    client.request,
                    {
                        "type": "register_session",
                        "control_hz": 50.0,
                        "task_uuid": TASK_UUID,
                        "task_text": TASK_TEXT,
                    },
                )
                session_id = registration["session_id"]
                started_at = time.perf_counter()
                startup = await loop.run_in_executor(
                    client_executor,
                    client.request,
                    {
                        "type": "step",
                        "session_id": session_id,
                        "observation": _observation(include_images=True),
                    },
                )
                assert time.perf_counter() - started_at < 0.5
                assert startup["action"] is None
                assert startup["monitoring"]["progress"] == 0.0
                assert await asyncio.to_thread(policy.started.wait, 1.0)

                sparse = await loop.run_in_executor(
                    client_executor,
                    client.request,
                    {
                        "type": "step",
                        "session_id": session_id,
                        "observation": _observation(include_images=False),
                    },
                )
                assert sparse["action"] is None
                assert sparse["monitoring"]["progress"] == 0.0
                assert sparse["monitoring_timestep"] == 1

                replacement = await loop.run_in_executor(
                    client_executor,
                    client.request,
                    {
                        "type": "register_session",
                        "control_hz": 50.0,
                        "task_uuid": TASK_UUID,
                        "task_text": TASK_TEXT,
                    },
                )
                new_session_id = replacement["session_id"]
                displaced = await loop.run_in_executor(
                    client_executor,
                    client.request,
                    {
                        "type": "step",
                        "session_id": session_id,
                        "observation": _observation(include_images=False),
                    },
                )
                assert displaced["error"] == "session_not_found"

                policy.release.set()
                await _wait_for_inference(handler)
                assert handler._active_session is not None
                assert handler._active_session.session_id == new_session_id
                assert handler._active_session.timeline == {}

                await loop.run_in_executor(
                    client_executor,
                    client.request,
                    {
                        "type": "step",
                        "session_id": new_session_id,
                        "observation": _observation(include_images=True),
                    },
                )
                await _wait_for_inference(handler)
                action_response = await loop.run_in_executor(
                    client_executor,
                    client.request,
                    {
                        "type": "step",
                        "session_id": new_session_id,
                        "observation": _observation(include_images=False),
                    },
                )
                assert action_response["action"]["left_arm_pose_pos"][0] == 1.0
                assert action_response["monitoring"]["progress"] == 0.0

                assert await loop.run_in_executor(
                    client_executor,
                    client.request,
                    {"type": "close_session", "session_id": new_session_id},
                ) == {"ok": True}
                closed = await loop.run_in_executor(
                    client_executor,
                    client.request,
                    {
                        "type": "step",
                        "session_id": new_session_id,
                        "observation": _observation(include_images=False),
                    },
                )
                assert closed["error"] == "session_not_found"
        finally:
            policy.release.set()
            if client is not None:
                await asyncio.get_running_loop().run_in_executor(client_executor, client.close)
            client_executor.shutdown(wait=True, cancel_futures=True)
            await handler.shutdown()

    asyncio.run(scenario())


def test_config_only_cli_defaults_and_overrides(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = tmp_path / "semihumanoid.yaml"
    source = Path("configs/embodiments/semihumanoid.yaml").read_text(encoding="utf-8")
    config_path.write_text(
        source.replace(
            "~/ml_data/outputs/gr00t/semihumanoid_20260820_202230/checkpoint-6321",
            str(model_path),
        ),
        encoding="utf-8",
    )

    resolved, profile = resolve_server_config(ServerConfig(config=str(config_path), port=0))
    assert resolved.model_path == model_path.resolve()
    assert resolved.port == 0
    assert profile.task_catalog.tasks
    with pytest.raises(ValueError, match="non-loopback host rejected"):
        resolve_server_config(ServerConfig(config=str(config_path), host="0.0.0.0"))
    with pytest.raises(ValueError, match="unknown log_level"):
        resolve_server_config(ServerConfig(config=str(config_path), log_level="verbose"))

    (model_path / "config.json").write_text(
        json.dumps({"model_type": "Gr00tN1d7", "action_horizon": 40}),
        encoding="utf-8",
    )
    native = ServerConfig(
        config=str(config_path),
        rtc_mode="native",
    )
    assert resolve_server_config(native)[0].serving.rtc_mode == "native"
    with pytest.raises(ValueError, match="does not advertise"):
        resolve_server_config(
            ServerConfig(
                config=str(config_path),
                rtc_mode="trained_prefix",
            )
        )
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "Gr00tN1d7",
                "action_horizon": 40,
                "rtc_training_max_prefix_steps": 4,
            }
        ),
        encoding="utf-8",
    )
    trained = ServerConfig(
        config=str(config_path),
        rtc_mode="trained_prefix",
        rtc_max_prefix_steps=4,
    )
    assert resolve_server_config(trained)[0].serving.rtc_max_prefix_steps == 4
    assert build_service_provenance(model_path) == {
        "model_path": str(model_path.resolve()),
        "checkpoint_model_type": "Gr00tN1d7",
        "checkpoint_action_horizon": 40,
        "checkpoint_rtc_training_max_prefix_steps": 4,
    }
