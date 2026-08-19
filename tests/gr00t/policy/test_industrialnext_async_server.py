# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic unit tests for the Industrial Next async GR00T server."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import Any

import cv2
from gr00t.policy.industrialnext.adapter import ACTION_HORIZON, IMAGE_KEY_TO_MODEL_KEY
from gr00t.policy.industrialnext.async_server import (
    IndustrialNextAsyncServer,
    IndustrialNextServingConfig,
)
from gr00t.policy.industrialnext.task_catalog import TaskCatalog, TaskCatalogEntry
import numpy as np
import pytest


TASK_UUID = "generic_pick"
TASK_TEXT = "Pick the grounded target object."
IDENTITY_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class _FakePolicy:
    def __init__(self, *, initially_released: bool = False, fail: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        if initially_released:
            self.release.set()
        self.fail = fail
        self.call_count = 0

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        assert observation["language"]["annotation.human.task_description"] == [[TASK_TEXT]]
        self.call_count += 1
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("test did not release fake inference")
        if self.fail:
            raise RuntimeError("synthetic inference failure")
        return _decoded_action(), {}


def _catalog() -> TaskCatalog:
    return TaskCatalog(
        schema_version=1,
        task_family="generic_pick_and_place",
        catalog_version="test",
        tasks=(TaskCatalogEntry(TASK_UUID, TASK_TEXT, "Pick"),),
    )


def _server(
    policy: _FakePolicy,
    *,
    max_staleness_steps: int = 5,
    min_usable_action_steps: int = 1,
    idle_session_timeout_s: float = 300.0,
) -> IndustrialNextAsyncServer:
    return IndustrialNextAsyncServer(
        policy=policy,
        executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="groot-test"),
        task_catalog=_catalog(),
        config=IndustrialNextServingConfig(
            max_image_staleness_steps=max_staleness_steps,
            min_usable_action_steps=min_usable_action_steps,
            idle_session_timeout_s=idle_session_timeout_s,
            stats_log_interval_steps=0,
        ),
        service_provenance={"model_path": "/test/model"},
        embodiment_tag="new_embodiment",
    )


def _register(server: IndustrialNextAsyncServer) -> str:
    response = server.handle_request(
        {
            "type": "register_session",
            "control_hz": 50.0,
            "task_uuid": TASK_UUID,
            "task_text": TASK_TEXT,
        }
    )
    assert "error" not in response
    return response["session_id"]


def _step(
    server: IndustrialNextAsyncServer,
    session_id: str,
    *,
    include_images: bool = False,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return server.handle_request(
        {
            "type": "step",
            "session_id": session_id,
            "observation": (
                _wire_observation(include_images=include_images)
                if observation is None
                else observation
            ),
        }
    )


def _wire_observation(*, include_images: bool) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "left_arm_pose_pos": [0.1, 0.2, 0.3],
        "left_arm_pose_rot": IDENTITY_ROT6D,
        "left_gripper": [0.25],
        "left_ft": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
        "right_arm_pose_pos": [-0.1, -0.2, -0.3],
        "right_arm_pose_rot": IDENTITY_ROT6D,
        "right_gripper": [0.75],
        "right_ft": [-1.0, -2.0, -3.0, -0.1, -0.2, -0.3],
        "task_uuid": TASK_UUID,
        "task_text": TASK_TEXT,
    }
    if include_images:
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
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
        for image_name in IMAGE_KEY_TO_MODEL_KEY:
            observation[image_name] = encoded.tobytes()
            observation["images_meta"][image_name] = dict(metadata)
    return observation


def _decoded_action() -> dict[str, np.ndarray]:
    left_eef = np.zeros((1, ACTION_HORIZON, 9), dtype=np.float32)
    right_eef = np.zeros((1, ACTION_HORIZON, 9), dtype=np.float32)
    left_eef[0, :, 0] = np.arange(ACTION_HORIZON)
    right_eef[0, :, 0] = -np.arange(ACTION_HORIZON)
    left_eef[0, :, 3:] = IDENTITY_ROT6D
    right_eef[0, :, 3:] = IDENTITY_ROT6D
    return {
        "left_eef": left_eef,
        "left_gripper": np.full((1, ACTION_HORIZON, 1), 0.2, dtype=np.float32),
        "right_eef": right_eef,
        "right_gripper": np.full((1, ACTION_HORIZON, 1), 0.8, dtype=np.float32),
    }


async def _wait_until(predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not met")
        await asyncio.sleep(0.005)


def test_metadata_and_configuration_contract() -> None:
    async def scenario() -> None:
        policy = _FakePolicy(initially_released=True)
        server = _server(policy, min_usable_action_steps=7)
        try:
            metadata = server.get_metadata()
            service = metadata["service_metadata"]
            assert service["async_protocol_version"] == 2
            assert service["async_capabilities"] == [
                "error_envelope_v2",
                "monitoring_in_step",
                "server_owned_gripper_snap",
            ]
            assert service["effective_gripper_snap_config"]["enabled"] is False
            assert service["min_usable_action_steps"] == 7
            assert set(metadata["request_format"]) == {
                "register_session",
                "step",
                "close_session",
            }
        finally:
            await server.shutdown()

    asyncio.run(scenario())
    with pytest.raises(ValueError, match="exactly 50"):
        IndustrialNextServingConfig(control_hz=49.0)
    with pytest.raises(ValueError, match=r"\[1, 40\]"):
        IndustrialNextServingConfig(min_usable_action_steps=0)


def test_protocol_metadata_cannot_be_overridden_by_provenance() -> None:
    async def scenario() -> None:
        policy = _FakePolicy(initially_released=True)
        server = IndustrialNextAsyncServer(
            policy=policy,
            executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="groot-test"),
            task_catalog=_catalog(),
            config=IndustrialNextServingConfig(stats_log_interval_steps=0),
            service_provenance={"async_serving": False, "model_path": "/test/model"},
            embodiment_tag="new_embodiment",
        )
        try:
            service = server.get_metadata()["service_metadata"]
            assert service["async_serving"] is True
            assert service["model_path"] == "/test/model"
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_step_is_nonblocking_and_timeline_is_age_corrected() -> None:
    async def scenario() -> None:
        policy = _FakePolicy()
        server = _server(policy)
        try:
            session_id = _register(server)
            started_at = time.perf_counter()
            startup = _step(server, session_id, include_images=True)
            elapsed = time.perf_counter() - started_at
            assert elapsed < 0.1
            assert startup["action"] is None
            assert startup["timestep"] == 0
            assert startup["monitoring"]["progress"] == 0.0
            assert startup["monitoring_timestep"] == 0
            assert await asyncio.to_thread(policy.started.wait, 1.0)

            policy.release.set()
            await _wait_until(lambda: server._inference_future is None)
            response = _step(server, session_id)
            assert response["timestep"] == 1
            assert response["action"]["left_arm_pose_pos"][0] == 1.0
            assert response["monitoring"]["progress"] == 0.0
            assert response["monitoring_timestep"] == 1
            assert response["total_actions_served"] == 1
        finally:
            policy.release.set()
            await server.shutdown()

    asyncio.run(scenario())


def test_invalid_step_is_transactional_and_errors_are_schema_complete() -> None:
    async def scenario() -> None:
        policy = _FakePolicy(initially_released=True)
        server = _server(policy)
        try:
            session_id = _register(server)
            invalid = _wire_observation(include_images=True)
            del invalid["left_ft"]
            response = _step(server, session_id, observation=invalid)
            assert response["error"].startswith("left_ft")
            assert response["action"] is None
            assert response["timestep"] == -1
            assert server._active_session is not None
            assert server._active_session.timestep == -1
            assert server._active_session.image_cache == {}

            valid = _step(server, session_id, include_images=True)
            assert "error" not in valid
            assert valid["timestep"] == 0

            unknown = _step(server, "not-a-session")
            assert unknown["error"] == "session_not_found"
            expected_fields = server.get_metadata()["response_format"]["step"]
            assert set(expected_fields).issubset(unknown)
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_registration_replaces_generation_and_discards_late_result() -> None:
    async def scenario() -> None:
        policy = _FakePolicy()
        server = _server(policy)
        try:
            old_session_id = _register(server)
            _step(server, old_session_id, include_images=True)
            assert await asyncio.to_thread(policy.started.wait, 1.0)

            new_session_id = _register(server)
            assert new_session_id != old_session_id
            assert _step(server, old_session_id)["error"] == "session_not_found"
            policy.release.set()
            await _wait_until(lambda: server._inference_future is None)
            assert server._active_session is not None
            assert server._active_session.session_id == new_session_id
            assert server._active_session.timeline == {}
            assert server._active_session.total_inferences == 0

            assert server.handle_request(
                {"type": "close_session", "session_id": new_session_id}
            ) == {"ok": True}
            assert server.handle_request(
                {"type": "close_session", "session_id": new_session_id}
            ) == {"ok": True}
        finally:
            policy.release.set()
            await server.shutdown()

    asyncio.run(scenario())


def test_stale_pending_snapshot_is_discarded_after_long_inference() -> None:
    async def scenario() -> None:
        policy = _FakePolicy()
        server = _server(policy, max_staleness_steps=2)
        try:
            session_id = _register(server)
            _step(server, session_id, include_images=True)
            assert await asyncio.to_thread(policy.started.wait, 1.0)
            _step(server, session_id)
            _step(server, session_id)
            stale = _step(server, session_id)
            assert stale["monitoring"]["null_reason"] == "stale_images"

            policy.release.set()
            await _wait_until(lambda: server._inference_future is None)
            assert policy.call_count == 1
            assert server._active_session is not None
            assert server._active_session.stats.stale_pending_snapshots == 1
            assert sorted(server._active_session.timeline) == list(range(4, ACTION_HORIZON))
        finally:
            policy.release.set()
            await server.shutdown()

    asyncio.run(scenario())


def test_inference_failure_and_minimum_tail_are_fail_closed() -> None:
    async def failure_scenario() -> None:
        policy = _FakePolicy(initially_released=True, fail=True)
        server = _server(policy)
        try:
            session_id = _register(server)
            _step(server, session_id, include_images=True)
            await _wait_until(lambda: server._inference_future is None)
            response = _step(server, session_id)
            assert response["action"] is None
            # The failed call is reported while this accepted step immediately retries.
            assert response["inference_status"] == "running"
            assert response["monitoring"]["null_reason"] == "inference_error"
            assert "synthetic inference failure" in response["monitoring"]["latest_inference_error"]
        finally:
            await server.shutdown()

    async def tail_scenario() -> None:
        policy = _FakePolicy()
        server = _server(policy, min_usable_action_steps=40)
        try:
            session_id = _register(server)
            _step(server, session_id, include_images=True)
            assert await asyncio.to_thread(policy.started.wait, 1.0)
            policy.release.set()
            await _wait_until(lambda: server._inference_future is None)
            assert server._active_session is not None
            assert server._active_session.timeline == {}
            assert server._active_session.inference_status == "insufficient_tail"
            assert server._active_session.stats.rejected_tails == 1
        finally:
            policy.release.set()
            await server.shutdown()

    asyncio.run(failure_scenario())
    asyncio.run(tail_scenario())


def test_idle_session_expiry_invalidates_session() -> None:
    async def scenario() -> None:
        policy = _FakePolicy(initially_released=True)
        server = _server(policy, idle_session_timeout_s=0.03)
        try:
            session_id = _register(server)
            await asyncio.sleep(0.06)
            assert _step(server, session_id)["error"] == "session_not_found"
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_cancelled_shutdown_still_closes_the_inference_executor() -> None:
    async def scenario() -> None:
        policy = _FakePolicy()
        server = _server(policy)
        session_id = _register(server)
        _step(server, session_id, include_images=True)
        assert await asyncio.to_thread(policy.started.wait, 1.0)

        shutdown_task = asyncio.create_task(server.shutdown())
        await asyncio.sleep(0)
        shutdown_task.cancel()
        policy.release.set()
        with pytest.raises(asyncio.CancelledError):
            await shutdown_task
        with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
            server.executor.submit(lambda: None)

    asyncio.run(scenario())
