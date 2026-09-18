# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic unit tests for the Industrial Next async GR00T server."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time
from typing import Any

import cv2
from gr00t.policy.industrialnext.adapter import ACTION_HORIZON, IMAGE_KEY_TO_MODEL_KEY
from gr00t.policy.industrialnext.async_server import (
    IndustrialNextAsyncServer,
    IndustrialNextServingConfig,
)
from gr00t.policy.industrialnext.profile_config import load_industrialnext_profile
from gr00t.policy.industrialnext.task_catalog import TaskCatalog, TaskCatalogEntry
import numpy as np
import pytest


TASK_UUID = "generic_pick"
TASK_TEXT = "Pick the grounded target object."
IDENTITY_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
PROFILE = load_industrialnext_profile("configs/embodiments/semihumanoid.yaml")


class _FakePolicy:
    def __init__(self, *, initially_released: bool = False, fail: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        if initially_released:
            self.release.set()
        self.fail = fail
        self.call_count = 0
        self.options_history: list[dict[str, Any]] = []
        self.echo_prefix = True

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        options = {} if options is None else options
        self.options_history.append(options)
        assert observation["language"]["annotation.human.task_description"] == [[TASK_TEXT]]
        self.call_count += 1
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("test did not release fake inference")
        if self.fail:
            raise RuntimeError("synthetic inference failure")
        action = _decoded_action()
        if self.echo_prefix and options.get("rtc_mode") in {"native", "trained_prefix"}:
            prefix = options["action_prefix"]
            prefix_steps = next(iter(prefix.values())).shape[1]
            for key, value in prefix.items():
                action[key][:, :prefix_steps] = value
        return action, {"rtc_mode": options.get("rtc_mode", "off")}


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
    **config_overrides: Any,
) -> IndustrialNextAsyncServer:
    config_overrides.setdefault("ensemble_strategy", "latest_only")
    config_overrides.setdefault("chunk_transition_frames", 0)
    # Legacy worker/lifecycle tests advance ticks without pacing. Timing laws
    # are covered below with a deterministic clock and production tolerances.
    config_overrides.setdefault("max_control_clock_drift_s", 10.0)
    config_overrides.setdefault("max_action_lateness_s", 10.0)
    return IndustrialNextAsyncServer(
        policy=policy,
        executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="groot-test"),
        config=IndustrialNextServingConfig(
            max_image_staleness_steps=max_staleness_steps,
            min_usable_action_steps=min_usable_action_steps,
            idle_session_timeout_s=idle_session_timeout_s,
            stats_log_interval_steps=0,
            **config_overrides,
        ),
        service_provenance={"model_path": "/test/model"},
        embodiment_tag="new_embodiment",
        profile=replace(PROFILE, task_catalog=_catalog()),
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
    assert IndustrialNextServingConfig(action_horizon=8, rtc_mode="off").action_horizon == 8
    with pytest.raises(ValueError, match="must leave"):
        IndustrialNextServingConfig(
            action_horizon=8,
            rtc_mode="native",
            ensemble_strategy="latest_only",
            chunk_transition_frames=0,
        )


def test_protocol_metadata_cannot_be_overridden_by_provenance() -> None:
    async def scenario() -> None:
        policy = _FakePolicy(initially_released=True)
        server = IndustrialNextAsyncServer(
            policy=policy,
            executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="groot-test"),
            config=IndustrialNextServingConfig(stats_log_interval_steps=0),
            service_provenance={"async_serving": False, "model_path": "/test/model"},
            embodiment_tag="new_embodiment",
            profile=replace(PROFILE, task_catalog=_catalog()),
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
            assert response["error"] == "session_unusable"
            assert "synthetic inference failure" in response["reason"]
            assert _step(server, session_id)["error"] == "session_unusable"
            replacement = _register(server)
            assert replacement != session_id
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
            assert server._active_session.terminal_reason.startswith("insufficient_usable_tail")
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


@pytest.mark.parametrize("rtc_mode", ["native", "trained_prefix"])
def test_rtc_refresh_uses_launch_time_contiguous_prefix(rtc_mode: str) -> None:
    async def scenario() -> None:
        policy = _FakePolicy(initially_released=True)
        server = _server(
            policy,
            rtc_mode=rtc_mode,
            rtc_initial_frozen_steps=1,
            rtc_delay_margin_steps=0,
            rtc_max_prefix_steps=4,
            rtc_native_overlap_steps=4,
            rtc_min_new_tail_steps=16,
        )
        try:
            session_id = _register(server)
            _step(server, session_id, include_images=True)
            await _wait_until(lambda: server._inference_future is None)

            served = _step(server, session_id)
            assert served["action"]["left_arm_pose_pos"][0] == 1.0
            await _wait_until(lambda: server._inference_future is None)
            assert len(policy.options_history) == 2
            rtc_options = policy.options_history[1]
            assert rtc_options["rtc_mode"] == rtc_mode
            if rtc_mode == "native":
                assert rtc_options["rtc_frozen_steps"] == 1
                assert rtc_options["rtc_overlap_steps"] == 4
                assert rtc_options["action_prefix"]["left_eef"].shape == (1, 4, 9)
            else:
                assert rtc_options["rtc_prefix_steps"] == 1
                assert rtc_options["action_prefix"]["left_eef"].shape == (1, 1, 9)
            assert server._active_session is not None
            assert server._active_session.inference_status == "ready"
            assert server._active_session.latest_prefix_position_error == 0.0
            assert server._active_session.served_history[1]["left_arm_pose_pos"][0] == 1.0
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_rtc_delay_underestimate_rejects_result_without_replacing_timeline() -> None:
    async def scenario() -> None:
        policy = _FakePolicy(initially_released=True)
        server = _server(
            policy,
            max_staleness_steps=0,
            rtc_mode="trained_prefix",
            rtc_initial_frozen_steps=1,
            rtc_delay_margin_steps=0,
            rtc_max_prefix_steps=4,
            rtc_native_overlap_steps=4,
            rtc_min_new_tail_steps=16,
        )
        try:
            session_id = _register(server)
            _step(server, session_id, include_images=True)
            await _wait_until(lambda: server._inference_future is None)
            policy.release.clear()

            _step(server, session_id, include_images=True)
            await _wait_until(lambda: policy.call_count == 2)
            _step(server, session_id)
            old_target = min(server._active_session.timeline)  # type: ignore[union-attr]
            policy.release.set()
            await _wait_until(lambda: server._inference_future is None)

            assert server._active_session is not None
            assert server._active_session.inference_status == "delay_underestimate"
            assert server._active_session.stats.rejected_delay_underestimates == 1
            assert min(server._active_session.timeline) == old_target
        finally:
            policy.release.set()
            await server.shutdown()

    asyncio.run(scenario())


def test_rtc_missing_contiguous_prefix_does_not_launch_inference() -> None:
    async def scenario() -> None:
        policy = _FakePolicy(initially_released=True)
        server = _server(
            policy,
            rtc_mode="trained_prefix",
            rtc_initial_frozen_steps=4,
            rtc_delay_margin_steps=0,
            rtc_max_prefix_steps=4,
            rtc_native_overlap_steps=4,
            rtc_min_new_tail_steps=16,
        )
        try:
            session_id = _register(server)
            _step(server, session_id, include_images=True)
            await _wait_until(lambda: server._inference_future is None)
            assert server._active_session is not None
            del server._active_session.timeline[2]

            response = _step(server, session_id)

            assert response["inference_status"] == "missing_prefix"
            assert policy.call_count == 1
            assert server._active_session.stats.missing_prefixes == 1
            assert server._active_session.timeline
        finally:
            await server.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("rtc_mode", ["native", "trained_prefix"])
def test_rtc_delay_prediction_above_runtime_bound_fails_closed(rtc_mode: str) -> None:
    async def scenario() -> None:
        policy = _FakePolicy(initially_released=True)
        server = _server(
            policy,
            rtc_mode=rtc_mode,
            rtc_initial_frozen_steps=1,
            rtc_delay_margin_steps=1,
            rtc_max_prefix_steps=4,
            rtc_native_overlap_steps=4,
            rtc_min_new_tail_steps=16,
        )
        try:
            session_id = _register(server)
            _step(server, session_id, include_images=True)
            await _wait_until(lambda: server._inference_future is None)
            assert server._active_session is not None
            server._active_session.observed_delays = [4]

            response = _step(server, session_id)

            assert response["inference_status"] == "prefix_out_of_range"
            assert response["monitoring"]["rtc_predicted_delay_steps"] == 5
            assert "required prefix 5" in response["monitoring"]["latest_inference_error"]
            assert policy.call_count == 1
            assert server._active_session.timeline
            assert server._active_session.requires_reregistration is False

            server._active_session.timeline.clear()
            exhausted = _step(server, session_id)
            assert exhausted["error"] == "session_unusable"
            assert "required prefix 5" in exhausted["reason"]
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_rtc_prefix_mismatch_is_rejected_atomically() -> None:
    async def scenario() -> None:
        policy = _FakePolicy(initially_released=True)
        server = _server(
            policy,
            rtc_mode="native",
            rtc_initial_frozen_steps=1,
            rtc_delay_margin_steps=0,
            rtc_max_prefix_steps=4,
            rtc_native_overlap_steps=4,
            rtc_min_new_tail_steps=16,
        )
        try:
            session_id = _register(server)
            _step(server, session_id, include_images=True)
            await _wait_until(lambda: server._inference_future is None)
            policy.echo_prefix = False
            _step(server, session_id)
            await _wait_until(lambda: server._inference_future is None)
            assert server._active_session is not None
            assert server._active_session.inference_status == "prefix_mismatch"
            assert server._active_session.stats.rejected_prefix_mismatches == 1
            assert server._active_session.timeline
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_action_dynamics_limit_rejects_without_clipping_output() -> None:
    async def scenario() -> None:
        policy = _FakePolicy(initially_released=True)
        server = _server(policy, max_position_step_m=0.5)
        try:
            session_id = _register(server)
            _step(server, session_id, include_images=True)
            await _wait_until(lambda: server._inference_future is None)
            assert server._active_session is not None
            assert server._active_session.timeline == {}
            assert server._active_session.terminal_reason.startswith("action_dynamics_limit")
            assert server._active_session.stats.rejected_dynamics == 1
            assert server._active_session.requires_reregistration is True
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_xarm_profile_schedules_model_row_zero_at_observation_offset_one() -> None:
    class XarmPolicy:
        def get_action(
            self, observation: dict[str, Any], options: dict[str, Any] | None = None
        ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
            del observation, options
            eef = np.zeros((1, 40, 9), dtype=np.float32)
            eef[0, :, 0] = np.arange(40)
            eef[0, :, 3:] = IDENTITY_ROT6D
            return {
                "right_eef": eef,
                "right_hand": np.zeros((1, 40, 6), dtype=np.float32),
            }, {}

    async def scenario() -> None:
        profile = load_industrialnext_profile("configs/embodiments/xarm_psyonic_mori_bracket.yaml")
        server = IndustrialNextAsyncServer(
            policy=XarmPolicy(),
            executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="xarm-offset-test"),
            config=IndustrialNextServingConfig(
                action_horizon=profile.action_horizon,
                stats_log_interval_steps=0,
                rtc_max_prefix_steps=12,
                rtc_native_overlap_steps=12,
            ),
            service_provenance={"model_path": "/test/xarm"},
            embodiment_tag="new_embodiment",
            profile=profile,
        )
        task = profile.task_catalog.tasks[0]
        registration = server.handle_request(
            {
                "type": "register_session",
                "control_hz": 50.0,
                "task_uuid": task.task_uuid,
                "task_text": task.task_text,
            }
        )
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
        observation: dict[str, Any] = {
            "right_arm_pose_pos": [0.0] * 3,
            "right_arm_pose_rot": IDENTITY_ROT6D,
            "right_hand": [0.0] * 6,
            "task_uuid": task.task_uuid,
            "task_text": task.task_text,
            "images_meta": {},
        }
        for key in profile.wire_image_to_model:
            observation[key] = encoded.tobytes()
            observation["images_meta"][key] = dict(metadata)
        try:
            first = server.handle_request(
                {
                    "type": "step",
                    "session_id": registration["session_id"],
                    "observation": observation,
                }
            )
            assert first["timestep"] == 0 and first["action"] is None
            await _wait_until(lambda: server._inference_future is None)
            second = server.handle_request(
                {
                    "type": "step",
                    "session_id": registration["session_id"],
                    "observation": {
                        key: value
                        for key, value in observation.items()
                        if key not in profile.wire_image_to_model and key != "images_meta"
                    },
                }
            )
            assert second["timestep"] == 1
            assert second["action"]["right_arm_pose_pos"][0] == 0.0
            assert second["monitoring_gripper_values"] == {}
        finally:
            await server.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("offset,target_offset,first_row", [(2, 0, 7), (0, 0, 5), (2, 1, 6)])
def test_execution_offset_maps_original_rows_with_virtual_clock(offset, target_offset, first_row):
    from gr00t.policy.industrialnext.adapter import ObservationSnapshot
    from gr00t.policy.industrialnext.async_server import InferenceRequest, InferenceResult

    async def scenario():
        server = _server(_FakePolicy(), action_offset=offset)
        now = [100.0]
        server.clock = lambda: now[0]
        server.profile = replace(server.profile, action_start_offset_steps=target_offset)
        try:
            session_id = _register(server)
            for tick in range(5):
                now[0] = 100 + tick / 50
                _step(server, session_id)
            snapshot = ObservationSnapshot(
                {}, {}, TASK_UUID, TASK_TEXT, 0, server._active_session.generation, 100.0
            )
            rows = server.profile.map_action_chunk(_decoded_action())
            result = InferenceResult(InferenceRequest(snapshot, "off"), rows, 80, 0, {})
            server._admit_inference_result(server._active_session, result)
            now[0] = 100.1
            response = _step(server, session_id)
            assert response["action"]["left_arm_pose_pos"][0] == first_row
            assert (
                response["monitoring"]["emitted_action"]["contributions"][0]["model_row"]
                == first_row
            )
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_actual_command_jump_terminates_and_late_result_cannot_revive():
    from gr00t.policy.industrialnext.adapter import ObservationSnapshot
    from gr00t.policy.industrialnext.async_server import InferenceRequest, InferenceResult

    async def scenario():
        server = _server(_FakePolicy(), max_position_step_m=0.01)
        server.clock = lambda: 100.0
        try:
            session_id = _register(server)
            _step(server, session_id)
            session = server._active_session
            base = server.profile.map_action_chunk(_decoded_action())[0]
            session.served_history[0] = base
            changed = {key: list(value) for key, value in base.items()}
            changed["left_arm_pose_pos"] = [1, 0, 0]
            snapshot = ObservationSnapshot({}, {}, TASK_UUID, TASK_TEXT, 0, session.generation, 100)
            result = InferenceResult(InferenceRequest(snapshot, "off"), (changed,) * 40, 0, 0, {})
            server._admit_inference_result(session, result)
            response = _step(server, session_id)
            assert response["error"] == "session_unusable"
            assert "emitted_action_dynamics_limit" in response["reason"]
            assert response["action"] is None
            server._admit_inference_result(session, result)
            assert not session.timeline
            assert list(session.served_history) == [0]
        finally:
            await server.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("expired", [False, True])
def test_failed_refresh_only_preserves_unexpired_predictions(expired):
    from gr00t.policy.industrialnext.adapter import ObservationSnapshot
    from gr00t.policy.industrialnext.async_server import InferenceRequest
    from gr00t.policy.industrialnext.execution import Contribution, ExecutionSlot, freeze_action

    async def scenario():
        server = _server(_FakePolicy())
        server.clock = lambda: 100.0
        try:
            _register(server)
            session = server._active_session
            row = server.profile.map_action_chunk(_decoded_action())[0]
            session.timeline[1] = ExecutionSlot(
                (
                    Contribution(
                        freeze_action(row),
                        0,
                        1,
                        99.0,
                        99.5 if expired else 100.5,
                    ),
                )
            )
            snapshot = ObservationSnapshot({}, {}, TASK_UUID, TASK_TEXT, 0, session.generation, 99)
            future = asyncio.get_running_loop().create_future()
            future.set_exception(RuntimeError("refresh failed"))
            server._inference_future = future
            server._complete_inference(InferenceRequest(snapshot, "off"), future)
            assert session.requires_reregistration is expired
            assert bool(session.timeline) is not expired
            if expired:
                assert server._terminal_response(session)["error"] == "session_unusable"
        finally:
            await server.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "strategy,frames",
    [("temporal_exponential", 0), ("latest_only", 4), ("temporal_exponential", 4)],
)
def test_ensemble_and_transition_are_independent_and_expire(strategy, frames):
    from gr00t.policy.industrialnext.adapter import ObservationSnapshot
    from gr00t.policy.industrialnext.async_server import InferenceRequest, InferenceResult

    async def scenario():
        server = _server(
            _FakePolicy(),
            ensemble_strategy=strategy,
            chunk_transition_frames=frames,
            ensemble_coeff=0.2,
            max_ensemble_chunks=2,
            max_action_lateness_s=0.04,
        )
        now = [100.0]
        server.clock = lambda: now[0]
        try:
            session_id = _register(server)
            _step(server, session_id)
            session = server._active_session
            base = server.profile.map_action_chunk(_decoded_action())[0]

            def result(source, value, received):
                row = {name: list(v) for name, v in base.items()}
                row["left_gripper"] = [value]
                snapshot = ObservationSnapshot(
                    {}, {}, TASK_UUID, TASK_TEXT, source, session.generation, received
                )
                return InferenceResult(InferenceRequest(snapshot, "off"), (row,) * 40, 0, 0, {})

            server._admit_inference_result(session, result(0, 0, 100))
            now[0] = 100.02
            session.timestep = 1
            server._admit_inference_result(session, result(1, 1, 100.02))
            slot = session.timeline[2]
            value = server._resolve_slot(slot)[0]["left_gripper"][0]
            blended = 1.0 if strategy == "latest_only" else 1 / (1 + np.exp(-0.2))
            assert value == pytest.approx(blended * (0.15625 if frames else 1))
            # A refresh halfway through the transition anchors to the already planned command.
            session.timestep = 1
            now[0] = 100.025
            server._admit_inference_result(session, result(2, 0.5, 100.025))
            assert len(session.timeline[2].contributions) <= 2
            if frames:
                assert session.timeline[2].anchor.action["left_gripper"][0] == pytest.approx(value)
                assert session.timeline[2].anchor.deadline_s <= slot.anchor.deadline_s
            server._expire_timeline(session, 105, 2)
            assert not session.timeline
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_clock_drift_and_command_gap_fail_closed_without_restarting():
    async def scenario():
        for reason in ("control_clock_drift", "command_gap_expired"):
            server = _server(
                _FakePolicy(), max_action_lateness_s=0.04, max_control_clock_drift_s=0.1
            )
            now = [100.0]
            server.clock = lambda: now[0]
            try:
                session_id = _register(server)
                _step(server, session_id)
                if reason == "command_gap_expired":
                    server._active_session.last_emitted_at_s = 100
                    now[0] = 100.081
                else:
                    now[0] = 105
                response = _step(server, session_id)
                assert response["error"] == "session_unusable"
                assert reason in response["reason"]
                assert _step(server, session_id)["error"] == "session_unusable"
            finally:
                await server.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("inference_cycles", [15, 18])
def test_elapsed_clock_serves_delayed_inference_with_missed_client_cycles(inference_cycles):
    from gr00t.policy.industrialnext.adapter import ObservationSnapshot
    from gr00t.policy.industrialnext.async_server import InferenceRequest, InferenceResult

    async def scenario():
        server = _server(
            _FakePolicy(),
            control_clock_mode="elapsed_time",
            action_offset=2,
            max_control_clock_drift_s=0.1,
            max_action_lateness_s=0.04,
            max_command_gap_s=0.2,
        )
        now = [100.0]
        server.clock = lambda: now[0]
        try:
            session_id = _register(server)
            _step(server, session_id)
            session = server._active_session
            rows = server.profile.map_action_chunk(_decoded_action())
            snapshot = ObservationSnapshot(
                {}, {}, TASK_UUID, TASK_TEXT, 0, session.generation, now[0]
            )
            actions = 0
            # A 40 Hz request stream used to exceed 100 ms cumulative drift.
            # Complete an inference every 375/450 ms, using its original snapshot.
            for step in range(1, 121):
                now[0] = 100 + step * 0.025
                if step % inference_cycles == 0:
                    result = InferenceResult(
                        InferenceRequest(snapshot, "off"), rows, inference_cycles * 25, 0, {}
                    )
                    server._admit_inference_result(session, result)
                response = _step(server, session_id)
                assert "error" not in response
                # Either neighbor is valid at a floating-point half-tick boundary.
                assert abs(response["timestep"] - step * 1.25) <= 0.5 + 1e-9
                assert response["monitoring_timestep"] == step
                if response["action"] is not None:
                    actions += 1
                    source = response["monitoring"]["emitted_action"]["contributions"][0]
                    assert source["model_row"] == response["timestep"] - source["source_tick"] + 2
                    assert now[0] <= source["received_at_s"] + (source["model_row"] - 2) / 50 + 0.04
                if step % inference_cycles == 0:
                    snapshot = ObservationSnapshot(
                        {}, {}, TASK_UUID, TASK_TEXT, session.timestep, session.generation, now[0]
                    )
            assert actions > 50
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_command_gap_budget_does_not_extend_action_deadlines_and_logs_once(caplog):
    from gr00t.policy.industrialnext.execution import Contribution, ExecutionSlot, freeze_action

    async def scenario():
        server = _server(
            _FakePolicy(),
            control_clock_mode="elapsed_time",
            max_action_lateness_s=0.04,
            max_command_gap_s=0.2,
            max_control_clock_drift_s=0.1,
        )
        now = [100.0]
        server.clock = lambda: now[0]
        try:
            session_id = _register(server)
            _step(server, session_id)
            session = server._active_session
            session.last_emitted_at_s = 100.0
            row = freeze_action(server.profile.map_action_chunk(_decoded_action())[0])
            # A row is expired even though the independently configured gap permits recovery.
            session.timeline[6] = ExecutionSlot((Contribution(row, 0, 6, 100, 100.10),))
            now[0] = 100.12
            response = _step(server, session_id)
            assert "error" not in response
            assert response["action"] is None
            assert response["monitoring"]["command_gap_s"] == pytest.approx(0.12)
            session.timeline[7] = ExecutionSlot((Contribution(row, 6, 1, 100.12, 100.18),))
            now[0] = 100.14
            assert _step(server, session_id)["action"] is not None
            now[0] = 100.35
            response = _step(server, session_id)
            assert response["error"] == "session_unusable"
            assert response["reason"] == "command_gap_expired"
            assert _step(server, session_id)["reason"] == "command_gap_expired"
            messages = [
                r.message for r in caplog.records if "GR00T session terminated" in r.message
            ]
            assert len(messages) == 1
            assert "reason=command_gap_expired" in messages[0]
            assert "command_gap_limit_s=0.200" in messages[0]
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_elapsed_clock_still_rejects_requests_running_too_fast():
    async def scenario():
        server = _server(
            _FakePolicy(), control_clock_mode="elapsed_time", max_control_clock_drift_s=0.1
        )
        server.clock = lambda: 100.0
        try:
            session_id = _register(server)
            for _ in range(7):
                response = _step(server, session_id)
            assert response["error"] == "session_unusable"
            assert "control_clock_drift" in response["reason"]
        finally:
            await server.shutdown()

    asyncio.run(scenario())


def test_rotation_mean_and_transition_use_so3_not_six_coordinate_averaging():
    from gr00t.policy.industrialnext.execution import (
        blend_actions,
        interpolate_actions,
        rotation_matrix,
    )
    from scipy.spatial.transform import Rotation

    def row(degrees):
        matrix = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
        return {"rot": tuple(matrix[:, :2].T.reshape(-1)), "hand": (0.2,) * 20}

    left, right = row(170), row(-170)
    mean = blend_actions([left, right], np.array([0.5, 0.5]), ("rot",))
    transition = interpolate_actions(left, right, 0.5, ("rot",))
    expected = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    np.testing.assert_allclose(rotation_matrix(mean["rot"]), expected, atol=1e-7)
    np.testing.assert_allclose(rotation_matrix(transition["rot"]), expected, atol=1e-7)
    assert mean["hand"] == [0.2] * 20
    with pytest.raises(ValueError, match="ill-conditioned"):
        blend_actions([row(0), row(180)], np.array([0.5, 0.5]), ("rot",))
