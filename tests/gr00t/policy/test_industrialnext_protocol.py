# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loopback conformance tests using the real direct WebSocket transport."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
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
import tyro
import yaml


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
    document = yaml.safe_load(source)
    document["serving"]["model_path"] = str(model_path)
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

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
        ensemble_strategy="latest_only",
        chunk_transition_frames=0,
    )
    assert resolve_server_config(native)[0].serving.rtc_mode == "native"
    with pytest.raises(ValueError, match="does not advertise"):
        resolve_server_config(
            ServerConfig(
                config=str(config_path),
                rtc_mode="trained_prefix",
                ensemble_strategy="latest_only",
                chunk_transition_frames=0,
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
        ensemble_strategy="latest_only",
        chunk_transition_frames=0,
        rtc_max_prefix_steps=4,
    )
    assert resolve_server_config(trained)[0].serving.rtc_max_prefix_steps == 4
    assert build_service_provenance(model_path) == {
        "model_path": str(model_path.resolve()),
        "checkpoint_model_type": "Gr00tN1d7",
        "checkpoint_action_horizon": 40,
        "checkpoint_rtc_training_max_prefix_steps": 4,
    }


@pytest.mark.parametrize("variant", ["100", "full"])
def test_inx_launcher_resolves_timing_recipe_and_overrides(tmp_path, variant):
    # Capture the actual shell arguments without loading a model or binding a socket.
    shim = tmp_path / "uv"
    shim.write_text('#!/bin/bash\nprintf "%s\\0" "$@"\n')
    shim.chmod(0o755)
    for overrides, gap in [([], 0.2), (["--max-command-gap-s", "0.3"], 0.3)]:
        argv = (
            subprocess.check_output(
                ["bash", "inx_serve.sh", variant, "--model-path", str(tmp_path), *overrides],
                env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
            )
            .decode()
            .split("\0")[:-1]
        )
        args = argv[argv.index("gr00t/eval/run_gr00t_industrialnext_server.py") + 1 :]
        resolved, profile = resolve_server_config(tyro.cli(ServerConfig, args=args))
        assert profile.name == f"taro_exp_{variant}"
        assert resolved.serving.action_offset == 2
        assert resolved.serving.control_clock_mode == "elapsed_time"
        assert resolved.serving.max_action_lateness_s == 0.04
        assert resolved.serving.max_control_clock_drift_s == 0.1
        assert resolved.serving.max_command_gap_s == gap
        with ThreadPoolExecutor(max_workers=1) as executor:
            server = IndustrialNextAsyncServer(
                policy=_BlockingPolicy(),
                executor=executor,
                config=resolved.serving,
                service_provenance={},
                embodiment_tag=resolved.embodiment_tag,
                profile=profile,
            )
            metadata = server._service_metadata()["execution"]
            assert metadata["tick_clock"] == "elapsed_time"
            assert metadata["max_command_gap_s"] == gap


def test_serving_recipe_precedence_and_absolute_metadata(tmp_path):
    document = yaml.safe_load(Path("configs/embodiments/taro_exp_100.yaml").read_text())
    document["serving"].update(
        model_path=str(tmp_path),
        action_offset=1,
        ensemble_coeff=0.2,
        max_ensemble_chunks=2,
        chunk_transition_frames=3,
        control_clock_mode="elapsed_time",
        max_action_lateness_s=0.05,
        max_control_clock_drift_s=0.15,
        max_command_gap_s=0.2,
    )
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(document))
    resolved, profile = resolve_server_config(
        ServerConfig(config=str(path), action_offset=2, ensemble_coeff=0.3, max_command_gap_s=0.3)
    )
    assert resolved.serving.action_offset == 2
    assert profile.action_start_offset_steps == 0
    assert resolved.serving.ensemble_coeff == 0.3
    assert resolved.serving.max_ensemble_chunks == 2
    assert resolved.serving.chunk_transition_frames == 3
    assert resolved.serving.control_clock_mode == "elapsed_time"
    assert resolved.serving.max_action_lateness_s == 0.05
    assert resolved.serving.max_control_clock_drift_s == 0.15
    assert resolved.serving.max_command_gap_s == 0.3
    metadata = profile.service_metadata()
    assert metadata["state_dim"] == 38
    assert metadata["action_dim"] == 29
    assert metadata["internal_action_fields"] == metadata["action_fields"]
    assert {field["representation"] for field in metadata["action_fields"]} == {"absolute"}
    assert metadata["action_fields"][-1]["length"] == 20
    with pytest.raises(ValueError, match="RTC requires"):
        IndustrialNextServingConfig(rtc_mode="native")


@pytest.mark.parametrize(
    "settings",
    [
        {"action_offset": True},
        {"action_offset": -1},
        {"action_offset": 40},
        {"ensemble_coeff": float("nan")},
        {"ensemble_coeff": True},
        {"chunk_transition_frames": True},
        {"max_ensemble_chunks": 0},
        {"max_action_lateness_s": 0},
        {"max_control_clock_drift_s": float("inf")},
        {"max_command_gap_s": 0},
        {"max_command_gap_s": True},
        {"max_command_gap_s": float("nan")},
        {"control_clock_mode": "unknown"},
    ],
)
def test_invalid_serving_recipe_fails_before_model_loading(settings):
    with pytest.raises(ValueError):
        IndustrialNextServingConfig(**settings)
