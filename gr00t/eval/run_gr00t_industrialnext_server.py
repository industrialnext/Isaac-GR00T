# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a strict GR00T policy through the Industrial Next async WebSocket contract."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass
from functools import partial
import hashlib
import ipaddress
import json
import logging
from pathlib import Path
import signal
import subprocess
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.industrialnext import (
    IndustrialNextAsyncServer,
    IndustrialNextServingConfig,
    assert_semihumanoid_policy_contract,
    build_synthetic_model_observation,
    load_task_catalog,
    map_action_chunk,
)
from industrialnext_rpc.direct.server import DirectServer
import tyro


logger = logging.getLogger(__name__)
DEFAULT_PORT = 10012
DEFAULT_MAX_MESSAGE_SIZE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the Industrial Next-compatible GR00T server."""

    model_path: str
    """Local GR00T checkpoint directory; required explicitly."""

    task_catalog_path: str
    """Industrial Next task-catalog YAML; required explicitly."""

    embodiment_tag: str = "new_embodiment"
    """Saved checkpoint embodiment tag."""

    device: str = "cuda"
    """Torch device used for model inference."""

    host: str = "127.0.0.1"
    """WebSocket bind host. Loopback is required by default because requests are pickled."""

    port: int = DEFAULT_PORT
    """WebSocket bind port."""

    control_hz: float = 50.0
    """Robot policy control frequency; the current ROS client contract requires 50 Hz."""

    max_image_staleness_steps: int = 5
    """Maximum cached RGB age in 50 Hz control steps."""

    min_usable_action_steps: int = 1
    """Minimum unexpired rows required to admit a 40-step model result."""

    idle_session_timeout_s: float = 300.0
    """Seconds before an abandoned session is invalidated."""

    max_message_size_bytes: int = DEFAULT_MAX_MESSAGE_SIZE_BYTES
    """Maximum accepted WebSocket message size."""

    stats_log_interval_steps: int = 250
    """Accepted-step interval for aggregate diagnostics; zero disables periodic logging."""

    rtc_mode: str = "off"
    """Chunk refresh mode: off, native, or trained_prefix."""

    rtc_initial_frozen_steps: int = 1
    """Bootstrap committed-prefix estimate before latency history is available."""

    rtc_delay_window_size: int = 20
    """Number of observed committed-prefix lengths in the rolling maximum."""

    rtc_delay_margin_steps: int = 1
    """Safety margin added to the rolling maximum committed-prefix length."""

    rtc_max_prefix_steps: int = 24
    """Hard runtime prefix bound; trained_prefix may not exceed checkpoint training support."""

    rtc_native_overlap_steps: int = 24
    """Maximum prior-action overlap supplied to the native RTC sampler."""

    rtc_min_new_tail_steps: int = 16
    """Minimum independently generated postfix rows required per accepted chunk."""

    rtc_ramp_rate: float = 6.0
    """Exponential velocity-ramp rate used only by native RTC."""

    rtc_position_tolerance: float = 1e-4
    """Maximum absolute-position hard-prefix round-trip error in meters."""

    rtc_orientation_tolerance_rad: float = 1e-3
    """Maximum SO(3) hard-prefix round-trip error in radians."""

    rtc_gripper_tolerance: float = 1e-4
    """Maximum hard-prefix gripper round-trip error."""

    max_position_step_m: float | None = None
    """Optional fail-closed maximum translation change between adjacent output rows."""

    max_orientation_step_rad: float | None = None
    """Optional fail-closed maximum SO(3) change between adjacent output rows."""

    max_gripper_step: float | None = None
    """Optional fail-closed maximum gripper change between adjacent output rows."""

    max_position_second_difference_m: float | None = None
    """Optional fail-closed position second-finite-difference bound."""

    max_gripper_second_difference: float | None = None
    """Optional fail-closed gripper second-finite-difference bound."""

    allow_unsafe_non_loopback: bool = False
    """Explicitly permit a non-loopback bind despite pickle remote-code-execution risk."""

    log_level: str = "INFO"
    """Python logging level."""


def validate_server_config(config: ServerConfig) -> tuple[Path, Path]:
    """Validate all cheap safety/configuration gates before allocating a model."""
    model_path = Path(config.model_path).expanduser().resolve()
    catalog_path = Path(config.task_catalog_path).expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError(f"model_path must be an existing directory: {model_path}")
    if not catalog_path.is_file():
        raise ValueError(f"task_catalog_path must be an existing file: {catalog_path}")
    if not isinstance(getattr(logging, config.log_level.upper(), None), int):
        raise ValueError(f"unknown log_level {config.log_level!r}")
    if (
        not isinstance(config.port, int)
        or isinstance(config.port, bool)
        or not 0 <= config.port < 65536
    ):
        raise ValueError("port must be an integer within [0, 65535]")
    if (
        not isinstance(config.max_message_size_bytes, int)
        or isinstance(config.max_message_size_bytes, bool)
        or config.max_message_size_bytes <= 0
    ):
        raise ValueError("max_message_size_bytes must be a positive integer")
    serving_config = IndustrialNextServingConfig(
        control_hz=config.control_hz,
        max_image_staleness_steps=config.max_image_staleness_steps,
        min_usable_action_steps=config.min_usable_action_steps,
        idle_session_timeout_s=config.idle_session_timeout_s,
        stats_log_interval_steps=config.stats_log_interval_steps,
        rtc_mode=config.rtc_mode,
        rtc_initial_frozen_steps=config.rtc_initial_frozen_steps,
        rtc_delay_window_size=config.rtc_delay_window_size,
        rtc_delay_margin_steps=config.rtc_delay_margin_steps,
        rtc_max_prefix_steps=config.rtc_max_prefix_steps,
        rtc_native_overlap_steps=config.rtc_native_overlap_steps,
        rtc_min_new_tail_steps=config.rtc_min_new_tail_steps,
        rtc_ramp_rate=config.rtc_ramp_rate,
        rtc_position_tolerance=config.rtc_position_tolerance,
        rtc_orientation_tolerance_rad=config.rtc_orientation_tolerance_rad,
        rtc_gripper_tolerance=config.rtc_gripper_tolerance,
        max_position_step_m=config.max_position_step_m,
        max_orientation_step_rad=config.max_orientation_step_rad,
        max_gripper_step=config.max_gripper_step,
        max_position_second_difference_m=config.max_position_second_difference_m,
        max_gripper_second_difference=config.max_gripper_second_difference,
    )
    validate_checkpoint_rtc_compatibility(model_path, serving_config)
    if not _is_loopback_host(config.host):
        if not config.allow_unsafe_non_loopback:
            raise ValueError(
                "non-loopback host rejected: the direct protocol unpickles input; "
                "pass --allow-unsafe-non-loopback only on a trusted network"
            )
        logger.warning(
            "UNSAFE non-loopback pickle server explicitly enabled on %s:%d",
            config.host,
            config.port,
        )
    return model_path, catalog_path


async def serve_forever(config: ServerConfig) -> None:
    """Load, validate, warm up, then serve until SIGINT or SIGTERM."""
    model_path, catalog_path = validate_server_config(config)
    catalog = load_task_catalog(catalog_path)
    serving_config = IndustrialNextServingConfig(
        control_hz=config.control_hz,
        max_image_staleness_steps=config.max_image_staleness_steps,
        min_usable_action_steps=config.min_usable_action_steps,
        idle_session_timeout_s=config.idle_session_timeout_s,
        stats_log_interval_steps=config.stats_log_interval_steps,
        rtc_mode=config.rtc_mode,
        rtc_initial_frozen_steps=config.rtc_initial_frozen_steps,
        rtc_delay_window_size=config.rtc_delay_window_size,
        rtc_delay_margin_steps=config.rtc_delay_margin_steps,
        rtc_max_prefix_steps=config.rtc_max_prefix_steps,
        rtc_native_overlap_steps=config.rtc_native_overlap_steps,
        rtc_min_new_tail_steps=config.rtc_min_new_tail_steps,
        rtc_ramp_rate=config.rtc_ramp_rate,
        rtc_position_tolerance=config.rtc_position_tolerance,
        rtc_orientation_tolerance_rad=config.rtc_orientation_tolerance_rad,
        rtc_gripper_tolerance=config.rtc_gripper_tolerance,
        max_position_step_m=config.max_position_step_m,
        max_orientation_step_rad=config.max_orientation_step_rad,
        max_gripper_step=config.max_gripper_step,
        max_position_second_difference_m=config.max_position_second_difference_m,
        max_gripper_second_difference=config.max_gripper_second_difference,
    )
    embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="groot-inference")
    handler: IndustrialNextAsyncServer | None = None
    loop = asyncio.get_running_loop()
    try:
        logger.info("Loading strict GR00T policy from %s on %s", model_path, config.device)
        policy = await loop.run_in_executor(
            executor,
            partial(
                Gr00tPolicy,
                model_path=str(model_path),
                embodiment_tag=embodiment_tag,
                device=config.device,
                strict=True,
            ),
        )
        assert_semihumanoid_policy_contract(policy)
        first_task = catalog.tasks[0]
        logger.info("Running one synthetic finite inference before binding the server")
        await loop.run_in_executor(executor, _warmup_policy, policy, first_task.task_text)
        provenance = build_service_provenance(model_path)
        handler = IndustrialNextAsyncServer(
            policy=policy,
            executor=executor,
            task_catalog=catalog,
            config=serving_config,
            service_provenance=provenance,
            embodiment_tag=embodiment_tag.value,
            owns_executor=True,
        )

        stop_event = asyncio.Event()
        installed_signals = _install_signal_handlers(loop, stop_event)
        try:
            async with DirectServer(
                config.host,
                config.port,
                handler,
                max_size_bytes=config.max_message_size_bytes,
            ):
                logger.info(
                    "Industrial Next GR00T server ready at ws://%s:%d | model=%s | "
                    "control_hz=%.1f | min_usable_action_steps=%d | rtc_mode=%s",
                    config.host,
                    config.port,
                    model_path,
                    config.control_hz,
                    config.min_usable_action_steps,
                    config.rtc_mode,
                )
                await stop_event.wait()
        finally:
            for sig in installed_signals:
                loop.remove_signal_handler(sig)
    finally:
        if handler is not None:
            await handler.shutdown()
        else:
            executor.shutdown(wait=True, cancel_futures=True)


def build_service_provenance(model_path: Path) -> dict[str, Any]:
    """Collect lightweight, reviewable model and source provenance."""
    index_path = model_path / "model.safetensors.index.json"
    processor_path = _processor_file(model_path, "processor_config.json")
    checkpoint_config_path = model_path / "config.json"
    if not index_path.is_file():
        raise ValueError(f"missing model index: {index_path}")
    if not processor_path.is_file():
        raise ValueError(f"missing processor config: {processor_path}")
    if not checkpoint_config_path.is_file():
        raise ValueError(f"missing checkpoint config: {checkpoint_config_path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"invalid model index {index_path}: {exc}") from exc
    shards = []
    for shard_name in shard_names:
        shard_path = model_path / shard_name
        if not shard_path.is_file() or shard_path.stat().st_size <= 0:
            raise ValueError(f"model index references a missing or empty shard: {shard_path}")
        shards.append(
            {
                "name": shard_name,
                "size_bytes": shard_path.stat().st_size,
                "sha256": _sha256(shard_path),
            }
        )
    try:
        checkpoint_config = json.loads(checkpoint_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint config {checkpoint_config_path}: {exc}") from exc
    repository_root = Path(__file__).resolve().parents[2]
    return {
        "model_path": str(model_path.resolve()),
        "model_index_sha256": _sha256(index_path),
        "processor_config_sha256": _sha256(processor_path),
        "checkpoint_config_sha256": _sha256(checkpoint_config_path),
        "checkpoint_model_type": checkpoint_config.get("model_type"),
        "checkpoint_action_horizon": checkpoint_config.get("action_horizon"),
        "checkpoint_rtc_training_max_prefix_steps": checkpoint_config.get(
            "rtc_training_max_prefix_steps", 0
        ),
        "model_shards": shards,
        "groot_revision": _git_revision(repository_root),
        "industrialnext_rpc_revision": _git_revision(
            repository_root / "packages" / "industrialnext_rpc"
        ),
    }


def validate_checkpoint_rtc_compatibility(
    model_path: Path, config: IndustrialNextServingConfig
) -> None:
    """Fail closed before model allocation for unsupported checkpoint/mode pairs."""
    if config.rtc_mode == "off":
        return
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"rtc_mode={config.rtc_mode!r} requires checkpoint config.json")
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint config {config_path}: {exc}") from exc
    if model_config.get("model_type") != "Gr00tN1d7":
        raise ValueError("RTC requires a Gr00tN1d7 checkpoint")
    if model_config.get("action_horizon") != 40:
        raise ValueError("RTC serving requires a saved 40-step action horizon")
    if config.rtc_mode == "trained_prefix":
        trained_max = model_config.get("rtc_training_max_prefix_steps", 0)
        if isinstance(trained_max, bool) or not isinstance(trained_max, int) or trained_max <= 0:
            raise ValueError("checkpoint does not advertise trained-prefix support")
        if config.rtc_max_prefix_steps > trained_max:
            raise ValueError(
                f"runtime prefix maximum {config.rtc_max_prefix_steps} exceeds checkpoint "
                f"trained maximum {trained_max}"
            )


def _warmup_policy(policy: Gr00tPolicy, task_text: str) -> None:
    observation = build_synthetic_model_observation(task_text)
    action, _ = policy.get_action(observation)
    map_action_chunk(action)


def _processor_file(model_path: Path, filename: str) -> Path:
    nested = model_path / "processor" / filename
    return nested if nested.is_file() else model_path / filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event
) -> tuple[signal.Signals, ...]:
    installed = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop_event.set)
            installed.append(sig)
    return tuple(installed)


def main(config: ServerConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(serve_forever(config))


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
