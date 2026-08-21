# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a config-driven strict GR00T policy through Industrial Next DirectServer."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass
from functools import partial
import ipaddress
import json
import logging
import os
from pathlib import Path
import signal
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.industrialnext import (
    ConfigDrivenIndustrialNextProfile,
    IndustrialNextAsyncServer,
    IndustrialNextServingConfig,
    load_industrialnext_profile,
)
from industrialnext_rpc.direct.server import DirectServer
import tyro


logger = logging.getLogger(__name__)
DEFAULT_MAX_MESSAGE_SIZE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class ServerConfig:
    """Operator settings; only the embodiment config is required."""

    config: str
    model_path: str | None = None
    embodiment_tag: str | None = None
    device: str | None = None
    host: str | None = None
    port: int | None = None
    control_hz: float | None = None
    rtc_mode: str | None = None
    max_image_staleness_steps: int = 5
    min_usable_action_steps: int = 1
    idle_session_timeout_s: float = 300.0
    max_message_size_bytes: int = DEFAULT_MAX_MESSAGE_SIZE_BYTES
    stats_log_interval_steps: int = 250
    rtc_initial_frozen_steps: int = 1
    rtc_delay_window_size: int = 20
    rtc_delay_margin_steps: int = 1
    rtc_max_prefix_steps: int = 12
    rtc_native_overlap_steps: int = 12
    rtc_min_new_tail_steps: int = 16
    rtc_ramp_rate: float = 6.0
    rtc_position_tolerance: float = 1e-4
    rtc_orientation_tolerance_rad: float = 1e-3
    rtc_gripper_tolerance: float = 1e-4
    max_position_step_m: float | None = None
    max_orientation_step_rad: float | None = None
    max_gripper_step: float | None = None
    max_position_second_difference_m: float | None = None
    max_gripper_second_difference: float | None = None
    allow_unsafe_non_loopback: bool = False
    log_level: str = "INFO"


@dataclass(frozen=True)
class ResolvedServerConfig:
    model_path: Path
    embodiment_tag: str
    device: str
    host: str
    port: int
    max_message_size_bytes: int
    allow_unsafe_non_loopback: bool
    log_level: str
    serving: IndustrialNextServingConfig


def resolve_server_config(
    config: ServerConfig,
) -> tuple[ResolvedServerConfig, ConfigDrivenIndustrialNextProfile]:
    profile = load_industrialnext_profile(config.config)
    model_path = (
        profile.model_path
        if config.model_path is None
        else Path(config.model_path).expanduser().resolve()
    )
    rtc_mode = profile.rtc_mode if config.rtc_mode is None else config.rtc_mode
    if rtc_mode not in profile.supported_rtc_modes:
        raise ValueError(
            f"rtc_mode {rtc_mode!r} is not supported by profile {profile.profile_name!r}"
        )
    serving = IndustrialNextServingConfig(
        control_hz=profile.control_hz if config.control_hz is None else config.control_hz,
        action_horizon=profile.action_horizon,
        max_image_staleness_steps=config.max_image_staleness_steps,
        min_usable_action_steps=config.min_usable_action_steps,
        idle_session_timeout_s=config.idle_session_timeout_s,
        stats_log_interval_steps=config.stats_log_interval_steps,
        rtc_mode=rtc_mode,
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
    resolved = ResolvedServerConfig(
        model_path=model_path,
        embodiment_tag=profile.embodiment_tag
        if config.embodiment_tag is None
        else config.embodiment_tag,
        device=profile.device if config.device is None else config.device,
        host=profile.host if config.host is None else config.host,
        port=profile.port if config.port is None else config.port,
        max_message_size_bytes=config.max_message_size_bytes,
        allow_unsafe_non_loopback=config.allow_unsafe_non_loopback,
        log_level=config.log_level,
        serving=serving,
    )
    validate_server_config(resolved, profile)
    return resolved, profile


def validate_server_config(
    config: ResolvedServerConfig,
    profile: ConfigDrivenIndustrialNextProfile,
) -> Path:
    if not config.model_path.is_dir():
        raise ValueError(f"model_path must be an existing directory: {config.model_path}")
    if not isinstance(getattr(logging, config.log_level.upper(), None), int):
        raise ValueError(f"unknown log_level {config.log_level!r}")
    if (
        isinstance(config.port, bool)
        or not isinstance(config.port, int)
        or not 0 <= config.port < 65536
    ):
        raise ValueError("port must be an integer within [0, 65535]")
    if (
        isinstance(config.max_message_size_bytes, bool)
        or not isinstance(config.max_message_size_bytes, int)
        or config.max_message_size_bytes <= 0
    ):
        raise ValueError("max_message_size_bytes must be a positive integer")
    if config.serving.action_horizon != profile.action_horizon:
        raise ValueError("serving action horizon differs from profile")
    validate_checkpoint_rtc_compatibility(config.model_path, config.serving)
    if not _is_loopback_host(config.host):
        if not config.allow_unsafe_non_loopback:
            raise ValueError(
                "non-loopback host rejected: the direct protocol unpickles input; "
                "pass --allow-unsafe-non-loopback only on a trusted network"
            )
        logger.warning(
            "UNSAFE non-loopback pickle server enabled on %s:%d", config.host, config.port
        )
    return config.model_path


async def serve_forever(
    config: ResolvedServerConfig,
    profile: ConfigDrivenIndustrialNextProfile,
) -> None:
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="groot-inference")
    handler: IndustrialNextAsyncServer | None = None
    loop = asyncio.get_running_loop()
    try:
        logger.info("Loading strict GR00T policy from %s on %s", config.model_path, config.device)
        policy = await loop.run_in_executor(
            executor,
            partial(
                Gr00tPolicy,
                model_path=str(config.model_path),
                embodiment_tag=embodiment_tag,
                device=config.device,
                strict=True,
            ),
        )
        profile.assert_policy_contract(policy)
        first_task = profile.task_catalog.tasks[0]
        logger.info("Running configured synthetic finite inference before binding")
        await loop.run_in_executor(executor, _warmup_policy, policy, profile, first_task.task_text)
        handler = IndustrialNextAsyncServer(
            policy=policy,
            executor=executor,
            config=config.serving,
            service_provenance={"model_path": str(config.model_path)},
            embodiment_tag=embodiment_tag.value,
            profile=profile,
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
                    "Industrial Next GR00T server ready at ws://%s:%d | profile=%s | model=%s",
                    config.host,
                    config.port,
                    profile.profile_name,
                    config.model_path,
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
    """Return lightweight checkpoint facts for existing evaluation report callers."""
    model_path = model_path.expanduser().resolve()
    checkpoint_config_path = model_path / "config.json"
    checkpoint_config: dict[str, Any] = {}
    if checkpoint_config_path.is_file():
        try:
            checkpoint_config = json.loads(checkpoint_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint config {checkpoint_config_path}: {exc}") from exc
    return {
        "model_path": str(model_path),
        "checkpoint_model_type": checkpoint_config.get("model_type"),
        "checkpoint_action_horizon": checkpoint_config.get("action_horizon"),
        "checkpoint_rtc_training_max_prefix_steps": checkpoint_config.get(
            "rtc_training_max_prefix_steps", 0
        ),
    }


def validate_checkpoint_rtc_compatibility(
    model_path: Path, config: IndustrialNextServingConfig
) -> None:
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
    if model_config.get("action_horizon") != config.action_horizon:
        raise ValueError("RTC action horizon differs from the checkpoint")
    if config.rtc_mode == "trained_prefix":
        trained_max = model_config.get("rtc_training_max_prefix_steps", 0)
        if isinstance(trained_max, bool) or not isinstance(trained_max, int) or trained_max <= 0:
            raise ValueError("checkpoint does not advertise trained-prefix support")
        if config.rtc_max_prefix_steps > trained_max:
            raise ValueError(
                f"runtime prefix maximum {config.rtc_max_prefix_steps} exceeds checkpoint "
                f"trained maximum {trained_max}"
            )


def _warmup_policy(policy: Any, profile: ConfigDrivenIndustrialNextProfile, task_text: str) -> None:
    action, _ = policy.get_action(profile.build_synthetic_model_observation(task_text))
    profile.map_action_chunk(action)


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
    os.environ.setdefault("GROOT_HF_LOCAL_FIRST", "1")
    os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    resolved, profile = resolve_server_config(config)
    logging.basicConfig(
        level=getattr(logging, resolved.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(serve_forever(resolved, profile))


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
