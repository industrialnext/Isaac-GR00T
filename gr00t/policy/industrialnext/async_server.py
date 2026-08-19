# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-blocking Industrial Next direct-RPC server for a GR00T policy."""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any, Mapping, Protocol
import uuid

from industrialnext_rpc.direct.metadata import Metadata

from .adapter import (
    ACTION_HORIZON,
    IMAGE_HEIGHT,
    IMAGE_KEY_TO_MODEL_KEY,
    IMAGE_WIDTH,
    CachedImage,
    ObservationAdmission,
    ObservationSnapshot,
    admit_observation,
    build_model_observation,
    map_action_chunk,
    snapshot_is_fresh,
)
from .task_catalog import TaskCatalog


logger = logging.getLogger(__name__)

ASYNC_PROTOCOL_VERSION = 2
ASYNC_CAPABILITIES = [
    "error_envelope_v2",
    "monitoring_in_step",
    "server_owned_gripper_snap",
]


class Policy(Protocol):
    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


@dataclass(frozen=True)
class IndustrialNextServingConfig:
    """Runtime settings for the single-session GR00T server."""

    control_hz: float = 50.0
    max_image_staleness_steps: int = 5
    min_usable_action_steps: int = 1
    idle_session_timeout_s: float = 300.0
    stats_log_interval_steps: int = 250

    def __post_init__(self) -> None:
        if not math.isfinite(self.control_hz) or self.control_hz != 50.0:
            raise ValueError("control_hz must be exactly 50.0")
        if (
            not isinstance(self.max_image_staleness_steps, int)
            or isinstance(self.max_image_staleness_steps, bool)
            or self.max_image_staleness_steps < 0
        ):
            raise ValueError("max_image_staleness_steps must be a non-negative integer")
        if (
            not isinstance(self.min_usable_action_steps, int)
            or isinstance(self.min_usable_action_steps, bool)
            or not 1 <= self.min_usable_action_steps <= ACTION_HORIZON
        ):
            raise ValueError(f"min_usable_action_steps must be within [1, {ACTION_HORIZON}]")
        if not math.isfinite(self.idle_session_timeout_s) or self.idle_session_timeout_s <= 0:
            raise ValueError("idle_session_timeout_s must be finite and positive")
        if (
            not isinstance(self.stats_log_interval_steps, int)
            or isinstance(self.stats_log_interval_steps, bool)
            or self.stats_log_interval_steps < 0
        ):
            raise ValueError("stats_log_interval_steps must be a non-negative integer")


@dataclass(frozen=True)
class InferenceResult:
    snapshot: ObservationSnapshot
    rows: tuple[dict[str, list[float]], ...]
    inference_latency_ms: float
    image_decode_latency_ms: float


@dataclass
class SessionStats:
    missing_rgb_steps: int = 0
    stale_rgb_steps: int = 0
    ignored_depth_fields: int = 0
    inference_failures: int = 0
    rejected_tails: int = 0
    expired_rows: int = 0
    stale_pending_snapshots: int = 0
    null_reasons: dict[str, int] = field(default_factory=dict)
    gripper_min: float | None = None
    gripper_max: float | None = None

    def record_null(self, reason: str) -> None:
        self.null_reasons[reason] = self.null_reasons.get(reason, 0) + 1

    def record_grippers(self, rows: tuple[dict[str, list[float]], ...]) -> None:
        values = [
            float(row[field_name][0])
            for row in rows
            for field_name in ("left_gripper", "right_gripper")
        ]
        if not values:
            return
        observed_min = min(values)
        observed_max = max(values)
        self.gripper_min = (
            observed_min if self.gripper_min is None else min(self.gripper_min, observed_min)
        )
        self.gripper_max = (
            observed_max if self.gripper_max is None else max(self.gripper_max, observed_max)
        )


@dataclass
class ActiveSession:
    session_id: str
    generation: int
    task_uuid: str
    task_text: str
    control_hz: float
    created_at_s: float
    last_activity_s: float
    timestep: int = -1
    monitoring_timestep: int = -1
    image_cache: dict[str, CachedImage] = field(default_factory=dict)
    timeline: dict[int, dict[str, list[float]]] = field(default_factory=dict)
    inference_status: str = "idle"
    inference_latency_ms: float = 0.0
    image_decode_latency_ms: float = 0.0
    total_inferences: int = 0
    total_actions_served: int = 0
    latest_inference_error: str | None = None
    latest_source_timestep: int | None = None
    latest_image_ages: dict[str, int | None] = field(default_factory=dict)
    latest_null_reason: str = "startup"
    stats: SessionStats = field(default_factory=SessionStats)


class IndustrialNextAsyncServer:
    """Synchronous DirectServer handler with one asynchronous inference worker."""

    def __init__(
        self,
        *,
        policy: Policy,
        executor: Executor,
        task_catalog: TaskCatalog,
        config: IndustrialNextServingConfig,
        service_provenance: Mapping[str, Any],
        embodiment_tag: str,
        owns_executor: bool = True,
    ) -> None:
        self.policy = policy
        self.executor = executor
        self.task_catalog = task_catalog
        self.config = config
        self.service_provenance = dict(service_provenance)
        self.embodiment_tag = embodiment_tag
        self.owns_executor = owns_executor
        self.server_instance_id = str(uuid.uuid4())

        self._generation = 0
        self._active_session: ActiveSession | None = None
        self._inference_future: asyncio.Future[InferenceResult] | None = None
        self._pending_snapshot: ObservationSnapshot | None = None
        self._idle_timer: asyncio.TimerHandle | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def get_metadata(self) -> Metadata:
        """Return the static direct-RPC and service metadata."""
        service_metadata = self._service_metadata()
        return {
            "server_info": {
                "server_name": "gr00t_industrialnext_async_server",
                "service_type": "rl_inference",
            },
            "service_metadata": service_metadata,
            "request_format": {
                "register_session": {
                    "type": str,
                    "control_hz": float,
                    "task_uuid": str,
                    "task_text": str,
                },
                "step": {
                    "type": str,
                    "session_id": str,
                    "observation": dict,
                },
                "close_session": {
                    "type": str,
                    "session_id": str,
                },
            },
            "response_format": {
                "register_session": {
                    "session_id": str,
                    "metadata": dict,
                    "error": str,
                },
                "step": {
                    "action": object,
                    "timestep": int,
                    "queue_len": int,
                    "inference_status": str,
                    "inference_latency_ms": float,
                    "inference_batch_size": int,
                    "obs_buffer_len": int,
                    "session_uptime_s": float,
                    "total_inferences": int,
                    "total_actions_served": int,
                    "server_step_ms": float,
                    "monitoring": dict,
                    "monitoring_timestep": int,
                    "monitoring_gripper_values": dict,
                    "error": str,
                },
                "close_session": {
                    "ok": bool,
                    "error": str,
                },
            },
        }

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one validated request without blocking on model inference."""
        if self._closed:
            return self._error_response(str(request.get("type")), "server_closed")
        self._bind_event_loop()
        request_type = request.get("type")
        try:
            if request_type == "register_session":
                return self._register_session(request)
            if request_type == "step":
                return self._step(request)
            if request_type == "close_session":
                return self._close_session(request)
            raise ValueError(f"unsupported request type {request_type!r}")
        except Exception as exc:
            logger.warning("Industrial Next request %r failed: %s", request_type, exc)
            return self._error_response(str(request_type), str(exc))

    async def shutdown(self) -> None:
        """Stop admission, invalidate sessions, drain inference, and close the worker."""
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._active_session = None
        self._pending_snapshot = None
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        future = self._inference_future
        try:
            if future is not None:
                await asyncio.shield(future)
        except Exception:
            logger.exception("In-flight GR00T inference failed during shutdown")
        finally:
            self._inference_future = None
            if self.owns_executor:
                self.executor.shutdown(wait=True, cancel_futures=True)

    def _bind_event_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._event_loop is None:
            self._event_loop = loop
        elif self._event_loop is not loop:
            raise RuntimeError("server handler was called from a different event loop")
        return loop

    def _service_metadata(self) -> dict[str, Any]:
        task_catalog = self.task_catalog.to_metadata()
        return {
            **self.service_provenance,
            "async_serving": True,
            "async_protocol_version": ASYNC_PROTOCOL_VERSION,
            "async_capabilities": list(ASYNC_CAPABILITIES),
            "expert_camera_height": IMAGE_HEIGHT,
            "expert_camera_width": IMAGE_WIDTH,
            "effective_gripper_snap_config": {
                "enabled": False,
                "gripper_action_fields": ["left_gripper", "right_gripper"],
                "snap_range": None,
                "signal_range": None,
            },
            "task_conditioning": {
                "enabled": True,
                "mode": "text",
                "prompt_field_name": "task_text",
                "task_catalog_version": self.task_catalog.catalog_version,
                "task_catalog": task_catalog,
                "task_uuid_to_text": self.task_catalog.task_uuid_to_text,
            },
            "server_instance_id": self.server_instance_id,
            "embodiment_tag": self.embodiment_tag,
            "video_keys": list(IMAGE_KEY_TO_MODEL_KEY.values()),
            "state_keys": [
                "left_eef",
                "left_gripper",
                "left_ft",
                "right_eef",
                "right_gripper",
                "right_ft",
            ],
            "action_keys": ["left_eef", "left_gripper", "right_eef", "right_gripper"],
            "action_horizon": ACTION_HORIZON,
            "action_semantics": "absolute",
            "action_rotation_representation": "rot6d",
            "max_sessions": 1,
            "default_control_hz": self.config.control_hz,
            "max_image_staleness_steps": self.config.max_image_staleness_steps,
            "min_usable_action_steps": self.config.min_usable_action_steps,
            "idle_session_timeout_s": self.config.idle_session_timeout_s,
        }

    def _register_session(self, request: Mapping[str, Any]) -> dict[str, Any]:
        control_hz = request.get("control_hz")
        if isinstance(control_hz, bool) or not isinstance(control_hz, int | float):
            raise ValueError("control_hz must be numeric")
        control_hz = float(control_hz)
        if not math.isfinite(control_hz) or control_hz != self.config.control_hz:
            raise ValueError(f"control_hz must be exactly {self.config.control_hz}")
        task_uuid = request.get("task_uuid")
        task_text = request.get("task_text")
        if not isinstance(task_uuid, str) or not isinstance(task_text, str):
            raise ValueError("task_uuid and task_text must be strings")
        pinned_task_text = self.task_catalog.resolve(task_uuid, task_text)

        previous_session_id = (
            None if self._active_session is None else self._active_session.session_id
        )
        self._generation += 1
        now_s = time.monotonic()
        session = ActiveSession(
            session_id=str(uuid.uuid4()),
            generation=self._generation,
            task_uuid=task_uuid,
            task_text=pinned_task_text,
            control_hz=control_hz,
            created_at_s=now_s,
            last_activity_s=now_s,
        )
        self._active_session = session
        self._pending_snapshot = None
        self._schedule_idle_expiry(session)
        if previous_session_id is not None:
            logger.warning(
                "Replacing Industrial Next session old=%s new=%s",
                previous_session_id,
                session.session_id,
            )
        else:
            logger.info("Registered Industrial Next session %s", session.session_id)
        return {
            "session_id": session.session_id,
            "metadata": self._service_metadata(),
        }

    def _close_session(self, request: Mapping[str, Any]) -> dict[str, Any]:
        session_id = str(request.get("session_id", ""))
        if self._active_session is not None and self._active_session.session_id == session_id:
            self._generation += 1
            self._active_session = None
            self._pending_snapshot = None
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            logger.info("Closed Industrial Next session %s", session_id)
        return {"ok": True}

    def _step(self, request: Mapping[str, Any]) -> dict[str, Any]:
        step_started_at = time.perf_counter()
        session = self._session(str(request.get("session_id", "")))
        candidate_timestep = session.timestep + 1
        observation = request.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError("observation must be a mapping")
        admission = admit_observation(
            observation,
            image_cache=session.image_cache,
            timestep=candidate_timestep,
            task_uuid=session.task_uuid,
            task_text=session.task_text,
            generation=session.generation,
            max_image_staleness_steps=self.config.max_image_staleness_steps,
        )

        session.timestep = candidate_timestep
        session.monitoring_timestep += 1
        session.last_activity_s = time.monotonic()
        session.latest_image_ages = dict(admission.image_ages)
        session.stats.missing_rgb_steps += int(bool(admission.missing_images))
        session.stats.stale_rgb_steps += int(bool(admission.stale_images))
        session.stats.ignored_depth_fields += admission.ignored_depth_fields
        self._schedule_idle_expiry(session)

        expired_targets = [target for target in session.timeline if target < candidate_timestep]
        for target in expired_targets:
            del session.timeline[target]
        session.stats.expired_rows += len(expired_targets)

        action = session.timeline.pop(candidate_timestep, None)
        if action is None:
            null_reason = self._null_reason(session, admission)
            session.latest_null_reason = null_reason
            session.stats.record_null(null_reason)
        else:
            session.total_actions_served += 1
            session.latest_null_reason = ""

        if admission.snapshot is not None:
            if self._inference_future is None:
                self._launch_inference(admission.snapshot)
            else:
                self._pending_snapshot = admission.snapshot
        elif self._inference_future is None and not session.timeline:
            session.inference_status = "waiting_for_images"

        server_step_ms = (time.perf_counter() - step_started_at) * 1000.0
        response = self._step_success_response(
            session=session,
            action=action,
            server_step_ms=server_step_ms,
        )
        self._maybe_log_stats(session, server_step_ms=server_step_ms)
        return response

    def _session(self, session_id: str) -> ActiveSession:
        session = self._active_session
        if session is None or session.session_id != session_id:
            raise SessionNotFoundError("session_not_found")
        return session

    def _launch_inference(self, snapshot: ObservationSnapshot) -> None:
        session = self._active_session
        if session is None or not snapshot_is_fresh(
            snapshot,
            current_timestep=session.timestep,
            active_generation=session.generation,
            max_staleness_steps=self.config.max_image_staleness_steps,
        ):
            if session is not None:
                session.stats.stale_pending_snapshots += 1
            return
        if self._inference_future is not None:
            self._pending_snapshot = snapshot
            return
        loop = self._bind_event_loop()
        session.inference_status = "running"
        future = loop.run_in_executor(self.executor, self._run_inference, snapshot)
        self._inference_future = future
        future.add_done_callback(lambda completed: self._complete_inference(snapshot, completed))

    def _run_inference(self, snapshot: ObservationSnapshot) -> InferenceResult:
        started_at = time.perf_counter()
        model_observation = build_model_observation(snapshot)
        decode_ms = (time.perf_counter() - started_at) * 1000.0
        action, _ = self.policy.get_action(model_observation)
        rows = map_action_chunk(action)
        return InferenceResult(
            snapshot=snapshot,
            rows=rows,
            inference_latency_ms=(time.perf_counter() - started_at) * 1000.0,
            image_decode_latency_ms=decode_ms,
        )

    def _complete_inference(
        self,
        snapshot: ObservationSnapshot,
        future: asyncio.Future[InferenceResult],
    ) -> None:
        if future is not self._inference_future:
            return
        self._inference_future = None
        session = self._active_session
        try:
            result = future.result()
        except Exception as exc:
            if session is not None and session.generation == snapshot.generation:
                session.inference_status = "error"
                session.latest_inference_error = f"{type(exc).__name__}: {exc}"
                session.stats.inference_failures += 1
                logger.error(
                    "GR00T inference failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
        else:
            if (
                not self._closed
                and session is not None
                and session.generation == result.snapshot.generation
            ):
                self._admit_inference_result(session, result)
        self._launch_pending_if_valid()

    def _admit_inference_result(self, session: ActiveSession, result: InferenceResult) -> None:
        future_rows = {
            result.snapshot.source_timestep + index: row
            for index, row in enumerate(result.rows)
            if result.snapshot.source_timestep + index > session.timestep
        }
        session.inference_latency_ms = result.inference_latency_ms
        session.image_decode_latency_ms = result.image_decode_latency_ms
        session.total_inferences += 1
        session.latest_source_timestep = result.snapshot.source_timestep
        session.stats.record_grippers(result.rows)
        if len(future_rows) < self.config.min_usable_action_steps:
            session.stats.rejected_tails += 1
            session.latest_inference_error = (
                "insufficient_usable_tail: "
                f"{len(future_rows)} < {self.config.min_usable_action_steps}"
            )
            session.inference_status = "insufficient_tail"
            return
        session.timeline = future_rows
        session.latest_inference_error = None
        session.inference_status = "ready"

    def _launch_pending_if_valid(self) -> None:
        pending = self._pending_snapshot
        self._pending_snapshot = None
        if self._closed or pending is None:
            return
        session = self._active_session
        if session is None or not snapshot_is_fresh(
            pending,
            current_timestep=session.timestep,
            active_generation=session.generation,
            max_staleness_steps=self.config.max_image_staleness_steps,
        ):
            if session is not None:
                session.stats.stale_pending_snapshots += 1
            return
        self._launch_inference(pending)

    def _schedule_idle_expiry(self, session: ActiveSession) -> None:
        loop = self._bind_event_loop()
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = loop.call_later(
            self.config.idle_session_timeout_s,
            self._expire_session_if_idle,
            session.session_id,
            session.generation,
        )

    def _expire_session_if_idle(self, session_id: str, generation: int) -> None:
        self._idle_timer = None
        session = self._active_session
        if session is None or session.session_id != session_id or session.generation != generation:
            return
        idle_s = time.monotonic() - session.last_activity_s
        remaining_s = self.config.idle_session_timeout_s - idle_s
        if remaining_s > 0:
            loop = self._event_loop
            if loop is not None:
                self._idle_timer = loop.call_later(
                    remaining_s,
                    self._expire_session_if_idle,
                    session_id,
                    generation,
                )
            return
        logger.warning("Expiring idle Industrial Next session %s", session_id)
        self._generation += 1
        self._active_session = None
        self._pending_snapshot = None

    def _null_reason(self, session: ActiveSession, admission: ObservationAdmission) -> str:
        if admission.missing_images:
            return "missing_images"
        if admission.stale_images:
            return "stale_images"
        if session.inference_status == "error":
            return "inference_error"
        if session.inference_status == "insufficient_tail":
            return "insufficient_tail"
        if self._inference_future is not None:
            return "inference_running"
        if session.total_inferences == 0:
            return "startup"
        return "timeline_gap"

    def _step_success_response(
        self,
        *,
        session: ActiveSession,
        action: dict[str, list[float]] | None,
        server_step_ms: float,
    ) -> dict[str, Any]:
        now_s = time.monotonic()
        action_age = (
            None
            if session.latest_source_timestep is None
            else session.timestep - session.latest_source_timestep
        )
        monitoring = {
            "progress": 0.0,
            "classification": "unknown",
            "scene_valid": True,
            "confidence": 0.0,
            "latest_inference_error": session.latest_inference_error,
            "action_source_age_steps": action_age,
            "usable_tail_count": len(session.timeline),
            "image_age_steps": dict(session.latest_image_ages),
            "image_decode_latency_ms": session.image_decode_latency_ms,
            "null_reason": session.latest_null_reason,
        }
        monitoring_grippers = (
            {}
            if action is None
            else {
                "left_gripper": list(action["left_gripper"]),
                "right_gripper": list(action["right_gripper"]),
            }
        )
        return {
            "action": action,
            "timestep": session.timestep,
            "queue_len": len(session.timeline),
            "inference_status": session.inference_status,
            "inference_latency_ms": float(session.inference_latency_ms),
            "inference_batch_size": 1,
            "obs_buffer_len": 1,
            "session_uptime_s": float(now_s - session.created_at_s),
            "total_inferences": session.total_inferences,
            "total_actions_served": session.total_actions_served,
            "server_step_ms": float(server_step_ms),
            "monitoring": monitoring,
            "monitoring_timestep": session.monitoring_timestep,
            "monitoring_gripper_values": monitoring_grippers,
        }

    def _error_response(self, request_type: str, error: str) -> dict[str, Any]:
        if request_type == "register_session":
            return {"session_id": "", "metadata": {}, "error": error}
        if request_type == "close_session":
            return {"ok": False, "error": error}
        return {
            "action": None,
            "timestep": -1,
            "queue_len": 0,
            "inference_status": "error",
            "inference_latency_ms": 0.0,
            "inference_batch_size": 0,
            "obs_buffer_len": 0,
            "session_uptime_s": 0.0,
            "total_inferences": 0,
            "total_actions_served": 0,
            "server_step_ms": 0.0,
            "monitoring": {},
            "monitoring_timestep": -1,
            "monitoring_gripper_values": {},
            "error": error,
        }

    def _maybe_log_stats(self, session: ActiveSession, *, server_step_ms: float) -> None:
        interval = self.config.stats_log_interval_steps
        if interval <= 0 or (session.timestep + 1) % interval != 0:
            return
        logger.info(
            "GR00T async stats session=%s step=%d server_step_ms=%.3f "
            "inference_ms=%.1f decode_ms=%.1f queue=%d missing_rgb=%d stale_rgb=%d "
            "ignored_depth=%d inference_failures=%d expired_rows=%d rejected_tails=%d "
            "stale_pending=%d null_reasons=%s gripper_range=(%s,%s)",
            session.session_id,
            session.timestep,
            server_step_ms,
            session.inference_latency_ms,
            session.image_decode_latency_ms,
            len(session.timeline),
            session.stats.missing_rgb_steps,
            session.stats.stale_rgb_steps,
            session.stats.ignored_depth_fields,
            session.stats.inference_failures,
            session.stats.expired_rows,
            session.stats.rejected_tails,
            session.stats.stale_pending_snapshots,
            session.stats.null_reasons,
            session.stats.gripper_min,
            session.stats.gripper_max,
        )


class SessionNotFoundError(ValueError):
    """Exact sentinel error required by the ROS reconnect path."""
