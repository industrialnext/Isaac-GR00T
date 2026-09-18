# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorded, masked observations through the production request-driven server."""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor, Future

import cv2
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.state_action.rot6d import rot6d_groot_to_source
from gr00t.policy.industrialnext.async_server import IndustrialNextAsyncServer
from gr00t.policy.industrialnext.execution import rotation_matrix
import numpy as np
import torch


class TraceExecutor(Executor):
    """Compute one inference at a time and deliver it at a recorded virtual delay."""

    def __init__(self, clock, delays, seeds):
        self.clock = clock
        self.delays = delays
        self.seeds = seeds
        self.calls = 0
        self.pending = None

    def submit(self, fn, /, *args, **kwargs):
        if self.pending is not None:
            raise RuntimeError("trace executor received overlapping inference")
        future = Future()
        torch.manual_seed(self.seeds[self.calls % len(self.seeds)])
        due = self.clock() + self.delays[self.calls % len(self.delays)] / 50.0
        self.calls += 1
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            result = exc
        self.pending = (due, future, result)
        return future

    def complete_ready(self):
        if self.pending is None or self.pending[0] > self.clock() + 1e-9:
            return
        _, future, result = self.pending
        self.pending = None
        if isinstance(result, Exception):
            future.set_exception(result)
        else:
            future.set_result(result)

    def shutdown(self, wait=True, *, cancel_futures=False):
        if self.pending is not None:
            self.pending[1].cancel()
            self.pending = None


def _wire_state(profile, data):
    state = {}
    for layout in profile.state_layouts:
        values = data.states[layout.key][-1].copy()
        start = 0
        for index, (name, width) in enumerate(zip(layout.fields, layout.widths)):
            value = values[start : start + width]
            if index == layout.rot6d_index:
                value = rot6d_groot_to_source(value)
            state[name] = value.tolist()
            start += width
    return state


def _wire_observation(profile, data, task):
    observation = {
        **_wire_state(profile, data),
        "task_uuid": task.task_uuid,
        "task_text": task.task_text,
        "images_meta": {},
    }
    for wire, key in profile.wire_image_to_model.items():
        rgb = np.asarray(data.images[key][-1], dtype=np.uint8)
        if rgb.shape != (profile.image_height, profile.image_width, 3):
            raise ValueError(f"Recorded view {key} geometry differs from the serving profile")
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError(f"Failed to encode recorded view {key}")
        observation[wire] = encoded.tobytes()
        observation["images_meta"][wire] = {
            "format": "jpeg",
            "dtype": "uint8",
            "channels": 3,
            "height": rgb.shape[0],
            "width": rgb.shape[1],
        }
    return observation


def physical_target_errors(profile, trajectory, index, action):
    """Keep metric units separate and exclude invalid target coordinates."""
    metrics = {"position_m": [], "orientation_rad": [], "hand_native_abs": []}
    if not 0 <= index < len(trajectory):
        return metrics
    for layout in profile.action_layouts:
        target = np.asarray(trajectory[f"action.{layout.key}"].iloc[index])
        mask_key = f"action_validity.{layout.key}"
        mask = (
            np.asarray(trajectory[mask_key].iloc[index], dtype=bool)
            if mask_key in trajectory
            else np.ones(layout.width, dtype=bool)
        )
        start = 0
        for field_index, (name, width) in enumerate(zip(layout.fields, layout.widths)):
            expected = target[start : start + width]
            valid = mask[start : start + width]
            actual = np.asarray(action[name])
            if field_index == layout.rot6d_index:
                if valid.all():
                    expected = rot6d_groot_to_source(expected)
                    relative = rotation_matrix(expected).T @ rotation_matrix(actual)
                    metrics["orientation_rad"].append(
                        float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
                    )
            elif name in profile.position_action_fields:
                if valid.all():
                    metrics["position_m"].append(float(np.linalg.norm(actual - expected)))
            else:
                metrics["hand_native_abs"].extend(np.abs(actual[valid] - expected[valid]).tolist())
            start += width
    return metrics


async def replay_production(policy, loader, trajectory_id, embodiment, profile, serving, config):
    """Use real admission, expiry, RTC prefix, postprocessing and emission owners."""
    trajectory = loader[trajectory_id]
    modalities = dict(loader.modality_configs)
    modalities.pop("action")
    now = [1000.0]
    executor = TraceExecutor(lambda: now[0], config.delay_trace, config.seeds)
    first = extract_step_data(trajectory, 0, modalities, embodiment)
    task = next((t for t in profile.task_catalog.tasks if t.task_text == first.text), None)
    if task is None:
        raise ValueError("Recorded task text is absent from the serving profile task catalog")
    # Pin the recorded task; no synthetic prompt substitution during replay.
    server = IndustrialNextAsyncServer(
        policy=policy,
        executor=executor,
        config=serving,
        profile=profile,
        embodiment_tag=profile.embodiment_tag,
        service_provenance={},
        clock=lambda: now[0],
    )
    session_id = server.handle_request(
        {
            "type": "register_session",
            "control_hz": 50.0,
            "task_uuid": task.task_uuid,
            "task_text": task.task_text,
        }
    )["session_id"]
    records = []
    metrics = {
        scope: {"position_m": [], "orientation_rad": [], "hand_native_abs": []}
        for scope in ("same_time", "lookahead_target")
    }
    invalid_observations = 0
    source_data_indices = {}
    try:
        for tick in range(min(config.replay_steps, len(trajectory))):
            now[0] = 1000 + tick / 50
            executor.complete_ready()
            # Deliver the executor Future, its asyncio wrapper and completion callback.
            for _ in range(3):
                await asyncio.sleep(0)
            eligible = (
                bool(trajectory["observation.valid"].iloc[tick])
                if "observation.valid" in trajectory
                else True
            )
            if not eligible:
                invalid_observations += 1
                records.append({"tick": tick, "observation_eligible": False})
                continue
            data = extract_step_data(trajectory, tick, modalities, embodiment)
            if data.text != task.task_text:
                raise ValueError("Task text changed inside the recorded episode")
            response = server.handle_request(
                {
                    "type": "step",
                    "session_id": session_id,
                    "observation": _wire_observation(profile, data, task),
                }
            )
            if not response.get("error"):
                source_data_indices[response["timestep"]] = tick
            monitoring = response.get("monitoring", {})
            action = response.get("action")
            record = {
                "tick": tick,
                "observation_eligible": True,
                "has_action": action is not None,
                "error": response.get("error"),
                "reason": response.get("reason"),
                "monitoring": monitoring,
            }
            if action is not None:
                for scope, target in [
                    ("same_time", tick - profile.action_start_offset_steps),
                    (
                        "lookahead_target",
                        tick + serving.action_offset - profile.action_start_offset_steps,
                    ),
                ]:
                    values = physical_target_errors(profile, trajectory, target, action)
                    for name, errors in values.items():
                        metrics[scope][name].extend(errors)
                record["selected_original_targets"] = [
                    {
                        "source_tick": source["source_tick"],
                        "model_row": source["model_row"],
                        "ensemble_weight": source["weight"],
                        "dataset_target_index": source_data_indices[source["source_tick"]]
                        + source["model_row"],
                        "emitted_action_errors": physical_target_errors(
                            profile,
                            trajectory,
                            source_data_indices[source["source_tick"]] + source["model_row"],
                            action,
                        ),
                    }
                    for source in monitoring["emitted_action"]["contributions"]
                ]
                record["action"] = action
            records.append(record)
            if response.get("error"):
                break
    finally:
        server._pending_snapshot = None
        if executor.pending is not None:
            now[0] = executor.pending[0] + 1e-9
            executor.complete_ready()
        await server.shutdown()
    summary = {}
    for scope, groups in metrics.items():
        summary[scope] = {
            name: {
                "count": len(values),
                "mean": float(np.mean(values)),
                "p99": float(np.percentile(values, 99)),
            }
            if values
            else None
            for name, values in groups.items()
        }
    emitted = sum(bool(r.get("has_action")) for r in records)
    requested = sum(bool(r.get("observation_eligible")) for r in records)
    terminal = sum(bool(r.get("error")) for r in records)
    return {
        "status": "terminal" if terminal else "ok" if emitted else "no_admitted_predictions",
        "calls": executor.calls,
        "requested_steps": requested,
        "emitted_steps": emitted,
        "coverage": emitted / max(1, requested),
        "null_rate": (requested - emitted - terminal) / max(1, requested),
        "terminal_count": terminal,
        "invalid_observations": invalid_observations,
        "physical_errors": summary,
        "steps": records,
        "qualification": "recorded open-loop replay; not robot success or deployment latency",
    }
