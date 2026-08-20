# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible checkpoint-latency and held-out RTC replay for Industrial Next."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.eval.run_gr00t_industrialnext_server import build_service_provenance
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.industrialnext import ACTION_HORIZON, build_synthetic_model_observation
import numpy as np
import torch
import tyro


SUPPORTED_MODES = ("off", "native", "trained_prefix")


@dataclass(frozen=True)
class BenchmarkConfig:
    model_path: str
    output_dir: str
    embodiment_tag: str = "new_embodiment"
    device: str = "cuda"
    task_text: str = "Pick the grounded target object."
    modes: list[str] = field(default_factory=lambda: list(SUPPORTED_MODES))
    warmup_calls: int = 5
    steady_calls: int = 100
    control_hz: float = 50.0
    prefix_steps: int = 12
    native_overlap_steps: int = 12
    min_new_tail_steps: int = 16
    rtc_ramp_rate: float = 6.0
    rtc_position_tolerance: float = 1e-4
    rtc_orientation_tolerance_rad: float = 1e-3
    rtc_gripper_tolerance: float = 1e-4
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    dataset_path: str | None = None
    trajectory_ids: list[int] = field(default_factory=lambda: [0])
    replay_steps: int = 100
    delay_trace: list[int] = field(default_factory=lambda: [4, 4, 5, 4, 6, 4])
    run_latency: bool = True
    run_replay: bool = True

    def __post_init__(self) -> None:
        if not self.run_latency and (not self.run_replay or self.dataset_path is None):
            raise ValueError("at least one benchmark workload must be enabled")
        if not self.modes or any(mode not in SUPPORTED_MODES for mode in self.modes):
            raise ValueError(f"modes must be selected from {SUPPORTED_MODES}")
        if self.warmup_calls < 0 or self.steady_calls <= 0:
            raise ValueError("warmup_calls must be non-negative and steady_calls must be positive")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if not 1 <= self.prefix_steps <= ACTION_HORIZON - self.min_new_tail_steps:
            raise ValueError("prefix_steps must leave min_new_tail_steps")
        if (
            not self.prefix_steps
            <= self.native_overlap_steps
            <= (ACTION_HORIZON - self.min_new_tail_steps)
        ):
            raise ValueError("native_overlap_steps must cover prefix_steps and leave a new tail")
        if not self.seeds or any(not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("seeds must be a non-empty integer list")
        if not self.delay_trace or any(delay < 1 for delay in self.delay_trace):
            raise ValueError("delay_trace must contain positive committed-step lengths")
        for name in (
            "rtc_position_tolerance",
            "rtc_orientation_tolerance_rad",
            "rtc_gripper_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def _checkpoint_config(model_path: Path) -> dict[str, Any]:
    path = model_path / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint config is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _unsupported_reason(
    mode: str, checkpoint_config: dict[str, Any], prefix_steps: int
) -> str | None:
    if mode == "off":
        return None
    if checkpoint_config.get("model_type") != "Gr00tN1d7":
        return "RTC requires a Gr00tN1d7 checkpoint"
    if checkpoint_config.get("action_horizon") != ACTION_HORIZON:
        return f"RTC requires action_horizon={ACTION_HORIZON}"
    if mode == "trained_prefix":
        trained_max = checkpoint_config.get("rtc_training_max_prefix_steps", 0)
        if isinstance(trained_max, bool) or not isinstance(trained_max, int) or trained_max <= 0:
            return "checkpoint does not advertise trained-prefix support"
        if prefix_steps > trained_max:
            return f"requested prefix {prefix_steps} exceeds trained maximum {trained_max}"
    return None


def _mode_options(
    mode: str,
    prefix: dict[str, np.ndarray] | None,
    *,
    prefix_steps: int,
    overlap_steps: int,
    ramp_rate: float,
) -> dict[str, Any]:
    if mode == "off":
        return {"rtc_mode": "off"}
    if prefix is None:
        raise ValueError(f"mode {mode!r} requires a physical action prefix")
    if mode == "native":
        return {
            "rtc_mode": mode,
            "action_prefix": {key: value[:, :overlap_steps] for key, value in prefix.items()},
            "rtc_frozen_steps": prefix_steps,
            "rtc_overlap_steps": overlap_steps,
            "rtc_ramp_rate": ramp_rate,
        }
    return {
        "rtc_mode": mode,
        "action_prefix": {key: value[:, :prefix_steps] for key, value in prefix.items()},
        "rtc_prefix_steps": prefix_steps,
    }


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def benchmark_checkpoint_latency(
    policy: Gr00tPolicy,
    config: BenchmarkConfig,
    checkpoint_config: dict[str, Any],
) -> dict[str, Any]:
    observation = build_synthetic_model_observation(config.task_text)
    torch.manual_seed(config.seeds[0])
    baseline_action, _ = policy.get_action(observation, options={"rtc_mode": "off"})
    results: dict[str, Any] = {}
    for mode in config.modes:
        reason = _unsupported_reason(mode, checkpoint_config, config.prefix_steps)
        if reason is not None:
            results[mode] = {"status": "unsupported", "reason": reason}
            continue
        options = _mode_options(
            mode,
            baseline_action,
            prefix_steps=config.prefix_steps,
            overlap_steps=config.native_overlap_steps,
            ramp_rate=config.rtc_ramp_rate,
        )
        warmup_totals = []
        for index in range(config.warmup_calls):
            torch.manual_seed(config.seeds[index % len(config.seeds)])
            started_at = time.perf_counter()
            policy.get_action(observation, options=options)
            warmup_totals.append((time.perf_counter() - started_at) * 1000.0)

        timings = {name: [] for name in ("preprocessing_ms", "generation_ms", "decode_ms")}
        total_ms: list[float] = []
        for index in range(config.steady_calls):
            torch.manual_seed(config.seeds[index % len(config.seeds)])
            started_at = time.perf_counter()
            _, info = policy.get_action(observation, options=options)
            total_ms.append((time.perf_counter() - started_at) * 1000.0)
            for name in timings:
                timings[name].append(float(info[name]))
        total_summary = _summary(total_ms)
        results[mode] = {
            "status": "ok",
            "warmup_total_ms": warmup_totals,
            "steady_calls": config.steady_calls,
            "preprocessing_ms": _summary(timings["preprocessing_ms"]),
            "generation_ms": _summary(timings["generation_ms"]),
            "decode_ms": _summary(timings["decode_ms"]),
            "total_ms": total_summary,
            "p99_committed_steps_sizing_proxy": max(
                1, math.ceil(total_summary["p99"] * config.control_hz / 1000.0)
            ),
        }
    return results


def _model_observation(data_point: Any, modality_configs: dict[str, Any]) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    for key, value in data_point.states.items():
        observation[f"state.{key}"] = value
    for key, value in data_point.images.items():
        observation[f"video.{key}"] = np.asarray(value)
    for language_key in modality_configs["language"].modality_keys:
        observation[language_key] = data_point.text
    return parse_observation_gr00t(observation, modality_configs)


def _action_row(action: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return {key: value[0, index].copy() for key, value in action.items()}


def _prefix_from_timeline(
    timeline: dict[int, dict[str, np.ndarray]], source: int, count: int
) -> dict[str, np.ndarray] | None:
    rows = []
    for target in range(source, source + count):
        if target not in timeline:
            return None
        rows.append(timeline[target])
    keys = rows[0]
    return {key: np.stack([row[key] for row in rows], axis=0)[None, ...] for key in keys}


def _flat_row(row: dict[str, np.ndarray], action_keys: list[str]) -> np.ndarray:
    return np.concatenate([np.atleast_1d(row[key]) for key in action_keys])


def _rotation_matrix_groot(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64).reshape(2, 3)
    first = rows[0] / np.linalg.norm(rows[0])
    second = rows[1] - np.dot(first, rows[1]) * first
    second = second / np.linalg.norm(second)
    return np.stack((first, second, np.cross(first, second)), axis=0)


def _so3_error(left: np.ndarray, right: np.ndarray) -> float:
    delta = _rotation_matrix_groot(left) @ _rotation_matrix_groot(right).T
    return math.acos(float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))


def _chunk_dynamics(action: dict[str, np.ndarray]) -> tuple[float, float, float, float, float]:
    position_step = 0.0
    orientation_step = 0.0
    gripper_step = 0.0
    position_second = 0.0
    gripper_second = 0.0
    for side in ("left", "right"):
        eef = action[f"{side}_eef"][0]
        gripper = action[f"{side}_gripper"][0, :, 0]
        position_step = max(
            position_step,
            float(np.linalg.norm(np.diff(eef[:, :3], axis=0), axis=1).max()),
        )
        orientation_step = max(
            orientation_step,
            max(_so3_error(eef[index, 3:], eef[index - 1, 3:]) for index in range(1, len(eef))),
        )
        gripper_step = max(gripper_step, float(np.abs(np.diff(gripper)).max()))
        position_second = max(
            position_second,
            float(np.linalg.norm(np.diff(eef[:, :3], n=2, axis=0), axis=1).max()),
        )
        gripper_second = max(gripper_second, float(np.abs(np.diff(gripper, n=2)).max()))
    return position_step, orientation_step, gripper_step, position_second, gripper_second


def _prefix_errors(
    action: dict[str, np.ndarray], prefix: dict[str, np.ndarray], steps: int
) -> tuple[float, float, float]:
    position_error = 0.0
    orientation_error = 0.0
    gripper_error = 0.0
    for side in ("left", "right"):
        actual_eef = action[f"{side}_eef"][0, :steps]
        expected_eef = prefix[f"{side}_eef"][0, :steps]
        position_error = max(
            position_error,
            float(np.linalg.norm(actual_eef[:, :3] - expected_eef[:, :3], axis=1).max()),
        )
        orientation_error = max(
            orientation_error,
            max(
                _so3_error(actual_eef[index, 3:], expected_eef[index, 3:]) for index in range(steps)
            ),
        )
        actual_gripper = action[f"{side}_gripper"][0, :steps]
        expected_gripper = prefix[f"{side}_gripper"][0, :steps]
        gripper_error = max(
            gripper_error,
            float(np.max(np.abs(actual_gripper - expected_gripper))),
        )
    return position_error, orientation_error, gripper_error


def replay_trajectory(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    trajectory_id: int,
    embodiment: EmbodimentTag,
    mode: str,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    trajectory = loader[trajectory_id]
    observation_modalities = deepcopy(loader.modality_configs)
    observation_modalities.pop("action")
    action_keys = loader.modality_configs["action"].modality_keys
    timeline: dict[int, dict[str, np.ndarray]] = {}
    errors: list[np.ndarray] = []
    position_seams: list[float] = []
    orientation_seams: list[float] = []
    gripper_seams: list[float] = []
    dynamics: list[tuple[float, float, float, float, float]] = []
    prefix_errors: list[tuple[float, float, float]] = []
    covered_targets: set[int] = set()
    holds = 0
    rejections = 0
    call_index = 0
    source = 0
    max_source = min(config.replay_steps, len(trajectory) - ACTION_HORIZON)
    while source < max_source:
        delay = config.delay_trace[call_index % len(config.delay_trace)]
        if delay >= ACTION_HORIZON or source + delay >= len(trajectory):
            break
        step_data = extract_step_data(
            trajectory,
            source,
            observation_modalities,
            embodiment,
        )
        observation = _model_observation(step_data, loader.modality_configs)
        effective_mode = "off" if call_index == 0 else mode
        requested_prefix = config.native_overlap_steps if effective_mode == "native" else delay
        prefix = None
        if effective_mode != "off":
            prefix = _prefix_from_timeline(timeline, source, requested_prefix)
            if prefix is None:
                holds += delay
                source += delay
                call_index += 1
                continue
        options = _mode_options(
            effective_mode,
            prefix,
            prefix_steps=delay,
            overlap_steps=config.native_overlap_steps,
            ramp_rate=config.rtc_ramp_rate,
        )
        torch.manual_seed(config.seeds[call_index % len(config.seeds)])
        action, _ = policy.get_action(observation, options=options)
        dynamics.append(_chunk_dynamics(action))
        if prefix is not None:
            hard_steps = delay
            prefix_error = _prefix_errors(action, prefix, hard_steps)
            prefix_errors.append(prefix_error)
            if (
                prefix_error[0] > config.rtc_position_tolerance
                or prefix_error[1] > config.rtc_orientation_tolerance_rad
                or prefix_error[2] > config.rtc_gripper_tolerance
            ):
                rejections += 1
                source += delay
                call_index += 1
                continue

        first_target = source + delay
        predicted = _action_row(action, delay)
        ground_truth = {
            key: np.asarray(trajectory[f"action.{key}"].iloc[first_target]) for key in action_keys
        }
        errors.append(_flat_row(predicted, action_keys) - _flat_row(ground_truth, action_keys))
        previous = timeline.get(first_target)
        if previous is not None:
            for side in ("left_eef", "right_eef"):
                position_seams.append(
                    float(np.linalg.norm(predicted[side][:3] - previous[side][:3]))
                )
                orientation_seams.append(_so3_error(predicted[side][3:], previous[side][3:]))
            for side in ("left_gripper", "right_gripper"):
                gripper_seams.append(float(np.max(np.abs(predicted[side] - previous[side]))))

        completion = source + delay - 1
        timeline = {
            source + index: _action_row(action, index)
            for index in range(ACTION_HORIZON)
            if source + index > completion
        }
        covered_targets.update(timeline)
        source += delay
        call_index += 1

    if not errors:
        return {
            "status": "no_admitted_predictions",
            "holds": holds,
            "rejections": rejections,
        }
    error = np.stack(errors)
    requested_targets = max(1, min(config.replay_steps, len(trajectory)))
    covered_within_request = sum(0 <= target < requested_targets for target in covered_targets)
    dynamics_array = np.asarray(dynamics)
    prefix_error_array = np.asarray(prefix_errors) if prefix_errors else None
    return {
        "status": "ok",
        "calls": call_index,
        "first_executable_row_mse": float(np.mean(error**2)),
        "first_executable_row_mae": float(np.mean(np.abs(error))),
        "target_timestep_coverage": covered_within_request / requested_targets,
        "hold_rate": holds / requested_targets,
        "rejections": rejections,
        "prefix_position_error_m": (
            _summary(prefix_error_array[:, 0].tolist()) if prefix_error_array is not None else None
        ),
        "prefix_orientation_error_rad": (
            _summary(prefix_error_array[:, 1].tolist()) if prefix_error_array is not None else None
        ),
        "prefix_gripper_error": (
            _summary(prefix_error_array[:, 2].tolist()) if prefix_error_array is not None else None
        ),
        "position_seam": _summary(position_seams) if position_seams else None,
        "orientation_seam_rad": _summary(orientation_seams) if orientation_seams else None,
        "gripper_seam": _summary(gripper_seams) if gripper_seams else None,
        "max_position_step_m": _summary(dynamics_array[:, 0].tolist()),
        "max_orientation_step_rad": _summary(dynamics_array[:, 1].tolist()),
        "max_gripper_step": _summary(dynamics_array[:, 2].tolist()),
        "max_position_second_difference_m": _summary(dynamics_array[:, 3].tolist()),
        "max_gripper_second_difference": _summary(dynamics_array[:, 4].tolist()),
    }


def benchmark_replay(
    policy: Gr00tPolicy,
    config: BenchmarkConfig,
    checkpoint_config: dict[str, Any],
) -> dict[str, Any] | None:
    if config.dataset_path is None:
        return None
    embodiment = EmbodimentTag.resolve(config.embodiment_tag)
    loader = LeRobotEpisodeLoader(
        dataset_path=config.dataset_path,
        modality_configs=policy.get_modality_config(),
    )
    results: dict[str, Any] = {}
    for mode in config.modes:
        reason = _unsupported_reason(mode, checkpoint_config, max(config.delay_trace))
        if reason is not None:
            results[mode] = {"status": "unsupported", "reason": reason}
            continue
        mode_results = {}
        for trajectory_id in config.trajectory_ids:
            if not 0 <= trajectory_id < len(loader):
                mode_results[str(trajectory_id)] = {"status": "trajectory_out_of_range"}
                continue
            mode_results[str(trajectory_id)] = replay_trajectory(
                policy,
                loader,
                trajectory_id,
                embodiment,
                mode,
                config,
            )
        results[mode] = {"status": "ok", "trajectories": mode_results}
    return results


def _text_summary(report: dict[str, Any]) -> str:
    lines = ["Industrial Next RTC benchmark", f"checkpoint: {report['model_path']}"]
    latency = report.get("latency")
    if latency is not None:
        for mode, result in latency.items():
            if result["status"] != "ok":
                lines.append(f"{mode}: unsupported ({result['reason']})")
                continue
            total = result["total_ms"]
            lines.append(
                f"{mode}: total p50={total['p50']:.2f} ms p95={total['p95']:.2f} ms "
                f"p99={total['p99']:.2f} ms max={total['max']:.2f} ms "
                f"p99_steps={result['p99_committed_steps_sizing_proxy']}"
            )
    replay = report.get("replay")
    if replay is not None:
        lines.append("held-out replay:")
        for mode, result in replay.items():
            if result["status"] != "ok":
                lines.append(f"{mode}: unsupported ({result['reason']})")
                continue
            for trajectory_id, trajectory in result["trajectories"].items():
                if trajectory["status"] != "ok":
                    lines.append(f"{mode}/trajectory-{trajectory_id}: {trajectory['status']}")
                    continue
                position_seam = trajectory["position_seam"]
                orientation_seam = trajectory["orientation_seam_rad"]
                gripper_seam = trajectory["gripper_seam"]
                lines.append(
                    f"{mode}/trajectory-{trajectory_id}: "
                    f"mse={trajectory['first_executable_row_mse']:.6g} "
                    f"mae={trajectory['first_executable_row_mae']:.6g} "
                    f"coverage={trajectory['target_timestep_coverage']:.4f} "
                    f"hold={trajectory['hold_rate']:.4f} "
                    f"rejections={trajectory['rejections']} "
                    f"prefix_position_max="
                    f"{trajectory['prefix_position_error_m']['max'] if trajectory['prefix_position_error_m'] else float('nan'):.6g} "
                    f"prefix_orientation_max="
                    f"{trajectory['prefix_orientation_error_rad']['max'] if trajectory['prefix_orientation_error_rad'] else float('nan'):.6g} "
                    f"prefix_gripper_max="
                    f"{trajectory['prefix_gripper_error']['max'] if trajectory['prefix_gripper_error'] else float('nan'):.6g} "
                    f"seam_position_p99="
                    f"{position_seam['p99'] if position_seam else float('nan'):.6g} "
                    f"seam_orientation_p99="
                    f"{orientation_seam['p99'] if orientation_seam else float('nan'):.6g} "
                    f"seam_gripper_p99="
                    f"{gripper_seam['p99'] if gripper_seam else float('nan'):.6g} "
                    f"step_position_max={trajectory['max_position_step_m']['max']:.6g} "
                    f"step_orientation_max={trajectory['max_orientation_step_rad']['max']:.6g} "
                    f"step_gripper_max={trajectory['max_gripper_step']['max']:.6g} "
                    f"second_position_max="
                    f"{trajectory['max_position_second_difference_m']['max']:.6g} "
                    f"second_gripper_max="
                    f"{trajectory['max_gripper_second_difference']['max']:.6g}"
                )
    return "\n".join(lines) + "\n"


def main(config: BenchmarkConfig) -> None:
    model_path = Path(config.model_path).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"benchmark output directory already exists: {output_dir}")
    checkpoint_config = _checkpoint_config(model_path)
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=str(model_path),
        device=config.device,
        strict=True,
    )
    report = {
        "schema_version": 1,
        "model_path": str(model_path),
        "provenance": build_service_provenance(model_path),
        "checkpoint_rtc_training_max_prefix_steps": checkpoint_config.get(
            "rtc_training_max_prefix_steps", 0
        ),
        "config": {
            key: value
            for key, value in config.__dict__.items()
            if key not in {"model_path", "output_dir"}
        },
        "latency": (
            benchmark_checkpoint_latency(policy, config, checkpoint_config)
            if config.run_latency
            else None
        ),
        "replay": (
            benchmark_replay(policy, config, checkpoint_config) if config.run_replay else None
        ),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.txt").write_text(_text_summary(report), encoding="utf-8")


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))
