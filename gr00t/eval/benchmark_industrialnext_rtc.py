# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible checkpoint-latency and held-out RTC replay for Industrial Next."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.industrialnext_replay import replay_production
from gr00t.eval.run_gr00t_industrialnext_server import build_service_provenance
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.industrialnext import ACTION_HORIZON, load_industrialnext_profile
from gr00t.policy.industrialnext.async_server import IndustrialNextServingConfig
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
    modes: list[str] = field(default_factory=lambda: ["off"])
    profile_config: str = "configs/embodiments/semihumanoid.yaml"
    action_offset: int = 2
    ensemble_strategy: str = "temporal_exponential"
    ensemble_coeff: float = 0.1
    max_ensemble_chunks: int = 3
    chunk_transition_frames: int = 4
    run_ablations: bool = True
    min_usable_action_steps: int = 1
    max_action_lateness_s: float = 0.04
    max_control_clock_drift_s: float = 0.1
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
    trajectory_ids: list[int] | None = None
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
        if self.control_hz != 50.0:
            raise ValueError("control_hz must be exactly 50.0 for production replay")
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
        if not self.delay_trace or any(
            isinstance(delay, bool) or not isinstance(delay, int) or delay < 1
            for delay in self.delay_trace
        ):
            raise ValueError("delay_trace must contain positive committed-step lengths")
        if (
            isinstance(self.replay_steps, bool)
            or not isinstance(self.replay_steps, int)
            or self.replay_steps < 1
        ):
            raise ValueError("replay_steps must be a positive integer")
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
    profile = load_industrialnext_profile(config.profile_config)
    profile.assert_policy_contract(policy)
    observation = profile.build_synthetic_model_observation(config.task_text)
    torch.manual_seed(config.seeds[0])
    baseline_action, _ = policy.get_action(observation, options={"rtc_mode": "off"})
    results: dict[str, Any] = {}
    for mode in config.modes:
        reason = _unsupported_reason(mode, checkpoint_config, config.prefix_steps)
        if mode not in profile.supported_rtc_modes:
            reason = "RTC mode is not supported by the serving profile"
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


def replay_trajectory(policy, loader, trajectory_id, embodiment, mode, config):
    profile = load_industrialnext_profile(config.profile_config)
    profile.assert_policy_contract(policy)
    serving = IndustrialNextServingConfig(
        action_horizon=profile.action_horizon,
        action_offset=config.action_offset,
        ensemble_strategy=config.ensemble_strategy,
        ensemble_coeff=config.ensemble_coeff,
        max_ensemble_chunks=config.max_ensemble_chunks,
        chunk_transition_frames=config.chunk_transition_frames,
        min_usable_action_steps=config.min_usable_action_steps,
        max_action_lateness_s=config.max_action_lateness_s,
        max_control_clock_drift_s=config.max_control_clock_drift_s,
        rtc_mode=mode,
        rtc_initial_frozen_steps=config.prefix_steps,
        rtc_max_prefix_steps=config.prefix_steps,
        rtc_native_overlap_steps=config.native_overlap_steps,
        rtc_min_new_tail_steps=config.min_new_tail_steps,
        rtc_ramp_rate=config.rtc_ramp_rate,
        rtc_position_tolerance=config.rtc_position_tolerance,
        rtc_orientation_tolerance_rad=config.rtc_orientation_tolerance_rad,
        rtc_gripper_tolerance=config.rtc_gripper_tolerance,
        stats_log_interval_steps=0,
    )
    return asyncio.run(
        replay_production(policy, loader, trajectory_id, embodiment, profile, serving, config)
    )


def benchmark_replay(
    policy: Gr00tPolicy,
    config: BenchmarkConfig,
    checkpoint_config: dict[str, Any],
) -> dict[str, Any] | None:
    if config.dataset_path is None:
        return None
    embodiment = EmbodimentTag.resolve(config.embodiment_tag)
    root = Path(config.dataset_path).expanduser().resolve()
    paths = (
        [root]
        if (root / "meta/info.json").is_file()
        else sorted(p.parent.parent for p in root.glob("*_val/meta/info.json"))
    )
    if not paths:
        raise ValueError("dataset_path must contain LeRobot metadata or explicit *_val datasets")
    results: dict[str, Any] = {}
    profile = load_industrialnext_profile(config.profile_config)
    recipes = [("configured", config)]
    if config.run_ablations:
        recipes = [
            (
                name,
                replace(
                    config,
                    action_offset=offset,
                    ensemble_strategy=strategy,
                    chunk_transition_frames=frames,
                ),
            )
            for name, offset, strategy, frames in (
                ("control", 0, "latest_only", 0),
                ("offset_only", 2, "latest_only", 0),
                ("ensemble_only", 0, "temporal_exponential", 0),
                ("transition_only", 0, "latest_only", 4),
                ("postprocessing_offset0", 0, "temporal_exponential", 4),
                ("default_offset2", 2, "temporal_exponential", 4),
            )
        ]
    for mode in config.modes:
        reason = _unsupported_reason(mode, checkpoint_config, config.prefix_steps)
        if mode not in profile.supported_rtc_modes:
            reason = "RTC mode is not supported by the serving profile"
        if reason is not None:
            results[mode] = {"status": "unsupported", "reason": reason}
            continue
        for dataset_path in paths:
            loader = LeRobotEpisodeLoader(
                dataset_path=str(dataset_path), modality_configs=policy.get_modality_config()
            )
            for name, recipe in recipes if mode == "off" else [("configured", config)]:
                trajectories = {}
                ids = range(len(loader)) if config.trajectory_ids is None else config.trajectory_ids
                for trajectory_id in ids:
                    if not 0 <= trajectory_id < len(loader):
                        raise ValueError(f"trajectory index out of range: {trajectory_id}")
                    trajectories[str(trajectory_id)] = replay_trajectory(
                        policy, loader, trajectory_id, embodiment, mode, recipe
                    )
                results[f"{mode}/{name}/{dataset_path.name}"] = {
                    "status": "ok",
                    "trajectories": trajectories,
                    "dataset_path": str(dataset_path),
                    "dataset_info_sha256": hashlib.sha256(
                        (dataset_path / "meta/info.json").read_bytes()
                    ).hexdigest(),
                    "recipe": {
                        "action_offset": recipe.action_offset,
                        "ensemble_strategy": recipe.ensemble_strategy,
                        "ensemble_coeff": recipe.ensemble_coeff,
                        "max_ensemble_chunks": recipe.max_ensemble_chunks,
                        "chunk_transition_frames": recipe.chunk_transition_frames,
                    },
                }
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
                lines.append(
                    f"{mode}/trajectory-{trajectory_id}: coverage={trajectory['coverage']:.4f} "
                    f"null_rate={trajectory['null_rate']:.4f} "
                    f"physical_errors={trajectory['physical_errors']}"
                )
    return "\n".join(lines) + "\n"


def _validation_identity(config: BenchmarkConfig, model_path: Path) -> dict[str, Any]:
    """Bind small configuration/source files once, outside the request path."""
    root = Path(__file__).resolve().parents[2]
    files = [Path(config.profile_config).expanduser().resolve(), model_path / "config.json"]
    files.extend(model_path.glob("*processor*.json"))
    files.extend(model_path.glob("*stat*.json"))
    files.extend(
        root / path
        for path in (
            "gr00t/policy/industrialnext/async_server.py",
            "gr00t/policy/industrialnext/execution.py",
            "gr00t/policy/industrialnext/profile_config.py",
            "gr00t/policy/industrialnext/adapter.py",
            "gr00t/policy/gr00t_policy.py",
            "gr00t/eval/industrialnext_replay.py",
            "gr00t/eval/benchmark_industrialnext_rtc.py",
        )
    )
    return {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        ),
        "files_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
            if path.is_file()
        },
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": config.device,
        "weights_hashed": False,
    }


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
        "schema_version": 2,
        "validation_identity": _validation_identity(config, model_path),
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
