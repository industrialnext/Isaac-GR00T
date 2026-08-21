# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strictly map recorded xArm/PSYONIC observations through a GR00T policy server.

This utility is for loopback and recorded-data validation only. It does not publish robot
commands, implement safety clamps, or authorize motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.rot6d import rot6d_groot_to_source, rot6d_source_to_groot
from gr00t.data.types import ActionFormat, ActionRepresentation, ActionType
from gr00t.policy.server_client import PolicyClient
import numpy as np
import tyro


ACTION_HORIZON = 40
OBSERVATION_KEYS = {
    "right_arm_pose_pos",
    "right_arm_pose_rot",
    "right_hand",
    "static_center_rgb",
    "eoat_right_bottom_rgb",
    "task_text",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_modality_contract(modality: dict[str, Any]) -> None:
    expected = {
        "video": ([0], ["front", "wrist"]),
        "state": ([0], ["right_eef", "right_hand"]),
        "action": (list(range(ACTION_HORIZON)), ["right_eef", "right_hand"]),
        "language": ([0], ["annotation.human.task_description"]),
    }
    if set(modality) != set(expected):
        raise ValueError(f"modality names differ: {sorted(modality)} != {sorted(expected)}")
    for name, (delta_indices, keys) in expected.items():
        config = modality[name]
        if config.delta_indices != delta_indices or config.modality_keys != keys:
            raise ValueError(
                f"{name} contract differs: delta={config.delta_indices}, keys={config.modality_keys}"
            )
    action_configs = modality["action"].action_configs
    if action_configs is None or len(action_configs) != 2:
        raise ValueError("action contract must contain two action configs")
    expected_actions = (
        (ActionRepresentation.RELATIVE, ActionType.EEF, ActionFormat.XYZ_ROT6D, "right_eef"),
        (ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT, None),
    )
    actual_actions = tuple(
        (config.rep, config.type, config.format, config.state_key) for config in action_configs
    )
    if actual_actions != expected_actions:
        raise ValueError(f"action representations differ: {actual_actions} != {expected_actions}")


class XarmPsyonicBracketAdapter:
    """Translate the Industrial Next wire surface to and from GR00T conventions."""

    def __init__(self, policy_client: PolicyClient):
        self.policy = policy_client
        self.modality = policy_client.get_modality_config()
        validate_modality_contract(self.modality)

    @staticmethod
    def observation_to_model(observation: dict[str, Any]) -> dict[str, Any]:
        if set(observation) != OBSERVATION_KEYS:
            missing = sorted(OBSERVATION_KEYS - set(observation))
            extra = sorted(set(observation) - OBSERVATION_KEYS)
            raise ValueError(f"observation keys differ: missing={missing}, extra={extra}")
        position = np.asarray(observation["right_arm_pose_pos"], dtype=np.float32)
        rotation = np.asarray(observation["right_arm_pose_rot"], dtype=np.float32)
        hand = np.asarray(observation["right_hand"], dtype=np.float32)
        if position.shape != (3,) or rotation.shape != (6,) or hand.shape != (6,):
            raise ValueError(
                f"state shapes differ: position={position.shape}, rotation={rotation.shape}, "
                f"hand={hand.shape}"
            )
        if not all(np.isfinite(value).all() for value in (position, rotation, hand)):
            raise ValueError("state contains non-finite values")
        images = {}
        for wire_key, model_key in (
            ("static_center_rgb", "front"),
            ("eoat_right_bottom_rgb", "wrist"),
        ):
            image = np.asarray(observation[wire_key])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{wire_key} must be an HxWx3 uint8 image, got {image.shape}")
            images[model_key] = image[None, None, ...]
        task_text = observation["task_text"]
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError("task_text must be a non-empty string")
        right_eef = np.concatenate((position, rot6d_source_to_groot(rotation))).astype(np.float32)
        return {
            "video": images,
            "state": {
                "right_eef": right_eef[None, None, :],
                "right_hand": hand[None, None, :],
            },
            "language": {"annotation.human.task_description": [[task_text]]},
        }

    @staticmethod
    def action_from_model(action: dict[str, Any]) -> dict[str, np.ndarray]:
        if set(action) != {"right_eef", "right_hand"}:
            raise ValueError(f"action keys differ: {sorted(action)}")
        eef = np.asarray(action["right_eef"], dtype=np.float32)
        hand = np.asarray(action["right_hand"], dtype=np.float32)
        if eef.shape != (1, ACTION_HORIZON, 9) or hand.shape != (1, ACTION_HORIZON, 6):
            raise ValueError(
                f"action shapes differ: right_eef={eef.shape}, right_hand={hand.shape}"
            )
        if not np.isfinite(eef).all() or not np.isfinite(hand).all():
            raise ValueError("action contains non-finite values")
        return {
            "right_arm_pose_pos": eef[0, :, :3].copy(),
            "right_arm_pose_rot": rot6d_groot_to_source(eef[0, :, 3:]).astype(np.float32),
            "right_hand": hand[0].copy(),
        }

    def get_action(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        action, info = self.policy.get_action(self.observation_to_model(observation))
        return self.action_from_model(action), info


@dataclass(frozen=True)
class ClientConfig:
    dataset_root: str
    trajectory_manifest: str
    output_dir: str
    host: str = "127.0.0.1"
    port: int = 5555
    timeout_ms: int = 120_000


def _load_manifest(path: Path) -> tuple[dict[str, list[int]], int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    datasets = raw.get("datasets")
    expected = {"xarm_psyonic_val", "manus_vive_val"}
    if raw.get("schema_version") != 1 or not isinstance(datasets, dict):
        raise ValueError("invalid trajectory manifest")
    if set(datasets) != expected:
        raise ValueError(f"trajectory datasets differ: {sorted(datasets)} != {sorted(expected)}")
    count = raw.get("samples_per_trajectory")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("samples_per_trajectory must be positive")
    return datasets, count


def _sample_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError("trajectory must be non-empty")
    return sorted(set(np.linspace(0, length - 1, min(length, count), dtype=int).tolist()))


def _source_observation(point: Any) -> dict[str, Any]:
    eef = np.asarray(point.states["right_eef"])[-1]
    return {
        "right_arm_pose_pos": eef[:3],
        "right_arm_pose_rot": rot6d_groot_to_source(eef[3:]).astype(np.float32),
        "right_hand": np.asarray(point.states["right_hand"])[-1],
        "static_center_rgb": np.asarray(point.images["front"])[-1],
        "eoat_right_bottom_rgb": np.asarray(point.images["wrist"])[-1],
        "task_text": point.text,
    }


def run_recorded_loopback(config: ClientConfig) -> dict[str, Any]:
    dataset_root = Path(config.dataset_root).expanduser().resolve()
    manifest_path = Path(config.trajectory_manifest).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"loopback output directory already exists: {output_dir}")
    trajectories, sample_count = _load_manifest(manifest_path)
    records = []
    errors = []
    with PolicyClient(
        host=config.host,
        port=config.port,
        timeout_ms=config.timeout_ms,
        strict=False,
    ) as client:
        if not client.ping():
            raise ConnectionError(f"policy server did not answer at {config.host}:{config.port}")
        adapter = XarmPsyonicBracketAdapter(client)
        modality = adapter.modality
        for dataset_name, trajectory_indices in trajectories.items():
            loader = LeRobotEpisodeLoader(dataset_root / dataset_name, modality)
            for trajectory_index in trajectory_indices:
                if trajectory_index >= len(loader):
                    raise IndexError(
                        f"trajectory {trajectory_index} is out of range for {dataset_name}"
                    )
                episode = loader[trajectory_index]
                for step_index in _sample_indices(len(episode), sample_count):
                    point = extract_step_data(
                        episode,
                        step_index,
                        {name: value for name, value in modality.items() if name != "action"},
                        EmbodimentTag.NEW_EMBODIMENT,
                    )
                    started_at = time.perf_counter()
                    try:
                        action, info = adapter.get_action(_source_observation(point))
                        records.append(
                            {
                                "dataset": dataset_name,
                                "trajectory_index": trajectory_index,
                                "step_index": step_index,
                                "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
                                "server_info": info,
                                "shapes": {key: list(value.shape) for key, value in action.items()},
                                "finite": all(
                                    np.isfinite(value).all() for value in action.values()
                                ),
                            }
                        )
                    except Exception as exc:
                        errors.append(
                            {
                                "dataset": dataset_name,
                                "trajectory_index": trajectory_index,
                                "step_index": step_index,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
    if errors or not records or not all(record["finite"] for record in records):
        raise RuntimeError(f"loopback failed: records={len(records)}, errors={errors}")
    latency = np.asarray([record["elapsed_ms"] for record in records], dtype=np.float64)
    report = {
        "schema_version": 1,
        "server": f"tcp://{config.host}:{config.port}",
        "dataset_root": str(dataset_root),
        "trajectory_manifest": str(manifest_path),
        "trajectory_manifest_sha256": _sha256_file(manifest_path),
        "requests": len(records),
        "protocol_errors": errors,
        "warmup_latency_ms": float(latency[0]),
        "steady_latency_ms": {
            "mean": float(latency[1:].mean()) if len(latency) > 1 else float(latency[0]),
            "median": float(np.median(latency[1:])) if len(latency) > 1 else float(latency[0]),
            "p99": float(np.percentile(latency[1:], 99)) if len(latency) > 1 else float(latency[0]),
            "max": float(latency[1:].max()) if len(latency) > 1 else float(latency[0]),
        },
        "records": records,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "loopback_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(config: ClientConfig) -> None:
    report = run_recorded_loopback(config)
    print(
        f"loopback passed: requests={report['requests']} "
        f"warmup={report['warmup_latency_ms']:.2f} ms "
        f"steady_p99={report['steady_latency_ms']['p99']:.2f} ms"
    )


if __name__ == "__main__":
    main(tyro.cli(ClientConfig))
