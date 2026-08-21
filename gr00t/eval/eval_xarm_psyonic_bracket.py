# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate an xArm/PSYONIC bracket checkpoint on immutable held-out trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import tyro


EXPECTED_DATASETS = ("xarm_psyonic_val", "manus_vive_val")
EXPECTED_ACTION_KEYS = ("right_eef", "right_hand")
EXPECTED_ACTION_HORIZON = 40


@dataclass(frozen=True)
class EvalConfig:
    model_path: str
    dataset_root: str
    trajectory_manifest: str
    output_dir: str
    embodiment_tag: str = "NEW_EMBODIMENT"
    device: str = "cuda:0"
    seed: int = 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_model_hash(model_path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash deployable model/processor content while excluding optimizer state."""
    included: list[Path] = []
    for path in sorted(item for item in model_path.rglob("*") if item.is_file()):
        relative = path.relative_to(model_path)
        if path.suffix == ".safetensors" or relative.parts[0] in {"experiment_cfg", "processor"}:
            included.append(path)
        elif relative.as_posix() in {
            "config.json",
            "embodiment_id.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "processor_config.json",
            "statistics.json",
        }:
            included.append(path)
    if not included or not any(path.suffix == ".safetensors" for path in included):
        raise FileNotFoundError(f"no deployable model weights found under {model_path}")
    inventory = [
        {
            "path": path.relative_to(model_path).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in included
    ]
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), inventory


def rotation_matrix_from_row_rot6d(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64).reshape(2, 3)
    first_norm = np.linalg.norm(rows[0])
    if not np.isfinite(first_norm) or first_norm <= 1e-12:
        raise ValueError("invalid first rot6d row")
    first = rows[0] / first_norm
    second = rows[1] - np.dot(first, rows[1]) * first
    second_norm = np.linalg.norm(second)
    if not np.isfinite(second_norm) or second_norm <= 1e-12:
        raise ValueError("invalid second rot6d row")
    second /= second_norm
    return np.stack((first, second, np.cross(first, second)), axis=0)


def rotation_errors_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape or prediction.shape[-1] != 6:
        raise ValueError("rotation arrays must have matching (..., 6) shapes")
    result = np.empty(prediction.shape[:-1], dtype=np.float64)
    for index in np.ndindex(result.shape):
        pred_matrix = rotation_matrix_from_row_rot6d(prediction[index])
        target_matrix = rotation_matrix_from_row_rot6d(target[index])
        cosine = np.clip((np.trace(pred_matrix @ target_matrix.T) - 1.0) / 2.0, -1.0, 1.0)
        result[index] = math.degrees(math.acos(float(cosine)))
    return result


def _summary(values: list[float], *, rmse: bool = False) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("metric input must be non-empty and finite")
    result = {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }
    if rmse:
        result["rmse"] = float(np.sqrt(np.mean(np.square(array))))
    return result


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot summarize an empty sample list")
    finite_count = sum(bool(sample["finite"]) for sample in samples)
    valid = [sample for sample in samples if sample["finite"]]
    if not valid:
        return {"samples": len(samples), "finite_output_rate": 0.0}

    hand_error = np.concatenate([sample["hand_abs_error"] for sample in valid], axis=0)
    return {
        "samples": len(samples),
        "finite_output_rate": finite_count / len(samples),
        "translation_error_mm": _summary(
            [value for sample in valid for value in sample["translation_error_mm"]], rmse=True
        ),
        "rotation_error_deg": _summary(
            [value for sample in valid for value in sample["rotation_error_deg"]]
        ),
        "hand_abs_error_rad": _summary(hand_error.reshape(-1).tolist()),
        "hand_per_joint_mae_rad": np.mean(hand_error, axis=0).tolist(),
        "hand_per_joint_p99_rad": np.percentile(hand_error, 99, axis=0).tolist(),
        "max_adjacent_position_step_mm": _summary(
            [sample["max_adjacent_position_step_mm"] for sample in valid]
        ),
        "max_adjacent_rotation_step_deg": _summary(
            [sample["max_adjacent_rotation_step_deg"] for sample in valid]
        ),
        "max_adjacent_hand_step_rad": _summary(
            [sample["max_adjacent_hand_step_rad"] for sample in valid]
        ),
        "first_step_position_seam_mm": _summary(
            [sample["first_step_position_seam_mm"] for sample in valid]
        ),
        "first_step_rotation_seam_deg": _summary(
            [sample["first_step_rotation_seam_deg"] for sample in valid]
        ),
        "first_step_hand_seam_rad": _summary(
            [sample["first_step_hand_seam_rad"] for sample in valid]
        ),
    }


def score_action(
    action: dict[str, np.ndarray], target: dict[str, np.ndarray], state: dict[str, np.ndarray]
) -> dict[str, Any]:
    if set(action) != set(EXPECTED_ACTION_KEYS):
        raise ValueError(f"unexpected action keys: {sorted(action)}")
    eef = np.asarray(action["right_eef"], dtype=np.float64)
    hand = np.asarray(action["right_hand"], dtype=np.float64)
    target_eef = np.asarray(target["right_eef"], dtype=np.float64)
    target_hand = np.asarray(target["right_hand"], dtype=np.float64)
    if eef.shape != (EXPECTED_ACTION_HORIZON, 9) or hand.shape != (
        EXPECTED_ACTION_HORIZON,
        6,
    ):
        raise ValueError(f"wrong action shapes: right_eef={eef.shape}, right_hand={hand.shape}")
    finite = bool(np.isfinite(eef).all() and np.isfinite(hand).all())
    if not finite:
        return {"finite": False}

    current_eef = np.asarray(state["right_eef"], dtype=np.float64)[-1]
    current_hand = np.asarray(state["right_hand"], dtype=np.float64)[-1]
    translation_error = np.linalg.norm(eef[:, :3] - target_eef[:, :3], axis=1) * 1000.0
    rotation_error = rotation_errors_deg(eef[:, 3:], target_eef[:, 3:])
    hand_error = np.abs(hand - target_hand)
    adjacent_rotation = rotation_errors_deg(eef[1:, 3:], eef[:-1, 3:])
    return {
        "finite": True,
        "translation_error_mm": translation_error.tolist(),
        "rotation_error_deg": rotation_error.tolist(),
        "hand_abs_error": hand_error,
        "max_adjacent_position_step_mm": float(
            np.linalg.norm(np.diff(eef[:, :3], axis=0), axis=1).max() * 1000.0
        ),
        "max_adjacent_rotation_step_deg": float(adjacent_rotation.max()),
        "max_adjacent_hand_step_rad": float(np.abs(np.diff(hand, axis=0)).max()),
        "first_step_position_seam_mm": float(np.linalg.norm(eef[0, :3] - current_eef[:3]) * 1000.0),
        "first_step_rotation_seam_deg": float(
            rotation_errors_deg(eef[None, 0, 3:], current_eef[None, 3:])[0]
        ),
        "first_step_hand_seam_rad": float(np.abs(hand[0] - current_hand).max()),
    }


def _persistence_action(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.repeat(np.asarray(state[key], dtype=np.float32)[-1:], EXPECTED_ACTION_HORIZON, 0)
        for key in EXPECTED_ACTION_KEYS
    }


def _model_observation(point: Any, modality: dict[str, Any]) -> dict[str, Any]:
    flat = {f"state.{key}": value for key, value in point.states.items()}
    flat.update({f"video.{key}": np.asarray(value) for key, value in point.images.items()})
    for key in modality["language"].modality_keys:
        flat[key] = point.text
    return parse_observation_gr00t(flat, modality)


def _sample_indices(length: int, count: int) -> list[int]:
    valid_starts = length - EXPECTED_ACTION_HORIZON + 1
    if valid_starts <= 0:
        raise ValueError(f"trajectory length {length} is shorter than the action horizon")
    return sorted(
        set(np.linspace(0, valid_starts - 1, min(count, valid_starts), dtype=int).tolist())
    )


def _validate_modality(modality: dict[str, Any]) -> None:
    expected = {
        "video": ["front", "wrist"],
        "state": list(EXPECTED_ACTION_KEYS),
        "action": list(EXPECTED_ACTION_KEYS),
    }
    for name, keys in expected.items():
        if modality[name].modality_keys != keys:
            raise ValueError(f"{name} keys differ: {modality[name].modality_keys} != {keys}")
    if modality["action"].delta_indices != list(range(EXPECTED_ACTION_HORIZON)):
        raise ValueError("checkpoint action horizon is not 40 contiguous steps")


def _load_trajectory_manifest(path: Path) -> tuple[dict[str, list[int]], int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("trajectory manifest schema_version must be 1")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(EXPECTED_DATASETS):
        raise ValueError(f"trajectory manifest must contain exactly {EXPECTED_DATASETS}")
    for name, indices in datasets.items():
        if (
            not isinstance(indices, list)
            or not indices
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
        ):
            raise ValueError(f"invalid trajectory indices for {name}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate trajectory indices for {name}")
    count = raw.get("samples_per_trajectory")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("samples_per_trajectory must be a positive integer")
    return datasets, count, _sha256_file(path)


def evaluate(config: EvalConfig) -> dict[str, Any]:
    model_path = Path(config.model_path).expanduser().resolve()
    dataset_root = Path(config.dataset_root).expanduser().resolve()
    manifest_path = Path(config.trajectory_manifest).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"evaluation output directory already exists: {output_dir}")
    trajectories, samples_per_trajectory, trajectory_hash = _load_trajectory_manifest(manifest_path)
    frozen_manifest = dataset_root / "_frozen_corpus_manifest.json"
    if not frozen_manifest.is_file():
        raise FileNotFoundError(f"frozen corpus manifest is missing: {frozen_manifest}")
    checkpoint_hash, checkpoint_inventory = checkpoint_model_hash(model_path)

    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=str(model_path),
        device=config.device,
        strict=True,
    )
    modality = policy.get_modality_config()
    _validate_modality(modality)
    embodiment = EmbodimentTag.resolve(config.embodiment_tag)
    model_samples: dict[str, list[dict[str, Any]]] = {}
    baseline_samples: dict[str, list[dict[str, Any]]] = {}
    expert_samples: dict[str, list[dict[str, Any]]] = {}
    evaluated_starts: dict[str, dict[str, list[int]]] = {}

    for dataset_name in EXPECTED_DATASETS:
        loader = LeRobotEpisodeLoader(dataset_root / dataset_name, modality)
        model_samples[dataset_name] = []
        baseline_samples[dataset_name] = []
        expert_samples[dataset_name] = []
        evaluated_starts[dataset_name] = {}
        for trajectory_index in trajectories[dataset_name]:
            if trajectory_index >= len(loader):
                raise IndexError(
                    f"trajectory {trajectory_index} is out of range for {dataset_name} ({len(loader)})"
                )
            episode = loader[trajectory_index]
            starts = _sample_indices(len(episode), samples_per_trajectory)
            evaluated_starts[dataset_name][str(trajectory_index)] = starts
            for step_index in starts:
                point = extract_step_data(episode, step_index, modality, embodiment)
                target = {key: np.asarray(value) for key, value in point.actions.items()}
                baseline_samples[dataset_name].append(
                    score_action(_persistence_action(point.states), target, point.states)
                )
                expert_samples[dataset_name].append(score_action(target, target, point.states))
                torch.manual_seed(config.seed + trajectory_index * 100_000 + step_index)
                action, _ = policy.get_action(_model_observation(point, modality))
                model_action = {key: np.asarray(value)[0] for key, value in action.items()}
                model_samples[dataset_name].append(score_action(model_action, target, point.states))

    def summaries(source: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        per_dataset = {name: summarize_samples(source[name]) for name in EXPECTED_DATASETS}
        per_dataset["aggregate"] = summarize_samples(
            [sample for name in EXPECTED_DATASETS for sample in source[name]]
        )
        return per_dataset

    report = {
        "schema_version": 2,
        "model_path": str(model_path),
        "checkpoint_model_sha256": checkpoint_hash,
        "checkpoint_hash_inventory": checkpoint_inventory,
        "dataset_root": str(dataset_root),
        "frozen_corpus_manifest_sha256": _sha256_file(frozen_manifest),
        "trajectory_manifest": str(manifest_path),
        "trajectory_manifest_sha256": trajectory_hash,
        "evaluated_starts": evaluated_starts,
        "seed": config.seed,
        "model": summaries(model_samples),
        "persistence_baseline": summaries(baseline_samples),
        "expert_target": summaries(expert_samples),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(config: EvalConfig) -> None:
    report = evaluate(config)
    for dataset_name in (*EXPECTED_DATASETS, "aggregate"):
        model = report["model"][dataset_name]
        baseline = report["persistence_baseline"][dataset_name]
        print(
            f"{dataset_name}: finite={model['finite_output_rate']:.3f} "
            f"translation={model['translation_error_mm']['mean']:.3f}/"
            f"{baseline['translation_error_mm']['mean']:.3f} mm "
            f"rotation={model['rotation_error_deg']['mean']:.3f}/"
            f"{baseline['rotation_error_deg']['mean']:.3f} deg "
            f"hand={model['hand_abs_error_rad']['mean']:.5f}/"
            f"{baseline['hand_abs_error_rad']['mean']:.5f} rad"
        )


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
