# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from gr00t.eval.eval_xarm_psyonic_bracket import (
    _load_trajectory_manifest,
    rotation_errors_deg,
    score_action,
    summarize_samples,
)
import numpy as np
import pytest


IDENTITY_ROT6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _eef(position: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [position, np.repeat(IDENTITY_ROT6D[None], len(position), axis=0)], axis=1
    )


def test_rotation_errors_use_geodesic_degrees():
    quarter_turn = np.array([0, -1, 0, 1, 0, 0], dtype=np.float32)
    error = rotation_errors_deg(quarter_turn[None], IDENTITY_ROT6D[None])
    assert error == pytest.approx([90.0])


def test_score_action_reports_physical_errors_and_dynamics():
    target_position = np.zeros((40, 3), dtype=np.float32)
    predicted_position = np.zeros((40, 3), dtype=np.float32)
    predicted_position[:, 0] = 0.001
    target = {"right_eef": _eef(target_position), "right_hand": np.zeros((40, 6))}
    action = {"right_eef": _eef(predicted_position), "right_hand": np.full((40, 6), 0.2)}
    state = {
        "right_eef": _eef(np.zeros((1, 3), dtype=np.float32)),
        "right_hand": np.zeros((1, 6), dtype=np.float32),
    }
    sample = score_action(action, target, state)
    summary = summarize_samples([sample])
    assert summary["finite_output_rate"] == 1.0
    assert summary["translation_error_mm"]["mean"] == pytest.approx(1.0)
    assert summary["rotation_error_deg"]["max"] == pytest.approx(0.0)
    assert summary["hand_per_joint_mae_rad"] == pytest.approx([0.2] * 6)
    assert summary["first_step_position_seam_mm"]["max"] == pytest.approx(1.0)


def test_manifest_requires_both_sources_and_unique_indices(tmp_path: Path):
    path = tmp_path / "trajectories.json"
    path.write_text(
        '{"schema_version":1,"datasets":{"xarm_psyonic_val":[0],'
        '"manus_vive_val":[0,1]},"samples_per_trajectory":4}\n'
    )
    datasets, count, digest = _load_trajectory_manifest(path)
    assert datasets["manus_vive_val"] == [0, 1]
    assert count == 4
    assert len(digest) == 64

    path.write_text(
        '{"schema_version":1,"datasets":{"xarm_psyonic_val":[0,0],'
        '"manus_vive_val":[0]},"samples_per_trajectory":4}\n'
    )
    with pytest.raises(ValueError, match="duplicate"):
        _load_trajectory_manifest(path)
