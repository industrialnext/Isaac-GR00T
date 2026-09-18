"""Projected supervision must survive statistics and per-step extraction."""

from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.stats import calculate_dataset_statistics
from gr00t.data.types import EmbodimentTag, ModalityConfig
import numpy as np
import pandas as pd
import pytest


def test_statistics_exclude_invalid_coordinates(tmp_path):
    path = tmp_path / "episode.parquet"
    pd.DataFrame(
        {
            "action": [np.array([2.0, 900.0]), np.array([800.0, 4.0]), np.array([6.0, 8.0])],
            "action_mask": [
                np.array([True, False]),
                np.array([False, True]),
                np.array([True, True]),
            ],
        }
    ).to_parquet(path)
    stats = calculate_dataset_statistics([path], ["action"])["action"]
    np.testing.assert_allclose(stats["mean"], [4.0, 6.0])
    np.testing.assert_allclose(stats["max"], [6.0, 8.0])


def test_statistics_reject_coordinates_without_supervision(tmp_path):
    path = tmp_path / "episode.parquet"
    pd.DataFrame({"action": [np.array([2.0])], "action_mask": [np.array([False])]}).to_parquet(path)
    with pytest.raises(ValueError, match="No valid supervision"):
        calculate_dataset_statistics([path], ["action"])


def test_extract_preserves_per_coordinate_action_masks():
    frame = pd.DataFrame(
        {
            "action.hand": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            "action_validity.hand": [np.array([True, False]), np.array([False, True])],
            "state.hand": [np.array([0.0, 0.0]), np.array([0.0, 0.0])],
            "language.task": ["pick", "pick"],
        }
    )
    modalities = {
        "action": ModalityConfig([0, 1], ["hand"]),
        "state": ModalityConfig([0], ["hand"]),
        "language": ModalityConfig([0], ["task"]),
    }
    step = extract_step_data(frame, 0, modalities, EmbodimentTag.NEW_EMBODIMENT)
    np.testing.assert_array_equal(step.action_validity["hand"], [[True, False], [False, True]])


def test_batched_relative_windows_match_pose_oracle():
    from gr00t.data.state_action.pose import EndEffectorPose
    from gr00t.data.stats import relative_rot6d_windows
    from gr00t.data.types import ActionFormat
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(7)
    states = np.concatenate(
        [
            rng.normal(size=(8, 3)),
            Rotation.random(8, random_state=rng).as_matrix()[:, :2, :].reshape(8, 6),
        ],
        axis=1,
    )
    actions = np.concatenate(
        [
            rng.normal(size=(8, 3)),
            Rotation.random(8, random_state=rng).as_matrix()[:, :2, :].reshape(8, 6),
        ],
        axis=1,
    )
    starts, offsets = np.array([0, 2, 4]), np.array([0, 1, 3])
    actual = relative_rot6d_windows(states, actions, starts, offsets)
    expected = np.array(
        [
            [
                (
                    EndEffectorPose.from_action_format(actions[i + d], ActionFormat.XYZ_ROT6D)
                    - EndEffectorPose.from_action_format(states[i], ActionFormat.XYZ_ROT6D)
                ).xyz_rot6d
                for d in offsets
            ]
            for i in starts
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)
