# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Industrial Next and GR00T rot6d convention conversions."""

from __future__ import annotations

from collections.abc import Callable

from gr00t.data.state_action.rot6d import rot6d_groot_to_source, rot6d_source_to_groot
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def _source_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def _groot_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[..., 0, :], matrix[..., 1, :]], axis=-1)


def _matrix_from_source(value: np.ndarray) -> np.ndarray:
    first = value[..., :3]
    second = value[..., 3:6]
    return np.stack([first, second, np.cross(first, second)], axis=-1)


def _matrix_from_groot(value: np.ndarray) -> np.ndarray:
    first = value[..., :3]
    second = value[..., 3:6]
    return np.stack([first, second, np.cross(first, second)], axis=-2)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(3),
        Rotation.from_euler("x", 90, degrees=True).as_matrix(),
        Rotation.from_euler("y", -45, degrees=True).as_matrix(),
        Rotation.from_euler("z", 180, degrees=True).as_matrix(),
    ],
)
def test_known_rotations_round_trip(matrix: np.ndarray) -> None:
    source = _source_from_matrix(matrix)
    groot = rot6d_source_to_groot(source)
    np.testing.assert_allclose(groot, _groot_from_matrix(matrix), atol=1e-6)
    reconstructed_source = rot6d_groot_to_source(groot)
    np.testing.assert_allclose(reconstructed_source, source, atol=1e-6)


def test_batched_random_rotations_round_trip_by_matrix_and_angle() -> None:
    matrices = Rotation.random(100, random_state=np.random.default_rng(7)).as_matrix()
    source = _source_from_matrix(matrices)
    groot = rot6d_source_to_groot(source)
    reconstructed_source = rot6d_groot_to_source(groot)

    source_matrices = _matrix_from_source(reconstructed_source)
    groot_matrices = _matrix_from_groot(groot)
    np.testing.assert_allclose(source_matrices, matrices, atol=1e-6)
    np.testing.assert_allclose(groot_matrices, matrices, atol=1e-6)

    relative = np.swapaxes(source_matrices, -1, -2) @ matrices
    angular_error = Rotation.from_matrix(relative).magnitude()
    assert float(np.max(angular_error)) < 1e-6


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(6),
        np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, np.nan, 1.0, 0.0]),
        np.zeros(5),
        np.array(1.0),
    ],
)
@pytest.mark.parametrize("converter", [rot6d_source_to_groot, rot6d_groot_to_source])
def test_invalid_rot6d_is_rejected(
    value: np.ndarray, converter: Callable[[np.ndarray], np.ndarray]
) -> None:
    with pytest.raises(ValueError):
        converter(value)
