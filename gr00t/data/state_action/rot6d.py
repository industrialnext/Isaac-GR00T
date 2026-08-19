# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversions between Industrial Next and GR00T rot6d conventions."""

from __future__ import annotations

import numpy as np


_MIN_AXIS_NORM = 1e-12


def _as_rot6d(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 6:
        raise ValueError(f"rot6d must have last dim 6, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("rot6d contains non-finite values")
    return array


def _normalize_axis(value: np.ndarray, axis_name: str) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= _MIN_AXIS_NORM):
        raise ValueError(f"rot6d has a degenerate {axis_name} axis")
    return value / norm


def _orthonormal_pair(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = _normalize_axis(value[..., :3], "first")
    second = value[..., 3:6]
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = _normalize_axis(second, "second")
    return first, second


def rot6d_source_to_groot(value: np.ndarray) -> np.ndarray:
    """Convert source first-two-columns rot6d to GR00T first-two-rows rot6d."""
    first_column, second_column = _orthonormal_pair(_as_rot6d(value))
    third_column = np.cross(first_column, second_column)
    result = np.concatenate(
        [
            np.stack(
                [first_column[..., 0], second_column[..., 0], third_column[..., 0]],
                axis=-1,
            ),
            np.stack(
                [first_column[..., 1], second_column[..., 1], third_column[..., 1]],
                axis=-1,
            ),
        ],
        axis=-1,
    )
    return result


def rot6d_groot_to_source(value: np.ndarray) -> np.ndarray:
    """Convert GR00T first-two-rows rot6d to source first-two-columns rot6d."""
    first_row, second_row = _orthonormal_pair(_as_rot6d(value))
    third_row = np.cross(first_row, second_row)
    result = np.concatenate(
        [
            np.stack(
                [first_row[..., 0], second_row[..., 0], third_row[..., 0]],
                axis=-1,
            ),
            np.stack(
                [first_row[..., 1], second_row[..., 1], third_row[..., 1]],
                axis=-1,
            ),
        ],
        axis=-1,
    )
    return result
