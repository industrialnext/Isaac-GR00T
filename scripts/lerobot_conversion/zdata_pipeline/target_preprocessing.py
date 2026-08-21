# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-only, representation-aware preprocessing for zdata conversion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import functools
import math

import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation, Slerp

from .config import ActionPreprocessingConfig, ResolvedLayout


_BOUND_TOLERANCE = 1e-7
_LOG_MAP_SINGULARITY_MARGIN = 1e-3
_REGULARIZATION_WEIGHT = 1e-4
_SOLVER_TIME_LIMIT = 5.0
_SOLVER_MAX_ITER = 4000


class QPSmoothingError(RuntimeError):
    """Raised when enabled target smoothing cannot produce a valid trajectory."""


@dataclass(frozen=True)
class PreprocessingResult:
    """Processed absolute target values and JSON-serializable repair evidence."""

    values: np.ndarray
    evidence: dict[str, object]


@dataclass(frozen=True)
class _RepairSpan:
    left: int
    right: int
    candidates: tuple[int, ...]


def _source_rot6d_to_matrices(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 6 or not np.isfinite(array).all():
        raise ValueError(f"source rot6d must have finite shape [T, 6], got {array.shape}")
    first = array[:, :3]
    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    if np.any(first_norm <= 1e-12):
        raise ValueError("source rot6d has a degenerate first axis")
    first = first / first_norm
    second = array[:, 3:]
    second = second - np.sum(first * second, axis=1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=1, keepdims=True)
    if np.any(second_norm <= 1e-12):
        raise ValueError("source rot6d has a degenerate second axis")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1)


def _matrices_to_source_rot6d(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not np.isfinite(values).all():
        raise ValueError(f"rotation matrices must have finite shape [T, 3, 3], got {values.shape}")
    return np.concatenate([values[:, :, 0], values[:, :, 1]], axis=1).astype(np.float32)


def _rotation_steps(values: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_matrix(_source_rot6d_to_matrices(values))
    if len(rotations) <= 1:
        return np.empty(0, dtype=np.float64)
    return (rotations[:-1].inv() * rotations[1:]).magnitude()


def _position_steps(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.empty(0, dtype=np.float64)
    return np.linalg.norm(np.diff(values, axis=0), axis=1)


def _find_feasible_span(
    *,
    distance: Callable[[int, int], float],
    length: int,
    required_left: int,
    required_right: int,
    candidates: tuple[int, ...],
    minimum_intervals: int,
    maximum_intervals: int,
    maximum_step: float,
    label: str,
    field_name: str,
) -> _RepairSpan:
    best: tuple[int, float, int, int] | None = None
    maximum = min(maximum_intervals, max(0, length - 1))
    for interval_count in range(minimum_intervals, maximum + 1):
        minimum_left = max(0, required_right - interval_count)
        maximum_left = min(required_left, length - 1 - interval_count)
        left_candidates = sorted(
            range(minimum_left, maximum_left + 1),
            key=lambda left: (
                abs((required_left - left) - (left + interval_count - required_right)),
                left,
            ),
        )
        for left in left_candidates:
            right = left + interval_count
            endpoint_distance = distance(left, right)
            required = math.ceil(endpoint_distance / maximum_step)
            evidence = (required, endpoint_distance, left, right)
            if best is None or evidence < best:
                best = evidence
            if endpoint_distance <= interval_count * maximum_step + _BOUND_TOLERANCE:
                return _RepairSpan(left, right, candidates)
    attempted = "none"
    required: int | str = "unknown"
    endpoint_distance = math.inf
    if best is not None:
        required, endpoint_distance, left, right = best
        attempted = f"[{left},{right}]"
    raise ValueError(
        "unrepairable action target outlier: "
        f"episode={label} field={field_name} candidate_boundaries={candidates} "
        f"attempted_span={attempted} endpoint_distance={endpoint_distance:.9g} "
        f"required_minimum_intervals={required} configured_limit={maximum_intervals}"
    )


def _repair_spans(
    *,
    distance: Callable[[int, int], float],
    length: int,
    candidates: tuple[int, ...],
    maximum_step: float,
    maximum_intervals: int,
    label: str,
    field_name: str,
) -> tuple[_RepairSpan, ...]:
    spans = [
        _find_feasible_span(
            distance=distance,
            length=length,
            required_left=boundary,
            required_right=boundary + 1,
            candidates=(boundary,),
            minimum_intervals=2,
            maximum_intervals=maximum_intervals,
            maximum_step=maximum_step,
            label=label,
            field_name=field_name,
        )
        for boundary in candidates
    ]
    while True:
        merged: list[_RepairSpan] = []
        for span in sorted(spans, key=lambda item: (item.left, item.right, item.candidates)):
            if not merged or span.left > merged[-1].right:
                merged.append(span)
                continue
            previous = merged.pop()
            left = min(previous.left, span.left)
            right = max(previous.right, span.right)
            combined = tuple(sorted(set(previous.candidates + span.candidates)))
            if distance(left, right) <= (right - left) * maximum_step + _BOUND_TOLERANCE:
                merged.append(_RepairSpan(left, right, combined))
            else:
                merged.append(
                    _find_feasible_span(
                        distance=distance,
                        length=length,
                        required_left=left,
                        required_right=right,
                        candidates=combined,
                        minimum_intervals=right - left + 1,
                        maximum_intervals=maximum_intervals,
                        maximum_step=maximum_step,
                        label=label,
                        field_name=field_name,
                    )
                )
        signature = tuple((item.left, item.right, item.candidates) for item in spans)
        merged_signature = tuple((item.left, item.right, item.candidates) for item in merged)
        if signature == merged_signature:
            return tuple(merged)
        spans = merged


def _repair_field(
    values: np.ndarray,
    *,
    kind: str,
    threshold: float,
    maximum_step: float,
    maximum_intervals: int,
    label: str,
    field_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    original = np.asarray(values, dtype=np.float32)
    output = original.copy()
    if kind == "position":
        steps = _position_steps(original)

        def distance(left: int, right: int) -> float:
            return float(np.linalg.norm(original[right] - original[left]))

        rotations = None
    elif kind == "rotation":
        steps = _rotation_steps(original)
        rotations = Rotation.from_matrix(_source_rot6d_to_matrices(original))

        def distance(left: int, right: int) -> float:
            assert rotations is not None
            return float((rotations[left].inv() * rotations[right]).magnitude())
    else:
        raise ValueError(f"unsupported target field kind {kind!r}")
    candidates = tuple(int(index) for index in np.flatnonzero(steps > threshold))
    spans = (
        _repair_spans(
            distance=distance,
            length=len(original),
            candidates=candidates,
            maximum_step=maximum_step,
            maximum_intervals=maximum_intervals,
            label=label,
            field_name=field_name,
        )
        if candidates
        else ()
    )
    interval_reports: list[dict[str, object]] = []
    for span in spans:
        count = span.right - span.left
        if kind == "position":
            alpha = np.arange(1, count, dtype=np.float64)[:, None] / float(count)
            output[span.left + 1 : span.right] = (1.0 - alpha) * original[
                span.left
            ] + alpha * original[span.right]
        else:
            assert rotations is not None
            pair = Rotation.from_matrix(
                np.stack([rotations[span.left].as_matrix(), rotations[span.right].as_matrix()])
            )
            interpolated = Slerp([0.0, 1.0], pair)(
                np.arange(1, count, dtype=np.float64) / float(count)
            )
            output[span.left + 1 : span.right] = _matrices_to_source_rot6d(interpolated.as_matrix())
        after = _position_steps(output) if kind == "position" else _rotation_steps(output)
        maximum_after = float(after[span.left : span.right].max(initial=0.0))
        if maximum_after > maximum_step + _BOUND_TOLERANCE:
            raise ValueError(
                f"outlier repair exceeded bound for episode={label} field={field_name} "
                f"span=[{span.left},{span.right}] maximum_after={maximum_after:.9g} "
                f"configured_bound={maximum_step:.9g}"
            )
        interval_reports.append(
            {
                "left_index": span.left,
                "right_index": span.right,
                "candidate_boundaries": list(span.candidates),
                "max_step_before": float(steps[span.left : span.right].max(initial=0.0)),
                "max_step_after": maximum_after,
                "interpolated_row_count": max(0, count - 1),
            }
        )
    after_steps = _position_steps(output) if kind == "position" else _rotation_steps(output)
    return output, {
        "field_name": field_name,
        "signal_kind": kind,
        "candidate_boundaries": list(candidates),
        "repair_intervals": interval_reports,
        "max_step_before": float(steps.max(initial=0.0)),
        "max_step_after": float(after_steps.max(initial=0.0)),
    }


@functools.lru_cache(maxsize=8)
def _jerk_hessian(t_steps: int, dimensions: int) -> sp.csc_matrix:
    row_count = t_steps - 3
    if row_count <= 0:
        single = sp.csc_matrix((0, t_steps))
    else:
        coefficients = np.asarray([1.0, -3.0, 3.0, -1.0], dtype=np.float64)
        rows = np.repeat(np.arange(row_count), 4)
        columns = np.asarray([np.arange(row_count) + offset for offset in range(4)]).T.ravel()
        single = sp.csc_matrix(
            (np.tile(coefficients, row_count), (rows, columns)),
            shape=(row_count, t_steps),
        )
    jerk = sp.csc_matrix(sp.block_diag([single] * dimensions))
    variable_count = t_steps * dimensions
    hessian = jerk.T @ jerk + _REGULARIZATION_WEIGHT * sp.eye(variable_count)
    return sp.csc_matrix(sp.triu(hessian, format="csc"))


def _smooth_field(values: np.ndarray, maximum_deviation: float) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    if len(field) <= 3:
        return field.copy()
    try:
        import clarabel
    except ImportError as error:
        raise ImportError("target preprocessing requires clarabel==0.11.1") from error
    time_steps, dimensions = field.shape
    variable_count = time_steps * dimensions
    original = field.ravel(order="F")
    linear = -_REGULARIZATION_WEIGHT * original
    identity = sp.eye(variable_count, format="csc")
    constraints = sp.vstack([-identity, identity], format="csc")
    bounds = np.concatenate([-(original - maximum_deviation), original + maximum_deviation])
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = _SOLVER_MAX_ITER
    settings.time_limit = _SOLVER_TIME_LIMIT
    solver = clarabel.DefaultSolver(
        _jerk_hessian(time_steps, dimensions),
        linear,
        constraints,
        bounds,
        [clarabel.NonnegativeConeT(2 * variable_count)],
        settings,
    )
    result = solver.solve()
    if result.status not in {clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved}:
        raise QPSmoothingError(
            f"QP smoothing failed with status {result.status} for "
            f"t_steps={time_steps}, n_dims={dimensions}"
        )
    return np.asarray(result.x, dtype=np.float64).reshape(time_steps, dimensions, order="F")


def _smooth_rotations(values: np.ndarray, maximum_deviation: float) -> np.ndarray:
    original = Rotation.from_matrix(_source_rot6d_to_matrices(values))
    matrices = original.as_matrix().copy()
    start = 0
    segments: list[tuple[int, int]] = []
    anchor = original[0]
    for index in range(1, len(original)):
        if float((anchor.inv() * original[index]).magnitude()) >= (
            np.pi - _LOG_MAP_SINGULARITY_MARGIN
        ):
            segments.append((start, index))
            start = index
            anchor = original[index]
    segments.append((start, len(original)))
    for start, end in segments:
        if end - start <= 3:
            continue
        anchor = original[start]
        segment = original[start:end]
        tangents = (anchor.inv() * segment).as_rotvec()
        smoothed = anchor * Rotation.from_rotvec(_smooth_field(tangents, maximum_deviation))
        smoothed_matrices = smoothed.as_matrix()
        for index in range(end - start):
            delta = segment[index].inv() * smoothed[index]
            distance = float(delta.magnitude())
            if distance > maximum_deviation:
                smoothed_matrices[index] = (
                    segment[index]
                    * Rotation.from_rotvec(delta.as_rotvec() * maximum_deviation / distance)
                ).as_matrix()
        matrices[start:end] = smoothed_matrices
    return _matrices_to_source_rot6d(matrices)


def preprocess_action_segment(
    values: np.ndarray,
    *,
    layout: ResolvedLayout,
    config: ActionPreprocessingConfig,
    label: str,
) -> PreprocessingResult:
    """Repair and smooth one gap-safe absolute action-target segment."""
    trajectory = np.asarray(values, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != layout.action_dim:
        raise ValueError(
            f"action target trajectory must have shape [T, {layout.action_dim}], "
            f"got {trajectory.shape}"
        )
    if not np.isfinite(trajectory).all():
        raise ValueError("action target trajectory contains non-finite values")
    if not config.enabled:
        return PreprocessingResult(trajectory.copy(), {"enabled": False})

    output = trajectory.copy()
    field_reports: list[dict[str, object]] = []
    outlier = config.outlier_filter
    if outlier.enabled:
        assert outlier.max_repair_intervals is not None
        assert outlier.position_step_threshold_m is not None
        assert outlier.position_max_repaired_step_m is not None
        assert outlier.rotation_step_threshold_rad is not None
        assert outlier.rotation_max_repaired_step_rad is not None
        for entry in layout.action:
            if entry.entry.key in config.skip_fields or entry.entry.format != "XYZ_ROT6D":
                continue
            position = slice(entry.start, entry.start + 3)
            rotation = slice(entry.start + 3, entry.end)
            output[:, position], report = _repair_field(
                output[:, position],
                kind="position",
                threshold=outlier.position_step_threshold_m,
                maximum_step=outlier.position_max_repaired_step_m,
                maximum_intervals=outlier.max_repair_intervals,
                label=label,
                field_name=f"{entry.entry.key}.position",
            )
            field_reports.append(report)
            output[:, rotation], report = _repair_field(
                output[:, rotation],
                kind="rotation",
                threshold=outlier.rotation_step_threshold_rad,
                maximum_step=outlier.rotation_max_repaired_step_rad,
                maximum_intervals=outlier.max_repair_intervals,
                label=label,
                field_name=f"{entry.entry.key}.rotation",
            )
            field_reports.append(report)

    assert config.max_deviation_position is not None
    assert config.max_deviation_rotation is not None
    non_rotation_indices: list[int] = []
    rotation_slices: list[tuple[int, int]] = []
    for entry in layout.action:
        if entry.entry.key in config.skip_fields:
            continue
        if entry.entry.format == "XYZ_ROT6D":
            non_rotation_indices.extend(range(entry.start, entry.start + 3))
            rotation_slices.append((entry.start + 3, entry.end))
        else:
            non_rotation_indices.extend(range(entry.start, entry.end))
    if non_rotation_indices:
        indices = np.asarray(non_rotation_indices, dtype=np.intp)
        output[:, indices] = _smooth_field(
            output[:, indices], config.max_deviation_position
        ).astype(np.float32)
    for start, end in rotation_slices:
        output[:, start:end] = _smooth_rotations(
            output[:, start:end], config.max_deviation_rotation
        )
    if not np.isfinite(output).all():
        raise ValueError("target preprocessing produced non-finite values")

    summary: dict[str, int | float] = {}
    for kind in ("position", "rotation"):
        reports = [report for report in field_reports if report["signal_kind"] == kind]
        intervals = [interval for report in reports for interval in report["repair_intervals"]]
        summary[f"{kind}_candidate_count"] = sum(
            len(report["candidate_boundaries"]) for report in reports
        )
        summary[f"{kind}_repaired_span_count"] = len(intervals)
        summary[f"{kind}_interpolated_row_count"] = sum(
            int(interval["interpolated_row_count"]) for interval in intervals
        )
        summary[f"{kind}_max_step_before"] = max(
            (float(report["max_step_before"]) for report in reports), default=0.0
        )
        summary[f"{kind}_max_step_after"] = max(
            (float(report["max_step_after"]) for report in reports), default=0.0
        )
    return PreprocessingResult(
        values=output.astype(np.float32, copy=False),
        evidence={"enabled": True, "summary": summary, "fields": field_reports},
    )
