# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe contract tests for the Industrial Next RTC benchmark."""

from gr00t.eval.benchmark_industrialnext_rtc import (
    BenchmarkConfig,
    _mode_options,
    _prefix_errors,
    _summary,
    _text_summary,
    _unsupported_reason,
)
import numpy as np
import pytest


def test_old_checkpoint_reports_trained_prefix_as_unsupported() -> None:
    checkpoint = {"model_type": "Gr00tN1d7", "action_horizon": 40}
    assert _unsupported_reason("off", checkpoint, 12) is None
    assert _unsupported_reason("native", checkpoint, 12) is None
    assert "does not advertise" in _unsupported_reason("trained_prefix", checkpoint, 12)


def test_mode_options_do_not_stack_native_and_trained_contracts() -> None:
    prefix = {"action": np.zeros((1, 12, 3), dtype=np.float32)}
    native = _mode_options("native", prefix, prefix_steps=4, overlap_steps=12, ramp_rate=6.0)
    trained = _mode_options(
        "trained_prefix", prefix, prefix_steps=4, overlap_steps=12, ramp_rate=6.0
    )
    assert "rtc_overlap_steps" in native and "rtc_prefix_steps" not in native
    assert "rtc_prefix_steps" in trained and "rtc_overlap_steps" not in trained
    assert trained["action_prefix"]["action"].shape == (1, 4, 3)


def test_prefix_errors_compare_physical_pose_and_gripper_contract() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action = {}
    for side in ("left", "right"):
        action[f"{side}_eef"] = np.concatenate(
            (np.zeros((1, 2, 3)), np.tile(identity, (1, 2, 1))), axis=-1
        )
        action[f"{side}_gripper"] = np.zeros((1, 2, 1))
    prefix = {key: value.copy() for key, value in action.items()}
    action["left_eef"][0, 1, 0] = 0.01
    action["right_gripper"][0, 0, 0] = 0.02

    position, orientation, gripper = _prefix_errors(action, prefix, 2)

    assert position == pytest.approx(0.01)
    assert orientation == pytest.approx(0.0)
    assert gripper == pytest.approx(0.02)


def test_latency_summary_and_bounds_are_deterministic() -> None:
    assert _summary([1.0, 2.0, 3.0])["p50"] == 2.0
    with pytest.raises(ValueError, match="leave"):
        BenchmarkConfig(
            model_path="checkpoint",
            output_dir="report",
            prefix_steps=25,
        )
    with pytest.raises(ValueError, match="workload"):
        BenchmarkConfig(
            model_path="checkpoint",
            output_dir="report",
            run_latency=False,
            run_replay=False,
        )


def test_text_summary_includes_replay_metrics() -> None:
    scalar = {"mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0, "max": 1.0}
    trajectory = {
        "status": "ok",
        "first_executable_row_mse": 0.1,
        "first_executable_row_mae": 0.2,
        "target_timestep_coverage": 0.9,
        "hold_rate": 0.1,
        "rejections": 0,
        "prefix_position_error_m": scalar,
        "prefix_orientation_error_rad": scalar,
        "prefix_gripper_error": scalar,
        "position_seam": scalar,
        "orientation_seam_rad": scalar,
        "gripper_seam": scalar,
        "max_position_step_m": scalar,
        "max_orientation_step_rad": scalar,
        "max_gripper_step": scalar,
        "max_position_second_difference_m": scalar,
        "max_gripper_second_difference": scalar,
    }
    report = {
        "model_path": "checkpoint",
        "latency": {
            "off": {
                "status": "ok",
                "total_ms": scalar,
                "p99_committed_steps_sizing_proxy": 1,
            }
        },
        "replay": {
            "off": {"status": "ok", "trajectories": {"0": trajectory}},
            "trained_prefix": {"status": "unsupported", "reason": "old checkpoint"},
        },
    }

    summary = _text_summary(report)

    assert "held-out replay:" in summary
    assert "off/trajectory-0: mse=0.1" in summary
    assert "trained_prefix: unsupported (old checkpoint)" in summary


def test_text_summary_allows_replay_only_report() -> None:
    report = {
        "model_path": "checkpoint",
        "latency": None,
        "replay": {
            "off": {
                "status": "ok",
                "trajectories": {"0": {"status": "no_admitted_predictions"}},
            }
        },
    }

    summary = _text_summary(report)

    assert "held-out replay:" in summary
    assert "off/trajectory-0: no_admitted_predictions" in summary
