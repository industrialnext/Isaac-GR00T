# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe contract tests for the Industrial Next RTC benchmark."""

from gr00t.eval.benchmark_industrialnext_rtc import (
    BenchmarkConfig,
    _mode_options,
    _summary,
    _text_summary,
    _unsupported_reason,
)
from gr00t.policy.industrialnext.async_server import _prefix_errors
from gr00t.policy.industrialnext.profile_config import load_industrialnext_profile
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
            (np.zeros((1, 40, 3)), np.tile(identity, (1, 40, 1))), axis=-1
        )
        action[f"{side}_gripper"] = np.zeros((1, 40, 1))
    prefix = {key: value.copy() for key, value in action.items()}
    action["left_eef"][0, 1, 0] = 0.01
    action["right_gripper"][0, 0, 0] = 0.02

    profile = load_industrialnext_profile("configs/embodiments/semihumanoid.yaml")
    position, orientation, gripper = _prefix_errors(
        profile.map_action_chunk(prefix), profile.map_action_chunk(action), 2, profile
    )

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


@pytest.mark.parametrize(
    "override", [{"control_hz": 25}, {"delay_trace": [1.5]}, {"replay_steps": 0}]
)
def test_replay_rejects_unrepresentable_timing(override):
    with pytest.raises(ValueError):
        BenchmarkConfig(model_path="checkpoint", output_dir="report", **override)


def test_replay_passes_nondefault_rtc_controls_to_production(monkeypatch):
    from types import SimpleNamespace

    from gr00t.eval import benchmark_industrialnext_rtc as benchmark

    profile = SimpleNamespace(action_horizon=40, assert_policy_contract=lambda policy: None)
    monkeypatch.setattr(benchmark, "load_industrialnext_profile", lambda path: profile)

    async def capture(policy, loader, trajectory_id, embodiment, profile, serving, config):
        return serving

    monkeypatch.setattr(benchmark, "replay_production", capture)
    config = BenchmarkConfig(
        model_path="checkpoint",
        output_dir="report",
        action_offset=0,
        ensemble_strategy="latest_only",
        chunk_transition_frames=0,
        rtc_ramp_rate=2.5,
        rtc_position_tolerance=0.002,
        rtc_orientation_tolerance_rad=0.003,
        rtc_gripper_tolerance=0.004,
    )
    serving = benchmark.replay_trajectory(None, None, 0, None, "native", config)
    assert serving.rtc_ramp_rate == 2.5
    assert serving.rtc_position_tolerance == 0.002
    assert serving.rtc_orientation_tolerance_rad == 0.003
    assert serving.rtc_gripper_tolerance == 0.004


def test_text_summary_includes_replay_metrics() -> None:
    scalar = {"mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0, "max": 1.0}
    trajectory = {
        "status": "ok",
        "coverage": 0.9,
        "null_rate": 0.1,
        "physical_errors": {"same_time": {"position_m": {"mean": 0.1}}},
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
    assert "off/trajectory-0: coverage=0.9000 null_rate=0.1000" in summary
    assert "physical_errors=" in summary
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


def test_taro_replay_uses_production_offset_ensemble_and_masks():
    import asyncio
    from types import SimpleNamespace

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import ModalityConfig
    from gr00t.eval.industrialnext_replay import replay_production
    from gr00t.policy.industrialnext import load_industrialnext_profile
    from gr00t.policy.industrialnext.async_server import IndustrialNextServingConfig
    import pandas as pd

    profile = load_industrialnext_profile("configs/embodiments/taro_exp_100.yaml")
    observation = profile.build_synthetic_model_observation(profile.task_catalog.tasks[0].task_text)
    data = {}
    for key, value in observation["state"].items():
        data[f"state.{key}"] = [value[0, 0]] * 20
    for key, value in observation["video"].items():
        data[f"video.{key}"] = [value[0, 0]] * 20
    data["language.annotation.human.task_description"] = [
        profile.task_catalog.tasks[0].task_text
    ] * 20
    actions = {
        "right_eef": np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0], (1, 40, 1)),
        "right_hand": np.zeros((1, 40, 20)),
    }
    for key, value in actions.items():
        data[f"action.{key}"] = [value[0, 0].copy() for _ in range(20)]
        data[f"action_validity.{key}"] = [np.ones(value.shape[-1], dtype=bool) for _ in range(20)]
    # A very wrong value in an unsupervised hand coordinate cannot affect error.
    for target, mask in zip(data["action.right_hand"], data["action_validity.right_hand"]):
        target[0] = 1000
        mask[0] = False
    data["observation.valid"] = [True] * 20
    trajectory = pd.DataFrame(data)

    class Loader:
        modality_configs = {
            modality: ModalityConfig(delta_indices=[0], modality_keys=list(observation[modality]))
            for modality in ("video", "state")
        }
        modality_configs["language"] = ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        )
        modality_configs["action"] = ModalityConfig(
            delta_indices=list(range(40)), modality_keys=list(actions)
        )

        def __getitem__(self, index):
            return trajectory

    class Policy:
        def get_action(self, observation, options=None):
            return actions, {}

    result = asyncio.run(
        replay_production(
            Policy(),
            Loader(),
            0,
            EmbodimentTag.NEW_EMBODIMENT,
            profile,
            IndustrialNextServingConfig(action_offset=2),
            SimpleNamespace(delay_trace=[2], seeds=[42], replay_steps=12),
        )
    )
    assert result["status"] == "ok"
    assert result["emitted_steps"] == 10
    assert result["physical_errors"]["same_time"]["hand_native_abs"]["mean"] == 0
    assert result["physical_errors"]["same_time"]["hand_native_abs"]["count"] == 190
    first = next(step for step in result["steps"] if step.get("has_action"))
    assert first["monitoring"]["emitted_action"]["contributions"][0]["model_row"] == 4
