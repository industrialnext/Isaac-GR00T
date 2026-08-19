# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in checkpoint smoke test for the semihumanoid Industrial Next adapter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.gpu
def test_real_semihumanoid_checkpoint_inference(
    load_hf_model_weights, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_model_path = os.environ.get("GR00T_SEMIHUMANOID_MODEL_PATH")
    if not raw_model_path:
        pytest.skip("set GR00T_SEMIHUMANOID_MODEL_PATH to run the real-checkpoint smoke test")
    model_path = Path(raw_model_path).expanduser().resolve()
    assert model_path.is_dir(), f"checkpoint directory does not exist: {model_path}"

    # Developers may point this opt-in test at an already-populated cache. This
    # is useful because the repository-wide pytest fixture otherwise isolates
    # Hugging Face downloads under its shared test cache.
    if raw_hf_home := os.environ.get("GR00T_SEMIHUMANOID_HF_HOME"):
        hf_home = Path(raw_hf_home).expanduser().resolve()
        monkeypatch.setenv("HF_HOME", str(hf_home))
        monkeypatch.setenv("HF_HUB_CACHE", str(hf_home / "hub"))
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
        monkeypatch.setenv("TRANSFORMERS_CACHE", str(hf_home / "hub"))

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.industrialnext import (
        ACTION_HORIZON,
        assert_semihumanoid_policy_contract,
        build_synthetic_model_observation,
        map_action_chunk,
    )

    with load_hf_model_weights():
        policy = Gr00tPolicy(
            model_path=str(model_path),
            embodiment_tag="new_embodiment",
            device="cuda",
            strict=True,
        )
    assert_semihumanoid_policy_contract(policy)
    observation = build_synthetic_model_observation("Pick the grounded target object.")
    action, _ = policy.get_action(observation)
    rows = map_action_chunk(action)
    assert len(rows) == ACTION_HORIZON
    assert set(rows[0]) == {
        "left_arm_pose_pos",
        "left_arm_pose_rot",
        "left_gripper",
        "right_arm_pose_pos",
        "right_arm_pose_rot",
        "right_gripper",
    }
