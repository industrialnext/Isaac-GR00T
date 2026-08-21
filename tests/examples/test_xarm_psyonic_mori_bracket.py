# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys

from gr00t.data.state_action.rot6d import rot6d_groot_to_source, rot6d_source_to_groot
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
import numpy as np
import pytest


_MODULE_PATH = (
    Path(__file__).parents[2] / "examples" / "xarm_psyonic_mori_bracket" / "server_client.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "xarm_psyonic_mori_bracket_server_client", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
XarmPsyonicBracketAdapter = _MODULE.XarmPsyonicBracketAdapter
validate_modality_contract = _MODULE.validate_modality_contract


def _modality():
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["front", "wrist"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["right_eef", "right_hand"]),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=["right_eef", "right_hand"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                    state_key="right_eef",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }


class _Client:
    def get_modality_config(self):
        return _modality()

    def get_action(self, observation):
        assert observation["video"]["front"].shape == (1, 1, 8, 10, 3)
        eef = np.repeat(observation["state"]["right_eef"], 40, axis=1)
        hand = np.repeat(observation["state"]["right_hand"], 40, axis=1)
        return {"right_eef": eef, "right_hand": hand}, {"test": True}


def _observation():
    return {
        "right_arm_pose_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "right_arm_pose_rot": np.array([1, 0, 0, 0, 0, 1], dtype=np.float32),
        "right_hand": np.arange(6, dtype=np.float32),
        "static_center_rgb": np.zeros((8, 10, 3), dtype=np.uint8),
        "eoat_right_bottom_rgb": np.zeros((8, 10, 3), dtype=np.uint8),
        "task_text": "Pick and place the bracket.",
    }


def test_adapter_validates_contract_and_round_trips_wire_rotation():
    adapter = XarmPsyonicBracketAdapter(_Client())
    observation = _observation()
    action, info = adapter.get_action(observation)
    assert info == {"test": True}
    assert action["right_arm_pose_pos"].shape == (40, 3)
    assert action["right_arm_pose_rot"].shape == (40, 6)
    assert action["right_hand"].shape == (40, 6)
    expected = rot6d_source_to_groot(observation["right_arm_pose_rot"])
    round_trip = rot6d_source_to_groot(action["right_arm_pose_rot"])
    assert np.allclose(round_trip, expected)
    assert np.allclose(rot6d_groot_to_source(expected), action["right_arm_pose_rot"][0], atol=1e-6)


def test_adapter_rejects_missing_extra_and_nonfinite_inputs():
    adapter = XarmPsyonicBracketAdapter(_Client())
    missing = _observation()
    missing.pop("task_text")
    with pytest.raises(ValueError, match="missing"):
        adapter.observation_to_model(missing)

    extra = _observation()
    extra["unexpected"] = 1
    with pytest.raises(ValueError, match="extra"):
        adapter.observation_to_model(extra)

    nonfinite = _observation()
    nonfinite["right_hand"][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        adapter.observation_to_model(nonfinite)


def test_adapter_rejects_wrong_output_shape_and_contract():
    with pytest.raises(ValueError, match="action shapes differ"):
        XarmPsyonicBracketAdapter.action_from_model(
            {"right_eef": np.zeros((1, 39, 9)), "right_hand": np.zeros((1, 40, 6))}
        )

    modality = _modality()
    modality["action"].delta_indices = list(range(39))
    with pytest.raises(ValueError, match="action contract differs"):
        validate_modality_contract(modality)
