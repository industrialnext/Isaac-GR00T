# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modality config for the IndustrialNext semihumanoid bimanual robot.

Matches the datasets produced by ``scripts/lerobot_conversion/convert_semihumanoid.py``.

State (32) and action (20) layouts are fixed by that converter; the ``start``/``end``
slices in each dataset's ``meta/modality.json`` must agree with the keys below.

Action horizon is 40, which equals the internal pipeline's ``n_action_steps`` at 50 Hz
(0.8 s) and is exactly the base checkpoint's ``action_horizon`` ceiling.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


semihumanoid_config = {
    # Three RGB views; current frame only. Keys are canonical -- the physical
    # camera each one came from is recorded per dataset in meta/modality.json.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "left_wrist", "right_wrist"],
    ),
    # Proprioception: per arm a 9-dim [xyz, rot6d] EEF pose, a 1-dim gripper in
    # [0, 1], and a 6-dim force/torque wrench in the TCP frame.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_eef",
            "left_gripper",
            "left_ft",
            "right_eef",
            "right_gripper",
            "right_ft",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "left_eef",
            "left_gripper",
            "right_eef",
            "right_gripper",
        ],
        action_configs=[
            # EEF poses are RELATIVE -- N1.7's native action space. state_key names
            # the 9-dim pose the delta is taken against; without it the processor
            # would look for a state key matching the action key.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="left_eef",
            ),
            # Grippers are ABSOLUTE: already normalized to [0, 1], and near-binary
            # targets train better as positions than as deltas.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
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
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(semihumanoid_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
