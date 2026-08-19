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

"""Tests for the semihumanoid -> GR00T LeRobot v2 converter.

The rot6d tests are the important ones: the source encodes a rotation as the first two
*columns* of R and GR00T decodes the first two *rows*, so a converter that forwards the
source bytes unchanged produces R-transpose. That trains without error and yields a policy
that rotates wrongly, so it is pinned here against GR00T's own decoder.

Run with h5py overlaid (the field-selection tests build in-memory HDF5 groups)::

    uv run --no-sync --with h5py python -m pytest \
        scripts/lerobot_conversion/test_convert_semihumanoid.py -q
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parent))

from convert_semihumanoid import (  # noqa: E402
    ACTION_FIELDS,
    ACTION_SLICES,
    CAMERA_MAP,
    STATE_FIELDS,
    STATE_SLICES,
    assign_split,
    gather_fields,
    rot6d_source_to_groot,
)
from gr00t.data.state_action.pose import EndEffectorPose  # noqa: E402


def _source_rot6d(R: np.ndarray) -> np.ndarray:
    """Encode R the way the source pipeline does: first two columns, concatenated."""
    return np.concatenate([R[:, 0], R[:, 1]])


def _random_rotations(n: int) -> list[np.ndarray]:
    from scipy.spatial.transform import Rotation

    return [Rotation.random(random_state=i).as_matrix() for i in range(n)]


# --- rot6d convention -------------------------------------------------------------


def test_rot6d_roundtrips_through_groot_decoder():
    """convert(source_encoding(R)) must decode back to exactly R under GR00T's reader."""
    for R in _random_rotations(200):
        converted = rot6d_source_to_groot(_source_rot6d(R))
        back = EndEffectorPose._rot6d_to_matrix(np.asarray(converted, dtype=float))
        assert np.allclose(back, R, atol=1e-9), "converted rot6d did not decode to R"


def test_naive_passthrough_is_wrong():
    """Guard the guard: without the transpose, GR00T reconstructs R-transpose.

    If this ever starts passing, either the source or GR00T changed convention and
    ``rot6d_source_to_groot`` must be revisited rather than silently kept.
    """
    from scipy.spatial.transform import Rotation

    R = _random_rotations(1)[0]
    naive = EndEffectorPose._rot6d_to_matrix(np.asarray(_source_rot6d(R), dtype=float))
    assert not np.allclose(naive, R, atol=1e-6)
    assert np.allclose(naive, R.T, atol=1e-9)
    err_deg = np.degrees(Rotation.from_matrix(naive @ R.T).magnitude())
    assert err_deg > 1.0, f"expected a large error, got {err_deg}"


def test_rot6d_output_is_a_valid_rotation():
    for R in _random_rotations(50):
        conv = np.asarray(rot6d_source_to_groot(_source_rot6d(R)), dtype=float)
        M = EndEffectorPose._rot6d_to_matrix(conv)
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(M), 1.0, atol=1e-9)


def test_rot6d_is_vectorised_over_leading_axis():
    Rs = _random_rotations(8)
    stacked = np.stack([_source_rot6d(R) for R in Rs])
    out = rot6d_source_to_groot(stacked)
    assert out.shape == (8, 6)
    for i, R in enumerate(Rs):
        assert np.allclose(rot6d_source_to_groot(_source_rot6d(R)), out[i])


def test_rot6d_rejects_wrong_width():
    with pytest.raises(ValueError, match="last dim 6"):
        rot6d_source_to_groot(np.zeros((4, 5)))


# --- field selection --------------------------------------------------------------


class _FakeGroup:
    """Minimal stand-in for an h5py group exposing field_names / field_slices."""

    def __init__(self, layout: list[tuple[str, int]]):
        names, slices, pos = [], [], 0
        for name, width in layout:
            names.append(name.encode())
            slices.append((pos, pos + width))
            pos += width
        self._d = {"field_names": np.array(names, dtype=object), "field_slices": np.array(slices)}
        self.width = pos

    def __getitem__(self, k):
        return self._d[k]


# The two real source schemas: 32-dim (no joints) and 46-dim (joints prepended per arm).
_LAYOUT_32 = [
    ("left_arm_pose_pos", 3),
    ("left_arm_pose_rot", 6),
    ("left_gripper", 1),
    ("left_ft", 6),
    ("right_arm_pose_pos", 3),
    ("right_arm_pose_rot", 6),
    ("right_gripper", 1),
    ("right_ft", 6),
]
_LAYOUT_46 = [
    ("left_arm_joints", 7),
    ("left_arm_pose_pos", 3),
    ("left_arm_pose_rot", 6),
    ("left_gripper", 1),
    ("left_ft", 6),
    ("right_arm_joints", 7),
    ("right_arm_pose_pos", 3),
    ("right_arm_pose_rot", 6),
    ("right_gripper", 1),
    ("right_ft", 6),
]


@pytest.mark.parametrize("layout,width", [(_LAYOUT_32, 32), (_LAYOUT_46, 46)])
def test_state_selection_is_name_based_not_positional(layout, width):
    """The same canonical 32-dim state must come out of both source schemas.

    This is the test that catches a converter that hardcodes ``[0:3]`` / ``[3:9]``: on the
    46-dim schema those slices land in ``left_arm_joints``.
    """
    from convert_semihumanoid import resolve_field_slices

    g = _FakeGroup(layout)
    assert g.width == width
    resolved = resolve_field_slices(g)

    rng = np.random.default_rng(0)
    flat = rng.normal(size=(5, width)).astype(np.float32)
    # make the rot blocks valid rotations so the transpose is well-defined
    from scipy.spatial.transform import Rotation

    for side in ("left", "right"):
        a, b = resolved[f"{side}_arm_pose_rot"]
        for t in range(5):
            R = Rotation.random(random_state=t).as_matrix()
            flat[t, a:b] = np.concatenate([R[:, 0], R[:, 1]])

    out = gather_fields(flat, resolved, STATE_FIELDS, "state")
    assert out.shape == (5, 32)
    # the gripper column must be the source gripper column, wherever it lived
    la, _ = resolved["left_gripper"]
    assert np.allclose(out[:, STATE_SLICES["left_gripper"][0]], flat[:, la])
    ra, _ = resolved["right_gripper"]
    assert np.allclose(out[:, STATE_SLICES["right_gripper"][0]], flat[:, ra])
    # ft blocks pass through untouched
    fa, fb = resolved["left_ft"]
    s, e = STATE_SLICES["left_ft"]
    assert np.allclose(out[:, s:e], flat[:, fa:fb])
    # eef position passes through; rotation does not (it is transposed)
    pa, pb = resolved["left_arm_pose_pos"]
    assert np.allclose(out[:, 0:3], flat[:, pa:pb])


def test_action_selection_yields_canonical_20():
    from convert_semihumanoid import resolve_field_slices

    layout = [
        ("left_arm_pose_pos", 3),
        ("left_arm_pose_rot", 6),
        ("left_gripper", 1),
        ("right_arm_pose_pos", 3),
        ("right_arm_pose_rot", 6),
        ("right_gripper", 1),
    ]
    g = _FakeGroup(layout)
    resolved = resolve_field_slices(g)
    flat = np.zeros((3, 20), dtype=np.float32)
    for side in ("left", "right"):
        a, b = resolved[f"{side}_arm_pose_rot"]
        flat[:, a:b] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    out = gather_fields(flat, resolved, ACTION_FIELDS, "action")
    assert out.shape == (3, 20)
    assert ACTION_SLICES["right_gripper"] == (19, 20)


def test_missing_field_raises():
    from convert_semihumanoid import resolve_field_slices

    g = _FakeGroup([("left_arm_pose_pos", 3)])
    with pytest.raises(ValueError, match="left_arm_pose_rot"):
        gather_fields(np.zeros((2, 3)), resolve_field_slices(g), STATE_FIELDS, "state")


def test_unexpected_field_width_raises():
    """A width change is a schema change and must fail loudly, not slice silently."""
    from convert_semihumanoid import resolve_field_slices

    bad = [(n, (w + 1 if n == "left_gripper" else w)) for n, w in _LAYOUT_32]
    g = _FakeGroup(bad)
    with pytest.raises(ValueError, match="left_gripper"):
        gather_fields(np.zeros((2, g.width)), resolve_field_slices(g), STATE_FIELDS, "state")


# --- layout invariants ------------------------------------------------------------


def test_canonical_slices_are_contiguous_and_complete():
    for slices, dim in ((STATE_SLICES, 32), (ACTION_SLICES, 20)):
        covered = sorted(slices.values())
        assert covered[0][0] == 0
        assert covered[-1][1] == dim
        for (_, end), (start, _) in zip(covered, covered[1:]):
            assert end == start, f"gap/overlap in {slices}"


def test_eef_blocks_are_nine_dim():
    """ActionType.EEF + XYZ_ROT6D requires exactly [xyz(3), rot6d(6)] contiguous."""
    for key in ("left_eef", "right_eef"):
        for slices in (STATE_SLICES, ACTION_SLICES):
            a, b = slices[key]
            assert b - a == 9, f"{key} in {slices} is {b - a}-dim, must be 9"


def test_camera_map_targets_bottom_wrist_cameras():
    assert CAMERA_MAP["head"] == "head_rgb"
    assert CAMERA_MAP["left_wrist"] == "eoat_left_bottom_rgb"
    assert CAMERA_MAP["right_wrist"] == "eoat_right_bottom_rgb"
    assert all(v.endswith("_rgb") for v in CAMERA_MAP.values()), "depth is not a GR00T modality"


# --- incremental growth -----------------------------------------------------------


def test_split_assignment_is_stable_and_position_independent():
    """Adding or backfilling episodes must never move an existing one between splits."""
    keys = [
        f"flexiv/2026/08/{d:02d}/2026{d:04d}_{i:03d}_expert/episode.h5"
        for d in (11, 12)
        for i in range(60)
    ]
    first = {k: assign_split(k, 20) for k in keys}
    # insert a backfilled earlier date and re-evaluate: existing assignments unchanged
    extra = [f"flexiv/2026/08/05/202608 05_{i:03d}_expert/episode.h5" for i in range(10)]
    second = {k: assign_split(k, 20) for k in extra + keys}
    for k in keys:
        assert second[k] == first[k], f"split for {k} moved when other episodes were added"


def test_split_ratio_is_approximately_one_in_n():
    keys = [
        f"s/2026/08/{d:02d}/ep_{i:04d}_expert/episode.h5" for d in range(1, 20) for i in range(80)
    ]
    val = sum(1 for k in keys if assign_split(k, 20) == "val")
    frac = val / len(keys)
    assert 0.02 < frac < 0.09, f"val fraction {frac:.3f} far from 1/20"


def test_val_every_zero_disables_split():
    assert all(assign_split(f"k{i}", 0) == "train" for i in range(20))
