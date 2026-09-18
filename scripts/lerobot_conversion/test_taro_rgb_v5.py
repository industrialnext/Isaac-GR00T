"""Contract tests for Taro projected-field compaction and observation admission."""

import h5py
import numpy as np
import pytest
from scripts.lerobot_conversion.prepare_taro_ablation import capture_group
from scripts.lerobot_conversion.zdata_pipeline.config import derive_layout, load_config
from scripts.lerobot_conversion.zdata_pipeline.taro_rgb_v5 import projected_arrays


def test_capture_groups_preserve_full_path_and_join_parts():
    assert capture_group("/bags/session/session_part_001") == capture_group(
        "/bags/session/session_part_002"
    )
    assert capture_group("/bags/a/robot_bag") != capture_group("/bags/b/robot_bag")


def test_masks_compaction_and_neutral_rotation(tmp_path):
    config = load_config("configs/embodiments/taro_exp_100.yaml")
    names = [
        "left_arm_pose_pos",
        "left_arm_pose_rot",
        "right_arm_pose_pos",
        "right_arm_pose_rot",
        "right_hand",
    ]
    widths = [3, 6, 3, 6, 25]
    compact_widths = dict(zip(names, [3, 6, 3, 6, 20]))
    layout = derive_layout(config, compact_widths, compact_widths, (256, 256, 3))
    offsets = np.r_[0, np.cumsum(widths)]
    with h5py.File(tmp_path / "episode.h5", "w") as h5:
        for scope in ("state", "action"):
            group = h5.create_group(scope)
            group["field_names"] = np.array(names, dtype=h5py.string_dtype())
            group["field_slices"] = np.stack([offsets[:-1], offsets[1:]], axis=1)
            values = np.ones((3, sum(widths)), dtype=np.float32)
            values[:, 3:9] = values[:, 12:18] = [1, 0, 0, 0, 1, 0]
            if scope == "action":
                values[1, 12:18] = 0  # Invalid pose: never feed this zero rotation to SO(3).
            group["flat" if scope == "state" else "expert"] = values
        masks = np.ones((3, sum(widths)), dtype=bool)
        masks[:, -5:] = False
        h5["extra/state_value_mask"] = masks
        masks[1, 9:18] = False
        masks[2, 18:38] = False
        for name in (
            "action_supervision_mask",
            "action_value_mask",
            "action_presence_mask",
            "action_ownership_mask",
        ):
            h5[f"extra/{name}"] = masks
        views = np.ones((3, 6), dtype=bool)
        views[1, 0] = False
        views[:, 3:] = False  # Excluded views cannot make an observation ineligible.
        h5["extra/view_mask"] = views
        state, action, sm, am, eligible = projected_arrays(h5, layout)
        assert state.shape == (3, 38) and action.shape == (3, 29)
        assert eligible.tolist() == [True, False, True]
        assert not am[1, :9].any() and not am[2, 9:].any()
        assert np.isfinite(state).all() and np.isfinite(action).all()
        h5["extra/action_supervision_mask"][0, -1] = True
        with pytest.raises(ValueError, match="padding"):
            projected_arrays(h5, layout)
