# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the config-driven zdata_hdf5 pipeline."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import pytest


sys.path.insert(0, str(Path(__file__).parent))

from gr00t.data.state_action.pose import EndEffectorPose  # noqa: E402
import zdata_pipeline.check as check_module  # noqa: E402
from zdata_pipeline.config import (  # noqa: E402
    derive_layout,
    load_config,
    modality_json,
    render_modality_module,
    resolve_source_subsets,
)
from zdata_pipeline.convert import recover_transactions, sync_config  # noqa: E402
from zdata_pipeline.source import (  # noqa: E402
    assign_split,
    discover_episodes,
    gather_fields,
    inspect_source,
    resolve_field_slices,
    rot6d_source_to_groot,
    stage_source,
)


REPO_ROOT = Path(__file__).parents[2]
SEMIHUMANOID_CONFIG = REPO_ROOT / "configs/embodiments/semihumanoid.yaml"


def _minimal_yaml(root: Path, output: Path, *, state_extra: str = "", root_extra: str = "") -> str:
    return f"""
name: testbot
{root_extra}
source:
  root: {root}
  subsets: ["source_*"]
  fps: 50
output:
  root: {output}
  robot_type: testbot
  strip_subset_prefix: "source_"
cameras:
  head: head_rgb
state:
  - key: eef
    fields: [pose_pos, pose_rot]
    rot6d: source_columns_to_groot_rows
    {state_extra}
action:
  source: executed
  horizon: 40
  keys:
    - key: eef
      fields: [pose_pos, pose_rot]
      rot6d: source_columns_to_groot_rows
      rep: RELATIVE
      type: EEF
      format: XYZ_ROT6D
      state_key: eef
"""


def test_semihumanoid_config_derives_expected_layout_and_generated_module():
    config = load_config(SEMIHUMANOID_CONFIG)
    state_widths = {
        "left_arm_pose_pos": 3,
        "left_arm_pose_rot": 6,
        "left_gripper": 1,
        "left_ft": 6,
        "right_arm_pose_pos": 3,
        "right_arm_pose_rot": 6,
        "right_gripper": 1,
        "right_ft": 6,
    }
    action_widths = {key: value for key, value in state_widths.items() if not key.endswith("_ft")}
    layout = derive_layout(config, state_widths, action_widths, (256, 256, 3))

    assert layout.state_dim == 32
    assert layout.action_dim == 20
    assert layout.state_slices["right_eef"] == (16, 25)
    assert layout.action_slices["right_gripper"] == (19, 20)
    assert config.cameras == {
        "head": "head_rgb",
        "left_wrist": "eoat_left_bottom_rgb",
        "right_wrist": "eoat_right_bottom_rgb",
    }
    for slices, dimension in (
        (layout.state_slices, layout.state_dim),
        (layout.action_slices, layout.action_dim),
    ):
        covered = sorted(slices.values())
        assert covered[0][0] == 0 and covered[-1][1] == dimension
        assert all(end == next_start for (_, end), (next_start, _) in zip(covered, covered[1:]))
    assert (
        render_modality_module(config)
        == (REPO_ROOT / "examples/semihumanoid/semihumanoid_config.py").read_text()
    )


def test_generated_semihumanoid_modality_module_imports_in_clean_process():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples/semihumanoid/semihumanoid_config.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_output_only_module_does_not_import_optional_h5py():
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(REPO_ROOT / 'scripts/lerobot_conversion')!r}); "
        "import zdata_pipeline.check; "
        "assert 'h5py' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_config_defaults_paths_and_non_layout_unknown_warning(tmp_path: Path):
    source = tmp_path / "sources"
    (source / "source_one").mkdir(parents=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _minimal_yaml(source, Path("~/pipeline-output"), root_extra="future_option: true")
    )
    with pytest.warns(UserWarning, match="future_option"):
        config = load_config(config_path)

    assert config.video.crf == 23
    assert config.output.chunks_size == 1000
    assert config.output.root == Path.home() / "pipeline-output"
    assert [path.name for path in resolve_source_subsets(config)] == ["source_one"]


def test_unknown_layout_key_fails(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_minimal_yaml(tmp_path, tmp_path / "out", state_extra="typo: true"))
    with pytest.raises(ValueError, match="unknown layout keys.*typo"):
        load_config(config_path)


def test_invalid_name_and_action_enum_fail(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _minimal_yaml(tmp_path, tmp_path / "out").replace("testbot", "test-bot", 1)
    )
    with pytest.raises(ValueError, match="valid Python identifier"):
        load_config(config_path)

    config_path.write_text(
        _minimal_yaml(tmp_path, tmp_path / "out").replace("rep: RELATIVE", "rep: MAYBE")
    )
    with pytest.raises(ValueError, match="rep must be one of"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ('  exclude_path_contains: "failed"\n', "exclude_path_contains"),
        ('select:\n  require_valid_for_training: "false"\n', "must be a boolean"),
        ("warn:\n  camera_coverage_below: 1.2\n", "camera_coverage_below"),
        ("train:\n  base_model: null\n", "train.base_model"),
    ],
)
def test_config_rejects_ambiguous_scalar_types_and_invalid_warning_ranges(
    tmp_path: Path, extra: str, message: str
):
    config_path = tmp_path / "config.yaml"
    text = _minimal_yaml(tmp_path, tmp_path / "out")
    if extra.startswith("  "):
        text = text.replace("  fps: 50\n", f"  fps: 50\n{extra}")
    else:
        text += extra
    config_path.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_layout_rejects_ambiguous_rot6d_and_non_nine_dimensional_eef(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_minimal_yaml(tmp_path, tmp_path / "out"))
    config = load_config(config_path)

    with pytest.raises(ValueError, match="exactly one 6D field"):
        derive_layout(
            config, {"pose_pos": 6, "pose_rot": 6}, {"pose_pos": 3, "pose_rot": 6}, (8, 8, 3)
        )
    with pytest.raises(ValueError, match="must be 9D"):
        derive_layout(
            config, {"pose_pos": 3, "pose_rot": 6}, {"pose_pos": 4, "pose_rot": 6}, (8, 8, 3)
        )
    with pytest.raises(ValueError, match="relative action width 9 differs from state.eef width 10"):
        derive_layout(
            config, {"pose_pos": 4, "pose_rot": 6}, {"pose_pos": 3, "pose_rot": 6}, (8, 8, 3)
        )


def test_source_subsets_must_map_to_unique_output_names(tmp_path: Path):
    source = tmp_path / "sources"
    (source / "source_one").mkdir(parents=True)
    (source / "one").mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _minimal_yaml(source, tmp_path / "out").replace(
            'subsets: ["source_*"]', 'subsets: ["source_one", "one"]'
        )
    )
    config = load_config(config_path)
    with pytest.raises(ValueError, match="both map to output"):
        resolve_source_subsets(config)


def _source_rot6d(rotation: np.ndarray) -> np.ndarray:
    return np.concatenate([rotation[:, 0], rotation[:, 1]])


def _random_rotations(count: int) -> list[np.ndarray]:
    from scipy.spatial.transform import Rotation

    return [Rotation.random(random_state=index).as_matrix() for index in range(count)]


def test_rot6d_roundtrips_through_groot_decoder():
    for rotation in _random_rotations(100):
        converted = rot6d_source_to_groot(_source_rot6d(rotation))
        decoded = EndEffectorPose._rot6d_to_matrix(np.asarray(converted, dtype=float))
        assert np.allclose(decoded, rotation, atol=1e-9)


def test_naive_rot6d_passthrough_remains_wrong():
    rotation = _random_rotations(1)[0]
    decoded = EndEffectorPose._rot6d_to_matrix(np.asarray(_source_rot6d(rotation), dtype=float))
    assert not np.allclose(decoded, rotation, atol=1e-6)
    assert np.allclose(decoded, rotation.T, atol=1e-9)


def test_rot6d_is_vectorized_valid_and_rejects_wrong_width():
    rotations = _random_rotations(8)
    encoded = np.stack([_source_rot6d(rotation) for rotation in rotations])
    converted = rot6d_source_to_groot(encoded)
    assert converted.shape == (8, 6)
    for value, rotation in zip(converted, rotations):
        decoded = EndEffectorPose._rot6d_to_matrix(value)
        assert np.allclose(decoded @ decoded.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(decoded), 1.0, atol=1e-9)
        assert np.allclose(decoded, rotation, atol=1e-9)
    with pytest.raises(ValueError, match="last dim 6"):
        rot6d_source_to_groot(np.zeros((4, 5)))


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(6),
        np.asarray([1, 0, 0, 2, 0, 0]),
        np.asarray([1, 0, 0, np.nan, 1, 0]),
    ],
)
def test_rot6d_rejects_degenerate_or_non_finite_axes(value: np.ndarray):
    with pytest.raises(ValueError, match="rot6d"):
        rot6d_source_to_groot(value)


class _FakeGroup:
    def __init__(self, fields: list[tuple[str, int]]):
        names: list[bytes] = []
        slices: list[tuple[int, int]] = []
        offset = 0
        for name, width in fields:
            names.append(name.encode())
            slices.append((offset, offset + width))
            offset += width
        self._values = {
            "field_names": np.asarray(names, dtype=object),
            "field_slices": np.asarray(slices),
        }
        self.width = offset

    def __getitem__(self, key: str):
        return self._values[key]


LAYOUT_32 = [
    ("left_arm_pose_pos", 3),
    ("left_arm_pose_rot", 6),
    ("left_gripper", 1),
    ("left_ft", 6),
    ("right_arm_pose_pos", 3),
    ("right_arm_pose_rot", 6),
    ("right_gripper", 1),
    ("right_ft", 6),
]
LAYOUT_46 = [
    ("left_arm_joints", 7),
    *LAYOUT_32[:4],
    ("right_arm_joints", 7),
    *LAYOUT_32[4:],
]


@pytest.mark.parametrize("native_layout", [LAYOUT_32, LAYOUT_46])
def test_named_field_selection_supports_current_32_and_46_dimensional_states(native_layout):
    config = load_config(SEMIHUMANOID_CONFIG)
    group = _FakeGroup(native_layout)
    source_slices = resolve_field_slices(group)
    state_widths = {name: end - start for name, (start, end) in source_slices.items()}
    action_widths = {
        "left_arm_pose_pos": 3,
        "left_arm_pose_rot": 6,
        "left_gripper": 1,
        "right_arm_pose_pos": 3,
        "right_arm_pose_rot": 6,
        "right_gripper": 1,
    }
    layout = derive_layout(config, state_widths, action_widths, (256, 256, 3))
    flat = np.zeros((4, group.width), dtype=np.float32)
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    for side in ("left", "right"):
        start, end = source_slices[f"{side}_arm_pose_rot"]
        flat[:, start:end] = identity
    gathered = gather_fields(flat, source_slices, layout.state)
    assert gathered.shape == (4, 32)
    left_gripper, _ = source_slices["left_gripper"]
    assert np.allclose(gathered[:, 9], flat[:, left_gripper])


def test_split_assignment_is_stable_under_insertion():
    keys = [f"robot/2026/08/18/episode_{index:04d}_expert/episode.h5" for index in range(200)]
    first = {key: assign_split(key, 20) for key in keys}
    backfill = [f"robot/2026/08/05/episode_{index:04d}_expert/episode.h5" for index in range(20)]
    second = {key: assign_split(key, 20) for key in backfill + keys}
    assert all(second[key] == split for key, split in first.items())


def test_split_ratio_and_disabled_mode():
    keys = [
        f"source/2026/08/{day:02d}/episode_{index:04d}_expert/episode.h5"
        for day in range(1, 20)
        for index in range(80)
    ]
    validation_fraction = sum(assign_split(key, 20) == "val" for key in keys) / len(keys)
    assert 0.02 < validation_fraction < 0.09
    assert all(assign_split(key, 0) == "train" for key in keys[:20])


def _jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _fake_encode_video(config, h5, physical, start, end, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"video")


def _file_snapshot(root: Path) -> dict[Path, tuple[int, bytes]]:
    return {
        path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def _write_episode(
    path: Path,
    *,
    frame_count: int = 50,
    valid_for_training: bool | None = True,
    gap_at: int | None = None,
    position_value: float = 0.0,
    task_id: str = "task_one",
    task_text: str = "Do the test task.",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pose = np.tile(np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32), (frame_count, 1))
    pose[:, 0] = position_value
    with h5py.File(path, "w") as h5:
        h5.attrs.update(
            {
                "frame_count": frame_count,
                "sampling_hz": 50.0,
                "image_height": 8,
                "image_width": 8,
                "task_uuid": task_id,
                "task_text": task_text,
            }
        )
        if valid_for_training is not None:
            h5.attrs["valid_for_training"] = valid_for_training
        state = h5.create_group("state")
        state.create_dataset("flat", data=pose)
        state.create_dataset("field_names", data=np.asarray([b"pose_pos", b"pose_rot"]))
        state.create_dataset("field_slices", data=np.asarray([[0, 3], [3, 9]]))
        action = h5.create_group("action")
        action.create_dataset("executed", data=pose)
        action.create_dataset("residual", data=np.zeros_like(pose))
        action.create_dataset("field_names", data=np.asarray([b"pose_pos", b"pose_rot"]))
        action.create_dataset("field_slices", data=np.asarray([[0, 3], [3, 9]]))
        frame = h5.create_group("frame")
        elapsed = np.arange(frame_count, dtype=np.float64) * 20.0
        if gap_at is not None:
            elapsed[gap_at:] += 100.0
        frame.create_dataset("elapsed_ms", data=elapsed)
        done = np.zeros(frame_count, dtype=bool)
        done[-1] = True
        frame.create_dataset("done", data=done)
        images = h5.create_group("images")
        camera = images.create_group("head_rgb")
        jpeg = _jpeg_bytes()
        camera.attrs["image_count"] = 1
        camera.create_dataset("blob", data=np.frombuffer(jpeg, dtype=np.uint8))
        camera.create_dataset("offsets", data=np.asarray([0, len(jpeg)], dtype=np.int64))
        camera.create_dataset("frame_ref_index", data=np.zeros(frame_count, dtype=np.int64))
        camera.create_dataset("frame_age_ms", data=np.zeros(frame_count, dtype=np.float32))


def _test_config(tmp_path: Path, *, gap_ms: float | None = None) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "sources"
    output_root = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    text = _minimal_yaml(source_root, output_root)
    text += "\nselect:\n  require_valid_for_training: true\n  policy_types: [expert]\n"
    if gap_ms is not None:
        text += f"\ncontinuity:\n  split_on_gap_ms: {gap_ms}\n"
    config_path.write_text(text)
    return config_path, source_root, output_root


def test_discovery_exclusion_matches_substrings_in_relative_paths(tmp_path: Path):
    config_path, source_root, _ = _test_config(tmp_path)
    subset = source_root / "source_one"
    included = subset / "robot/2026/08/18/episode_ok_expert/episode.h5"
    excluded = subset / "robot_failed_recordings_copy/2026/08/18/episode_bad_expert/episode.h5"
    _write_episode(included)
    _write_episode(excluded)

    assert discover_episodes(load_config(config_path), subset) == [included]


def test_inspection_separates_selection_warnings_and_gap_segments(tmp_path: Path):
    config_path, source_root, _ = _test_config(tmp_path, gap_ms=40)
    subset = source_root / "source_one"
    accepted = subset / "robot/2026/08/18/episode_ok_expert/episode.h5"
    _write_episode(accepted, frame_count=90, valid_for_training=True, gap_at=45)
    config = load_config(config_path)
    description = inspect_source(config, subset, accepted)
    assert description.segments == ((0, 45), (45, 90))

    missing = subset / "robot/2026/08/18/episode_missing_expert/episode.h5"
    _write_episode(missing, valid_for_training=None)
    missing_description = inspect_source(config, subset, missing)
    assert missing_description.skip_reason == "valid_for_training is missing"

    rejected = subset / "robot/2026/08/18/episode_bad_expert/episode.h5"
    _write_episode(rejected, valid_for_training=False)
    skipped = inspect_source(config, subset, rejected)
    assert skipped.skip_reason == "valid_for_training=false"


def test_structural_camera_failure_is_not_downgraded_to_warning(tmp_path: Path):
    config_path, source_root, _ = _test_config(tmp_path)
    subset = source_root / "source_one"
    source = subset / "robot/2026/08/18/episode_bad_expert/episode.h5"
    _write_episode(source)
    with h5py.File(source, "r+") as h5:
        h5["images/head_rgb/frame_ref_index"][0] = 3
    with pytest.raises(ValueError, match="frame_ref_index range"):
        inspect_source(load_config(config_path), subset, source)


def test_field_slices_cannot_extend_beyond_flat_tensor(tmp_path: Path):
    config_path, source_root, _ = _test_config(tmp_path)
    subset = source_root / "source_one"
    source = subset / "robot/2026/08/18/episode_bad_expert/episode.h5"
    _write_episode(source)
    with h5py.File(source, "r+") as h5:
        h5["state/field_slices"][1, 1] = 20
    with pytest.raises(ValueError, match="state field_slices exceed state/flat width"):
        inspect_source(load_config(config_path), subset, source)


def test_video_encoder_reaps_ffmpeg_after_broken_pipe(tmp_path: Path, monkeypatch):
    import zdata_pipeline.source as source_module

    config_path, _, _ = _test_config(tmp_path)

    class FakeStdin:
        def write(self, value):
            raise BrokenPipeError("closed")

        def close(self):
            pass

    class FakeProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.waited = False

        def wait(self):
            self.waited = True
            return 1

    process = FakeProcess()
    monkeypatch.setattr(source_module.subprocess, "Popen", lambda *args, **kwargs: process)
    h5 = {
        "images": {
            "head_rgb": {
                "offsets": np.asarray([0, 1], dtype=np.int64),
                "frame_ref_index": np.asarray([0], dtype=np.int64),
                "blob": np.asarray([0], dtype=np.uint8),
            }
        }
    }

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        source_module._encode_video(
            load_config(config_path), h5, "head_rgb", 0, 1, tmp_path / "video.mp4"
        )
    assert process.waited


def test_staged_gap_segments_reset_indices_timestamps_and_done(tmp_path: Path, monkeypatch):
    config_path, source_root, _ = _test_config(tmp_path, gap_ms=40)
    subset = source_root / "source_one"
    source = subset / "robot/2026/08/18/episode_ok_expert/episode.h5"
    _write_episode(source, frame_count=90, gap_at=45)
    config = load_config(config_path)
    description = inspect_source(config, subset, source)

    monkeypatch.setattr("zdata_pipeline.source._encode_video", _fake_encode_video)
    staged = stage_source(config, description, tmp_path / "stage")
    assert [segment.length for segment in staged.segments] == [45, 45]
    for segment in staged.segments:
        frame = pd.read_parquet(segment.parquet)
        assert frame["frame_index"].tolist() == list(range(45))
        assert frame["timestamp"].iloc[0] == pytest.approx(0.0)
        assert bool(frame["next.done"].iloc[-1])


def test_sync_commits_successes_without_index_gaps_and_is_noop(tmp_path: Path, monkeypatch):
    import zdata_pipeline.convert as convert_module
    import zdata_pipeline.source as source_module

    config_path, source_root, output_root = _test_config(tmp_path)
    subset = source_root / "source_one"
    for index in range(3):
        _write_episode(subset / f"robot/2026/08/18/episode_{index}_expert/episode.h5")
    config = load_config(config_path)
    monkeypatch.setattr(convert_module, "REPO_ROOT", tmp_path / "repo")

    monkeypatch.setattr(source_module, "_encode_video", _fake_encode_video)
    real_worker = convert_module.stage_source_worker
    real_next_indices = convert_module._next_indices
    real_task_table = convert_module._task_table
    next_index_calls: list[str] = []
    task_table_calls: list[str] = []

    def counting_next_indices(ledger, dataset_name):
        next_index_calls.append(dataset_name)
        return real_next_indices(ledger, dataset_name)

    def counting_task_table(config, ledger, dataset_name):
        task_table_calls.append(dataset_name)
        return real_task_table(config, ledger, dataset_name)

    monkeypatch.setattr(convert_module, "_next_indices", counting_next_indices)
    monkeypatch.setattr(convert_module, "_task_table", counting_task_table)

    def failing_worker(config, description, stage_directory):
        if "episode_1_expert" in description.key:
            return description.key, None, "injected failure"
        return real_worker(config, description, stage_directory)

    monkeypatch.setattr(convert_module, "stage_source_worker", failing_worker)
    assert sync_config(config, workers=1) == 1
    ledger_path = output_root / "_ledgers/one.json"
    ledger = json.loads(ledger_path.read_text())
    completed = [record for record in ledger["sources"].values() if record["status"] == "complete"]
    assert [record["segments"][0]["episode_index"] for record in completed] == [0, 1]
    assert [record["segments"][0]["index_offset"] for record in completed] == [0, 50]
    assert not list(output_root.glob("*/.sync_transaction.json"))
    assert next_index_calls == ["one"]
    assert task_table_calls == ["one"]

    monkeypatch.setattr(convert_module, "stage_source_worker", real_worker)
    monkeypatch.setattr(convert_module, "_next_indices", real_next_indices)
    monkeypatch.setattr(convert_module, "_task_table", real_task_table)
    assert sync_config(config, workers=1) == 0
    stats = output_root / "one/meta/stats.json"
    relative_stats = output_root / "one/meta/relative_stats.json"
    stats.write_text("stats")
    relative_stats.write_text("relative")
    _write_episode(subset / "robot/2026/08/18/episode_3_expert/episode.h5")
    assert sync_config(config, workers=1) == 0
    assert not stats.exists()
    assert not relative_stats.exists()
    before = _file_snapshot(output_root)
    assert sync_config(config, workers=1) == 0
    after = _file_snapshot(output_root)
    assert after == before

    key = "robot/2026/08/18/episode_0_expert/episode.h5"
    selector = f"source_one/{key}"
    ledger_before = json.loads(ledger_path.read_text())
    assignment_before = ledger_before["sources"][key]["segments"][0].copy()
    episode_index = assignment_before["episode_index"]
    parquet = (
        output_root
        / "one"
        / (f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet")
    )
    stats.write_text("stats")
    relative_stats.write_text("relative")
    _write_episode(subset / key, position_value=2.0)
    assert sync_config(config, workers=1, reconvert=[selector]) == 0
    repaired = pd.read_parquet(parquet)
    assert repaired["observation.state"].iloc[0][0] == pytest.approx(2.0)
    ledger_after = json.loads(ledger_path.read_text())
    assert ledger_after["sources"][key]["segments"][0] == assignment_before
    assert not stats.exists() and not relative_stats.exists()

    parquet_before = parquet.read_bytes()
    ledger_bytes_before = ledger_path.read_bytes()
    _write_episode(subset / key, frame_count=51, position_value=3.0)
    assert sync_config(config, workers=1, reconvert=[selector]) == 1
    assert parquet.read_bytes() == parquet_before
    assert ledger_path.read_bytes() == ledger_bytes_before


def test_task_table_grows_without_reindexing_and_keeps_existing_text(tmp_path: Path, monkeypatch):
    config, output_root = _synced_test_output(
        tmp_path, monkeypatch, subsets=1, episodes_per_subset=1
    )
    subset = config.source.root / "source_0"
    second = subset / "robot/2026/08/18/episode_01_expert/episode.h5"
    _write_episode(second, task_id="task_two", task_text="Do the second task.")
    assert sync_config(config, workers=1) == 0
    tasks_path = output_root / "0/meta/tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    assert tasks == [
        {"task_index": 0, "task": "Do the test task."},
        {"task_index": 1, "task": "Do the second task."},
    ]

    first = subset / "robot/2026/08/18/episode_00_expert/episode.h5"
    _write_episode(first, task_text="Changed source wording.")
    selector = "source_0/robot/2026/08/18/episode_00_expert/episode.h5"
    assert sync_config(config, workers=1, reconvert=[selector]) == 0
    assert [json.loads(line) for line in tasks_path.read_text().splitlines()] == tasks


def test_task_override_updates_existing_metadata_without_new_data_or_stats_invalidation(
    tmp_path: Path, monkeypatch
):
    config, output_root = _synced_test_output(
        tmp_path, monkeypatch, subsets=1, episodes_per_subset=1
    )
    dataset = output_root / "0"
    stats_paths = [dataset / "meta" / filename for filename in check_module.STATS_FILES]
    for path in stats_paths:
        path.write_text(path.name)
    updated = replace(
        config,
        tasks=replace(config.tasks, text_overrides={"task_one": "Use the preferred wording."}),
    )

    assert sync_config(updated, workers=1) == 0
    tasks = [
        json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text().splitlines() if line
    ]
    assert tasks == [{"task_index": 0, "task": "Use the preferred wording."}]
    episodes = [
        json.loads(line)
        for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()
        if line
    ]
    assert episodes[0]["tasks"] == ["Use the preferred wording."]
    assert [path.read_text() for path in stats_paths] == [path.name for path in stats_paths]
    before = _file_snapshot(output_root)
    assert sync_config(updated, workers=1) == 0
    after = _file_snapshot(output_root)
    assert after == before


def test_skipped_path_requires_explicit_reconvert_before_it_can_be_admitted(
    tmp_path: Path, monkeypatch
):
    import zdata_pipeline.convert as convert_module
    import zdata_pipeline.source as source_module

    config_path, source_root, output_root = _test_config(tmp_path)
    subset = source_root / "source_one"
    source = subset / "robot/2026/08/18/episode_00_expert/episode.h5"
    _write_episode(source, valid_for_training=False)
    config = load_config(config_path)
    monkeypatch.setattr(convert_module, "REPO_ROOT", tmp_path / "generated_repo")

    monkeypatch.setattr(source_module, "_encode_video", _fake_encode_video)
    assert sync_config(config, workers=1) == 0
    ledger_path = output_root / "_ledgers/one.json"
    ledger = json.loads(ledger_path.read_text())
    key = source.relative_to(subset).as_posix()
    assert ledger["sources"][key]["status"] == "skipped"

    _write_episode(source, valid_for_training=True)
    before = ledger_path.read_bytes()
    assert sync_config(config, workers=1) == 0
    assert ledger_path.read_bytes() == before
    assert not (output_root / "one/meta/info.json").exists()

    selector = f"source_one/{key}"
    assert sync_config(config, workers=1, reconvert=[selector]) == 0
    admitted = json.loads(ledger_path.read_text())["sources"][key]
    assert admitted["status"] == "complete"
    assert admitted["segments"][0]["episode_index"] == 0


@pytest.mark.parametrize("replaced_count", [0, 1, 2])
def test_transaction_journal_rolls_forward_from_every_replacement_boundary(
    tmp_path: Path, replaced_count: int
):
    output = tmp_path / "output"
    dataset = output / "one"
    transaction = output / "_staging/transactions/test"
    transaction.mkdir(parents=True)
    worker_stage = output / "_staging/runs/test/source"
    worker_stage.mkdir(parents=True)
    (worker_stage / "staged.parquet").write_text("staged")
    dataset.mkdir(parents=True)
    replacements = []
    for index in range(2):
        staging = transaction / f"staging-{index}"
        final = dataset / f"final-{index}"
        staging.write_text(f"new-{index}")
        final.write_text(f"old-{index}")
        replacements.append((staging, final))
    stats = dataset / "meta/stats.json"
    relative_stats = dataset / "meta/relative_stats.json"
    stats.parent.mkdir()
    stats.write_text("stale")
    relative_stats.write_text("stale")
    journal = {
        "version": 1,
        "transaction_directory": "_staging/transactions/test",
        "replacements": [
            {
                "staging": staging.relative_to(output).as_posix(),
                "final": final.relative_to(output).as_posix(),
            }
            for staging, final in replacements
        ],
        "invalidate_stats": [
            stats.relative_to(output).as_posix(),
            relative_stats.relative_to(output).as_posix(),
        ],
        "cleanup_paths": [worker_stage.relative_to(output).as_posix()],
    }
    journal_path = dataset / ".sync_transaction.json"
    journal_path.write_text(json.dumps(journal))
    for staging, final in replacements[:replaced_count]:
        staging.replace(final)

    recover_transactions(output)

    assert [final.read_text() for _, final in replacements] == ["new-0", "new-1"]
    assert not stats.exists() and not relative_stats.exists()
    assert not journal_path.exists() and not transaction.exists() and not worker_stage.exists()


def test_version_one_ledger_migration_is_read_only_in_preview_and_preserves_stats(
    tmp_path: Path, monkeypatch
):
    import zdata_pipeline.convert as convert_module

    config_path, source_root, output_root = _test_config(tmp_path)
    subset = source_root / "source_one"
    source = subset / "robot/2026/08/18/episode_0_expert/episode.h5"
    _write_episode(source)
    config = load_config(config_path)
    description = inspect_source(config, subset, source)
    dataset = output_root / "one"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Do the test task."}) + "\n"
    )
    (meta / "modality.json").write_text(
        json.dumps(modality_json(config, description.layout), indent=4) + "\n"
    )
    (meta / "stats.json").write_text("absolute-stats")
    (meta / "relative_stats.json").write_text("relative-stats")
    ledger_path = output_root / "_ledgers/one.json"
    ledger_path.parent.mkdir(parents=True)
    key = source.relative_to(subset).as_posix()
    ledger_path.write_text(
        json.dumps(
            {
                "version": 1,
                "camera_map": {"head": "head_rgb"},
                "episodes": {
                    key: {
                        "dataset": "one",
                        "split": "train",
                        "episode_index": 0,
                        "index_offset": 0,
                        "length": 50,
                        "task_uuid": "task_one",
                        "source": "/stale/absolute/path/episode.h5",
                    }
                },
            },
            indent=4,
        )
    )
    monkeypatch.setattr(convert_module, "REPO_ROOT", tmp_path / "repo")
    before = _file_snapshot(output_root)
    assert sync_config(config, workers=1, dry_run=True) == 0
    after_preview = _file_snapshot(output_root)
    assert after_preview == before

    assert sync_config(config, workers=1) == 0
    migrated = json.loads(ledger_path.read_text())
    assert migrated["version"] == 2
    assert migrated["sources"][key]["segments"][0]["episode_index"] == 0
    assert "source" not in migrated["sources"][key]
    assert (dataset / "_layout.json").exists()
    assert (meta / "stats.json").read_text() == "absolute-stats"
    assert (meta / "relative_stats.json").read_text() == "relative-stats"


def _synced_test_output(
    tmp_path: Path, monkeypatch, *, subsets: int = 1, episodes_per_subset: int = 2
):
    import zdata_pipeline.convert as convert_module
    import zdata_pipeline.source as source_module

    config_path, source_root, output_root = _test_config(tmp_path)
    for subset_index in range(subsets):
        subset = source_root / f"source_{subset_index}"
        for episode_index in range(episodes_per_subset):
            _write_episode(
                subset / f"robot/2026/08/18/episode_{episode_index:02d}_expert/episode.h5"
            )
    config = load_config(config_path)
    generated_repo = tmp_path / "generated_repo"
    monkeypatch.setattr(convert_module, "REPO_ROOT", generated_repo)
    monkeypatch.setattr(check_module, "REPO_ROOT", generated_repo)

    monkeypatch.setattr(source_module, "_encode_video", _fake_encode_video)
    assert sync_config(config, workers=1) == 0
    return config, output_root


def test_stats_skips_current_datasets_and_summarizes_failures(tmp_path: Path, monkeypatch):
    config, output_root = _synced_test_output(tmp_path, monkeypatch, subsets=2)
    datasets = sorted(path.parent.parent for path in output_root.glob("*/meta/info.json"))
    called: list[str] = []

    def successful_stats(config, dataset):
        called.append(dataset.name)
        for filename in check_module.STATS_FILES:
            (dataset / "meta" / filename).write_text("{}")
        return dataset, 0

    monkeypatch.setattr(check_module, "_run_stats", successful_stats)
    assert check_module.generate_missing_stats(config, jobs=2) == 0
    assert sorted(called) == ["0", "1"]
    assert check_module.generate_missing_stats(config, jobs=2) == 0
    assert len(called) == 2

    (datasets[0] / "meta/relative_stats.json").unlink()

    def failed_stats(config, dataset):
        return dataset, 7

    monkeypatch.setattr(check_module, "_run_stats", failed_stats)
    assert check_module.generate_missing_stats(config, jobs=1) == 1
    command = check_module.stats_command(config, datasets[0])
    assert command[:2] == [sys.executable, str(check_module.REPO_ROOT / "gr00t/data/stats.py")]
    assert command[command.index("--dataset-path") + 1] == str(datasets[0])
    assert command[command.index("--embodiment-tag") + 1] == "NEW_EMBODIMENT"


def test_default_and_full_checks_are_bounded_and_detect_later_index_gap(
    tmp_path: Path, monkeypatch
):
    config, output_root = _synced_test_output(tmp_path, monkeypatch)
    monkeypatch.setattr(check_module, "_decode_one_frame", lambda path: None)
    monkeypatch.setattr(check_module, "_video_frame_count", lambda path: 50)
    assert check_module.check_outputs(config) == 0

    dataset = output_root / "0"
    second = dataset / "data/chunk-000/episode_000001.parquet"
    frame = pd.read_parquet(second)
    frame.loc[10, "frame_index"] = 999
    frame.to_parquet(second)
    assert check_module.check_outputs(config) == 0
    assert check_module.check_outputs(config, full=True) == 1


def test_default_check_detects_missing_selected_video_and_samples_loader_when_stats_exist(
    tmp_path: Path, monkeypatch
):
    config, output_root = _synced_test_output(tmp_path, monkeypatch)
    decoded: list[Path] = []
    loader_samples: list[str] = []
    monkeypatch.setattr(check_module, "_decode_one_frame", decoded.append)
    monkeypatch.setattr(
        check_module, "_loader_sample", lambda config, dataset: loader_samples.append(dataset.name)
    )
    dataset = output_root / "0"
    for filename in check_module.STATS_FILES:
        (dataset / "meta" / filename).write_text("{}")
    assert check_module.check_outputs(config) == 0
    assert len(decoded) == 1
    assert loader_samples == ["0"]

    decoded[0].unlink()
    assert check_module.check_outputs(config) == 1


@pytest.mark.parametrize("field", ["total_videos", "total_chunks"])
def test_default_check_detects_incorrect_aggregate_metadata(
    tmp_path: Path, monkeypatch, field: str
):
    config, output_root = _synced_test_output(tmp_path, monkeypatch)
    monkeypatch.setattr(check_module, "_decode_one_frame", lambda path: None)
    info_path = output_root / "0/meta/info.json"
    info = json.loads(info_path.read_text())
    info[field] += 1
    info_path.write_text(json.dumps(info))

    assert check_module.check_outputs(config) == 1


def test_default_check_detects_task_text_mismatch(tmp_path: Path, monkeypatch):
    config, output_root = _synced_test_output(tmp_path, monkeypatch)
    monkeypatch.setattr(check_module, "_decode_one_frame", lambda path: None)
    episodes_path = output_root / "0/meta/episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    episodes[0]["tasks"] = ["stale text"]
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in episodes))

    assert check_module.check_outputs(config) == 1


def test_checks_refuse_incomplete_sync_transaction(tmp_path: Path, monkeypatch):
    config, output_root = _synced_test_output(tmp_path, monkeypatch)
    (output_root / "0/.sync_transaction.json").write_text("{}")
    with pytest.raises(RuntimeError, match="incomplete sync transaction"):
        check_module.check_outputs(config)


def test_training_command_covers_epoch_max_steps_wandb_fresh_and_resume(
    tmp_path: Path, monkeypatch
):
    config, output_root = _synced_test_output(
        tmp_path, monkeypatch, subsets=2, episodes_per_subset=2
    )
    datasets, validation = check_module.find_datasets(output_root)
    assert len(datasets) == 2 and validation == []
    out_base = tmp_path / "training"
    config = replace(
        config,
        train=replace(
            config.train,
            out_base=out_base,
            gpus=2,
            batch=4,
            epochs=2.5,
            max_steps=None,
            use_wandb=True,
            wandb_project="test-project",
            rtc_training_max_prefix_steps=3,
        ),
    )
    instant = datetime(2026, 8, 19, 12, 34, 56, tzinfo=timezone.utc)
    command, output_directory, steps, starts = check_module.build_train_command(
        config, datasets, now=lambda: instant
    )
    assert starts == 2 * 2 * (50 - config.action.horizon + 1)
    assert steps == math.ceil(2.5 * starts / 4)
    assert output_directory == out_base / "testbot_20260819_123456"
    assert command[:6] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "2",
    ]
    assert command[command.index("--dataset-path") + 1] == os.pathsep.join(
        str(dataset) for dataset in datasets
    )
    assert command[command.index("--max-steps") + 1] == str(steps)
    assert command[command.index("--wandb-project") + 1] == "test-project"
    assert command[command.index("--rtc-training-max-prefix-steps") + 1] == "3"
    assert "--use-wandb" in command
    assert "--experiment-name" not in command
    assert "--resume-from-checkpoint" not in command

    resume_directory = tmp_path / "resume"
    resume_directory.mkdir()
    max_step_config = replace(
        config, train=replace(config.train, max_steps=7, epochs=None, use_wandb=False)
    )
    resumed, resumed_output, resumed_steps, _ = check_module.build_train_command(
        max_step_config, datasets, resume_from=resume_directory
    )
    assert resumed_output == resume_directory
    assert resumed_steps == 7
    assert "--resume-from-checkpoint" in resumed
    assert "--use-wandb" not in resumed

    indivisible = replace(config, train=replace(config.train, batch=5))
    with pytest.raises(ValueError, match="must be divisible"):
        check_module.build_train_command(indivisible, datasets, now=lambda: instant)

    output_directory.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="fresh training directory already exists"):
        check_module.build_train_command(config, datasets, now=lambda: instant)


def test_training_manifest_binds_converted_content_and_source_stats(tmp_path: Path, monkeypatch):
    config, output_root = _synced_test_output(tmp_path, monkeypatch, episodes_per_subset=1)
    datasets, _ = check_module.find_datasets(output_root)
    for dataset in datasets:
        for filename in check_module.STATS_FILES:
            (dataset / "meta" / filename).write_text("{}\n")
    monkeypatch.setattr(check_module, "check_outputs", lambda config, full: 0)
    assert check_module.freeze_corpus(config) == 0
    monkeypatch.setattr(check_module, "_base_model_revision", lambda model: "test-revision")
    monkeypatch.setattr(check_module, "_repository_state", lambda: ("test-head", "diff-hash"))
    output_directory = tmp_path / "training" / "run"
    command = ["train", "--deterministic-test"]

    manifest_path = check_module.create_training_manifest(
        config,
        datasets,
        command,
        output_directory,
        steps=2,
        starts=11,
        batch=1,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["base_model_revision"] == "test-revision"
    assert manifest["command"] == command
    assert manifest["source_stat_inventory"]
    assert manifest["converted_artifact_inventory"]
    check_module.verify_training_manifest(manifest_path)

    bound_artifact = Path(manifest["converted_artifact_inventory"][0]["path"])
    bound_artifact.write_bytes(bound_artifact.read_bytes() + b"mutation")
    with pytest.raises(RuntimeError, match="verification failed"):
        check_module.verify_training_manifest(manifest_path)


def test_freeze_manifest_blocks_pipeline_writes(tmp_path: Path, monkeypatch):
    config, output_root = _synced_test_output(tmp_path, monkeypatch, episodes_per_subset=1)
    datasets, _ = check_module.find_datasets(output_root)
    for dataset in datasets:
        for filename in check_module.STATS_FILES:
            (dataset / "meta" / filename).write_text("{}\n")
    monkeypatch.setattr(check_module, "check_outputs", lambda config, full: 0)

    assert check_module.freeze_corpus(config) == 0
    assert check_module.freeze_corpus(config) == 0
    with pytest.raises(RuntimeError, match="converted corpus is frozen"):
        sync_config(config, workers=1)

    (datasets[0] / "meta/stats.json").write_text("mutated\n")
    with pytest.raises(RuntimeError, match="frozen corpus verification failed"):
        check_module.verify_frozen_corpus(check_module.frozen_corpus_manifest_path(output_root))


def test_freeze_rejects_source_changed_after_conversion(tmp_path: Path, monkeypatch):
    config, output_root = _synced_test_output(tmp_path, monkeypatch, episodes_per_subset=1)
    datasets, _ = check_module.find_datasets(output_root)
    for dataset in datasets:
        for filename in check_module.STATS_FILES:
            (dataset / "meta" / filename).write_text("{}\n")
    source = next(resolve_source_subsets(config)[0].glob(config.source.episode_glob))
    source.touch()

    with pytest.raises(RuntimeError, match="source changed after conversion"):
        check_module.freeze_corpus(config)


def test_resource_reporting_is_advisory_for_malformed_gpu_output_and_missing_output_path(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setattr(check_module.shutil, "which", lambda executable: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        check_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="malformed\n"),
    )

    check_module._report_resources(tmp_path / "not-yet-created" / "nested" / "outputs")

    captured = capsys.readouterr()
    assert "ignoring malformed nvidia-smi output" in captured.err
    assert "training output free disk" in captured.out


def test_training_smoke_validation_requires_wired_rtc_and_objective_coverage(tmp_path: Path):
    output = tmp_path / "smoke"
    checkpoint = output / "checkpoint-30"
    experiment = output / "experiment_cfg"
    checkpoint.mkdir(parents=True)
    experiment.mkdir()
    (output / "config.json").write_text('{"rtc_training_max_prefix_steps": 3}\n')
    (experiment / "final_model_config.json").write_text('{"rtc_training_max_prefix_steps": 3}\n')
    state = {
        "log_history": [
            {"loss": 1.0},
            {
                "rtc_postfix_valid_elements": 100.0,
                "rtc_prefix_count_0": 2.0,
                "rtc_prefix_count_1": 3.0,
            },
        ]
    }
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))

    check_module.validate_training_smoke(output, 3)

    (experiment / "final_model_config.json").write_text('{"rtc_training_max_prefix_steps": 0}\n')
    with pytest.raises(RuntimeError, match="instead of 3"):
        check_module.validate_training_smoke(output, 3)


def test_existing_layout_rejects_semantic_or_chunk_change_without_new_sources(
    tmp_path: Path, monkeypatch
):
    config, output_root = _synced_test_output(tmp_path, monkeypatch, episodes_per_subset=1)
    module_path = check_module.REPO_ROOT / "examples/testbot/testbot_config.py"
    module_before = module_path.read_bytes()

    changed_action = replace(
        config.action,
        keys=(replace(config.action.keys[0], rep="ABSOLUTE"), *config.action.keys[1:]),
    )
    with pytest.raises(ValueError, match="stored action semantics differ from config"):
        sync_config(replace(config, action=changed_action), workers=1)
    assert module_path.read_bytes() == module_before

    changed_output = replace(config.output, chunks_size=config.output.chunks_size + 1)
    with pytest.raises(ValueError, match="stored output/video layout differs from config"):
        sync_config(replace(config, output=changed_output), workers=1)
    assert module_path.read_bytes() == module_before

    info_path = output_root / "0/meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["shape"] = [999]
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="feature 'action' differs from stored layout"):
        sync_config(config, workers=1)
