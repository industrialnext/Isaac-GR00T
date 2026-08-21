# GR00T Industrial Next experimental pipeline

This is the reusable experimental path for internal zdata datasets:

```text
zdata HDF5 -> LeRobot v2 -> GR00T fine-tune -> Industrial Next DirectServer
  -> industrialnext_ros2 async policy client -> controller-owned robot safety
```

The embodiment YAML is the single operator configuration. The checked-in examples are
`configs/embodiments/xarm_psyonic_mori_bracket.yaml` and
`configs/embodiments/semihumanoid.yaml`.

## Convert, check, and fine-tune

From the repository root, the normal experimental command is:

```bash
uv run --no-sync --with h5py python \
  scripts/lerobot_conversion/run_zdata_pipeline.py \
  --config configs/embodiments/xarm_psyonic_mori_bracket.yaml
```

With no subcommand it runs incremental `sync -> stats -> check -> train`. Training is mutable and
unfrozen by default. It still generates missing statistics, imports the generated modality module,
loads one sample from every training dataset, prints the full training command, and records every
dataset and training argument in a lightweight `run_manifest.json`.

If the output root has an old `_frozen_corpus_manifest.json`, the experimental command removes the
marker before sync. This prevents stale content-bound metadata from remaining attached to changed
data. Use `--freeze` only when the slower content-bound workflow is intentionally wanted:

```bash
uv run --no-sync --with h5py python \
  scripts/lerobot_conversion/run_zdata_pipeline.py \
  --config configs/embodiments/xarm_psyonic_mori_bracket.yaml \
  --freeze
```

The `sync`, `stats`, `check`, `freeze`, and `train` subcommands remain available. For example, a
short one-GPU training check is:

```bash
uv run --no-sync --with h5py python \
  scripts/lerobot_conversion/run_zdata_pipeline.py train \
  --config configs/embodiments/xarm_psyonic_mori_bracket.yaml \
  --smoke-max-steps 2 --smoke-batch 1
```

Do not run the config-only command merely to inspect a config: its final stage launches training.

## Select and configure the checkpoint

Set `serving.model_path` in the same YAML to the checkpoint directory to serve. Relative paths are
resolved from the Isaac-GR00T repository root; `~` paths are expanded. The xArm example currently
selects `checkpoint-5778`.

The serving profile derives these values from the existing data sections:

- `cameras`: model video key to robot wire image key;
- `state`: ordered wire fields assembled into each model state key;
- `action.keys`: model action keys split back into ordered wire fields;
- `action.horizon` and `action.observation_offset`: output size and async timeline offset;
- `tasks.text_overrides`: accepted task UUID and exact model instruction.

The `serving` section adds the information unavailable from converted data: wire field lengths,
image size, ignored optional observations, units/frame notes, checkpoint/device/network defaults,
supported RTC modes, and gripper monitoring fields.

`source_columns_to_groot_rows` is a named matrix conversion. Industrial Next's column rot6d and
GR00T's row rot6d must never be copied numerically in either direction.

## Start the robot-facing server

The normal command needs only the config:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync --extra industrialnext python \
  gr00t/eval/run_gr00t_industrialnext_server.py \
  --config configs/embodiments/xarm_psyonic_mori_bracket.yaml
```

The xArm config binds `127.0.0.1:10012`, uses `cuda:0`, strict checkpoint preprocessing,
`NEW_EMBODIMENT`, a 40-row action horizon, offset one, RTC off, and no server-side gripper snap.
CLI flags such as `--model-path`, `--device`, `--host`, `--port`, `--control-hz`, and `--rtc-mode`
are optional debugging overrides.

This entry point runs a profile-aware synthetic inference before it binds. Use
`run_gr00t_server.py` only for model-only ZMQ debugging; it is not the Industrial Next robot
protocol.

## Test without publishing robot commands

With the server running, a config-driven synthetic DirectClient smoke is:

```bash
uv run --no-sync --extra industrialnext python \
  gr00t/eval/smoke_industrialnext_loopback.py \
  --config configs/embodiments/xarm_psyonic_mori_bracket.yaml
```

Each run writes a uniquely named JSON report under `/tmp`; pass `--output-json-path` when a fixed
report location is useful.

The xArm recorded-data example uses the same async protocol:

```bash
uv run --no-sync --extra industrialnext python \
  examples/xarm_psyonic_mori_bracket/industrialnext_client.py \
  --config configs/embodiments/xarm_psyonic_mori_bracket.yaml \
  --dataset-path data/training_data/gr00t/xarm_psyonic_mori_bracket_v1_20260821/xarm_psyonic_val
```

On the deployment machine, first run the real `industrialnext_ros2` client without command
publication. Compare registration metadata, sparse image keys, state/action fields, units, EEF
frame, rot6d convention, offset behavior, reconnect behavior, and latency. Then use the established
controller safety path for a supervised reduced-speed trial. Workspace, collision, joint/velocity,
emergency-stop, and command ownership remain controller/ROS responsibilities.

## Add another zdata embodiment

Copy an embodiment YAML and update the source/output paths, cameras, state/action field groups,
field lengths, tasks, action offset, ignored observations, RTC modes, gripper keys, and selected
checkpoint. Identity fields and `source_columns_to_groot_rows` are currently supported.

A different source HDF5 schema still needs conversion code. A genuinely new online transform also
needs one small named implementation. Production artifact sealing, hashes, release admission,
deployment bundles, and supervisor integration are intentionally deferred while GR00T remains an
experimental policy source.
