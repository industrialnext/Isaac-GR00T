# Internal zdata datasets: conversion and GR00T N1.7 training

This is the maintained reference for training GR00T N1.7 on Industrial Next
`zdata_hdf5` recordings. Continue with [WebSocket serving](industrialnext_serving.md)
after selecting a checkpoint.

```text
zdata episode.h5 → LeRobot v2.1 datasets → statistics and loader checks
  → GR00T N1.7 fine-tune → checkpoint + saved processor → WebSocket serving
```

The implementation is experimental. A successful conversion or training run does not
establish source semantics, policy quality, or robot readiness. The dated integration
plans have been consolidated here; historical run records remain in Git and `artifacts/`.
Their dataset counts, checkpoint choices, and hardware measurements are not current defaults.

## Environment and entry points

Run commands from the repository root. Initialize the pinned RPC workspace member before
resolving dependencies, even when initially working on conversion:

```bash
git submodule update --init --recursive packages/industrialnext_rpc
```

For a supported dGPU host with system/CUDA prerequisites installed:

```bash
uv sync --extra industrialnext
```

See the [deployment setup](../scripts/deployment/README.md) for platform prerequisites.
GR00T uses its own environment; do not substitute the `industrialnext_ai` environment.
The commands below use the dGPU environment. On Orin, Thor, or Spark, use the platform
installer and activation helper, then the activated `python` interpreter instead of
root-project `uv run`, as described in the
[fine-tuning guide](../getting_started/finetune_new_embodiment.md).
Conversion needs `h5py`, ffmpeg, and ffprobe; the dGPU commands overlay `h5py` without
changing the project dependency manifests.

| Owner | Responsibility |
|---|---|
| [run_zdata_pipeline.py](../scripts/lerobot_conversion/run_zdata_pipeline.py) | CLI orchestration |
| [config.py](../scripts/lerobot_conversion/zdata_pipeline/config.py) | YAML parsing, layout, generated modality module |
| [source.py](../scripts/lerobot_conversion/zdata_pipeline/source.py) | HDF5 reading, named fields, images, target alignment |
| [convert.py](../scripts/lerobot_conversion/zdata_pipeline/convert.py) | Incremental conversion, ledgers, transactions |
| [check.py](../scripts/lerobot_conversion/zdata_pipeline/check.py) | Statistics, validation, freezing, training launch |
| [launch_finetune.py](../gr00t/experiment/launch_finetune.py) | GR00T training configuration and execution |

## One embodiment YAML for training and serving

Start from a matching example and give a new experiment its own name and output paths:

| Example | Model inputs | Targets |
|---|---|---|
| [semihumanoid.yaml](../configs/embodiments/semihumanoid.yaml) | Head and two bottom wrist RGB views; bilateral EEF, gripper, F/T (32 state dimensions) | Executed bilateral EEF/gripper commands (20 dimensions), offset 0, horizon 40 |
| [xarm_psyonic_mori_bracket.yaml](../configs/embodiments/xarm_psyonic_mori_bracket.yaml) | Static and right wrist RGB; right EEF and six-dimensional hand (15 state dimensions) | Future observed EEF/hand (15 dimensions), offset 1, horizon 40 |

These configs include historical source/output/checkpoint paths and resource settings.
Review them before use. The semihumanoid example includes Matcha and Ube; listing a source
does not certify that every configured action coordinate is authoritative in that source.
The xArm example enables target preprocessing; copying it also copies that choice.

| YAML section | What to specify |
|---|---|
| `name` | Python identifier; also names the generated example module |
| `source` | Root, subsets/globs, episode glob, excluded path fragments, sampling FPS |
| `output` | Converted root, robot type, subset-prefix removal, `val_every`, chunk size |
| `cameras` | Ordered model video key → source HDF5 / live wire camera name |
| `state` | Ordered model keys and source field groups |
| `action` | Source, observation offset, horizon, optional preprocessing, ordered action keys and representations |
| `tasks` | Source task attributes and explicit UUID → instruction overrides |
| `select` | Admission attribute requirement and policy-type allowlist |
| `warn`, `continuity` | Quality warning thresholds and optional timestamp-gap splitting |
| `train` | Base model, output root, GPU/global batch, steps or epochs, optimizer, image preprocessing, RTC training |
| `serving` | Online field lengths, image size, units/frame notes, checkpoint, network and RTC profile; see the serving guide |

Configured paths expand `~`. Conversion/training paths are relative to the working
directory, so run from the repository root; they are not relative to the YAML's directory.
Serving model paths in the YAML resolve explicitly from the repository root.
Do not hand-edit the generated
`examples/<name>/<name>_config.py`: `sync` derives it from the YAML and registers
`NEW_EMBODIMENT`. One run requires compatible layouts across its datasets; this wrapper
does not build a heterogeneous multi-embodiment training mixture.

## Source and representation contract

The reader consumes `state/flat` and its `field_names`/`field_slices`, the selected action
array and its field metadata, `frame/elapsed_ms`, `frame/done`, task attributes, and
`images/<camera>/{blob,offsets,frame_ref_index}`. Read fields by name: native layouts can
insert joint arrays and shift every later slice. Extra, unselected fields are ignored.

- **Action source:** `executed` reads `action/executed`; `observation` reads `state/flat`.
  With observation offset `k`, converted row `t` pairs observation `t` with target `t+k`.
  Conversion removes the last `k` observation rows of each segment. Nonzero offsets are
  only allowed for observation targets. Selecting commands versus future measurements is
  a modeling decision, not an interchangeable format setting.
- **Rotations:** `source_columns_to_groot_rows` reconstructs the rotation matrix from
  Industrial Next's first two columns and encodes GR00T's first two rows. A six-number
  passthrough or permutation is wrong. The shared [rot6d helpers](../gr00t/data/state_action/rot6d.py)
  also implement the inverse serving conversion.
- **Relative EEF:** configure `rep: RELATIVE`, `type: EEF`, `format: XYZ_ROT6D`, and the
  matching `state_key`. Store physical targets in converted data; GR00T's processor owns
  relative transformation and normalization. The examples use absolute gripper/hand targets.
- **Images:** each source reference selects JPEG bytes for one observation row. Repeated
  references produce repeated video frames at the configured FPS, preserving alignment.
  This neither increases camera freshness nor creates extra visual information. This
  pipeline consumes RGB, not depth.
- **Continuity:** `split_on_gap_ms` partitions source episodes at large time gaps. Segments
  need at least `horizon + observation_offset` source rows. Segment timestamps/frame
  indices restart at zero and the final row is marked done. Model state/video history is
  `[0]`; action deltas are `0..horizon-1`. N1.7's configured limits are 132 state/action
  dimensions and horizon 40.

When `require_valid_for_training: true`, missing or false admission attributes are skipped.
Configured policy-type exclusions and too-short segments are also intentional skips.
Missing required fields/cameras, invalid references, non-finite data, degenerate rotations,
or incompatible dimensions/FPS fail conversion for the affected source.

Camera coverage/age and residual thresholds under `warn` are warnings, not automatic
quality filters. General per-field presence, freshness, and action-ownership masks are
not propagated into GR00T training by this converter. Finite values alone are insufficient:
audit representative raw episodes and exclude or appropriately project unsupported
targets before training. Do not label an unowned stationary coordinate as a valid hold
command merely because its values look constant.

Optional `action.preprocessing` repairs/smooths selected target fields within configured
bounds and records evidence in conversion bookkeeping; it does not rewrite source HDF5.
Review its effect against raw targets, including hand exclusions and SO(3) rotation handling.
Implementation: [target_preprocessing.py](../scripts/lerobot_conversion/zdata_pipeline/target_preprocessing.py).

## Convert, validate, and train

Choose the reviewed config. The following separates data work from GPU launch:

```bash
CFG=configs/embodiments/semihumanoid.yaml

uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
  sync --config "$CFG" --dry-run
uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
  sync --config "$CFG" --workers 4
uv run --no-sync python scripts/lerobot_conversion/run_zdata_pipeline.py \
  stats --config "$CFG" --jobs 4
uv run --no-sync python scripts/lerobot_conversion/run_zdata_pipeline.py \
  check --config "$CFG" --full

CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
  scripts/lerobot_conversion/run_zdata_pipeline.py train --config "$CFG" \
  --smoke-max-steps 30 --smoke-batch 8

uv run --no-sync python scripts/lerobot_conversion/run_zdata_pipeline.py \
  train --config "$CFG"
```

The smoke uses one GPU and a separate timestamped `_smoke` run. Choose a batch that fits
the available GPU. It verifies checkpoint creation, finite losses, positive postfix
coverage, and persistence of the RTC training setting. With prefix training enabled it
also requires a sampled zero prefix and varying prefix lengths; a two-step/batch-one
smoke can fail simply because it did not sample enough of that distribution.

The convenience command below **also launches training**:

```bash
uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
  --config "$CFG"
```

With no subcommand the flow is `sync → stats → check → train`. Use `sync --dry-run`
for inspection, not the convenience command. `check` samples each dataset;
`check --full` scans all episode records and verifies video frame counts with ffprobe.
Loader checks require statistics. `stats` covers train and validation outputs; `train`
automatically generates missing statistics only for training datasets and loads one sample
from each before launching.

## Outputs, appends, and recovery

```text
<output.root>/
  _ledgers/<subset>.json
  <dataset>/
    data/chunk-*/episode_*.parquet
    videos/chunk-*/observation.images.<key>/episode_*.mp4
    meta/{info,modality,stats,relative_stats}.json
    meta/{episodes,tasks}.jsonl
    _layout.json
  <dataset>_val/                 # when val_every is enabled
```

The Parquet columns include `observation.state`, `action`, `timestamp`, `frame_index`,
`episode_index`, `index`, `task_index`, and `next.done`. Language resolves through
`task_index` and the task table. A path-hash split gives approximately `1/val_every`
validation recordings; all segments from one source stay on the same side. This is not
content deduplication: renamed copies can leak across splits.

`sync` retains existing assignments and appends new source paths. It invalidates both
statistics files only for changed datasets. A writer lock and temporary transaction
journals protect commits; after interruption, rerun `sync` to roll forward before
stats/check/train. Successful sources survive other sources' failures, and the command
returns nonzero when failures occurred. Failed sources are retried; skipped sources are
remembered.

In-place source edits need explicit `sync --reconvert <subset>/<relative-path>`.
Existing segment counts and lengths must remain compatible with assigned indices.
Use a new output root for layout, target-semantics, or segment-topology changes.
The append ledger is not a content audit, and renamed sources can be converted twice.
Avoid converting actively written episodes or modifying a corpus while a run consumes it.

New task IDs append to conversion metadata. Existing text is retained on conflicts unless
an explicit override changes it. Serving accepts the catalog in `tasks.text_overrides`,
so adding a training task does not automatically add an online task: synchronize exact
UUID/text pairs with the deployment catalog.

## Training identity and optional freezing

The launcher uses `nvidia/GR00T-N1.7-3B` in the examples, the generated modality module,
`NEW_EMBODIMENT`, and `torch.distributed.run`. It joins non-`_val` dataset paths using
`os.pathsep`. Set resources in `train.gpus`, `train.batch`, and `train.workers`; global
batch must be divisible by GPU count. Historical four-GPU batch sizes are not portable
memory recommendations.

When `max_steps` is absent, steps are `ceil(epochs * trainable_starts / global_batch)`;
each converted segment contributes `max(0, length - horizon + 1)` starts. Fresh runs use
`<train.out_base>/<name>_<UTC timestamp>` and write `run_manifest.json`. Resume explicitly
with `train --config "$CFG" --resume-from /path/to/existing-run`, not a checkpoint
subdirectory; retain the intended data/config and resumable optimizer state.

Default runs are mutable experiments with a lightweight manifest of dataset paths,
counts, settings, and command. For content-bound runs, finish stats, then:

```bash
uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
  freeze --config "$CFG"
uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
  train --config "$CFG" --freeze
```

`freeze` performs a full check and writes `_frozen_corpus_manifest.json`, binding the
YAML, generated module, converted artifacts and statistics by hash, and source files by
size/mtime guards. Source guards are not raw HDF5 content hashes or semantic admission.
`train --freeze` requires and verifies this marker; the config-only command with `--freeze`
can create it after conversion or reuse it and skip sync. Freezing is a pipeline guard,
not filesystem immutability.

**Plain `train` and the config-only command without `--freeze` remove an existing freeze
marker.** Keep using `--freeze` for a frozen run/resume. Do not edit the frozen YAML,
including its serving checkpoint path; use a serving CLI override or a separate deployment
copy instead. A different experiment should use a new output root.

For RTC training, `train.rtc_training_max_prefix_steps: M` samples integer prefixes
uniformly from `0..M`, supplies a clean prefix, and trains the postfix. Zero disables
this objective. The wrapper requires at least 16 postfix steps. Verify the saved
`config.json` records `M`; runtime `trained_prefix` cannot exceed it.

## Evaluate and hand off

Evaluate each held-out subset with the checkpoint's saved modality configuration:

```bash
MODEL=/absolute/path/to/selected/checkpoint
VAL=/absolute/path/to/converted/subset_val
uv run --no-sync python gr00t/eval/open_loop_eval.py \
  --model-path "$MODEL" --dataset-path "$VAL" \
  --embodiment-tag NEW_EMBODIMENT --traj-ids 0 1 2 \
  --execution-horizon 40 --steps 400 \
  --save-plot-path /tmp/gr00t-open-loop
```

Use valid trajectory IDs and an execution horizon no larger than the trained horizon.
Compare train versus held-out accuracy by task/source, inspect physical position and
orientation errors, and compare checkpoints rather than declaring the last one best.
The wrapper excludes `_val` datasets from training; its smoke does not measure generalization.
For RTC, also evaluate continuity, rejection/hold rate, and latency as described in the
[serving guide](industrialnext_serving.md#rtc-modes-and-timing).

Transfer the complete selected model/config and saved `processor/` (or root processor)
assets, the reviewed YAML, and the run/evaluation records. Provide the tokenizer/image
processor assets referenced by the checkpoint, including the Cosmos cache when required.
Keep training normalization with the checkpoint; do not replace it with validation or live
data statistics.

## Validation and troubleshooting

For converter or representation changes, the focused regression command is:

```bash
uv run --no-sync --with h5py python -m pytest \
  scripts/lerobot_conversion/test_zdata_pipeline.py \
  tests/gr00t/data/test_rot6d_conventions.py -q
```

| Symptom | Check |
|---|---|
| Workspace dependency cannot resolve | Initialize the pinned RPC submodule before `uv sync` |
| HDF5 images cannot be read | Resolve external links relative to the episode, configured camera names, offsets and references |
| No selected sources | Root/subset/episode glob, admission attributes, policy type and minimum segment length |
| Append layout mismatch | Use a new root for intentional camera/field/target changes |
| Missing loader statistics | Run `stats`; do not retain old stats after changing episode data |
| Frozen verification fails | Restore the bound config/artifacts or create a new corpus; do not silently bypass with plain `train` |
| Implausible orientation with finite values | Check row/column rot6d conversion and paired EEF state key |
| Loss improves but robot behavior does not | Audit target ownership, offset, camera identity, units/frame, masks and held-out quality |

Optional source survey helpers live beside the converter, but inspect their source-path
assumptions before using them for a new embodiment. GPU training, checkpoint evaluation,
WebSocket loopback, and real ROS validation establish different parts of readiness.
