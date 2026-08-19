# Config-Driven zdata_hdf5 → GR00T Pipeline — Design

**Date:** 2026-08-19

**Status:** Implemented and validated

**Supersedes:** the removed semihumanoid-only converter, dataset helper, test module, and
training shell entry point

**Related:**
[`2026_0818_semihumanoid_gr00t_conversion_plan.md`](2026_0818_semihumanoid_gr00t_conversion_plan.md),
which records the completed semihumanoid conversion and training work. Where that plan
describes a sealed release manifest, catalog-version admission, or a full-corpus migration
gate, this document defines the simpler behavior for the generalized pipeline.

---

## Purpose

Internal zdata datasets are working datasets. During collection, new episode directories
may arrive several times a day, and users should be able to append them and try another run
without creating a release, pinning a source snapshot, or satisfying an admission process.

The existing semihumanoid converter already handles the hard data-format details, but its
camera map, selected fields, action source, task order, quality rules, and GR00T modality
keys are Python constants. Copying the script for every embodiment would make those copies
drift. The generalized pipeline moves only those layout choices into one small YAML file
and keeps the zdata_hdf5 reader and LeRobot writer shared.

The target workflow is deliberately short:

```bash
# Append source episodes not processed before.
uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py sync \
  --config configs/embodiments/semihumanoid.yaml

# Refresh required statistics if data changed, then launch training.
uv run --no-sync python scripts/lerobot_conversion/run_zdata_pipeline.py train \
  --config configs/embodiments/semihumanoid.yaml
```

`sync` never launches training. Re-running it after more collection is the normal operating
model, not a special release operation. `h5py` remains an on-demand conversion dependency;
`stats`, `check`, and `train` run in the normal project environment without that overlay.

## Design principles

1. **The corpus may keep growing.** Counts, source timestamps, and source-tree contents are
   observations, not a sealed contract.
2. **One config describes the data layout.** The same state, action, camera, and annotation
   entries drive conversion, `meta/modality.json`, and the Python `ModalityConfig`.
3. **Only unusable data stops conversion.** Structural problems that prevent correct
   reading or writing are errors. Quality anomalies are reported and, unless explicitly
   selected out by the config, do not block an append.
4. **Successful work is retained.** A bad episode does not roll back other episodes. A
   later `sync` retries unfinished inputs.
5. **Meaning remains an operator decision.** The config says which action array and fields
   to use. The pipeline checks that they can be read consistently; it does not try to prove
   that the selected action represents the intended command semantics.

### Non-goals

- Dataset sealing, immutable release manifests, content checksums, source fingerprints, or
  strict source-identity checks.
- A reader plug-in framework. This design supports the shared `zdata_hdf5` format with
  different layouts. A second reader abstraction is justified only by a second format.
- Multi-embodiment training in one GR00T run.
- A config-driven evaluation harness. GR00T evaluation loads the modality config saved in
  the checkpoint.
- Automatic inference of action ownership, task meaning, or camera meaning.
- Depth, progress prediction, or state-history support not consumed by GR00T N1.7.
- Full-corpus scans on every append.

---

## Minimal architecture

```text
scripts/lerobot_conversion/
├── run_zdata_pipeline.py          # sync, stats, check, and train commands
└── zdata_pipeline/
    ├── common.py                  # HDF5-free stats names and transaction-journal check
    ├── config.py                  # YAML parsing and derived canonical layout
    ├── source.py                  # zdata_hdf5 reading, field assembly, rot6d conversion
    ├── convert.py                 # discovery, selection, LeRobot writing, append ledger
    └── check.py                   # small post-write checks and optional full diagnostics

configs/embodiments/semihumanoid.yaml
examples/semihumanoid/semihumanoid_config.py     # generated from the YAML
```

The CLI contains orchestration and training command construction. `source.py` is the only
module that knows the HDF5 structure. Video writing and `meta/` generation stay together in
`convert.py` because they are one conversion transaction. `common.py` prevents `stats`,
`check`, and `train` from importing the optional `h5py` dependency through conversion code.
Separate writer, layout, QC, modality, and validation frameworks would add interfaces
without adding another use case.

## Configuration

The schema is intentionally permissive about extra source fields and attributes. It names
only what the conversion consumes. Optional sections may be omitted. Unknown keys in a
state, action, or camera layout are errors because silently ignoring a misspelled layout
key would change the training tensors; unknown keys elsewhere are warnings.

```yaml
name: semihumanoid

source:
  root: data/training_data/semihumanoid
  subsets: ["flexiv_*"]
  episode_glob: "*/20*/*/*/*/episode.h5"
  exclude_path_contains: ["_failed_recordings"]
  fps: 50

output:
  root: data/training_data/gr00t/semihumanoid_260818
  robot_type: semihumanoid_bimanual
  strip_subset_prefix: "flexiv_"
  val_every: 20                    # deterministic path-based split; 0 disables validation
  chunks_size: 1000

cameras:                           # canonical GR00T key -> zdata image group
  head: head_rgb
  left_wrist: eoat_left_bottom_rgb
  right_wrist: eoat_right_bottom_rgb

video:
  codec: libx264
  crf: 23
  preset: veryfast
  pixel_format: yuv420p

state:                              # order defines observation.state
  - key: left_eef
    fields: [left_arm_pose_pos, left_arm_pose_rot]
    rot6d: source_columns_to_groot_rows
  - {key: left_gripper, fields: [left_gripper]}
  - {key: left_ft, fields: [left_ft]}
  - key: right_eef
    fields: [right_arm_pose_pos, right_arm_pose_rot]
    rot6d: source_columns_to_groot_rows
  - {key: right_gripper, fields: [right_gripper]}
  - {key: right_ft, fields: [right_ft]}

action:
  source: executed                 # reads action/executed; chosen by the dataset owner
  horizon: 40
  keys:
    - key: left_eef
      fields: [left_arm_pose_pos, left_arm_pose_rot]
      rot6d: source_columns_to_groot_rows
      rep: RELATIVE
      type: EEF
      format: XYZ_ROT6D
      state_key: left_eef
    - {key: left_gripper, fields: [left_gripper], rep: ABSOLUTE,
       type: NON_EEF, format: DEFAULT}
    - key: right_eef
      fields: [right_arm_pose_pos, right_arm_pose_rot]
      rot6d: source_columns_to_groot_rows
      rep: RELATIVE
      type: EEF
      format: XYZ_ROT6D
      state_key: right_eef
    - {key: right_gripper, fields: [right_gripper], rep: ABSOLUTE,
       type: NON_EEF, format: DEFAULT}

tasks:
  id_attr: task_uuid
  text_attr: task_text
  text_overrides:                  # optional task ID -> preferred instruction text
    generic_pick: Pick the grounded target object and hold it securely in the gripper.
    generic_place: Place the currently held object at the grounded destination and release it.
    bracket_handover: Hand over the bracket from the gripper holding it to the opposite gripper and secure it in the receiving gripper.

select:                             # cheap, intentional source selection only
  require_valid_for_training: true
  policy_types: [expert]

warn:                               # observations, not admission gates
  camera_coverage_below: 0.45
  camera_coverage_above: 0.80
  camera_age_p99_ms_above: 150
  frame_gap_ms_above: 40
  nonzero_action_residual: true

continuity:
  split_on_gap_ms: null             # null preserves current episode boundaries

train:
  base_model: nvidia/GR00T-N1.7-3B
  out_base: outputs/gr00t
  gpus: 4
  batch: 256
  epochs: 2.2                       # max_steps may be supplied instead
  max_steps: null
  lr: 2.8e-4
  workers: 8
  save_steps: 1000
  save_total_limit: 5
  state_dropout_prob: 0.2
  shortest_image_edge: 256
  crop_fraction: 0.95
  color_jitter: {brightness: 0.15, contrast: 0.15, saturation: 0.2, hue: 0.1}
  use_wandb: false
  wandb_project: gr00t-semihumanoid
  weight_decay: 1.0e-5
  warmup_ratio: 0.05
```

Extra HDF5 fields, unselected cameras, and unrelated attributes are ignored. Required
field widths are read from each episode's `field_slices`; the assembled canonical widths
must remain the same within an output dataset. An EEF action entry must resolve to 9 values
(XYZ + rot6d), because that is required by GR00T's EEF representation.

With `require_valid_for_training: true`, an explicit false source value is skipped. A
missing value is accepted with a warning, so older recordings do not need a metadata
migration. A configured `policy_types` list is an intentional selection rule; omit it to
accept every policy type.

`rot6d` is explicit because the source recorder and GR00T use different 6D rotation
conventions. On an entry that sets `rot6d`, the transform applies to its single 6D rotation
field, not to the position field or the concatenated block; the entry is invalid if that
rotation field cannot be identified unambiguously or its two source axes are degenerate. A
field-name heuristic would recreate the earlier silent orientation error.

The example batch of 256 was measured on four local RTX 4090 cards reporting 49,140 MiB
each. It is not a default for ordinary 24 GiB 4090 cards; those users must select a batch
that fits their hardware.

### Tasks grow with the data

Task IDs and text come from each episode by default. The first occurrence of a new task ID
appends it to that output dataset's `meta/tasks.jsonl`; existing task indices never move.
Later catalog-version changes are ignored. If the same task ID arrives with different text,
`sync` warns and keeps the existing text unless `tasks.text_overrides` explicitly replaces
it. Editing an override and running `sync` also updates the existing `tasks.jsonl` and
`episodes.jsonl` text without reprocessing episodes or invalidating numeric statistics. New
tasks therefore require no catalog release or schema update.

---

## Output layout

Each source subset maps to one LeRobot v2.1 training dataset and, when `val_every` is
enabled, a sibling `<name>_val` dataset. The on-disk shape remains the one consumed by the
current GR00T loader:

```text
<output-root>/
├── _ledgers/<subset>.json
├── _staging/                       # transient work, normally empty after sync
├── <name>/
│   ├── data/chunk-*/episode_*.parquet
│   ├── videos/chunk-*/observation.images.<camera>/episode_*.mp4
│   ├── meta/{info,modality,stats,relative_stats}.json
│   ├── meta/{episodes,tasks}.jsonl
│   ├── .sync_transaction.json      # present only during interrupted finalization
│   └── _layout.json
└── <name>_val/                    # only when validation splitting is enabled
```

Each parquet row contains `observation.state`, `action`, `timestamp`, `frame_index`,
`episode_index`, global `index`, `task_index`, and `next.done`. For each configured camera,
the writer follows the source `frame_ref_index` and emits one video frame per state row at
the configured dataset rate. `stats.json` and `relative_stats.json` may be absent between a
successful append and the next `stats` or `train` command.

---

## Single layout source

The pipeline derives canonical slice offsets from the ordered `state` and `action.keys`
entries and emits both:

1. `<dataset>/meta/modality.json`, including state/action slices, canonical video mappings,
   and `annotation.human.task_description` with `original_key: task_index`. No separate
   annotation column is written to parquet.
2. `examples/<name>/<name>_config.py`, which registers the same modality keys under
   `EmbodimentTag.NEW_EMBODIMENT` for `--modality-config-path`.

The generated `ModalityConfig` uses `delta_indices=[0]` for video and state and
`delta_indices=list(range(action.horizon))` for action. It translates each action entry's
`rep`, `type`, `format`, and optional `state_key` directly into GR00T `ActionConfig`
values.

The generated Python file carries a short “generated from YAML” header and is overwritten
when its layout changes. It contains no whole-config hash: changing a training parameter,
source glob, warning threshold, or output path must not create a false layout mismatch.

For each output dataset, the converter also writes a small `_layout.json` containing the
direct camera-group mapping, selected state/action fields and transforms, action source and
representation values, sampling rate, observed image shape, derived slices, chunk size,
robot type, and declared video codec/pixel format. When appending, those values,
`meta/modality.json`, and the corresponding direct values in `meta/info.json` must still
agree with the YAML—even when there are no new source paths. This is the one compatibility
check the append path needs: mixing different dimensions, action semantics, path geometry,
or physical viewpoints under existing canonical columns would make the output internally
ambiguous. An intentional layout change uses a new output directory or an explicit full
rebuild. `_layout.json` contains no source paths, hashes, corpus counts, or source identity
and does not bind the output to a snapshot.

---

## Commands and normal flow

### `sync`

`sync` is the everyday command. For every configured subset it:

1. Discovers episode files and skips paths already marked complete in the append ledger.
2. Applies the small `select` section. Episodes explicitly marked invalid, from an
   unselected policy type, or too short to produce the configured action horizon are
   recorded as skipped; they do not fail the run.
3. Reads configured fields by name, converts rot6d where requested, and writes parquet and
   MP4 files. Source rows remain in their recorded order.
4. Appends successful episodes to `episodes.jsonl` and new task IDs to `tasks.jsonl`, then
   refreshes `info.json` and `modality.json` from current output metadata.
5. Removes `stats.json` and `relative_stats.json` only for datasets that changed, because
   the current GR00T stats cache does not notice episode-only appends.
6. Prints warning/skip lines as they occur, a compact discovered/complete/candidate/accepted
   summary per subset, commit counts per changed dataset, and a final failure summary.

Warnings for camera coverage, image age, timestamp gaps, task-text changes, or nonzero
`action/residual` are visible in the summary but do not block conversion. The dataset owner
can tighten `select` or add a continuity threshold later if a particular collection needs
it.

### Minimal append ledger

`<output>/_ledgers/<subset>.json` is operational state, not a release manifest. Version 2
stores completed segment assignments and concise reasons for intentionally skipped source
paths. For example:

```json
{
  "version": 2,
  "sources": {
    "robot/2026/08/17/episode_expert/episode.h5": {
      "status": "complete",
      "task_id": "generic_pick",
      "segments": [{
        "source_start": 0,
        "source_end": 386,
        "dataset": "matcha_v2",
        "split": "train",
        "episode_index": 0,
        "index_offset": 0,
        "length": 386,
        "task_index": 0
      }]
    },
    "robot/2026/08/17/too_short_expert/episode.h5": {
      "status": "skipped",
      "reason": "all segments are shorter than the action horizon",
      "segments": []
    }
  }
}
```

Workers first write new episodes under `_staging/`, without final dataset indices. The main
process commits successful staged results in relative-path order and assigns contiguous
episode/global indices. Before replacing any final file for a changed dataset, it atomically
writes `<dataset>/.sync_transaction.json` with the transaction directory, ordered
staging-to-final replacements, worker-stage cleanup paths, and stats paths to invalidate. It
then replaces parquet, videos, regenerated metadata, and the staged ledger; removes stale
stats and consumed stage paths; and deletes the journal and transaction directory.

On startup, `sync` acquires one advisory writer lock on the output directory and rolls any
remaining journal forward idempotently. `stats`, `check`, and `train` stop only while such a
journal is present, because those commands must not observe a half-committed dataset. Failed
inputs receive no ledger entry and are retried by the next `sync`; successful peers remain
committed. `sync --dry-run` performs discovery, parsing, compatibility checks, and planning
without creating a lock file, stage, generated module, layout, ledger, or metadata. No file
hash, mtime, source-tree checksum, Git SHA, catalog version, or config fingerprint is stored.

This intentionally assumes the collector appends new episode paths and does not silently
rewrite completed episodes in place. If an episode was repaired at the same path, the user
runs `sync --reconvert <source-subset>/<relative-path>`; the pipeline rewrites the same
output assignment and invalidates that dataset's statistics only when the new segment count
and lengths equal the old topology. A topology-changing repair uses a new output root or a
deliberate rebuild so later global offsets do not become invalid. A previously skipped path
has no assigned topology and may become complete when explicitly reprocessed. A renamed
completed source looks new and may be converted again, which is an accepted tradeoff for
the relaxed identity model.

The validation split is computed from the relative source path, so backfilled directories
do not move already converted episodes between train and validation. It is stable placement,
not a content identity check.

### Continuity

By default, the converter preserves one output episode per source episode and reports large
`frame/elapsed_ms` gaps. This matches the lightweight append workflow. If a collection is
known to contain pauses that should not be crossed by GR00T action windows, setting
`continuity.split_on_gap_ms` partitions only newly converted episodes at those gaps. The
setting is optional rather than a corpus-admission gate.

Changing the setting affects new inputs only. Use a new output directory or a deliberate
full rebuild if earlier episodes also need repartitioning; routine appends do not rewrite
history.

### `stats`

`stats` regenerates GR00T's `stats.json` and `relative_stats.json` for datasets missing
them, running independent datasets in parallel. It is also invoked automatically by
`train`. Unchanged datasets keep their existing files. Several `sync` runs can therefore
accumulate new recordings without paying the stats cost until training is actually wanted.

### `train`

`train` performs only the checks needed to construct a valid GR00T invocation:

- train dataset paths exist and do not include the generated `_val` datasets;
- required stats exist after the automatic stats step;
- global batch is divisible by GPU count;
- action horizon and canonical state/action dimensions fit the base checkpoint limits;
- the generated modality file can be imported; and
- one loader sample can be read from every training dataset before they are joined.

It joins the training dataset paths with `os.pathsep`, matching
`gr00t/experiment/launch_finetune.py`'s multi-dataset interface.

GPU occupancy and free disk are reported, not treated as hard gates. The command always
passes `shortest_image_edge` and `crop_fraction` so preprocessing does not fall back to a
different legacy crop. It does not pass `--experiment-name`, avoiding an unexpected nested
output directory.

Training launches through the active interpreter with
`-m torch.distributed.run --standalone --nproc-per-node <gpus>`, so no fixed rendezvous port
or second environment is introduced. A plain run writes to
`<out_base>/<name>_<UTC timestamp>` and refuses to reuse an existing computed directory.
Resumption is explicit with `train --resume-from <existing-run-directory>`, which reuses that
directory and passes `--resume-from-checkpoint`.

When `train.epochs` is used, the pipeline derives
`ceil(epochs * current_trainable_starts / batch)` immediately before launch and prints both
the resulting step count and effective epochs. `train.max_steps`, when non-null, overrides
the derivation. A growing corpus therefore does not silently reduce the requested training
coverage. `current_trainable_starts` is computed from train episodes after any configured
gap splitting, with each segment contributing `max(0, length - horizon + 1)`.

There is deliberately no `all` command. Data append and GPU launch remain two explicit
operator actions.

### `check` (on demand)

Normal `sync` checks only episodes it touches plus the metadata it rewrites. `check` exists
for debugging or a deliberate audit; it is not part of every append.

- `check` verifies metadata counts and reads one parquet/video sample per discovered dataset.
  If statistics are already present, it also opens one GR00T loader sample; otherwise it
  reports that the loader check is deferred to `train` after automatic stats generation.
- `check --full` scans every parquet/video record, verifies episode/frame/global indices and
  lengths, and uses `ffprobe -count_frames` for exact video-frame counts.

Source profiling remains a separate diagnostic activity using the existing survey helpers.
It is useful when onboarding a new embodiment, but it is not a prerequisite for routine
appends.

---

## Essential failure handling

An individual episode fails conversion only when the requested output cannot be produced
correctly, for example:

- a configured field or camera is absent;
- the episode sampling rate differs from the output dataset's configured rate;
- the image shape differs from the existing output dataset;
- canonical widths differ from the existing output layout;
- state, action, timestamp, or camera-reference lengths disagree;
- canonical state/action contains non-finite values;
- a configured rot6d field contains degenerate axes;
- a camera reference is out of bounds; or
- parquet or video writing fails.

The episode path and reason are printed, other successes are kept, and `sync` exits nonzero
after finishing the batch if any episodes failed. Re-running retries those records.
Intentional skips are remembered by path and concise reason so they are not re-inspected on
every frequent sync; warning observations and structural failures remain per-run output and
are not accumulated as provenance or admission artifacts.

Unit tests cover the few transformations whose failure can remain numerically plausible:

- source-column rot6d → GR00T-row rot6d round-trip;
- name-based selection from both current 32-dim and 46-dim source schemas;
- stable path-based split under insertion;
- derived state/action slices and generated modality agreement; and
- append/retry behavior around a partially failed episode.

---

## Migration result

The cutover was completed against the existing `semihumanoid_260818` working root without
rebuilding parquet or video data. A bounded real-data pilot covered one native 32D episode
and one native 46D episode; both canonical state/action arrays matched the legacy converter
exactly, real stats and `LeRobotEpisodeLoader` checks passed, and all three canonical videos
were decoded and visually sampled.

The live version-1 ledgers were then adopted in place. All 1,857 prior assignments retained
their dataset, split, episode index, global offset, length, and task mapping. One source path
that had arrived since the earlier conversion was appended at the next contiguous
assignment; only its changed validation dataset regenerated statistics. A subsequent live
sync changed no metadata bytes. Finally, a four-GPU, seven-train-dataset, one-step smoke run
completed with finite `train_loss: 1.328125`, after which the superseded semihumanoid-only
entry points were removed.

The initial direct layouts were also upgraded once, without touching stats or episode data,
to record action representation, output/chunk, and video-format values that must remain
internally consistent across future appends.

The small pre-migration ledger/meta backup is outside the working output root at
`data/training_data/gr00t/semihumanoid_260818_metadata_backup_20260819T070503Z`.
The generalized pipeline neither reads nor writes the pre-existing legacy `manifest.json`;
it remains an inert artifact rather than an admission contract.

## Tradeoffs accepted

| Choice | Benefit | Accepted limitation |
|---|---|---|
| Mutable working corpus | Frequent append-and-try iterations | A run must record its dataset paths and counts separately if exact reproduction is needed |
| Path-only append ledger | Small, fast, easy to inspect | In-place source edits need explicit `--reconvert`; renames can duplicate data |
| Warnings for quality anomalies | Collection can continue without admission ceremonies | Operators decide whether a warning matters for a particular experiment |
| One zdata reader | Minimal code and maintenance | A genuinely different source format requires a later reader abstraction |
| Automatic stats refresh before training | Correct normalization without a separate checklist | Changed datasets still pay GR00T's stats-computation cost |
| Optional gap splitting | Default behavior stays simple | With splitting disabled, action windows may cross recorded timestamp gaps |

## Decisions

1. The output is a continuously growing working corpus, not a sealed dataset release.
2. The normal interface is `sync` followed, when desired, by explicit `train`.
3. The only persistent append bookkeeping created by this pipeline is the minimal path
   ledger; it creates no source fingerprints or release manifests.
4. Routine checks are local to newly written episodes. Full profiling and full scans are
   optional diagnostics.
5. State/action/video layout comes from one YAML and is the only compatibility boundary for
   appending to an existing output directory.
6. Task IDs are learned and appended from episode attributes; catalog versions do not gate
   conversion.
7. Timestamp-gap splitting is optional and disabled by default.
8. Training duration is epoch-based by default and re-derived from the current corpus.

The implementation, live-corpus adoption, bounded pilot, and training smoke have validated
these decisions.
