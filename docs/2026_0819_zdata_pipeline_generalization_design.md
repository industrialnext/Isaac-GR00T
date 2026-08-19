# Config-Driven zdata_hdf5 → GR00T Pipeline — Design

**Date:** 2026-08-19
**Status:** Approved, ready for implementation planning
**Supersedes (partially):** the hardcoded `scripts/lerobot_conversion/convert_semihumanoid.py`
**Related:** the one-off semihumanoid pipeline this generalizes, shipped in commit `2e64dd6`
(`scripts/lerobot_conversion/convert_semihumanoid.py`). Its detailed execution plan and the
source-data survey artifacts are kept machine-local under `docs/local/` (git-ignored) and are
not required to implement this design.

---

## Motivation

The semihumanoid pipeline works — 1,762 train episodes / 679,671 frames converted, GR00T
finetuning verified end to end. But every fact about the data layout is a module constant
in `convert_semihumanoid.py`: `CAMERA_MAP`, `STATE_FIELDS`, `ACTION_FIELDS`, `TASK_ORDER`,
the QC thresholds, the rot6d handling, and the 50 Hz assumption. A second embodiment means
either editing that file (and breaking the first) or copying it (and diverging).

The layout is also declared **twice**: as slice offsets inside the converter, and again as
`ModalityConfig` keys in `examples/semihumanoid/semihumanoid_config.py`. Nothing checks that
the two agree. This session already produced one silent, plausible-looking failure from a
*convention* mismatch (source rot6d encodes the first two **columns** of R, GR00T decodes
the first two **rows** — a 170° error that trains without complaint). A *layout* mismatch is
the same class of bug, one edit away.

**Target state:** one YAML per embodiment is the single source of truth. It generates both
`meta/modality.json` and the Python `ModalityConfig`, so they cannot disagree. One command
runs profile → convert → stats → validate → train.

### Non-goals

- **A reader-plugin interface.** Confirmed: all future data comes from the same
  `zdata_hdf5` recorder, with different *layouts*. One reader stays in code. If a genuinely
  different source format ever appears, that is when to add the abstraction — not now.
- **Multi-embodiment training in a single run.** GR00T's sharded trainer takes one
  embodiment tag; mixing is out of scope.
- **A config-driven eval harness.** `gr00t/eval/open_loop_eval.py` already reads its
  modality config from the checkpoint (`open_loop_eval.py:323`), so it needs nothing here.
- **Depth, progress prediction, state history.** GR00T N1.7 has no depth modality and no
  progress head, and every in-tree embodiment uses `delta_indices=[0]` for state and video.

---

## Architecture

```
scripts/lerobot_conversion/
├── run_pipeline.py              CLI: stages, --set overrides, orchestration
└── zdata_pipeline/
    ├── config.py                YAML → validated dataclasses
    ├── reader.py                zdata_hdf5 access (the single supported format)
    ├── layout.py                field→key resolution, rot6d re-encode, canonical vectors
    ├── qc.py                    configurable quality gate
    ├── writer.py                LeRobot v2 emission, ledger, stats invalidation
    ├── modality.py              generates modality.json AND the Python ModalityConfig
    └── validate.py              loader round-trip, rot6d round-trip, stats ranges

configs/embodiments/semihumanoid.yaml
examples/semihumanoid/semihumanoid_config.py     (generated; committed for reproducibility)
```

Each module has one job and a narrow interface:

| Module | Responsibility | Depends on |
|---|---|---|
| `config.py` | Parse + validate YAML. Reject unknown keys, bad enums, non-contiguous EEF blocks, horizon > model ceiling. | nothing |
| `reader.py` | Open an `episode.h5`; expose attrs, `state/flat`, `action/<source>`, `field_names`/`field_slices`, and per-camera `blob`/`offsets`/`frame_ref_index`/`frame_age_ms`. | h5py |
| `layout.py` | Resolve configured field names against the episode's own `field_slices`; apply rot6d re-encoding; assemble canonical state/action arrays; derive slice offsets. | reader, config |
| `qc.py` | Evaluate one episode against the configured gate; return pass/fail plus per-camera measurements and reasons. | reader, config |
| `writer.py` | Ledger read/write, episode index + split assignment, parquet + MP4 emission, `meta/` regeneration, stats invalidation. | layout, config |
| `modality.py` | Emit `meta/modality.json` and the Python `ModalityConfig` file from the same config object. | config |
| `validate.py` | Post-conversion assertions against GR00T's own code paths. | gr00t, config |

`reader.py` is the only module that knows the HDF5 layout. If the recorder schema ever
changes, exactly one file moves.

---

## Config schema

```yaml
name: semihumanoid                  # dataset dir prefix; embodiment tag is always
                                    # NEW_EMBODIMENT (GR00T projector slot 10)

source:
  root: ~/ml_data/data/training_data/semihumanoid
  subsets: ["flexiv_*"]             # explicit list or glob
  episode_glob: "*/20*/*/*/*/episode.h5"
  exclude_path_contains: ["_failed_recordings"]
  fps: 50                           # asserted against each episode's sampling_hz
  image_hw: [256, 256]

output:
  root: ~/ml_data/data/training_data/gr00t/semihumanoid_260818
  name_transform: {strip_prefix: "flexiv_"}     # flexiv_ube_v4 -> ube_v4

cameras:                            # canonical key -> source group under images/
  head: head_rgb
  left_wrist: eoat_left_bottom_rgb
  right_wrist: eoat_right_bottom_rgb
video: {codec: libx264, crf: 23, preset: veryfast, gop: 50}

state:                              # ordered -> canonical vector + derived slices
  - {key: left_eef,      fields: [left_arm_pose_pos, left_arm_pose_rot], rot6d_transpose: true}
  - {key: left_gripper,  fields: [left_gripper]}
  - {key: left_ft,       fields: [left_ft]}
  - {key: right_eef,     fields: [right_arm_pose_pos, right_arm_pose_rot], rot6d_transpose: true}
  - {key: right_gripper, fields: [right_gripper]}
  - {key: right_ft,      fields: [right_ft]}

action:
  source: executed                  # action/<source>; asserts action/residual == 0 if present
  horizon: 40                       # must be <= the base checkpoint's action_horizon
  keys:
    - {key: left_eef,  fields: [left_arm_pose_pos, left_arm_pose_rot], rot6d_transpose: true,
       rep: RELATIVE, type: EEF, format: XYZ_ROT6D, state_key: left_eef}
    - {key: left_gripper,  fields: [left_gripper],  rep: ABSOLUTE, type: NON_EEF, format: DEFAULT}
    - {key: right_eef, fields: [right_arm_pose_pos, right_arm_pose_rot], rot6d_transpose: true,
       rep: RELATIVE, type: EEF, format: XYZ_ROT6D, state_key: right_eef}
    - {key: right_gripper, fields: [right_gripper], rep: ABSOLUTE, type: NON_EEF, format: DEFAULT}

qc:
  coverage: [0.45, 0.80]            # per canonical camera: image_count / frame_count
  age_p99_ms: 150
  min_frames: 41                    # > action.horizon, else the episode yields no chunk
  policy_types: [expert]            # episode dir suffix allowlist

tasks:                              # stable index order; text is the fallback when an
  generic_pick:      "Pick the grounded target object and hold it securely in the gripper."
  generic_place:     "Place the currently held object at the grounded destination and release it."
  bracket_handover:  "Hand over the bracket from the gripper holding it to the opposite gripper and secure it in the receiving gripper."

split: {val_every: 20}              # hash-based; 0 disables

train:
  base_model: nvidia/GR00T-N1.7-3B
  out_base: ~/ml_data/outputs/gr00t
  gpus: 4
  batch: 256                        # global, pre-accumulation; must divide gpus
  lr: 2.8e-4
  steps: 5250                       # sized to ~2.2 epochs of the CURRENT corpus; see note
  workers: 8
  save_steps: 1000
  save_total_limit: 5
  state_dropout_prob: 0.2
  shortest_image_edge: 256          # must match the base checkpoint's declared values
  crop_fraction: 0.95
  color_jitter: {brightness: 0.15, contrast: 0.15, saturation: 0.2, hue: 0.1}
  use_wandb: false
```

**`steps` must be re-derived whenever the corpus grows, and this is easy to forget.** It is an
absolute step count, so a fixed value silently trains *fewer* epochs as data is appended. The
2026-08-19 expansion (+417 episodes) took the trainable 40-step start indices from 440,280 to
610,953 (+39%), which turned the previously-correct 3,750 steps into 1.57 epochs instead of
2.18. At batch 256, ~2.2 epochs is now 5,250 steps (~2.77 h at the measured 134.7 samples/s).

Implementation note: `validate` should report `steps x batch / trainable_starts` as the
effective epoch count, so a stale `steps` is visible before the run rather than after.

### Two deliberate schema decisions

**Slice offsets are derived, never authored.** Widths come from each episode's own
`field_slices`, and the canonical layout is the concatenation of `state`/`action` entries in
order. This means a config cannot declare a 10-dim EEF block: `config.py` asserts that any
key with `type: EEF` resolves to exactly 9 dims (3 translation + 6 rot6d), which is what
`EndEffectorPose.from_action_format` requires (`pose.py:681`).

**`rot6d_transpose` is explicit per key, not inferred.** The current code guesses from a
`_pose_rot` name suffix. That is correct for this recorder and fragile for the next one.
Making it explicit forces whoever adds an embodiment to answer the question that already
cost us a silent 170° error.

---

## Single source of truth: generated modality config

`modality.py` emits two artifacts from one config object:

1. **`<dataset>/meta/modality.json`** — `state`/`action` slice maps, `video` canonical→
   `observation.images.<key>` mapping, and `annotation.human.task_description` pointing at
   `original_key: task_index` (the loader follows that indirection,
   `lerobot_episode_loader.py:381`, so no `annotation.*` parquet column is needed).
2. **`examples/<name>/<name>_config.py`** — a generated Python module that builds the
   `ModalityConfig` dict and calls `register_modality_config(...,
   EmbodimentTag.NEW_EMBODIMENT)`. Python is required because that is the only interface
   GR00T exposes for custom embodiments; `--modality-config-path` imports the file
   (`launch_finetune.py:31-40`).

The generated Python carries a `# GENERATED — edit configs/embodiments/<name>.yaml` header
and a hash of the source config. `validate` re-generates in memory and fails if the
committed file differs, so hand-edits are caught rather than silently overriding the YAML.

---

## Stages

`run_pipeline.py <stage> --config configs/embodiments/<name>.yaml [--set k=v ...]`

| Stage | Does | Idempotent |
|---|---|---|
| `profile` | Survey the source tree: episode counts, camera signatures, state/action dims and field names, tasks, policy types, fps, per-camera coverage/staleness. Writes a JSON report. **No writes to the corpus.** | yes |
| `convert` | QC-gate new episodes, convert them, regenerate `meta/`, invalidate stats. Append-only via the ledger. | yes |
| `stats` | Run `gr00t/data/stats.py` per dataset **in parallel** (one process each). | yes (fingerprint-cached) |
| `validate` | Assertions below. | yes |
| `train` | Preflight, then `torchrun` the finetune. | no |
| `all` | profile → convert → stats → validate → train | — |

`profile` exists because it is how this session caught the 46-dim state shift, the
mislabelled matcha_v1 cameras, and the 2026/08/12 camera-dropout day. Discovering that by
hand each time is the expensive part.

`stats` is a stage rather than a side effect because the relative pass is CPU-bound at
~2.5 s/episode/key and `DatasetFactory` otherwise does it serially on rank 0 before step 1
(`factory.py:59`) — roughly 40 minutes with three GPUs idle.

### Append-only guarantees (carried over, now config-driven)

- **Ledger** (`<out>/_ledgers/<subset>.json`) freezes `(dataset, episode_index, split,
  index_offset, length, task_uuid)` per source episode. Without it, indices are positional
  and a backfilled recording date renumbers every later episode, orphaning written files.
  The ledger also stores the camera map and refuses to run if the config's map changed.
- **Hash-based split**: `sha256(episode_path_relative_to_subset) % val_every == 0`. A
  positional rule moves existing episodes between train and val when earlier episodes are
  inserted — leaking held-out data into training on the next append.
- **Stats invalidation on append**: GR00T fingerprints `stats.json` over the `info.json`
  feature *schema* only (`stats.py:183`), which does not change when episodes are appended.
  Both stats files are deleted whenever an episode is written.

---

## Validation — what makes the refactor safe

**Equivalence gate (blocking, run once at migration).** The generic driver converts the
existing 7 subsets into a *fresh* output root, then asserts against the current corpus, per
dataset: identical episode count; identical `episodes.jsonl`; identical `info.json` and
`modality.json` (modulo key order); and for every episode, `observation.state` and `action`
arrays equal within 0 tolerance, plus identical `timestamp`/`task_index`/`index`/
`frame_index` columns. Video files are compared by frame count and decoded-frame checksums
at 3 sampled indices rather than byte-identity, since h264 encoding is not bit-reproducible
across runs. `convert_semihumanoid.py` is deleted only after this passes.

**Ongoing `validate` stage.** Per dataset:

1. `meta/stats.json` exists — the loader asserts it (`lerobot_episode_loader.py:180`).
2. `info.json.features` declares `action` and `observation.state` as `float32`, since
   `generate_stats` only computes statistics for features whose dtype contains `"float"`
   (`stats.py:251`). Wrong dtype ⇒ no statistics ⇒ silently unnormalized training.
3. `LeRobotEpisodeLoader` round-trip yields exactly the expected column set.
4. `relative_stats.json` contains exactly the `RELATIVE` action keys, each shaped
   `(horizon, dim)`.
5. Per-dim `q99 − q01` ≥ 1e-4 for every relative key — `normalize_values_minmax`
   (`data/utils.py:106`) zeroes dims where min≈max but *amplifies* tiny-but-nonzero ranges.
6. **rot6d round-trip on real converted data**: rebuild R from the written 9-dim block using
   GR00T's `_rot6d_to_matrix` and compare against the source decoded with the source
   convention; assert angular error < 1e-4 deg.
7. Structural invariants: parquet count == `episodes.jsonl` length; concatenated `index`
   contiguous from 0; per-episode `frame_index` 0..n−1; `info.json.total_frames` == sum.
8. Generated `ModalityConfig` matches the YAML (hash check).

**Unit tests** (no GPU, no corpus): rot6d round-trip vs GR00T's decoder plus a guard
asserting the naive passthrough is still wrong; name-based field selection across both the
32-dim and 46-dim schemas; config validation rejects non-9-dim EEF, horizon > ceiling,
batch not divisible by gpus, unknown enum values; hash split stability under insertion.

---

## Preflight for `train`

Refuses to launch unless: every train dataset has both stats files; no `_val` dataset is in
`--dataset-path`; `batch % gpus == 0`; `action.horizon` ≤ the base checkpoint's
`action_horizon`; state and action dims ≤ `max_state_dim`/`max_action_dim` (132); free disk
≥ `save_total_limit × 40 GB`; GPUs idle.

Two footguns it also handles: **`--experiment-name` is not passed**, because
`experiment.py:213` appends it to `--output-dir`, silently nesting the run one level deeper
than the summary looks for. And `--shortest-image-edge`/`--crop-fraction` are always passed
from config, because omitting them makes `launch_finetune.py` override the base checkpoint's
declared 0.95 crop with the legacy 0.898 path, diverging from pretraining preprocessing.

---

## Error handling

Fail loudly and early, with the episode path in the message. Specifically: a configured
field absent from an episode, or present with an unexpected width, is an error (not a
silent slice) — this is what protects against the 46-dim shift. A per-episode conversion
failure is collected, removed from the ledger so a re-run retries only it, and reported;
it does not abort the batch. QC drops are logged individually with their measured
coverage/age into `_conversion_report.json`, never silently.

---

## Migration

1. Build `zdata_pipeline/` + `configs/embodiments/semihumanoid.yaml` reproducing current behavior.
2. Generate `examples/semihumanoid/semihumanoid_config.py` from the YAML; confirm it matches the hand-written one.
3. Run the equivalence gate against the live corpus.
4. Delete `convert_semihumanoid.py` and `semihumanoid_datasets.py` (the latter's function becomes `run_pipeline.py` stages), keeping `test_convert_semihumanoid.py`'s assertions in the new test module.
5. Commit; push to the fork.

---

## Risks

| Risk | Mitigation |
|---|---|
| Refactor silently changes converted output | Equivalence gate is blocking and array-exact on state/action |
| YAML expressiveness gap for a future embodiment | `profile` reports what a source tree contains; config validation names the missing/extra fields explicitly. Accept that a genuinely new *format* needs code — that is the stated non-goal |
| Generated Python config hand-edited, diverging from YAML | Hash header + `validate` check |
| Config-owned training params make experiment sweeps edit the data config | `--set train.batch=128` overrides; the YAML stays the stable, versioned data description |
| More modules ⇒ more places to look | Each module has one responsibility and a stated interface; `reader.py` is the only one that knows HDF5 |

## Decisions taken (were open questions)

1. **YAML lives in `configs/embodiments/<name>.yaml`**; the *generated* Python
   `ModalityConfig` stays at `examples/<name>/<name>_config.py`. Rationale: configs sit
   together and are diffable as a set, while the generated file stays where
   `--modality-config-path` users and the existing SO100/LIBERO examples already look.
2. **`all` stops after `validate`; `train` is always explicit.** A typo in a data stage
   should not cost GPU hours. `all` therefore means profile → convert → stats → validate,
   and the operator runs `train` once the validation output looks right.

## Migration addendum

`examples/semihumanoid/finetune_semihumanoid.sh` is superseded by `run_pipeline.py train`
and is removed in the same commit. Its measured batch-scaling table moves into the
`semihumanoid.yaml` header comments so the numbers survive.

Note that `convert_semihumanoid.py`, `semihumanoid_datasets.py`,
`test_convert_semihumanoid.py`, and `finetune_semihumanoid.sh` are already committed and
pushed to the fork (`2e64dd6`). The refactor is therefore a follow-up commit that removes
three of them and rewrites the fourth's assertions into the new test module — not a
never-committed rewrite. Anyone who pulled `2e64dd6` gets a working pipeline either way.
