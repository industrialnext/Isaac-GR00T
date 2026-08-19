# Config-Driven zdata_hdf5 → GR00T Pipeline Implementation Plan

**Date:** 2026-08-19

**Status:** Implemented and validated

**Design:**
[`2026_0819_zdata_pipeline_generalization_design.md`](2026_0819_zdata_pipeline_generalization_design.md)

## Motivation

The pre-cutover semihumanoid path was proven end to end, but its layout and workflow were
split across an 813-line converter, a dataset-list helper, a generated-looking but
hand-maintained modality module, and a shell training launcher. Adding another zdata_hdf5
embodiment would have required copying and editing those files.

The target is one small YAML per embodiment and two routine commands:

```bash
# h5py remains an on-demand overlay; it is not added to the project dependency manifests.
uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
  sync --config configs/embodiments/semihumanoid.yaml

uv run --no-sync python scripts/lerobot_conversion/run_zdata_pipeline.py \
  train --config configs/embodiments/semihumanoid.yaml
```

The implementation must support frequent append-and-try iterations against the existing
working corpus. It must not introduce dataset sealing, source hashes, immutable manifests,
catalog-version admission, or routine full-corpus scans.

### Design-review clarifications applied by this plan

The design is implementation-ready. This plan makes four source-level details explicit:

1. `h5py` is not installed by the current project environment and the completed
   semihumanoid work deliberately used `uv run --with h5py`. `sync` keeps that overlay;
   `train`, `stats`, and output-only `check` use the normal environment. No `pyproject.toml`,
   platform manifest, or lockfile change is needed.
2. Optional timestamp-gap splitting means one source path may produce several output
   segments. The ledger therefore maps a source path to a list of segment assignments,
   while remaining path-only and checksum-free.
3. `--reconvert` may replace an existing assignment only when its segment count and segment
   lengths are unchanged. A repair that changes lengths or partitioning needs a new output
   root or a deliberate rebuild because later global indices are already assigned.
4. A plain `train` creates a timestamped run directory below `train.out_base`; resumption is
   explicit through `--resume-from <run-dir>`. This replaces the old shell launcher's
   implicit `RUN_NAME` state without reintroducing `--experiment-name` nesting.

Two metadata values currently hardcoded by the converter are completed as small output
config fields: `robot_type` and `chunks_size` (default `1000`). This is a schema completion,
not a new abstraction.

### Scope

- Generalize the existing zdata_hdf5 reader, field assembly, video conversion, LeRobot v2.1
  metadata generation, dataset discovery, stats generation, and finetune launch.
- Add the semihumanoid YAML and regenerate its Python `ModalityConfig` from that YAML.
- Adopt the existing `semihumanoid_260818` output and version-1 ledgers without rebuilding
  or deleting the corpus.
- Preserve bounded diagnostics and focused tests for the transformation and append logic.
- Remove the superseded semihumanoid-only converter, dataset helper, and training shell only
  after a bounded pilot and existing-corpus dry run pass.

### Non-goals

- Another source format or reader plug-in interface.
- Dataset release/version management, content identity, or exact experiment reproduction.
- Automatic action/task/camera semantic inference.
- Multi-embodiment training, evaluation orchestration, or deployment changes.
- A new general-purpose configuration framework or dependency.
- A mandatory full conversion, full source survey, or GPU training run on each append.

---

## Files and ownership

| Path | Change | Responsibility |
|---|---|---|
| `scripts/lerobot_conversion/run_zdata_pipeline.py` | Add | Thin `argparse` CLI for `sync`, `stats`, `check`, and `train`; lazy-load HDF5 code |
| `scripts/lerobot_conversion/zdata_pipeline/__init__.py` | Add | Declare the shared helper package without importing optional HDF5 dependencies |
| `scripts/lerobot_conversion/zdata_pipeline/common.py` | Add | HDF5-free stats filenames and incomplete-transaction guard shared by output-only commands |
| `scripts/lerobot_conversion/zdata_pipeline/config.py` | Add | Safe YAML parsing, dataclasses, layout derivation, modality/layout artifact rendering |
| `scripts/lerobot_conversion/zdata_pipeline/source.py` | Add | zdata_hdf5 discovery/read, named-field extraction, rot6d conversion, selection/warnings, optional gap segmentation |
| `scripts/lerobot_conversion/zdata_pipeline/convert.py` | Add | Staging, append ledger, task/index assignment, parquet/MP4 commit, LeRobot metadata, stats invalidation |
| `scripts/lerobot_conversion/zdata_pipeline/check.py` | Add | Lightweight output checks, optional full scan, dataset enumeration, stats and train helpers |
| `configs/embodiments/semihumanoid.yaml` | Add | Semihumanoid source/layout/selection/warning/training values |
| `examples/semihumanoid/semihumanoid_config.py` | Regenerate | Generated GR00T `ModalityConfig`; no hand-maintained layout constants |
| `scripts/lerobot_conversion/test_zdata_pipeline.py` | Add/replace | Unit and temporary-directory integration coverage |
| `scripts/lerobot_conversion/convert_semihumanoid.py` | Remove after cutover | Superseded hardcoded converter |
| `scripts/lerobot_conversion/semihumanoid_datasets.py` | Remove after cutover | Dataset enumeration moves into the shared pipeline |
| `scripts/lerobot_conversion/test_convert_semihumanoid.py` | Remove after test migration | Assertions move to the generalized test module |
| `examples/semihumanoid/finetune_semihumanoid.sh` | Remove after smoke test | Training launch moves into `run_zdata_pipeline.py train` |

Keep `survey_zdata_source.py` and `camera_health_zdata.py` as optional, standalone source
diagnostics. They do not become required pipeline stages.

## Pre-cutover code reused

The source paths in this section identify the baseline implementation used during the work;
the files marked for removal above no longer exist after the completed cutover.

- `scripts/lerobot_conversion/convert_semihumanoid.py:163-230` — move the proven rot6d
  conversion and name-based field assembly into `source.py`, replacing constants with parsed
  layout entries.
- `scripts/lerobot_conversion/convert_semihumanoid.py:283-337` — retain deterministic
  discovery, path-based split assignment, and atomic JSON replacement.
- `scripts/lerobot_conversion/convert_semihumanoid.py:343-396` — retain the JPEG-blob /
  `frame_ref_index` → ffmpeg video path; parameterize camera, fps, codec, CRF, and preset.
- `scripts/lerobot_conversion/convert_semihumanoid.py:399-504` — retain the LeRobot v2.1
  metadata shapes and explicit stats invalidation, but derive dimensions, cameras, and tasks.
- `scripts/lerobot_conversion/convert_semihumanoid.py:510-590` — reuse the standard parquet
  columns and source arrays while separating stage output from final index assignment.
- `scripts/lerobot_conversion/semihumanoid_datasets.py:48-135` — reuse dataset enumeration,
  stats-presence reporting, `os.pathsep` joining, and trainable-start counting; omit manifest
  writing.
- `scripts/lerobot_conversion/test_convert_semihumanoid.py:67-292` — preserve the rot6d,
  32/46-dim named-field, canonical-slice, camera-map, and stable-split assertions in
  config-driven form.
- `examples/semihumanoid/semihumanoid_config.py:38-102` — use as the expected generated
  modality output for the initial semihumanoid config.
- `gr00t/data/dataset/lerobot_episode_loader.py:144-202` — required metadata and stats files;
  a loader check must run after stats exist.
- `gr00t/data/dataset/lerobot_episode_loader.py:346-438` — exact annotation, joint-slice, and
  video-key behavior that generated `modality.json` must satisfy.
- `gr00t/data/stats.py:251-291` and `gr00t/data/stats.py:415-475` — invoke the existing
  absolute-then-relative stats path in independent subprocesses; do not reimplement it.
- `gr00t/experiment/launch_finetune.py:29-130` and
  `gr00t/configs/finetune_config.py:22-211` — build the supported finetune CLI, dataset-path,
  preprocessing, batch, checkpoint, and resume arguments.
- `gr00t/configs/model/gr00t_n1d7.py:73-81` — current limits are state/action dimension 132,
  action horizon 40, and state history 1.
- `gr00t/configs/base_config.py:54-75` — follow the repository's `yaml.safe_load` security
  pattern; do not use object-constructing YAML loaders.

---

## Implementation contracts

### Configuration and generated artifacts

- Parse YAML with `yaml.safe_load` into explicit dataclasses. Missing optional sections use
  documented defaults. Unknown layout-entry keys fail; unknown non-layout keys emit one
  warning and are otherwise ignored.
- Require `name` to be a valid Python identifier because it determines the generated module
  name; fail clearly instead of silently sanitizing it to a different embodiment name.
- Accept both user-relative and repository-relative paths and resolve them only at runtime.
  Documentation uses the repository's portable `data/` and `outputs/` symlinks rather than
  recording resolved machine paths.
- Derive state/action slice offsets from the configured order and the widths reported by a
  source episode. Extra source fields and cameras are ignored.
- Require one unambiguous 6D rotation field for any entry declaring
  `source_columns_to_groot_rows`; require 9 total dimensions for `EEF/XYZ_ROT6D` actions.
- Render `meta/modality.json`, the generated Python module, and per-dataset `_layout.json`
  from the same normalized layout object. `_layout.json` stores direct fields, transforms,
  action representation values, camera mapping, sampling/image values, robot/chunk values,
  and declared video format—not a hash.
- Write generated text/JSON only when bytes differ, using temporary sibling files plus
  `os.replace`.
- `sync --dry-run` performs discovery, parsing, compatibility comparison, and planning in
  memory but writes nothing, including lock files, staging data, generated Python, layouts,
  ledgers, or metadata.

The semihumanoid YAML targets the existing working output, referred to throughout the
documentation as `data/training_data/gr00t/semihumanoid_260818`, and sets
`robot_type: semihumanoid_bimanual`, and preserves the current camera, state, action, video,
and training values. Populate `tasks.text_overrides` for the three currently known task IDs
to preserve their existing text during ledger migration; unseen task IDs still append
automatically.

### Minimal ledger and append transaction

Use one version-2 ledger per source subset. Completed and intentionally skipped paths are
both remembered so frequent no-op syncs do not repeatedly inspect old inputs; failed paths
receive no entry and are retried:

```json
{
  "version": 2,
  "sources": {
    "relative/path/episode.h5": {
      "status": "complete",
      "task_id": "generic_pick",
      "segments": [
        {
          "source_start": 0,
          "source_end": 386,
          "dataset": "matcha_v2",
          "split": "train",
          "episode_index": 0,
          "index_offset": 0,
          "length": 386,
          "task_index": 0
        }
      ]
    },
    "relative/path/too_short.h5": {
      "status": "skipped",
      "reason": "all segments are shorter than the action horizon",
      "segments": []
    }
  }
}
```

This is path bookkeeping, not source identity. Do not store absolute source paths, mtimes,
checksums, catalog versions, Git SHAs, or corpus fingerprints.

Conversion uses a single advisory writer lock under the output root to prevent two `sync`
processes from assigning the same indices. Workers stage source-relative canonical rows and
encoded videos under `_staging/`. The main process commits successful sources in
deterministic path order, injects final episode/global/task indices, and updates one dataset
at a time.

Before the first final replacement for a dataset, atomically write a transient
`<dataset>/.sync_transaction.json` recovery journal containing the transaction staging
directory, the ordered staging-to-final replacements (including the staged ledger), worker-
stage cleanup paths, and the two stats paths to invalidate. Then apply each `os.replace`,
replace the ledger and metadata, remove stale stats/consumed staging, and delete the journal
and transaction staging directory. On startup,
`sync` acquires the lock and rolls any journal forward idempotently: a remaining staging path
is replaced, while a missing staging path is accepted only when its final target exists.
`stats`, `check`, and `train` stop only while a journal exists because they must not observe a
half-committed dataset. This short recovery journal is the only transaction gate; beyond the
ledger's relative path bookkeeping it contains no hashes or source-identity fields, and it is
removed after a successful commit.

Version-1 ledger loading is a compatibility path, not a separate migration command:

- preserve every existing dataset, split, episode index, index offset, length, and task ID;
- wrap each old assignment as one `[0, length)` segment;
- discard the stale absolute `source` value;
- compare the old `camera_map`, current `meta/modality.json`, and YAML before seeding
  `_layout.json`; and
- let `--dry-run` preview the conversion, then write version 2 only during a successful
  non-dry `sync`.

A ledger/layout-only version-1 adoption does not invalidate statistics because it changes no
episode rows or task mappings.

### Selection, warnings, and failures

- Skip an episode only when the enabled selection rule sees
  `valid_for_training=false`, its parsed policy type is outside a configured allowlist, or
  all resulting segments are shorter than the action horizon.
- Record those intentional skips in the path ledger with a concise reason. A repeatable
  `--reconvert` selector re-evaluates either a completed or skipped path explicitly.
- Drop individual sub-horizon segments with a warning while converting any usable siblings.
  For every emitted segment, reset timestamp/frame indices to zero and force `next.done=true`
  on its final row so the new LeRobot episode boundary is self-consistent.
- Missing `valid_for_training` is accepted with a warning.
- Camera coverage, `frame_age_ms` p99, timestamp gaps, task-text drift, and nonzero residual
  are warnings. Missing optional warning inputs produce “measurement unavailable,” not a
  conversion failure.
- Missing configured arrays/cameras, inconsistent lengths, non-finite canonical tensors,
  out-of-range camera references, fps/image-shape mismatch, ambiguous rot6d selection, or
  writer failure fail that source only.
- Continue staging other sources, commit successful ones, report all failures, and return a
  nonzero exit status so automation notices the partial run.

### Stats, checks, and training

- `stats` discovers output datasets through `meta/info.json`, selects those missing either
  stats file, and launches `gr00t/data/stats.py` in bounded parallel subprocesses. `train`
  performs the same operation for train datasets only.
- `check` verifies metadata/file counts and samples one parquet/video per dataset without
  requiring stats. It chooses the lowest assigned episode index deterministically. If stats
  exist, it also constructs one `LeRobotEpisodeLoader` sample. `check --full` performs the
  optional complete structural scan and uses `ffprobe -count_frames` for exact video-frame
  counts.
- `train` enumerates all non-`_val` datasets, joins them with `os.pathsep`, derives trainable
  starts as `sum(max(0, segment_length - horizon + 1))`, and calculates
  `ceil(epochs * starts / batch)` unless `max_steps` is configured.
- Hard train errors are limited to incomplete sync transactions, no train data, missing
  stats after generation, non-divisible batch/GPU count, dimensions/horizon above current
  N1.7 limits, failed modality import, or failed loader sample. GPU occupancy and free disk
  remain warnings.
- Launch with `sys.executable -m torch.distributed.run --standalone --nproc-per-node
  <num_gpus>` so the already active uv environment is reused, the configured rank count is
  explicit, and a fixed master port is unnecessary.
- Default output is `<out_base>/<name>_<UTC timestamp>`. `--resume-from` uses the named
  existing directory and adds `--resume-from-checkpoint`. A fresh run fails if its computed
  directory already exists; it never silently reuses or resumes an old run.

---

## Phased breakdown

### Phase 1 — Configuration and layout generation

- **Problem:** Layout constants and GR00T modality declarations are maintained in separate
  Python files.
- **Solution:** Add the shared config dataclasses, safe YAML loader, layout derivation, and
  deterministic artifact renderers; add the semihumanoid YAML.
- **Impact:** One config describes every consumed field and generated modality surface, with
  no HDF5 or output mutation yet.

### Phase 2 — Incremental `sync` and existing-corpus adoption

- **Problem:** The current converter is embodiment-specific, quality thresholds block data,
  and partial worker failure can leave index gaps.
- **Solution:** Extract the proven reader/transforms, add warning-oriented selection,
  optional continuity segments, staged contiguous commits, minimal ledgers, and version-1
  adoption.
- **Impact:** Repeated `sync` calls append only new usable episodes, retain successes, retry
  failures, and adopt the live corpus without a rebuild.

### Phase 3 — `stats`, `check`, and `train`

- **Problem:** Corpus enumeration, stats refresh, epoch sizing, and training launch require
  separate helpers and manual arithmetic.
- **Solution:** Reuse existing GR00T stats and finetune entry points behind the same config,
  with bounded parallel stats and explicit training/resume behavior.
- **Impact:** `sync` and `train` are sufficient for normal operation; `stats` and `check`
  remain available on demand.

### Phase 4 — Bounded validation and cutover

- **Problem:** The generalized path must replace working tools without forcing a full
  recode/rebuild of a growing corpus.
- **Solution:** Run focused tests, a two-schema pilot, version-1 ledger dry-run/adoption, one
  ordinary append, and a short loader/training smoke test before deleting legacy wrappers.
- **Impact:** The generalized pipeline becomes the maintained path while the existing data
  and optional survey helpers remain intact.

---

## Detailed checklist

### Phase 1 — Configuration and layout generation

- [x] **1.1** Add the package skeleton and CLI subcommands
  (`scripts/lerobot_conversion/run_zdata_pipeline.py`,
  `scripts/lerobot_conversion/zdata_pipeline/__init__.py`, and helper modules).
  - Keep top-level imports HDF5-free so `train`, `stats`, and output-only `check` work in the
    normal uv environment.
  - If `sync` cannot import `h5py`, print the exact `uv run --no-sync --with h5py ...`
    command and exit without a traceback.
  - Acceptance: all four `--help` paths run; `sync --help` does not mutate the corpus or
    dependency files.

- [x] **1.2** Implement safe config parsing and normalized layout derivation
  (`zdata_pipeline/config.py`).
  - Add dataclasses for source, output, video, state/action entries, tasks, selection,
    warnings, continuity, and training.
  - Validate required layout fields/enums and derive canonical slices, dimensions, and
    `ActionConfig` values; warn on unrelated unknown keys. Reject two source subsets that
    resolve to the same output dataset name.
  - Include `output.robot_type` and `output.chunks_size=1000`.
  - Acceptance: unit tests cover defaults, path expansion, unknown-key behavior, contiguous
    slices, 9D EEF blocks, and invalid enum/rot6d layouts.

- [x] **1.3** Add `configs/embodiments/semihumanoid.yaml` and artifact rendering.
  - Preserve the current three cameras, 32D canonical state, 20D action, action horizon 40,
    executed-action source, transforms, task text, warning thresholds, and measured local
    training values.
  - Render `examples/semihumanoid/semihumanoid_config.py` and compare it structurally with
    the current hand-written module before replacing it.
  - Acceptance: importing the generated file registers `NEW_EMBODIMENT`; its video/state/
    action/language keys, action configs, and delta indices match the YAML exactly.

### Phase 2 — Incremental `sync` and existing-corpus adoption

- [x] **2.1** Extract and generalize the zdata reader (`zdata_pipeline/source.py`).
  - Move deterministic discovery, HDF5 metadata reads, named field resolution, canonical
    assembly, and rot6d conversion from the current converter.
  - Return a source description plus zero or more `[start, end)` continuity segments;
    preserve source order when splitting.
  - Compute selection reasons and warnings separately.
  - Acceptance: the current 32D and 46D native state layouts both yield the same configured
    32D state/20D action, and the GR00T rot6d round-trip remains numerically correct.

- [x] **2.2** Implement video/parquet staging and deterministic final commit
  (`zdata_pipeline/convert.py`).
  - Stage canonical row data without final episode/global/task indices and stage each
    camera MP4 using the existing ffmpeg pipe.
  - Commit successful sources in relative-path order, assign contiguous indices per output
    dataset, and atomically replace each final parquet/metadata/ledger file.
  - Use the writer lock and `.sync_transaction.json` roll-forward journal described above;
    write the journal before any final replacement and remove it only after stats
    invalidation.
  - Acceptance: injected worker failure leaves no index gap; successful peers remain staged
    or committed; interruption after each replacement boundary is recoverable; the next
    `sync` rolls the transaction forward and retries only unfinished input.

- [x] **2.3** Implement version-2 ledgers, stable tasks/splits, and `--reconvert`.
  - Store one completed source record with one or more segment assignments, or one skipped
    source record with its reason; leave failed sources unrecorded.
  - Within each output dataset, preserve existing `tasks.jsonl` indices and text, then append
    new task IDs by first committed output index. Warn on later text drift and keep the old
    text unless `tasks.text_overrides` supplies a replacement. Apply edited overrides to
    existing task/episode metadata on the next no-new-data sync without invalidating numeric
    statistics.
  - Assign the train/validation split once per source path and apply it to every segment from
    that source so continuity splitting cannot leak one recording across both sides.
  - Accept `--reconvert <subset>/<relative-source-path>` as a repeatable, unambiguous
    selector; do not scan or hash completed sources looking for changes.
  - Preserve old split assignments; use deterministic relative-path splitting only for new
    sources.
  - Reject in-place reconversion when segment count/length changes, with the new-root/full-
    rebuild remedy in the error. A previously skipped path has no topology to preserve and
    may produce new assignments when explicitly reprocessed.
  - Acceptance: insertion/backfill does not renumber or move old episodes; content repair of
    equal length rewrites the same outputs and invalidates only affected dataset stats.

- [x] **2.4** Generate LeRobot metadata and invalidate stats only after a changed commit.
  - Derive `info.json`, `episodes.jsonl`, `tasks.jsonl`, `modality.json`, and `_layout.json`
    from the completed ledger and normalized config.
  - Compare direct layout values before appending; do not compare source roots, counts,
    warning thresholds, or training settings.
  - Populate `info.json` with v2.1 paths, chunk/fps/count fields, every configured video
    feature, and declarations for `action`, `observation.state`, `timestamp`, `frame_index`,
    `episode_index`, `index`, and `task_index`. Declare `action` and `observation.state` as
    `float32` with their derived dimensions so the existing stats code includes them.
  - Keep `episodes.jsonl` lengths/tasks and `tasks.jsonl` indices/text consistent with the
    committed parquet `episode_index` and `task_index` columns.
  - Remove both stats files for changed train or validation datasets; leave unchanged
    datasets untouched.
  - Acceptance: metadata totals and canonical slices equal final files, and a no-op `sync`
    changes no output bytes.

- [x] **2.5** Add transparent version-1 ledger adoption for the existing working root.
  - Discover every current ledger at runtime; do not hardcode seven subsets or current
    episode counts.
  - Seed task ID/index associations using configured text overrides and existing
    `tasks.jsonl`; verify old modality slices and camera map before writing `_layout.json`.
  - Acceptance: `sync --dry-run` against `semihumanoid_260818` reports every ledger entry as
    already complete, reports any source paths appended since the previous conversion
    separately, proposes no rewrite of existing parquet/video files, and changes no bytes.
    A migration-only sync changes only ledger/layout metadata and leaves both stats files
    intact.

### Phase 3 — `stats`, `check`, and `train`

- [x] **3.1** Implement dataset enumeration and bounded parallel stats
  (`zdata_pipeline/check.py`, `run_zdata_pipeline.py`).
  - Reuse `meta/info.json` discovery and `_val` suffix separation.
  - Launch `gr00t/data/stats.py` with the generated modality path in independent subprocesses;
    default concurrency to `min(4, dataset_count)` and allow a `--jobs` override.
  - Acceptance: datasets with both stats files are skipped; missing absolute/relative files
    are regenerated in the required order; any subprocess failure is summarized and yields
    a nonzero command exit.

- [x] **3.2** Implement lightweight and optional full output checks.
  - Default: reconcile ledger/meta counts, verify expected files for the lowest-index episode
    in each dataset, inspect parquet columns/dtypes/shapes, and decode one frame per configured
    camera.
  - With stats: construct one `LeRobotEpisodeLoader` sample and verify configured columns.
  - `--full`: check every episode index, frame index, global index, length, and exact video
    frame count with `ffprobe`, without reading the source HDF5 tree.
  - Acceptance: default work is bounded per dataset; `--full` catches an intentional index
    gap or missing video in a temporary test corpus.

- [x] **3.3** Implement epoch-derived training and explicit resume.
  - Enumerate train datasets, auto-run missing stats, import the generated modality module,
    open one loader sample, calculate current starts/steps, and print the final command.
  - Map every configured value to supported `FinetuneConfig` flags, including explicit
    shortest-edge/crop preprocessing and no `--experiment-name`.
  - Launch under `torch.distributed.run --standalone --nproc-per-node <num_gpus>`; use a new
    timestamped output directory or the explicit existing `--resume-from` directory.
  - Acceptance: command-construction tests cover epoch and max-step modes, multi-dataset
    `os.pathsep`, GPU/batch divisibility, W&B off/on, fresh run, and resume.

### Phase 4 — Bounded validation and cutover

- [x] **4.1** Port the existing focused tests and add append transaction coverage
  (`scripts/lerobot_conversion/test_zdata_pipeline.py`).
  - Preserve all transformation/layout/split tests from the current suite.
  - Add temporary-config, generated-artifact, task-growth, warning-vs-failure, gap-segment,
    skipped-path/reconvert, legacy-ledger, read-only dry-run, no-op sync, journal-boundary
    recovery, and stats-invalidation tests.
  - Mock ffmpeg/subprocesses in unit tests; reserve real encoding for the bounded pilot.
  - Acceptance: the test module is corpus-independent and CPU-safe with the h5py overlay.

- [x] **4.2** Run a bounded real-data pilot without rebuilding the corpus.
  - Build a temporary source root containing one current 32D episode and one current 46D
    episode, convert to a temporary output root, and compare canonical state/action arrays
    with the current converter.
  - Generate stats, open both outputs with `LeRobotEpisodeLoader`, and visually sample the
    three canonical videos.
  - Acceptance: arrays and task/camera mapping agree; loader columns/shapes are correct;
    repeated `sync` is a byte-preserving no-op.

- [x] **4.3** Adopt the existing working corpus and exercise one ordinary append.
  - Back up only the small `_ledgers/` and `meta/` JSON/JSONL files before first migration;
    do not copy, delete, or rebuild parquet/video data.
  - Run dry-run, ledger adoption, default `check`, append any currently new source episodes,
    and parallel stats for changed datasets.
  - Acceptance: old episode/global indices are unchanged, new ones are contiguous, only
    changed datasets lose/regain stats, and no manifest/source-identity artifact is created.

- [x] **4.4** Run a short training smoke and cut over maintained entry points.
  - Use the complete multi-dataset train path with a temporary config setting a small
    `max_steps`; require finite loss and a clean exit.
  - Remove the three superseded semihumanoid-only entry points and old test after the smoke
    passes; keep survey helpers and the historical 2026-08-18 plan.
  - Update the generalized design's example command and illustrative YAML with the verified
    h5py overlay, existing working-root choice, the two completed output config fields, and
    the implemented segment/skipped ledger and transaction details.
  - Acceptance: repository search finds no active reference directing users to removed
    scripts, while the historical plan remains explicitly historical.

---

## Risk assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Legacy-ledger import changes assignments | Existing parquet/video no longer matches metadata | Preserve all v1 indices/offsets/splits; dry-run and back up only ledger/meta files before atomic migration |
| Parallel or interrupted sync creates gaps | Loader addresses the wrong episode or stats include partial data | Single writer lock, staging, contiguous main-process assignment, transient roll-forward journal, startup recovery tests |
| Source repair changes length | Later global offsets become invalid | Permit in-place reconvert only for unchanged segment topology; require new root/rebuild otherwise |
| Layout config changes under an existing canonical key | One dataset mixes incompatible dimensions, action semantics, or viewpoints | Compare direct `_layout.json` values only; use a new root for intentional layout changes |
| Stats survive an append | Training uses stale normalization | Delete both stats files only after changed commits; auto-regenerate before train |
| Warning accidentally becomes admission policy | Growing internal data is unexpectedly blocked | Model selection, warnings, and structural failures as separate result types and test each path |
| Generated Python and dataset modality drift | Loader slices different tensors than training expects | Render both from one normalized layout and test structural equality/importability |
| rot6d convention regresses | Numerically plausible but wrong orientations train silently | Keep GR00T-decoder round-trip and naive-passthrough guard tests |
| New task reorders old indices | Existing parquet task labels change meaning | Append task IDs by first committed episode index; preserve migrated indices |
| `h5py` becomes an undeclared permanent dependency | Platform manifests and lockfiles drift for a conversion-only tool | Keep the verified uv overlay and lazy imports; assert dependency files stay unchanged |
| Plain rerun resumes or writes into an old run unexpectedly | Checkpoints from different attempts mix | Timestamp fresh runs; require explicit `--resume-from` for continuation |

## Validation commands

```bash
# Focused unit/integration tests (no corpus or GPU required).
uv run --no-sync --with h5py python -m pytest \
  scripts/lerobot_conversion/test_zdata_pipeline.py -q

# Lint and formatting for the new Python surfaces.
uv run --no-sync ruff check \
  scripts/lerobot_conversion/run_zdata_pipeline.py \
  scripts/lerobot_conversion/zdata_pipeline \
  scripts/lerobot_conversion/test_zdata_pipeline.py \
  examples/semihumanoid/semihumanoid_config.py
uv run --no-sync ruff format --check \
  scripts/lerobot_conversion/run_zdata_pipeline.py \
  scripts/lerobot_conversion/zdata_pipeline \
  scripts/lerobot_conversion/test_zdata_pipeline.py \
  examples/semihumanoid/semihumanoid_config.py

# Existing-corpus preview: must not write or reconvert data.
uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
  sync --config configs/embodiments/semihumanoid.yaml --dry-run

# Bounded pilot config points to a temporary two-episode source/output tree.
uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
  sync --config docs/local/zdata_pipeline_pilot.yaml
uv run --no-sync python scripts/lerobot_conversion/run_zdata_pipeline.py \
  stats --config docs/local/zdata_pipeline_pilot.yaml
uv run --no-sync python scripts/lerobot_conversion/run_zdata_pipeline.py \
  check --config docs/local/zdata_pipeline_pilot.yaml

# Repository-level gates after cutover.
uv lock --locked
python tools/check_manifest_alignment.py
pre-commit run --all-files
git diff --check
```

The GPU smoke command uses a temporary copy of the semihumanoid YAML with a small
`train.max_steps` and temporary `train.out_base`; it is intentionally not part of routine
CPU validation.

## Implementation record

- The focused generalized pipeline suite passes all 46 tests with the on-demand `h5py`
  overlay. The repository CPU suite passes 703 tests, with 2 skipped and 35 GPU tests
  deselected.
- The bounded real-data pilot covered one native 32D UBE episode and one native 46D Matcha
  episode. Canonical arrays matched the legacy converter exactly; real ffmpeg, stats, full
  checks, loader shapes, three-view visual sampling, and a byte-preserving no-op all passed.
- Live adoption preserved 1,857 prior assignments and appended one newly discovered source
  at the next validation episode/global indices. All 14 outputs pass `check`; only the
  changed dataset regenerated stats. The metadata backup is
  `data/training_data/gr00t/semihumanoid_260818_metadata_backup_20260819T070503Z`.
- A four-GPU, seven-train-dataset, one-step smoke completed cleanly with finite
  `train_loss: 1.328125`. Its isolated 40 GiB temporary pilot/checkpoint tree was deleted
  afterward and is not recoverable.
- `pre-commit run --all-files`, `uv lock --locked`, manifest alignment, and
  `git diff --check` pass. The pipeline creates no manifest or source-identity artifact; it
  leaves the pre-existing legacy `manifest.json` inert and untouched.

## Completion criteria

- The ordinary user flow is `sync` and explicit `train`; no `all` command exists.
- New source paths append without changing existing assignments, and a no-op sync changes
  no corpus files.
- `sync --dry-run` is fully read-only, and interrupted finalization is recoverable from the
  transient roll-forward journal before any stats/check/train operation proceeds.
- The existing working corpus is adopted without parquet/video regeneration.
- Warnings do not block conversion; structural failures are isolated and retryable.
- Stats are refreshed only for changed datasets and are present before loader/training use.
- The generated modality module and every dataset's modality/layout agree with the YAML.
- Epoch-based training reflects the corpus size at launch and uses an explicit fresh or
  resumed output directory.
- Legacy semihumanoid-only entry points are removed only after the bounded pilot, append,
  loader, and training smoke pass.
- No release manifest, checksum, source fingerprint, catalog-version gate, or routine full
  scan is introduced.

There are no unresolved design decisions blocking implementation.
