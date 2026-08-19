# Semihumanoid → GR00T N1.7 Conversion & Finetune Plan

**Date:** 2026-08-18
**Status:** Planning
**Author:** generated from source inspection of `data/training_data/semihumanoid/`, the
external `<industrialnext-ai-repo>/`, and this repo
**Output dataset:** `data/training_data/gr00t/semihumanoid_260818/`

> ## Corpus state as of 2026-08-19 — READ THIS BEFORE ACTING ON ANY NUMBER BELOW
>
> Two subsets (`flexiv_matcha_v4`, `flexiv_ube_v4`) were collected after this plan was
> written and converted on 2026-08-19 with **zero QC drops**. The measurements in the body
> of this document are the **2026-08-18 five-subset snapshot** and are kept as the record of
> what was analysed then. Current reality:
>
> | | 2026-08-18 (body of this doc) | 2026-08-19 (current) |
> |---|---|---|
> | Subsets | 5 | **7** |
> | Collected episodes | 1,426 | **1,858** |
> | Post-QC train episodes | 1,345 | **1,762** |
> | Post-QC train frames | 492,735 | **679,671** (3.78 h) |
> | Val episodes | 80 | **95** |
> | Trainable 40-step starts | 440,280 | **610,953** |
> | `bracket_handover` episodes | 32 | **99** |
>
> Both new subsets are 46-dim state, 20-dim action, carry all three canonical cameras, and
> are 100% `*_expert` at 50/30 Hz, 256×256, rot6d — i.e. fully covered by the existing
> config. Across all 7 subsets: 1,054 episodes at 32-dim state and 804 at 46-dim, tasks
> `generic_pick` 900 / `generic_place` 859 / `bracket_handover` 99, `action/residual`
> identically zero, and every episode contains the three canonical cameras.
>
> **Consequence for Phase 5:** `--max-steps` is an absolute count, so the previously-correct
> 3,750 steps at batch 256 now covers only **1.57 epochs** instead of 2.18. Use **5,250
> steps** (~2.77 h at the measured 134.7 samples/s) to hold ~2.2 epochs. Phase 5.1 below has
> been updated; the launcher's committed default has not.

---

## Motivation

We want to finetune `nvidia/GR00T-N1.7-3B` on the internal `semihumanoid` bimanual
manipulation data currently consumed by the in-house DEFT-S24 pipeline
(`<industrialnext-ai-repo>/config/manipulation/semihumanoid/deft_s24_semihumanoid.yaml`).

The internal data is stored in a bespoke `zdata_hdf5` format (one `episode.h5` per
episode, JPEG-blob images, multi-rate state/image clocks). GR00T requires
GR00T-flavored **LeRobot v2**: per-episode parquet + per-camera MP4 + `meta/*.jsonl` +
`meta/modality.json`. This plan defines the conversion and the finetune run.

**Target state:** a set of self-contained LeRobot v2 datasets under
`data/training_data/gr00t/semihumanoid_260818/`, one per source subset, plus a
modality config registered under `EmbodimentTag.NEW_EMBODIMENT`, plus a validated
finetune command.

**Non-goals**
- Replacing the internal DEFT pipeline. This is an additive GR00T track.
- Depth. GR00T N1.7 has **no depth modality** (only RGB `video` keys). All
  `*_depth` streams are dropped.
- Progress prediction. GR00T has no progress head; `progress_prediction` in the
  internal config has no GR00T counterpart and is dropped (it is derivable from
  episode ratio anyway).
- 20-step state history. The internal model uses `n_obs_steps: 20`; GR00T N1.7 ships
  `state_history_length` unset and every in-tree embodiment uses `delta_indices=[0]`
  for state and video. We match GR00T (current frame only).

---

## Source Data Profile (measured, not assumed)

Surveyed **all 1,426 episodes** in the five subsets — not a sample. Zero read errors,
`valid_for_training=True` and `state/present` all-true on every episode, `residual`
identically zero on every episode.

Reproducible: the survey script and its raw output are kept beside this plan as
`scripts/lerobot_conversion/survey_zdata_source.py` (subsets auto-discovered, so it stays
correct as data is added; run with
`uv run --no-sync --with h5py python scripts/lerobot_conversion/survey_zdata_source.py <out.json>`).
The raw JSON output is not committed -- it is ~800 KB of machine-generated data, regenerable
in ~2 minutes.
Every count in this section is derived from that JSON, so any claim here can be
re-checked without re-reading 55 GB of HDF5.

| Subset | Episodes | Frames | Len min/med/max | State dim | RGB cameras |
|---|---|---|---|---|---|
| `flexiv_matcha_v2` | 262 | 96,721 | 234/328/1475 | 32 | head, left_bottom, left_top, right_bottom |
| `flexiv_matcha_v3` | 160 | 66,305 | 258/393/993 | **46** | head, left_bottom, left_top, right_bottom |
| `flexiv_ube_v1` | 322 | 127,559 | 268/362/1095 | 32 | head, left_bottom, right_bottom |
| `flexiv_ube_v2` | 470 | 151,451 | 181/302/1099 | 32 | head, left_bottom, right_bottom |
| `flexiv_ube_v3` | 212 | 79,110 | 238/338/1126 | **46** | head, left_bottom, right_bottom |
| **Total (in scope)** | **1,426** | **521,146** | — | — | — |

These are **as-collected** counts. One episode is later removed by the QC gate, leaving
**1,425 / 520,822** as the working scope — do not mix the two sets of numbers when writing
acceptance tests.

Uniform across all episodes: `schema_version=1.0`, `recording_mode=multi_rate`,
`rotation_mode=rot6d`, `sampling_hz=50.0`, `image_sampling_hz=30.0`, images 256×256
JPEG, action dim **20**.

**The camera set is homogeneous *within* each subset.** In scope there are two
signatures: 1,004 episodes with exactly `{head, eoat_left_bottom, eoat_right_bottom}` and
422 (matcha_v2/v3) that additionally carry an unused `eoat_left_top`. All 1,426 in-scope
episodes contain all three canonical cameras, so one global camera map suffices. State dim
is likewise constant within a subset.

**Tasks** (3 distinct strings, from `task_catalog.yaml` v2026-08-16). In scope, as
collected: `generic_pick` **710**, `generic_place` **684**, `bracket_handover` **32**
(ube_v3 only). After the single QC drop: 710 / 683 / 32.

**All five subsets are 100% `*_expert`.** The `_expert` criterion in the QC gate is
therefore a guard against future non-expert data, not an active filter.

**Native field layout** (read from `state/field_names` + `state/field_slices` per episode):

```
action  (20 dims, all subsets):
  left_arm_pose_pos [0:3]  left_arm_pose_rot [3:9]   left_gripper [9:10]
  right_arm_pose_pos [10:13] right_arm_pose_rot [13:19] right_gripper [19:20]

state (32 dims -- matcha_v2, ube_v1, ube_v2):
  left_arm_pose_pos [0:3] left_arm_pose_rot [3:9] left_gripper [9:10] left_ft [10:16]
  right_arm_pose_pos [16:19] right_arm_pose_rot [19:25] right_gripper [25:26] right_ft [26:32]

state (46 dims — matcha_v3, ube_v3) — adds joints, SHIFTING every downstream slice:
  left_arm_joints [0:7] left_arm_pose_pos [7:10] left_arm_pose_rot [10:16]
  left_gripper [16:17] left_ft [17:23] right_arm_joints [23:30] ...
```

### Findings that drive the design

1. **Multi-rate is already solved by the source.** State runs at 50 Hz, images at
   30 Hz, and each camera has its own image count (e.g. a ube_v2 episode holds 174 head
   images against 292 state frames; every episode in every subset is `multi_rate`). Every
   camera group carries `frame_ref_index` of length `frame_count`, mapping each state
   frame → image index, plus `frame_age_ms` (in-scope medians of `age_p99` are 49–55 ms).
   **We index images through `frame_ref_index`; no resampling or timestamp math is
   needed.**

2. **Field slices shift between subsets.** v3 subsets prepend 7-dim joint arrays per
   arm. A converter that hardcodes `[0:3]`/`[3:9]` silently corrupts the **371 in-scope v3
   episodes — 26% of the corpus**. **Selection must be by `field_names`, never by index.**

3. **rot6d convention mismatch — silent and severe.** Verified numerically:
   - internal `_matrix_to_rot6d` = `concat([R[:,0], R[:,1]])` → **first two columns**
     (`<industrialnext-ai-repo>/src/industrialnext_ai/common/rotation_representations.py:115`)
   - GR00T `_matrix_to_rot6d` = `R[:2,:].flatten()` → **first two rows**
     (`gr00t/data/state_action/pose.py:458`)

   Feeding internal rot6d straight into GR00T reconstructs **Rᵀ**, not R. On a random
   rotation this produced a **170.2° orientation error**. Because GR00T composes
   relative EEF transforms from this matrix, the corruption is systematic and would
   look like "training runs but the policy rotates wrongly" rather than an error.
   **The converter must transpose:** rebuild R from the internal column convention,
   then emit `R[:2,:].flatten()`. The inverse mapping is mandatory at deployment on
   GR00T's predicted rot6d.

4. **Action/observation alignment needs no shift.** The internal config sets
   `action_source: "action"`, and `_target_start_index`
   (`<industrialnext-ai-repo>/src/industrialnext_ai/datasets/zdata_hdf5_dataset.py:2637`)
   returns `frame_index` unchanged for that source — `obs_based_action_offset` applies
   only to `action_source: "observation"`. So the action chunk for observation *t* is
   `action/executed[t : t+40]`, which is exactly GR00T's `delta_indices=range(0,40)`.
   No off-by-one correction.

5. **`action/executed` is the right source and is identical to `expert`.**
   The internal resolver prefers `executed` when present
   (`zdata_hdf5_dataset.py:3170`), and `action/residual` is identically zero across
   all 1,426 episodes, so `executed == expert` everywhere.

6. **The data is overwhelmingly single-arm.** Only a small minority of episodes
   move both arms >5 mm. Over the **1,425 in-scope episodes**: **173 move both arms
   (12.1%), 48% have a static left arm, 40% a static right arm.** Which arm is active is
   subset-dependent (matcha → mostly right, ube → mostly left). R4 uses these figures.
   This is legitimate "hold still" supervision, but see Risk R4.

7. **Dimensions fit GR00T exactly.** Base checkpoint `config.json` reports
   `action_horizon: 40`, `max_state_dim: 132`, `max_action_dim: 132`,
   `image_target_size: [256, 256]`. Our 32-dim state / 20-dim action / 40-step horizon
   / 256×256 images all fit — the horizon fits *exactly* at the ceiling, and native
   image size matches GR00T's target with no rescale.

8. **Conversion is cheap.** Measured on one ube_v2 episode: piping source JPEG bytes
   straight into `ffmpeg -f image2pipe -c:v libx264 -crf 23` yields **≈1.13 KB/frame
   for all three cameras combined** (288 + 423 + 420 B) at ~1,500 frames/s/camera, and
   the result decodes cleanly through `gr00t.utils.video_utils.get_frames_by_indices`.

---

## Scope & Data Selection

**Scope: five subsets — `matcha_v2`, `matcha_v3`, `ube_v1`, `ube_v2`, `ube_v3`.**
These are the only semihumanoid subsets present on this machine (55 GB total);
`flexiv_matcha_v1` was removed on 2026-08-18 and now lives only in cloud backup.

**Camera map — one uniform mapping for all five subsets:**

| Canonical key | Physical stream |
|---|---|
| `head` | `head_rgb` |
| `left_wrist` | `eoat_left_bottom_rgb` |
| `right_wrist` | `eoat_right_bottom_rgb` |

All 1,426 in-scope episodes contain all three. Two camera signatures exist in scope
(1,004 episodes with exactly the canonical set; 422 in matcha_v2/v3 that additionally
carry an unused `eoat_left_top_*`), but since the extra stream is simply not mapped, the
mapping is uniform and can be asserted once at discovery. All `*_depth` streams are
dropped — GR00T N1.7 has no depth modality.

### Camera-health QC gate

A health survey measured per-camera coverage (`image_count / frame_count`; 0.60 is the
ceiling at a 30 Hz image clock against 50 Hz state) and staleness (`frame_age_ms` p99).
**The in-scope corpus is uniformly healthy** — coverage medians 0.61–0.62 with p10 ≥ 0.59,
`age_p99` medians 49–55 ms. Keep an episode iff **all** hold:

> - every canonical camera has `0.45 <= coverage <= 0.80`
> - every canonical camera has `frame_age_ms p99 <= 150`
> - `frame_count >= 41` (below this a 40-step horizon yields zero samples)
> - `policy_type == expert` (episode directory suffix `_expert`)

**On the current data the gate drops exactly one episode:** `matcha_v3`
`20260817_231631_expert`, which fails both coverage and staleness (`age_p99` = 2,535 ms).
The other three criteria fire zero times: no episode exceeds `coverage 0.80`, the shortest
episode is 181 frames, and all five subsets are 100% `*_expert`.

**Keep all four criteria even though three are inert.** They are not speculative — each
was written against a real failure observed in a subset that has since been removed
(coverage up to 4.16 from a truncated state stream, staleness to 9.2 s from a one-day
camera fault, and 6 `*_policy` rollouts mixed in with expert demos). Those pathologies can
recur in the next recording batch; the gate is the only thing that would catch them.

| | Episodes | Frames | Hours @50 Hz | Usable start indices (H=40) |
|---|---|---|---|---|
| Five subsets, as collected | 1,426 | 521,146 | 2.90 | — |
| **After QC gate (plan scope)** | **1,425** | **520,822** | **2.89** | **465,247** |
| Dropped | 1 | 324 | — | — |

Per-subset kept: matcha_v2 **262**, matcha_v3 **159**, ube_v1 **322**, ube_v2 **470**,
ube_v3 **212**.

Thresholds and per-episode measurements are reproducible via
`scripts/lerobot_conversion/camera_health_zdata.py`, which also prints the per-subset table the thresholds were read off.

---

## Growth Model — the corpus is append-only, not sealed

Data collection is ongoing, so the converted corpus must accept new episodes without a
rebuild and without disturbing what is already there. Three mechanisms make that safe;
all three live in `scripts/lerobot_conversion/convert_semihumanoid.py`.

**1. A per-subset ledger freezes episode identity.**
`<out-root>/_ledgers/<subset>.json` records, for every source episode,
`(dataset, episode_index, split, index_offset, length, task_uuid)`. Re-running the
converter converts only sources absent from the ledger. Without it, `episode_index` would
be the position in the sorted discovery list — so a **backfilled older recording date
would renumber every later episode**, orphaning already-written parquet/MP4 files and
silently remapping the dataset. The ledger also carries the `camera_map` and refuses to
run if the current mapping differs, so a changed camera assignment can never be blended
into an existing output.

**2. The train/val split is hash-based, not positional.**
`assign_split()` hashes the episode's path relative to its subset and sends
`hash % val_every == 0` to `<subset>_val`. The plan originally specified
`episode_index % 20 == 0`; that was **changed deliberately**, because a positional rule
moves existing episodes between train and val whenever earlier episodes are inserted —
which would leak held-out data into training on the next append. Assignments are frozen in
the ledger regardless, so the rule only ever decides for newly-seen episodes. Cost: the
val fraction is ~1/N in expectation rather than exactly 1/N.

**3. Cached stats are deleted whenever episodes are added.** This is the subtle one.
GR00T fingerprints `meta/stats.json` over the `info.json` feature *schema* only
(`gr00t/data/stats.py:183` — feature name, dtype, shape), and `relative_stats.json` over
the embodiment/action config (`gr00t/data/stats.py:388`). **Neither changes when episodes
are appended**, so both files would be considered fresh and the next run would normalize
against the old, smaller episode set — with no error and no warning. The converter
therefore removes both whenever it writes an episode, and logs that it did. They
regenerate on the next `gr00t/data/stats.py` run, or automatically at training start
(`gr00t/data/dataset/factory.py:59`).

### Adding data later

```bash
OUT=data/training_data/gr00t/semihumanoid_260818

# 1. convert (only new episodes are touched; --dry-run to preview)
uv run --no-sync --with h5py python scripts/lerobot_conversion/convert_semihumanoid.py \
    --source-subset <path/to/flexiv_new_subset> --out-root $OUT --workers 8

# 2. regenerate stats for the datasets whose episode set changed (the converter deleted
#    their cached stats). Run them in parallel -- the relative pass is CPU-bound at
#    ~2.5 s/episode/key, so serial is ~4x slower and training would otherwise do this
#    work at startup with the other GPUs idle.
for ds in $(python scripts/lerobot_conversion/semihumanoid_datasets.py --out-root $OUT \
              --print train | tr ':' ' '); do
  ( uv run python gr00t/data/stats.py --dataset-path $ds \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path examples/semihumanoid/semihumanoid_config.py ) &
done; wait

# 3. see what the corpus now contains, refresh the manifest, get the --dataset-path string
python scripts/lerobot_conversion/semihumanoid_datasets.py --out-root $OUT --write-manifest

# 4. re-train, letting the helper enumerate datasets so a new subset is never missed
uv run torchrun --nproc_per_node=4 gr00t/experiment/launch_finetune.py \
    --dataset-path "$(python scripts/lerobot_conversion/semihumanoid_datasets.py \
                        --out-root $OUT --print train)" \
    ... # remaining flags as in 5.1
```

A brand-new subset needs no code change: it becomes its own dataset directory plus a
`_val` sibling, and step 4 picks it up automatically. Never hand-write the dataset list —
that is how a new subset silently gets left out of a run.

`semihumanoid_datasets.py` also flags any dataset whose stats are missing, so a corpus
that has grown since its last stats run is visible at a glance.

---

## Target Format Mapping

**Canonical output layout** — one LeRobot v2 dataset *per subset* (not one merged
dataset). Rationale: every subset stays under 1,000 episodes so everything lands in
`chunk-000` and the LeRobot chunk arithmetic never comes up; per-subset `mix_ratio`
weighting and exclusion stay available at train time; conversion parallelizes; and
stats are per-subset with GR00T merging them by sampling weight.

```
data/training_data/gr00t/semihumanoid_260818/
├── manifest.json                     # provenance: source paths, camera map, QC drops, counts, git SHAs
├── matcha_v2/                        # 262 eps
│   ├── meta/{info,modality}.json  meta/{episodes,tasks}.jsonl
│   ├── data/chunk-000/episode_000000.parquet …
│   └── videos/chunk-000/observation.images.{head,left_wrist,right_wrist}/episode_000000.mp4
├── matcha_v3/   # 159   (160 collected − 1 QC-dropped)
├── ube_v1/      # 322
├── ube_v2/      # 470
├── ube_v3/      # 212
└── <subset>_val/  # 5% held-out sibling per subset (checklist 3.5), ~74 eps total
```
All five subsets stay under 1,000 episodes, so every dataset uses `chunk-000` only.

**Per-frame parquet columns**

| Column | Type | Content |
|---|---|---|
| `observation.state` | list[float32] (32) | canonical state, order below |
| `action` | list[float32] (20) | canonical action, order below |
| `timestamp` | float32 | `frame/elapsed_ms / 1000` |
| `task_index` | int64 | index into `meta/tasks.jsonl` |
| `episode_index`, `frame_index`, `index` | int64 | standard LeRobot |
| `next.done` | bool | `frame/done` |

No `annotation.*` column is written. `meta/modality.json` redirects the annotation
through `original_key: task_index`, which the loader honors
(`gr00t/data/dataset/lerobot_episode_loader.py:381`) — verified against the shipped
`demo_data/cube_to_bowl_5`, which does exactly this.

**Canonical state (32)** — chosen so each EEF pose is a contiguous 9-dim
`[xyz, rot6d]` block, which `ActionType.EEF` + `ActionFormat.XYZ_ROT6D` requires:

```
left_eef      [0:9]    = left_arm_pose_pos(3) ++ rot6d_transposed(left_arm_pose_rot)(6)
left_gripper  [9:10]
left_ft       [10:16]
right_eef     [16:25]  = right_arm_pose_pos(3) ++ rot6d_transposed(right_arm_pose_rot)(6)
right_gripper [25:26]
right_ft      [26:32]
```

**Canonical action (20)**

```
left_eef      [0:9]
left_gripper  [9:10]
right_eef     [10:19]
right_gripper [19:20]
```

Joint arrays present in v3 subsets are **dropped** so the state vector is uniform
across all output datasets (a single embodiment tag admits exactly one state layout).
This also matches the internal `shape_meta`, which does not list joints.

**`meta/modality.json`** (identical for all five datasets):

```json
{
  "state": {
    "left_eef":  {"start": 0,  "end": 9},
    "left_gripper": {"start": 9, "end": 10},
    "left_ft":   {"start": 10, "end": 16},
    "right_eef": {"start": 16, "end": 25},
    "right_gripper": {"start": 25, "end": 26},
    "right_ft":  {"start": 26, "end": 32}
  },
  "action": {
    "left_eef":  {"start": 0,  "end": 9},
    "left_gripper": {"start": 9, "end": 10},
    "right_eef": {"start": 10, "end": 19},
    "right_gripper": {"start": 19, "end": 20}
  },
  "video": {
    "head":        {"original_key": "observation.images.head"},
    "left_wrist":  {"original_key": "observation.images.left_wrist"},
    "right_wrist": {"original_key": "observation.images.right_wrist"}
  },
  "annotation": {"human.task_description": {"original_key": "task_index"}}
}
```

**Modality config** → `examples/semihumanoid/semihumanoid_config.py`:

```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

semihumanoid_config = {
    "video": ModalityConfig(delta_indices=[0],
        modality_keys=["head", "left_wrist", "right_wrist"]),
    "state": ModalityConfig(delta_indices=[0],
        modality_keys=["left_eef","left_gripper","left_ft",
                       "right_eef","right_gripper","right_ft"]),
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),          # == internal n_action_steps
        modality_keys=["left_eef","left_gripper","right_eef","right_gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF,
                         format=ActionFormat.XYZ_ROT6D, state_key="left_eef"),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                         format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF,
                         format=ActionFormat.XYZ_ROT6D, state_key="right_eef"),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                         format=ActionFormat.DEFAULT),
        ]),
    "language": ModalityConfig(delta_indices=[0],
        modality_keys=["annotation.human.task_description"]),
}
register_modality_config(semihumanoid_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

Grippers are `ABSOLUTE` (already normalized to [0,1] natively, and binary-ish targets
train better absolute — matches the shipped SO100 precedent). EEF poses are `RELATIVE`,
which is N1.7's headline action space and requires the paired `state_key`.

**Frame rate:** keep native **50 Hz**. A 40-step chunk covers 0.8 s, identical to the
internal `n_action_steps: 40` at `fps: 50`, so deployment control rate is unchanged.
Video frames are duplicated where `frame_ref_index` repeats — roughly **60% of written
frames are distinct** (30 Hz image clock ÷ 50 Hz state clock; the measured ube_v2
sample had 174 unique images across 292 state frames). h264 P-frames make repeated
frames nearly free, and GR00T reads only `delta_indices=[0]`, so the duplication is
harmless.

The alternative — decimating to 25 Hz — was rejected: it would halve the temporal
resolution of the action chunk and change the control rate the policy is trained for,
forcing controller-side interpolation at deployment. Storage is not a reason to do it
(the whole corpus is under 1 GB either way).

---

## Existing Modules to Reuse

All line numbers below were verified against the working tree at commit `376ba89`.

| Path | What it gives us |
|---|---|
| `gr00t/data/dataset/lerobot_episode_loader.py:144` | `_load_metadata` — authoritative list of required `meta/` files |
| `gr00t/data/dataset/lerobot_episode_loader.py:180` | `assert stats_path.exists()` — **the loader refuses to construct without `meta/stats.json`** |
| `gr00t/data/dataset/lerobot_episode_loader.py:60` | `DEFAULT_COLUMN_NAMES` = `{state: "observation.state", action: "action"}` — why our `modality.json` needs no `original_key` on state/action |
| `gr00t/data/dataset/lerobot_episode_loader.py:381` | annotation `original_key` indirection → lets us skip `annotation.*` columns |
| `gr00t/data/dataset/lerobot_episode_loader.py:429` | `assert original_key in self.feature_config` — video keys **must** appear in `info.json.features` |
| `gr00t/data/dataset/factory.py:59` | rank-0 auto-generation of `stats.json` + `relative_stats.json` at train start — we do **not** ship stats |
| `gr00t/data/stats.py:251` (`generate_stats`) | stats are computed **only** for `info.json.features` entries whose `dtype` contains `"float"` |
| `gr00t/data/stats.py:438` (`main`) | standalone stats CLI for pre-flight validation |
| `gr00t/data/stats.py:345` | EEF relative-stat path: reference pose built from `state.<state_key>` via `from_action_format` |
| `gr00t/data/state_action/pose.py:426,458` | GR00T rot6d ⇄ matrix (rows convention) — the reference for the transpose fix |
| `gr00t/data/state_action/pose.py:681` | `from_action_format`: `XYZ_ROT6D` ⇒ `translation=data[:3]`, `rotation=data[3:]` — confirms our 9-dim block layout |
| `gr00t/data/utils.py:106` | `mask = ~np.isclose(max_vals, min_vals)` — degenerate dims are zeroed, not NaN (see R4) |
| `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py:132` | `validate_action_horizons` — fails fast if horizon > 40 |
| `gr00t/utils/video_utils.py:70` | `get_frames_by_indices` — the exact decode path our MP4s must satisfy |
| `gr00t/experiment/launch_finetune.py:59` | `dataset_path.split(os.pathsep)` — multi-dataset joining |
| `gr00t/data/dataset/sharded_single_step_dataset.py:236` | `get_effective_episode_length` = `max(0, len - horizon + 1)` |
| `gr00t/eval/open_loop_eval.py:323` | `policy.get_modality_config()` — eval reads the modality config **from the checkpoint**, not the registry |
| `examples/SO100/{so100_config.py,modality.json}` | closest shipped precedent for a `NEW_EMBODIMENT` config |
| `demo_data/cube_to_bowl_5/meta/` | reference `info.json` / `episodes.jsonl` / `tasks.jsonl` shapes |
| `<industrialnext-ai-repo>/src/industrialnext_ai/common/rotation_representations.py:105,115` | internal rot6d (columns convention) — source side of the transpose |
| `<industrialnext-ai-repo>/src/industrialnext_ai/datasets/zdata_hdf5_dataset.py:2637,3170` | action-source + offset semantics we must preserve |

**Verified registry state:** `new_embodiment` is *absent* from the built-in
`MODALITY_CONFIGS` (only the 8 pretrain/posttrain tags are present), so
`register_modality_config`'s "already registered" assertion cannot fire for us.

---

## Phased Breakdown

### Phase 0 — Decisions & scaffolding
- **Problem:** the camera map and output location must be fixed before any bulk write.
- **Solution:** record the uniform camera map; create `examples/semihumanoid/` and the output
  root; write the per-subset camera map.
- **Impact:** conversion is deterministic and re-runnable.

### Phase 1 — Converter script
- **Problem:** no tool maps `zdata_hdf5` → GR00T LeRobot v2.
- **Solution:** `scripts/lerobot_conversion/convert_semihumanoid.py` — name-based field
  selection, rot6d transpose, `frame_ref_index` image indexing, JPEG→h264 piping,
  full `meta/` emission, per-subset parallelism, resumable per episode.
- **Impact:** one command per subset produces a loadable dataset.

### Phase 2 — Pilot on one subset
- **Problem:** bulk conversion errors are expensive to discover late.
- **Solution:** convert `matcha_v3` (160 eps — smallest *and* the 46-dim schema, so it
  exercises the field-shift path) and validate end to end.
- **Impact:** the risky code paths are proven on the hardest subset first.

### Phase 3 — Full conversion
- **Problem:** remaining four subsets (matcha_v3 is done in the Phase 2 pilot).
- **Solution:** run the converter over them, in parallel; write `manifest.json`.
- **Impact:** complete corpus at ~0.8 GB.

### Phase 4 — Dataset validation
- **Problem:** GR00T failures surface as flat predictions or huge MSE, not exceptions.
- **Solution:** loader round-trip, stats sanity, rot6d round-trip assertion, and a
  100-step smoke finetune.
- **Impact:** we enter the long run with the pipeline proven.

### Phase 5 — Finetune
- **Problem:** produce a usable checkpoint on 4×RTX 4090.
- **Solution:** multi-dataset `torchrun` run with tuned batch/steps.
- **Impact:** a checkpoint to evaluate.

### Phase 6 — Evaluation & deployment notes
- **Problem:** open-loop numbers plus the inverse rot6d mapping needed on-robot.
- **Solution:** `open_loop_eval.py` per subset; document the deployment transform.
- **Impact:** decision data for a real robot trial.

---

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| **R1 rot6d transpose omitted** | Systematically wrong rotations (~170° on random poses); trains fine, policy rotates wrongly. Silent. | Unit test asserting `groot_from6d(convert(internal_6d)) == R` for 1k random rotations; assert on real data that each reconstructed `R` satisfies `RᵀR ≈ I` and `det(R) ≈ +1`. Phase 1 checklist item 1.4. |
| **R2 hardcoded field slices** | 372 v3 episodes (20% of corpus) silently corrupted — joints read as pose. | Select by `field_names`; assert the resolved slice width equals the expected width per field; fail loudly on any unknown field name. |
| **R3 `frame_ref_index` misuse** | Off-by-one image/state misalignment; policy conditioned on stale or future frames. | Assert `len(frame_ref_index) == frame_count` and `0 <= idx < image_count` for every camera; log `frame_age_ms` p99 per subset. |
| **R4 near-degenerate relative-action stats** | 48%/40% of included episodes hold an arm still, so per-arm relative deltas cluster at 0. `normalize_values_minmax` (`gr00t/data/utils.py:106`) zeroes dims where `min≈max`, but a *tiny-but-nonzero* q01–q99 range divides by ~1e-6 and amplifies noise ~1e6×. | Stats are pooled per dataset and then merged across datasets by weight, so the moving 50% dominates — but verify explicitly: dump merged `relative_stats.json` per-dim ranges before the long run (Phase 4.3). If any EEF dim range < 1e-4, switch that key to `mean_std_embedding_keys` or pass `--no-use-percentiles`. |
| **R5 small corpus (2.89 h)** | 1,425 episodes / 2.89 h is modest for a 3B VLA. The repo FAQ suggests ~100 trajectories for simple fixed-location pick-and-place and ~500+ for multi-step or varied scenes; we have two distinct scenes (matcha, ube) and three tasks, one of which (`bracket_handover`) has only 32 episodes. Risk is under-fitting or scene-specific overfitting. | Measure it rather than guess: 6.5's train-vs-held-out gap is the signal. Levers if the gap is large are listed in 6.5. A previously-considered subset (`flexiv_matcha_v1`, ~338 QC-clean episodes) exists in cloud backup only — see Open Question 5 for the cost of restoring it. |
| ~~**R6 viewpoint mixing**~~ | **RETIRED** — all five subsets supply the same three physical viewpoints, so no canonical key ever blends two mounts. | n/a |
| **R14 non-expert episodes** | Behaviour cloning treats `*_policy` rollouts as expert targets, importing a worse action distribution and an off-policy state distribution. Such episodes are invisible in every field the loader reads — only the directory suffix distinguishes them. **Inert on current data** (all five subsets are 100% expert), but live the moment new data lands: policy rollouts have appeared in this corpus before. | Keep the `_expert` criterion in 1.16 as a standing guard. If DAgger-style training is wanted later, convert policy rollouts into a separate dataset and weight it explicitly rather than merging silently. |
| **R13 camera dropout / stale frames** | Episodes where a canonical camera drops out condition the policy on seconds-old observations while the state moves — plausible-looking data that teaches the wrong visuomotor mapping. **On current data this hits exactly 1 episode** (`matcha_v3 20260817_231631_expert`, `age_p99` = 2,535 ms), but a single bad recording day previously produced 57 such episodes with staleness to 9.2 s. | Apply the full QC gate at discovery (checklist 1.16) and log every drop with measured coverage/age into `_conversion_report.json`. Keep all four criteria even though three are currently inert — they are the guard against the next bad recording day. |
| **R7 episode shorter than horizon** | `get_effective_episode_length` (`sharded_single_step_dataset.py:236`) returns `max(0, len - 40 + 1)`, so a short episode contributes zero samples; if *every* episode were short the sharder raises `"No valid timesteps found … episode lengths may be shorter than action horizon"` (`sharded_single_step_dataset.py:195`). | **Inert on current data**: the shortest episode is 181 frames, so `frame_count >= 41` never fires. The criterion stays in 1.16 as a guard — short aborted recordings have occurred in this corpus before. |
| **R8 JPEG→h264 generational loss** | Slight visual degradation vs source. | Unavoidable for MP4. Use `-crf 20` (not 23) for the wrist cameras if a pilot A/B shows artifacts; measured cost is only ~0.42 KB/frame at crf 23. |
| **R9 disk / write amplification** | Reading **55 GB** of in-scope h5 to write ~0.70 GB. | I/O bound, one pass, resumable per episode. 849 GB free — ample. |
| **R10 stats written into a tracked tree** | Training regenerates `meta/stats.json`, dirtying git if data ever lives in-repo. | Output root is outside the repo. Keep it that way. |
| **R15 Phase 5 OOM / wall clock** | Phase 5 is the only step measured in hours, and it runs 3 cameras × 8 samples per GPU through the ViT — 50% more image tokens per step than the 2-camera shipped examples. An OOM 20 minutes in wastes the setup; an unmeasured step time makes the run unschedulable. 4×RTX 4090 also all-reduce ~3.2 GB of bf16 gradients per step over PCIe with no NVLink, so communication is a real fraction of step time. | Measure before committing: 30-step probe with VRAM watch (5.6), then read step time from the logs (5.3). Fall back to `--global-batch-size 16 --gradient-accumulation-steps 2` for the same effective batch at half the activation memory. Keep checkpoints resumable (5.5) so an interruption costs at most `--save-steps` of work. |
| **R11 `info.json.features` wrong or incomplete** | `generate_stats` (`gr00t/data/stats.py:251`) filters to features whose `dtype` contains `"float"`. Omit `action`/`observation.state`, or type them as e.g. `"double"`/`"list"`, and **no statistics are computed for them** — normalization then has no entries. The loader also asserts every video `original_key` is in `features` (`:429`). Both failures are config-time, not obviously data-related. | Checklist 1.11 acceptance test asserts the float-filter actually selects `action` and `observation.state`, and that declared shapes match written widths. |
| **R12 loader used before stats exist** | `LeRobotEpisodeLoader.__init__` asserts `meta/stats.json` (`:180`). Any ad-hoc inspection script written against a fresh dataset fails with an assertion that looks like a conversion bug. | Always run `gr00t/data/stats.py` before constructing a loader by hand. Encoded in checklist 2.4. Training itself is unaffected — `DatasetFactory` generates stats first (`factory.py:59`). |

---

## Detailed Checklist

### Phase 0 — Decisions & scaffolding
- [x] **0.1** ~~Confirm camera preset A vs B~~ — **resolved.** Scope is the five subsets present on disk, with one uniform camera map. Record the scope and the map in `manifest.json`.
- [x] **0.2** `mkdir -p data/training_data/gr00t/semihumanoid_260818`
- [x] **0.3** `mkdir -p examples/semihumanoid`
- [x] **0.4** Encode the **single** canonical→physical camera map — `head`←`head_rgb`, `left_wrist`←`eoat_left_bottom_rgb`, `right_wrist`←`eoat_right_bottom_rgb` — and assert all three keys exist in every episode of every subset (verified true for all 1,426).

### Phase 1 — Converter (`scripts/lerobot_conversion/convert_semihumanoid.py`)
- [x] **1.1** CLI: `--source-subset`, `--out-root`, `--crf`, `--workers`, `--limit`, `--overwrite`. (No camera-preset flag — the map is uniform across all five subsets.)

  **Interpreter — resolved.** Neither existing environment can run this alone: the GR00T
  venv has `pandas`/`pyarrow` but no `h5py`; `<industrialnext-ai-repo>/.venv` has `h5py 3.16.0`
  but no `pandas`/`pyarrow` (so it cannot write parquet). Do **not** add `h5py` to the root
  `pyproject.toml` — that forces a `uv.lock` regeneration, and `uv lock --locked` is a
  documented validation gate (`CLAUDE.md`). Verified working answer, which leaves
  `pyproject.toml` and `uv.lock` untouched:

  ```bash
  uv run --no-sync --with h5py python scripts/lerobot_conversion/convert_semihumanoid.py …
  ```

  This overlays `h5py` on the project environment (confirmed: h5py 3.16.0 + pandas 2.2.3
  + pyarrow 23.0.1 + `import gr00t` all available, `git status` on `uv.lock` clean).
  - Acceptance: `uv run --no-sync --with h5py python scripts/lerobot_conversion/convert_semihumanoid.py --help` prints usage; `git status --porcelain uv.lock pyproject.toml` is empty afterwards.
- [x] **1.2** Episode discovery: glob `<subset>/*/20*/*/*/*/episode.h5`, excluding any path containing `_failed_recordings`. Sort deterministically (path order) to fix `episode_index`.
  - Acceptance: counts match the profile table exactly per subset.
- [x] **1.3** Name-based field extraction: read `state/field_names` + `state/field_slices` and `action/field_names` + `action/field_slices`; build `{name: (start,end)}`; assert every required field is present with the expected width; **error** on unexpected width. (R2)
- [x] **1.4** `rot6d_internal_to_groot(v6)`: build R via the column convention (`_rot6d_to_matrix` semantics), return `R[:2,:].flatten()`. Add a unit test over 1,000 random rotations asserting exact round-trip through GR00T's `_rot6d_to_matrix`, plus `det(R)≈+1`. (R1)
- [x] **1.5** Assemble canonical state (32) and action (20) per the layout above, `float32`. Assert `state.shape == (n,32)`, `action.shape == (n,20)`.
- [x] **1.6** Action source: read `action/executed`; assert `action/residual` is all-zero and warn if not. No index shift (Finding 4).
- [x] **1.7** Video: for each canonical key, resolve the physical camera, then for `i in range(frame_count)` write `blob[offsets[j]:offsets[j+1]]` where `j = frame_ref_index[i]`, piped to `ffmpeg -f image2pipe -framerate 50 -c:v libx264 -preset veryfast -crf <crf> -pix_fmt yuv420p -g 50`. Assert bounds on `j`. (R3)
  - Acceptance: output frame count equals `frame_count` for every camera.
- [x] **1.8** Parquet per episode with the column set above; `timestamp = frame/elapsed_ms / 1000`.
- [x] **1.9** `meta/tasks.jsonl` — the 3 tasks with stable indices (`generic_pick`=0, `generic_place`=1, `bracket_handover`=2). **Read the task text from each episode's HDF5 root attrs (`task_uuid`, `task_text`), not from a `task_catalog.yaml` file** — every episode carries its own `task_uuid` / `task_text` / `task_catalog_version`, so the converter needs no external catalog and cannot drift from one. (The `ml_data` copy of `task_catalog.yaml` no longer exists; the surviving copy is `<industrialnext-ai-repo>/config/manipulation/semihumanoid/task_catalog.yaml`, useful only for cross-checking.)
  - Acceptance: assert every episode's `task_catalog_version` is `2026-08-16`, and that the set of distinct `(task_uuid, task_text)` pairs has exactly 3 members matching the catalog copy.
- [x] **1.10** `meta/episodes.jsonl` — `{"episode_index", "tasks": [task_text], "length"}` where `length` is the exact parquet row count.
- [x] **1.11** `meta/info.json`. **This file is load-bearing in two specific ways — get it wrong and failures are silent:**
  - `generate_stats` (`gr00t/data/stats.py:251`) reads `info.json["features"]` and computes statistics **only** for entries whose `dtype` string contains `"float"`. If `action` and `observation.state` are not declared with `dtype: "float32"`, no stats are produced for them and normalization has nothing to work with.
  - The loader asserts each video `original_key` is present in `features` (`lerobot_episode_loader.py:429`).

  Required keys: `data_path` and `chunks_size` are read unconditionally
  (`lerobot_episode_loader.py:198,201`); `features`, `video_path`, `fps` are read with
  defaults but all three are needed here. Write: `codebase_version: "v2.1"` (not
  validated by GR00T — present only for third-party LeRobot tooling), `fps: 50`,
  `chunks_size: 1000`, `robot_type: "semihumanoid_bimanual"`, `data_path`/`video_path`
  patterns matching the layout above, `total_episodes`/`total_frames`/`total_tasks`/`total_videos`,
  and a `features` block declaring `action` (shape `[20]`, `float32`),
  `observation.state` (shape `[32]`, `float32`), `timestamp` (`float32`), and the three
  `observation.images.*` video keys at 256×256.
  - Acceptance: `json.load` round-trips; `[f for f in features if "float" in features[f]["dtype"]]` contains `action` and `observation.state`; declared shapes equal the actual written array widths.
- [x] **1.12** `meta/modality.json` — exactly as specified above.
- [x] **1.13** Do **not** write `stats.json`/`relative_stats.json`; the factory generates them (`factory.py:59`).
- [x] **1.14** Resumability: skip an episode whose parquet and all three MP4s already exist unless `--overwrite`. Per-episode failures logged and collected, not fatal.
- [x] **1.15** Per-subset `_conversion_report.json`: episode count, frame count, dropped episodes with reasons, `frame_age_ms` p50/p99, camera map used, elapsed time.
- [x] **1.16** **QC gate — runs inside discovery (1.2), before any conversion work** (R7, R13, R14). This item is numbered last but executes early: 1.2 enumerates candidates, 1.16 filters them, and only survivors reach 1.3–1.12. For each candidate compute per canonical camera `coverage = image_count / frame_count` and `age_p99 = percentile(frame_age_ms, 99)`, then keep the episode only if **all** of:
  - every canonical camera has `0.45 <= coverage <= 0.80` — the upper bound guards against coverage >1.0 (more images than state frames), which indicates a truncated state stream and which a lower-bound-only test would admit (R13)
  - every canonical camera has `age_p99 <= 150` ms
  - `frame_count >= 41` — below this the 40-step horizon yields zero samples (R7)
  - the episode directory ends in `_expert` — guards against `*_policy` rollouts (R14); inert on current data, since all five subsets are 100% expert
  - Emit one line per drop into `_conversion_report.json` recording which criteria failed and the measured coverage/age per camera, so a drop is never silent.
  - Expected outcome over the 1,426 episodes: **exactly 1 dropped, 1,425 kept** — `matcha_v3 20260817_231631_expert`, failing both the coverage and staleness tests (`age_p99` = 2,535 ms). The other three criteria fire zero times.
  - Acceptance: kept counts equal matcha_v2 **262** / matcha_v3 **159** / ube_v1 **322** / ube_v2 **470** / ube_v3 **212**; total kept frames **520,822**. If the gate drops more than one episode, investigate before converting — the health distribution is tight and a second drop means something changed.
  - Raw measurements reproducible via `scripts/lerobot_conversion/camera_health_zdata.py`; the thresholds' insensitivity is documented in Scope & Data Selection.

### Phase 2 — Pilot (`matcha_v3`, 160 eps, 46-dim schema)
- [x] **2.1** Convert with `--limit 5`; hand-inspect one parquet (dims, dtypes, timestamp monotonic) and decode 3 frames from each MP4 via `get_frames_by_indices`.
- [x] **2.2** Convert the full subset; confirm **159 episodes / 65,981 rows** — matcha_v3 has 160 collected episodes, one of which the 1.16 QC gate drops (stale frames). If you see 160, the QC gate is not wired into discovery.
- [x] **2.3** Write `examples/semihumanoid/semihumanoid_config.py` (spec above).
- [x] **2.4** **Generate stats first, then** round-trip the loader. `LeRobotEpisodeLoader.__init__` asserts `meta/stats.json` exists (`lerobot_episode_loader.py:180`), so a bare loader call on a freshly converted dataset **will fail** — this ordering is mandatory, not optional:
  ```bash
  uv run python gr00t/data/stats.py --dataset-path <ds> \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/semihumanoid/semihumanoid_config.py
  ```
  Then assert the loaded column set is exactly
  `{language.annotation.human.task_description, state.{left_eef,left_gripper,left_ft,right_eef,right_gripper,right_ft}, action.{left_eef,left_gripper,right_eef,right_gripper}, video.{head,left_wrist,right_wrist}}` (13 columns).
  Note `gr00t/data/stats.py` itself orders this correctly — `generate_stats` (parquet-only, no modality config needed) runs before `generate_rel_stats`, which internally constructs a `LeRobotEpisodeLoader` and therefore depends on `stats.json` already existing.
- [x] **2.5** Independent rot6d check on real data: for 100 random frames, rebuild R from the written 9-dim EEF block using GR00T's reader and compare against the source pose reconstructed with the internal reader; assert angular error < 1e-5 rad.
- [x] **2.6** Visual spot-check: dump a 3-camera montage at frames 0/mid/last for 3 episodes and confirm the canonical keys hold the intended physical views.

### Phase 3 — Full conversion
- [x] **3.1** Convert `matcha_v2`, `ube_v1`, `ube_v2`, `ube_v3` (parallel across subsets; `matcha_v3` already done in Phase 2).
- [x] **3.2** Verify totals: **1,425 episodes / 520,822 rows** across the five subsets after the 1.16 QC gate, counting train + val together (`<subset>` + `<subset>_val` must reconcile to the post-gate count). Expected per-subset kept counts: matcha_v2 **262**, matcha_v3 **159**, ube_v1 **322**, ube_v2 **470**, ube_v3 **212**. After the 3.5 split that is roughly **1,351 train + 74 val**; assert the sums rather than hard-coding the halves.
- [x] **3.3** Write top-level `manifest.json`: source subset paths, camera map, per-subset counts (collected and post-QC), QC drops with reasons, converter git SHA, `task_catalog_version`, conversion timestamp.
- [x] **3.4** Record actual on-disk size; compare against the estimate: **~0.59 GB video** (520,822 frames × ~1.13 KB) + **~0.11 GB parquet** ≈ **0.70 GB total**, from 55 GB of source HDF5.
- [x] **3.5** **Hold out a validation split** (resolves Open Question 4; default = do it). Implemented in the converter as `--val-every 20` (default), not as a post-pass: each newly-seen episode is assigned by `assign_split()` and frozen in the ledger, and val episodes are written to a sibling `<subset>_val/` dataset that is a valid standalone LeRobot v2 tree with its own contiguous `episode_index`.
  - **Changed from the original plan**: the rule is a hash of the episode path, not `episode_index % 20`. A positional rule reassigns existing episodes when earlier ones are inserted, which would leak held-out data into training on the next append. See Growth Model.
  - Rationale for holding out at all: GR00T's sharded trainer asserts `eval_strategy == "no"` and `launch_finetune.py` exposes no eval flags, so a held-out set can only exist as a separate dataset. Without one, Phase 6 measures training-set fit only.
  - The ledger records each episode's split, so the assignment is auditable and stable.
  - Acceptance: `<subset>` + `<subset>_val` episode counts sum to the post-QC kept count per subset; no source appears in both; ~5% held out overall.
- [x] **3.6** **Verify structural invariants per dataset** (added during implementation; 3.2 checked totals but not internal consistency, and an append-only writer makes index arithmetic a live risk). For every dataset assert: parquet file count == `episodes.jsonl` length; the concatenated `index` column is contiguous from 0; each episode's `frame_index` runs 0..n-1; per-episode lengths match `episodes.jsonl`; `info.json.total_frames` equals the sum.
  - **Result: 10/10 datasets PASS.** This is the check that would catch a bad `index_offset` after an append — `matcha_v3` was deliberately converted in two passes (5 episodes, then the remaining 154) and its `index` column is still contiguous across the boundary.

### Phase 4 — Validation
- [x] **4.1** Per dataset: `uv run python gr00t/data/stats.py --dataset-path <ds> --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/semihumanoid/semihumanoid_config.py`
  - **Run the datasets in parallel, and always pre-generate before training.** Measured: the absolute pass (`stats.json`) is parquet-only and takes seconds, but the relative pass is ~**2.5 s/episode/key** — it constructs an `EndEffectorPose` per frame per 40-step chunk in Python (`gr00t/data/stats.py:345`), so it is CPU-bound, not I/O-bound. Sequentially over all 10 datasets that is roughly **50 minutes**; running one process per dataset (10 at once, trivial on 256 cores) brings it to ~**15 minutes**, bounded by the largest dataset.
  - This matters beyond convenience: `DatasetFactory` generates missing stats **on rank 0, sequentially, before the first training step** (`gr00t/data/dataset/factory.py:59`). Skipping 4.1 does not save the work, it just moves ~40 minutes of it into the head of the training run, with 3 GPUs idle. After every data addition, re-run this step in parallel before launching training.
  - `matcha_v3` already had stats generated in 2.4; re-running is safe and near-instant because both stats files are fingerprint-cached (`gr00t/data/stats.py`, `__fingerprints__` sidecar) and recompute only when the feature schema or action config changes.
- [x] **4.2** Confirm `relative_stats.json` contains **only** `left_eef` and `right_eef` (the two RELATIVE keys) and that each is shape (40, 9).
  - **Result: 10/10 datasets PASS** — exactly `[left_eef, right_eef]`, each `(40, 9)`. `left_gripper`/`right_gripper` correctly absent, since ABSOLUTE keys get no relative entry (`gr00t/data/stats.py:424`).
- [x] **4.3** **(R4)** Dump per-dim `q01/q99` and `min/max` for both EEF keys across all five datasets. Flag any dim whose q99−q01 < 1e-4. If flagged, switch that key to `mean_std_embedding_keys` or plan `--no-use-percentiles`.
  - **Result: PASS on all 10 datasets, no degenerate dims.** Worst per-(step, dim) `q99−q01` range is **1.573e-04** (matcha_v3_val), i.e. 1.6× above the threshold; the train datasets run looser (1.8e-04 to 1.03e-03) because they pool more samples. Percentile normalization is safe as configured; no switch to `mean_std_embedding_keys` or `--no-use-percentiles` needed.
  - **Pre-measured beforehand, and it held.** Relative EEF chunks were computed directly from the source HDF5 for 40 sampled episodes (1,922 chunks per arm) using GR00T's own `EndEffectorActionChunk.relative_chunking`: the smallest per-(step, dim) `q99−q01` range is **7.2e-4** — 7× above the threshold — with a median of 8.5e-2, and **0 of 360 (step, dim) cells** fall below 1e-4 for either arm. Treat a flag here as a signal that something upstream differs from the sampled data, not as a routine tuning step.
- [x] **4.4** 100-step smoke finetune, 1 GPU, `--global-batch-size 8`; require finite loss and no NaN. **Run it against the full multi-dataset `--dataset-path`, not a single dataset** — that is the only way this step exercises `merge_statistics` across datasets (`sharded_mixture_dataset.py:44`), which is where a per-dataset stats inconsistency would surface. Capture peak VRAM while it runs; at 3 cameras this doubles as the memory evidence 5.6 needs.
  - **Result: PASS.** exit 0, `train_loss` 1.0907 falling 1.094 → 1.048 over 100 steps, `grad_norm` 0.30–0.77, zero NaN/OOM mentions. All five datasets loaded (80+56+109+122+66 = 433 shards, 81,780 total sampleable steps) and `Overriding statistics for embodiment 'new_embodiment'` confirms the cross-dataset `merge_statistics` path ran.
  - **Measured, and it changes the Phase 5 estimate:** steady state **2.89 it/s = 0.346 s/step** at batch 8 / 1 GPU / **3 cameras**, with peak VRAM **39,024 MiB of 49,140 (79%)**. The 2-camera reference was 3.04 it/s at 39,028 MiB — so the third camera costs only **~5% step time and no measurable extra VRAM**, not the +35–100% assumed in 5.3. Note `nvidia-smi` reports caching-allocator *reserved* memory, which plateaus, so this bounds the risk rather than measuring the exact delta.
- [x] **4.5** Confirm `validate_action_horizons` passes (40 ≤ 40) — it runs at processor construction, so 4.4 passing covers this.

### Phase 5 — Finetune
- [ ] **5.1** Launch (paths joined by `:`; `os.pathsep` per `launch_finetune.py:59`):
  ```bash
  OUT=data/training_data/gr00t/semihumanoid_260818
  uv run torchrun --nproc_per_node=4 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "$OUT/matcha_v2:$OUT/matcha_v3:$OUT/ube_v1:$OUT/ube_v2:$OUT/ube_v3" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/semihumanoid/semihumanoid_config.py \
    --num-gpus 4 --output-dir outputs/gr00t/gr00t_semihumanoid_260818 \
    --max-steps 30000 --save-steps 2500 --save-total-limit 5 \
    --global-batch-size 32 --dataloader-num-workers 4 \
    --learning-rate 1e-4 --warmup-ratio 0.05 --weight-decay 1e-5 \
    --state-dropout-prob 0.2 \
    --color-jitter-params brightness 0.15 contrast 0.15 saturation 0.2 hue 0.1 \
    --use-wandb --wandb-project gr00t-semihumanoid
  ```
  The five paths are listed explicitly so the `<subset>_val/` datasets from 3.5 are excluded — do **not** replace this with a glob over `$OUT/*`, which would train on the validation split.
  Color-jitter values mirror the internal config's augmentation block.
  `--state-dropout-prob 0.2` (not the 0.8 model default) because FT and gripper state
  carry real signal for contact-rich picking.
- [ ] **5.2** Budget disk, then verify empirically. Measured twice in this repo: a short smoke run produces a **36 GB output directory** holding *both* the end-of-training model save *and* one `checkpoint-N` — so the per-checkpoint figure is **~18 GB**, not 36 GB. Checkpoints include optimizer state because `save_only_model` defaults to `False`. Plan for ~18 GB × (`save-total-limit 5` + 1 final save) ≈ **110 GB**, then confirm against the first real checkpoint with `du -sh`. Pass `--save-only-model` to roughly halve this if disk gets tight (it forfeits resumability).
- [ ] **5.3** After ~500 steps, record steady-state step time and confirm the budget against the projection below.
  - **Measured baseline** (30-step run on the shipped SO100 demo, 1×RTX 4090, `--global-batch-size 8`, **2** cameras): steady state **3.04 it/s = 0.329 s/step** (clean asymptote over steps 11–30; the first step costs ~30 s of warmup), peak VRAM **39.0 GB of 47.4 GB usable (79%)**.
  - **Measured directly at the real config (5.6), superseding two earlier estimates.** 200 steps on 4 GPUs at `--global-batch-size 32` ran at **1.63 it/s = 0.613 s/step** (`train_runtime` 150.3 s including warmup and the final save). So **30,000 steps ≈ 5.1 hours**.
  - Two earlier projections were wrong and are recorded here so the reasoning is not repeated: 4–8 h (assumed the 3rd camera cost +35–100%; it costs ~5%) and 3.3–4.0 h (assumed DDP overhead of +15–40%; it is **+77%**).
  - **DDP scaling is the real cost: ~56% efficiency.** Per-sample throughput is 52.2 samples/s on 4 GPUs versus 23.1 samples/s on 1 GPU — 2.3× for 4× the hardware. This is the PCIe all-reduce penalty R15 predicted (no NVLink on 4090s), and it is why 5.7 below is worth doing.
  - Useful denominator: the post-QC corpus yields **465,247 usable sampling start indices** across 1,425 episodes (520,822 frames minus a 39-frame tail per episode; 10.7% of frames cannot start a 40-step chunk). Subtract ~5% for the 3.5 validation split → **~442k trainable start indices**. At `--global-batch-size 32`, 30,000 steps ≈ 960k samples ≈ **2.2 passes** over the trainable set, which is a reasonable starting budget for a 2.89 h corpus.
  - `--episode-sampling-rate` (default 0.1) does **not** discard data: it splits each episode's shuffled timesteps into `1/rate = 10` interleaved subsets to diversify shards (`sharded_single_step_dataset.py:176-190`). All timesteps remain reachable. Leave it at the default.
- [ ] **5.4** Optional: `--ds-weights-alpha` if length-proportional mixing skews toward ube_v2 (470 eps) more than desired.
- [ ] **5.5** **Make the run survivable — Phase 5 is the only multi-hour step.** Do not pass `--save-only-model`: it strips optimizer/scheduler/RNG state, and `check_resume_compatibility` (`gr00t/configs/training/training_config.py:165`) then *refuses* to resume rather than silently re-initialising the optimizer. Keep the default (`False`).
  - To restart after an interruption, re-launch the identical command plus `--resume-from-checkpoint`; it picks up the latest `checkpoint-*` in `--output-dir`. Note the flag defaults to `False` **by design** so that a plain re-run starts fresh instead of silently merging with a previous experiment — meaning a bare re-launch after a crash would **discard all progress**. Always add the flag explicitly when resuming.
  - `--save-steps 2500` bounds the worst-case loss to 2,500 steps of work. If measured step time (5.3) puts 2,500 steps above ~2 h, lower it.
  - Acceptance: after the first checkpoint appears, verify it contains optimizer state (`optimizer.pt` / `scheduler.pt` present alongside the weights), i.e. that it is actually resumable.
- [x] **5.6** **Guard against OOM before committing to the long run** (R15). Launch with the real command but `--max-steps 30`, and watch peak VRAM (`nvidia-smi --query-gpu=memory.used --format=csv -l 5`). Our config puts **3 cameras × 8 samples per GPU** through the vision tower — 50% more image tokens per step than the 2-camera SO100 configuration that the shipped examples use.
  - **Partly de-risked, still required.** 4.4 ran the exact 3-camera config at the exact per-GPU batch (8) and peaked at **39,024 MiB of 49,140**, so single-process activation memory is known to fit. What 4.4 does *not* cover is DDP: 4-way training adds gradient buckets and reduction buffers (~3.2 GB of bf16 gradients plus workspace) on top of that per GPU, which could push toward 43–45 GB. Run this probe anyway — it is the only thing that measures that delta.
  - If it OOMs or peaks above ~90% of 47 GB: first drop `--global-batch-size` to 16 and set `--gradient-accumulation-steps 2` (same effective batch, half the activation memory); only then consider dropping a camera.
  - Acceptance: 30 steps complete with peak memory leaving headroom, and the loss is finite. Do not start the 30,000-step run until this passes.
  - **Result: PASS, with a large margin nobody expected.** A 200-step run at the real config (4×RTX 4090 via `torchrun`, `--global-batch-size 32` → `per_device_train_batch_size=8`, `world_size=4`, 3 cameras, real corpus) peaked at **17.9–18.1 GB per GPU — only 36–37% of 49,140 MiB**, leaving ~29 GB free per card. exit 0, zero NaN/OOM/NCCL/timeout messages.
  - **This contradicts the 1-GPU measurement and the 1-GPU number is the unreliable one.** The single-process runs reported ~39.0 GB at the *same* per-device batch of 8 — and reported it as 39,028 MiB with 2 cameras and 39,024 MiB with 3, i.e. essentially identical despite 50% more image tokens. A figure that does not move with the workload is not measuring the workload: that was the caching allocator holding a large reserved pool, which `nvidia-smi` reports verbatim. Treat the 4-GPU 18 GB as the real steady-state demand and the 39 GB as an allocator artifact.
  - **Consequence: the batch size is far too small for the hardware.** At 37% utilisation there is room to roughly double `--global-batch-size` (to 64, i.e. 16/GPU), which should also improve the poor DDP scaling noted in 5.3 by raising the compute-to-communication ratio. Worth one probe before the long run; see the new item 5.7.

### Phase 6 — Evaluation & deployment notes
- [x] **5.7** **Probe a larger batch before committing 5 hours** (added after 5.6 measured only 37% VRAM utilisation). Run 200 steps at `--global-batch-size 64` (16/GPU) and compare samples/s and peak VRAM against the batch-32 baseline (52.2 samples/s, 18 GB/GPU).
  - Rationale: at 37% memory use the GPUs are underfed, and larger per-device batches amortise the fixed PCIe all-reduce cost over more compute — the direct lever on the 56% scaling efficiency. If samples/s improves materially, prefer batch 64 and scale `--max-steps` down proportionally to keep the same number of samples seen.
  - Keep an eye on the effective learning rate: `--global-batch-size` changes the samples per optimizer step, so a doubled batch usually wants a modestly higher `--learning-rate` (or the same LR with fewer steps).
  - Acceptance: no OOM, finite loss, and a recorded samples/s to compare. If it does not help, stay at 32 and accept ~5.1 h.
  - **Result: it helps a lot.** 100-step probes at the real config (4 GPUs, 3 cameras, real corpus, corrected preprocessing), all rc=0 with zero OOM/NaN:

    | global batch | per-GPU | it/s | samples/s | peak VRAM | vs batch 32 |
    |---|---|---|---|---|---|
    | 32 | 8 | 1.630 | 52.2 | 17.6 GB (37%) | 1.00x |
    | 64 | 16 | 1.030 | 65.9 | 19.1 GB (40%) | 1.26x |
    | **128** | **32** | **0.787** | **100.8** | **22.7 GB (47%)** | **1.93x** |
    | 256 | 64 | 0.526 | 134.7 | 33.0 GB (69%) | 2.58x |

  - Marginal gains taper (1.26x, 1.53x, 1.34x) while memory climbs steeply. **Batch 128 is the chosen operating point**: near-double throughput, 47% VRAM, and it preserves 7,500 optimizer steps for the 960k-sample budget. Batch 512 was not probed -- at 69% already, an OOM hours into a run costs far more than the ~20 min it might save.
  - Time for the 960k-sample budget: batch 32 -> 5.11 h, 64 -> 4.05 h, **128 -> 2.65 h**, 256 -> 1.98 h.
  - **Learning rate is coupled to this choice.** Batch 128 is 4x the repo's validated batch-32 recipe, so the same sample budget means a quarter as many optimizer steps. Default is sqrt-scaled `--learning-rate 2e-4`; linear scaling (4e-4) is more aggressive than warranted for a finetune.

- [ ] **6.1** Open-loop per subset:
  ```bash
  uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path $OUT/ube_v2 --embodiment-tag NEW_EMBODIMENT \
    --model-path <ckpt> --traj-ids 0 1 2 --execution-horizon 40 --steps 400 \
    --modality-keys left_eef left_gripper right_eef right_gripper \
    --save-plot-path outputs/gr00t/gr00t_semihumanoid_260818/olp
  ```
  Note there is deliberately **no `--modality-config-path`** here: `open_loop_eval.py`
  has no such flag and instead reads the modality config back out of the checkpoint's
  saved processor (`open_loop_eval.py:323`, `policy.get_modality_config()`), which it
  then hands to `LeRobotEpisodeLoader`. So the checkpoint is self-describing for eval.
  Model serving also reads the modality config from the saved processor. Although
  `run_gr00t_server.py` exposes `--modality-config-path`, that option applies only to its
  replay-policy branch and does not override a model checkpoint. See
  [`2026_0819_semihumanoid_gr00t_model_serving.md`](2026_0819_semihumanoid_gr00t_model_serving.md).
  `--execution-horizon 40` is the maximum permitted value (the contract requires
  `1 <= n_action_steps <= action_horizon`, and our chunk length is 40); use a smaller
  value to re-plan more often.
- [ ] **6.2** Record MSE/MAE per checkpoint (2500…30000) and confirm a monotone decrease, per the repo's own guidance.
- [x] **6.3** Document the **deployment inverse transform**: GR00T emits rot6d in
  row-major convention; before sending to the Flexiv controller, rebuild R via
  GR00T's `_rot6d_to_matrix` and re-encode with the internal column convention
  (`concat([R[:,0], R[:,1]])`). Omitting this inverts the commanded rotation. This is
  covered by the model-serving guide linked above. (R1)
- [x] **6.4** Note that GR00T consumes only the current frame (no 20-step state
  history) and predicts 40 steps at 50 Hz — matching the internal `n_action_steps`,
  so the existing action-execution logic transfers unchanged. The serving guide records
  the resulting observation contract and receding-horizon execution flow.
- [ ] **6.5** **Evaluate on the held-out `<subset>_val/` datasets** created in 3.5 and
  report train-vs-held-out MSE/MAE side by side. This is the only number in the plan
  that speaks to generalization; 6.1–6.2 alone measure training-set fit. A large gap
  (held-out MSE ≫ train MSE) means data scarcity or overfitting rather than a pipeline
  fault. Levers in order, if the gap is large: (a) train longer, or unfreeze the vision
  tower with `--tune-visual`; (b) collect more data for the weakest task —
  `bracket_handover` has only 32 episodes, far below the ~100 the repo FAQ suggests even
  for a simple task; (c) restore `flexiv_matcha_v1` from cloud backup (Open Question 5),
  which is the largest single data increase available but the most operational work.
  - Note `open_loop_eval.py` needs no `--modality-config-path` for the val datasets
    either; it takes the config from the checkpoint. But the val datasets **do** need
    their own `meta/stats.json`, so run 4.1's stats command against each `<subset>_val/`
    before evaluating (R12).

---

## Validation Commands

```bash
# repo hygiene
pre-commit run --all-files
python -m pytest tests/ -m "not gpu" -q --timeout=300

# converter unit tests. The rot6d convention test imports gr00t.data.state_action.pose
# as the reference implementation and needs no h5py; the field-selection test reads a
# real episode.h5 and does, hence the --with overlay.
uv run --no-sync --with h5py python -m pytest \
  scripts/lerobot_conversion/test_convert_semihumanoid.py -q

# per-dataset stats (must precede any hand-written LeRobotEpisodeLoader call)
uv run python gr00t/data/stats.py --dataset-path <ds> \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/semihumanoid/semihumanoid_config.py

# smoke finetune
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B --dataset-path <ds> \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/semihumanoid/semihumanoid_config.py \
  --num-gpus 1 --output-dir outputs/gr00t/semihumanoid_smoke \
  --max-steps 100 --global-batch-size 8
```

---

## Open Questions

1. **~~Camera selection~~ — CLOSED.** All five subsets supply the same three physical
   viewpoints under one uniform camera map; no canonical key blends two mounts and there is
   no viewpoint trade-off to weigh.
2. **~~Are `left_ft`/`right_ft` in the same units across matcha and ube?~~ — measured,
   partially resolved.** Per-episode `|ft|max` over the in-scope subsets:

   | Robot | Episodes | median | p90 | p99 | max |
   |---|---|---|---|---|---|
   | `flexiv_matcha` | 422 | 0.846 | 8.040 | 15.477 | 27.315 |
   | `flexiv_ube` | 1004 | 2.769 | 7.780 | 9.975 | 19.140 |

   The **upper tails agree closely** (p90 8.04 vs 7.78; same order of magnitude at p99
   and max), which is strong evidence of a shared unit system — a unit mismatch would
   show a 10×/1000× gap, not this. The 3.27× difference in *medians* is therefore best
   read as a task-contact difference (ube makes sustained contact more often), not a
   scaling error. **Action:** no per-robot rescaling; treat FT as one shared feature.
   Still worth one confirmation from whoever owns the FT pipeline that both robots
   publish `/…/external_wrench_in_tcp` in N/Nm in the TCP frame — the episode summaries
   name the same topics, which is consistent with this conclusion.
3. **Should `bracket_handover` (32 episodes) be included?** It is the only genuinely
   bimanual task and is 2.2% of episodes. Included by default; worth a held-out check
   since 32 episodes is far below the ~100 the repo FAQ suggests for a simple task.
4. **~~Held-out split~~ — resolved with a default; override if you disagree.** The
   internal pipeline uses `val_ratio: 0.05` / `val_split_mode: mixed`, but GR00T's
   sharded trainer asserts `eval_strategy == "no"` and `launch_finetune.py` exposes no
   eval flags, so a held-out set can only exist as a *separate* converted dataset
   excluded from `--dataset-path`. **Default adopted: hold out 5% per subset** into
   `<subset>_val/` (checklist 3.5, evaluated in 6.5), which mirrors the internal
   `val_ratio` and costs ~71 episodes. Say so if you would rather train on 100% and
   skip generalization measurement.
5. **If more data is needed, is restoring `flexiv_matcha_v1` worth it?** It was removed
   from this machine on 2026-08-18 and exists only in cloud backup. Measured before removal:
   402 episodes / 201,405 frames, of which **338 pass the QC gate** — a ~24% increase in
   episodes over the current corpus. Two caveats make this more than a re-download:
   - Its wrist cameras were recorded with **wrong labels** (`eoat_left_top_*` /
     `eoat_right_top_*` for what were physically the *bottom* pair). A local in-place rename
     fixed this on 2026-08-18, *after* the cloud copy was made — so a restored copy will
     almost certainly carry the old `_top` names again and need the rename re-applied:
     `eoat_left_top_{rgb,depth}` → `eoat_left_bottom_{rgb,depth}` and likewise for `right`,
     on the HDF5 `images/` group links and the `cameras` keys in `episode.summary.json`.
     Verify which naming the backup holds before assuming either.
   - 64 of its 402 episodes fail QC (57 from a single-day camera fault on 2026/08/12 with
     staleness to 9.2 s, plus 5 truncated-state and 6 `*_policy` episodes), so the gate must
     run on it exactly as on everything else.

   Not required for the first run. Revisit only if 6.5 shows a large train/held-out gap.
