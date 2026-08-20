# Semihumanoid GR00T RTC, Fresh Finetune, and Real-Robot Implementation Plan

**Date:** 2026-08-19

**Status:** Reviewed; GPU-machine work complete, deployment-machine validation pending

**Primary repository:** this Isaac-GR00T checkout on the GPU machine

**Machine ownership:** the GPU machine owns code, corpus freezing, training, offline
evaluation, checkpoint-backed loopback validation, and the deployment handoff bundle. The
deployment machine owns real-ROS shadow validation and every step that can publish robot
commands.

**Related serving plan:**
[`2026_0819_semihumanoid_gr00t_real_robot_serving_implementation_plan.md`](2026_0819_semihumanoid_gr00t_real_robot_serving_implementation_plan.md)

## Outcome

Implement the complete action-prefix path needed by the Industrial Next async GR00T server,
train a fresh semihumanoid checkpoint with training-time action-prefix conditioning, compare
the available RTC modes offline and in a no-motion shadow run, and only then perform a
separately authorized, supervised robot rollout.

The fresh run is target-specific to Flexiv Ube. It uses only explicitly admitted
`flexiv_ube_v1` through `flexiv_ube_v4` releases. Matcha is not mixed into this run: its
producer contract owns only the moving right arm/gripper, while the current GR00T loader has
no per-source action-coordinate mask. Adding that generic masking surface is not required for
the Ube deployment and would put an unrelated data-model change on the critical path.

The production candidate will keep the existing 50 Hz ROS policy client and WebSocket
protocol. GR00T inference remains asynchronous and produces a 40-step, 0.8-second action
chunk. The server owns chunk alignment, delay prediction, action-prefix construction, and
RTC diagnostics. The ROS side continues to own command publication, robot-state trust,
manual stop, and the last-resort per-step action clamp.

The RPC `progress` field remains `0.0`. That protocol value is independent of this document's
implementation status. Robot trials are manually stopped; this work does not add an automatic
completion signal.

## Decisions

### 1. Implement two RTC modes, but run only one at a time

The repository and the research literature contain two different techniques:

| Mode | Checkpoint requirement | Prefix behavior | Intended use |
|---|---|---|---|
| `off` | Any compatible checkpoint | No prior action is supplied | Regression baseline and first-chunk generation |
| `native` | Any compatible GR00T N1.7 checkpoint | Use the checked-in GR00T prefix seeding, frozen prefix, and exponential velocity ramp | Immediate compatibility with the current checkpoint and an A/B baseline |
| `trained_prefix` | Checkpoint explicitly trained with action-prefix conditioning | Clamp a hard prefix and use tokenwise flow timesteps learned during finetuning | Fresh-checkpoint candidate that must earn selection against `native` and `off` |

`native` and `trained_prefix` are mutually exclusive. They must never be stacked. Native
GR00T RTC changes inference only and does **not** require retraining. Training-time action
conditioning is a separate method: it supplies a clean ground-truth prefix during training,
uses tokenwise flow timesteps, and excludes the prefix from the loss. The fresh finetune is
therefore required for `trained_prefix`, not for `native`.

The full gradient-guided PiGDM/VJP algorithm from the original RTC paper is not part of
this implementation. GR00T's current native primitive is a cheaper, repository-specific
prefix initialization and velocity-scaling path. Keeping those algorithms distinct avoids
claiming equivalence that the code does not provide.

### 2. Keep one physical action timeline as the server source of truth

The server will retain:

- recently served absolute action rows, keyed by target control timestep;
- the unserved future action timeline, keyed the same way; and
- the observation snapshot and source timestep for each inference.

When an inference is actually launched, the server assembles the contiguous prior actions
that overlap the new chunk. It does this at launch time, after the preceding result has been
admitted, rather than when a pending observation was first captured. This ordering is
required because a pending observation may have been recorded before the newest prior chunk
existed.

The first inference after session registration has no prefix and always uses `off`
generation. A later inference uses RTC only when a contiguous, generation-matched prefix is
available. A missing or discontinuous prefix does not silently fall back to an independently
sampled replacement chunk; the server retains the current timeline and eventually returns
null actions/holds if it cannot safely refresh it.

### 3. Normalize prefixes against the new observation and new horizon indices

The semihumanoid EEF actions are stored and served as absolute poses, but the checkpoint
uses state-relative EEF actions and horizon-dependent `relative_action` statistics. A prior
absolute trajectory cannot be copied directly into the model tensor.

For a new observation at source timestep $s$, the policy will:

1. convert each prior wire action from Industrial Next column-form rot6d into GR00T row-form
   rot6d;
2. re-anchor the absolute EEF poses relative to the state in the new observation;
3. normalize the prefix using statistics rows `0..prefix_length-1` of the **new** chunk;
4. place those already-normalized values in the tensor layout expected by the selected RTC
   sampler; and
5. decode the generated output against the same new observation state.

This is a hard correctness boundary. Normalizing the tail with the old chunk's horizon rows
and then decoding it as the new prefix changes translation and orientation even when the
physical action was intended to be identical.

### 4. Predict inference delay conservatively and reject underestimates

At 50 Hz, one action step is 20 ms. The source-timestep action has already been returned
when inference is launched, so even a sub-period inference has one committed prefix row.
For a result completed while the latest accepted step is $t_i$ and its source is $s_i$, the
observed committed-prefix length is:

$$
D_i = \max(1,\ t_i - s_i + 1)
$$

The server predicts the next frozen length from a bounded rolling window:

$$
d_{pred} = \min(d_{max},\ \max(D_i) + d_{margin})
$$

Wall-clock latency still records image decode, preprocessing, model generation, and action
decode, but it is diagnostic rather than the alignment source of truth. The bootstrap value
and safety margin are explicit server arguments. The training maximum prefix length is
selected only after repeating the steady-state latency measurement; the recent preliminary
run suggested roughly 4 committed steps at the median and 10 at the long tail, so 12 steps
is a provisional smoke-test value, not a robot-ready constant.

For `trained_prefix`, `d_pred` must not exceed the maximum prefix seen during training. For
both RTC modes, the server rejects a completed chunk if actual elapsed control steps exceed
the frozen prediction. It keeps serving the previous timeline if rows remain, updates the
delay estimator, and reports the rejection. This makes an underestimated latency a visible
hold/retry condition instead of a discontinuous chunk swap.

### 5. Do not hide unsafe model behavior with server-side clipping

RTC addresses cross-chunk continuity; it does not guarantee reasonable motion inside a
chunk. The server will measure and gate position, orientation, gripper, velocity, and
acceleration-like finite differences, but it will not mutate model output with a second
clipping or smoothing implementation.

The first commanded robot trials will enable the existing ROS
`action_delta_clamp_enabled` envelope. Clamp activation is evidence of a bad rollout, not a
normal operating mode: repeated activation is a no-go and triggers offline review. This
preserves the ownership boundary while retaining an independent last line of defense.

### 6. Train only on a content-bound, continuity-safe Ube corpus

The existing converted working root is mutable and was generated with
`continuity.split_on_gap_ms: null`. Its broad `flexiv_*` discovery also includes Matcha,
whose producer contract does not own the left-arm targets, and it includes a previously
identified stale-camera Matcha episode. It is valid historical baseline evidence, but it is
not the input to the fresh RTC run.

The fresh run uses a new output root and an explicit source list containing only
`flexiv_ube_v1`, `flexiv_ube_v2`, `flexiv_ube_v3`, and `flexiv_ube_v4`. Conversion splits
every source episode at observed control gaps greater than 40 ms and drops resulting segments
shorter than the 40-step horizon. Before conversion, the release record must bind the
producer-confirmed semantics that Ube rows pair observed state with authoritative absolute
commands for both arms/grippers and that the head camera is fixed for every included release.
If those semantics are not confirmed for a release, that release is excluded rather than
inferred from numerical equality or directory naming.

The corpus manifest binds every admitted source path and stat guard, every conversion ledger,
and SHA-256 inventories of the converted parquet, video, metadata, and statistics files. The
training launcher verifies this manifest immediately before starting. No sync or stats process
may mutate the frozen output after that verification.

### 7. Use one explicit training-delay distribution

`rtc_training_max_prefix_steps=M` means the checkpoint was trained on every integer prefix
length in `0..M`, sampled uniformly and independently per batch item. The upper bound is
inclusive so the advertised maximum actually receives supervision. Zero remains in the
distribution to preserve ordinary chunk generation. The model reports sampled prefix lengths
and postfix valid-element counts so the training smoke and long run can verify coverage.

### 8. Make the deployment handoff an artifact, not an informal command list

The GPU-machine deliverable is a versioned handoff directory containing repository revision,
checkpoint and processor hashes, corpus/run manifests, selected RTC settings, benchmark and
held-out results, server command, expected metadata, and deployment-machine validation steps.
The deployment operator verifies the hashes before no-motion shadow testing. No ROS config edit,
service restart, shadow connection, or commanded rollout is performed on the GPU machine.

## Verified current baseline

- The current baseline checkpoint is
  `outputs/gr00t/semihumanoid_20260819_080107`.
- Its model horizon is 40 actions, inference uses 4 denoising steps, the saved
  `rtc_ramp_rate` is 6.0, and the semihumanoid control rate is 50 Hz.
- Its saved action contract is relative `xyz+rot6d` for left/right EEF and absolute for both
  grippers.
- `Gr00tN1d7ActionHead.get_action_with_features()` already accepts a previous normalized
  action tensor and the four native RTC options, but only when `action_input["action"]` is
  present.
- `Gr00tPolicy._get_action()` currently calls `self.model.get_action(**collated_inputs)` and
  labels `options` unused, so neither the action tensor nor RTC options reach the model.
- `IndustrialNextAsyncServer._run_inference()` currently calls the policy without options,
  and `_admit_inference_result()` replaces the future timeline wholesale.
- The current training forward pass uses one flow timestep per batch item and learns to
  denoise the whole action chunk. It does not train an action prefix.
- `MultiEmbodimentActionEncoder` internally has a `(B,T)` sinusoidal timestep encoder but
  currently rejects `(B,T)` input. `DiT`/`AlternateVLDiT` apply one batch timestep through
  AdaNorm and the output projection, so training-time conditioning requires tokenwise
  support in both places.
- The current data configuration discovers all `flexiv_*` semihumanoid source subsets,
  converts 50 Hz actions with horizon 40, and trains from the base
  `nvidia/GR00T-N1.7-3B` checkpoint. The current converted working root contains seven
  source ledgers and 1,858 complete source records across Matcha v2-v4 and Ube v1-v4.
- That converted root is not fresh-run admissible: it uses no control-gap segmentation,
  includes Matcha without coordinate ownership masks, and includes the known stale-camera
  `20260817_231631_expert` Matcha episode. It remains read-only baseline evidence.
- At review time all four 49,140 MiB RTX 4090 GPUs are idle and the training volume has
  sufficient free space for a separate frozen Ube conversion and fresh checkpoints.
- The pinned `packages/industrialnext_rpc` submodule is not initialized in this checkout,
  so `uv` commands fail until Phase 0 restores that exact dependency revision.
- The ROS async client already has `inference_mode`, robot-state trust, manual stop, and an
  optional per-step action delta clamp. The Flexiv Ube deployment currently has the clamp
  configured but disabled.
- The deployment machine's ROS and Industrial Next worktrees are outside this GPU-machine
  implementation scope. Their status must be inspected again on the deployment machine;
  no handoff instruction may assume they are clean or use broad staging/cleanup commands.

## Current extension points

- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py:182-282` — current whole-chunk flow-matching
  training objective and action loss.
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py:325-435` — checked-in native prefix seeding,
  frozen region, exponential velocity ramp, and denoising loop.
- `gr00t/model/modules/embodiment_conditioned_mlp.py:181-220` — action encoder that
  currently expands only a `(B,)` timestep despite its internal tokenwise encoder.
- `gr00t/model/modules/dit.py:77-101,292-344` — AdaNorm and DiT output conditioning that
  currently broadcast one embedding over the complete state/action sequence. The same
  ownership repeats in `AlternateVLDiT.forward()`.
- `gr00t/policy/gr00t_policy.py:380-432` — observation processing, unused `options`, model
  call, and state-relative action decode.
- `gr00t/data/state_action/state_action_processor.py:373-419,421-526` — authoritative
  absolute/relative action normalization and inverse conversion; extend its public action
  normalization with an explicit no-clipping path for previously generated prefixes rather
  than duplicating pose math or mutating processor-global state.
- `gr00t/policy/industrialnext/adapter.py:297-339` — current one-way model-action-to-wire
  chunk conversion.
- `gr00t/policy/industrialnext/async_server.py:451-543` — inference launch, model call,
  whole-timeline replacement, and pending-snapshot launch ordering.
- `gr00t/eval/run_gr00t_industrialnext_server.py:35-151` — current serving CLI,
  configuration validation, policy construction, and startup metadata.
- `scripts/lerobot_conversion/zdata_pipeline/check.py:435-607` — derived train step count,
  finetune command construction, resource checks, and launch.
- `scripts/lerobot_conversion/zdata_pipeline/config.py:107-126,479-528` — training schema
  and YAML parsing.
- `gr00t/experiment/trainer.py:254-327` — training-output diagnostics; add reduced prefix
  histogram and postfix-valid-element logging here.
- Deployment repository `policy_control_node_async.py` — existing action-clamp
  configuration and publication-time application.
- Deployment repository `robots/flexiv_ube/task_config.yaml` — current 50 Hz async policy
  deployment and clamp setting; resolve its current line numbers and worktree diff during
  handoff rather than carrying stale GPU-machine assumptions.

## Scope

### Isaac-GR00T files

| Path | Planned change |
|---|---|
| `gr00t/configs/model/gr00t_n1d7.py` | Persist and validate training-time prefix support and its maximum trained delay |
| `gr00t/configs/finetune_config.py` | Add the opt-in finetune argument for training-time prefix conditioning |
| `gr00t/experiment/launch_finetune.py` | Propagate the finetune setting into the saved model config |
| `gr00t/model/modules/embodiment_conditioned_mlp.py` | Accept scalar or tokenwise flow timesteps without changing the scalar path |
| `gr00t/model/modules/dit.py` | Extend AdaNorm and output conditioning to tokenwise timesteps while preserving scalar behavior |
| `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | Implement training-time prefix loss and the explicit `native`/`trained_prefix` sampling paths |
| `gr00t/data/state_action/state_action_processor.py` | Add an explicit no-clipping action-normalization option for exact generated-prefix re-encoding |
| `gr00t/policy/gr00t_policy.py` | Validate RTC options, preprocess absolute prefixes correctly, and pass action/options to the model |
| `gr00t/policy/industrialnext/adapter.py` | Add the inverse wire-row-to-model-action mapping and physical prefix validation |
| `gr00t/policy/industrialnext/async_server.py` | Add served history, delay estimator, launch-time prefix assembly, RTC result admission, and metrics |
| `gr00t/eval/run_gr00t_industrialnext_server.py` | Add explicit RTC CLI/config/provenance and startup compatibility gates |
| `gr00t/eval/benchmark_industrialnext_rtc.py` | Add repeatable checkpoint latency and offline chunk-continuity evaluation |
| `scripts/lerobot_conversion/zdata_pipeline/config.py` | Add semihumanoid training-time prefix configuration |
| `scripts/lerobot_conversion/zdata_pipeline/check.py` | Propagate the setting and atomically create/verify content-bound corpus and training manifests before launch |
| `scripts/lerobot_conversion/run_zdata_pipeline.py` | Add bounded smoke-launch controls without requiring temporary YAML edits |
| `scripts/lerobot_conversion/zdata_pipeline/source.py` | Make required training-admission metadata fail closed while retaining continuity segmentation |
| `gr00t/experiment/trainer.py` | Reduce and log sampled-prefix coverage and postfix valid-element counts |
| `configs/embodiments/semihumanoid.yaml` | Pin the admitted Ube sources, new frozen output root, 40 ms continuity split, and measured training prefix bound |
| `getting_started/real_world_deployment.md` | Replace the "unwired" status after the implementation and document supported modes |
| Focused tests under `tests/gr00t/model/`, `tests/gr00t/policy/`, and `scripts/lerobot_conversion/test_zdata_pipeline.py` | Cover scalar compatibility, prefix math, normalization, timing, protocol behavior, source pinning, manifests, and launch refusal |

### ROS2 files

No ROS Python or RPC protocol change is required. Before the first commanded trial, make
one reviewed deployment-only edit to:

the deployment repository's `robots/flexiv_ube/task_config.yaml`

to enable the already-existing action delta clamp for the trial. That file is currently
modified by the user; the RTC-specific hunk must be reconciled and staged independently.
Shadow mode may instead use a runtime parameter override if that is the established operator
workflow at trial time.

### Explicit non-goals

- Replacing `industrialnext_rpc`, changing the direct WebSocket schema, or adding another
  ROS policy client.
- Implementing gradient-guided PiGDM RTC or a second temporal ensemble.
- Combining `native` and `trained_prefix` conditioning.
- Training a longer than 40-step model in this iteration.
- Adding depth, joint-state inputs, a progress head, automatic rollout completion, or
  supervisor/autostart integration.
- Treating action clamps as a model-quality fix.
- Moving or restarting the robot during software implementation, training, or no-motion
  validation.

## Runtime contract

### Prefix and timeline definitions

For a new inference snapshot with source timestep $s$ and horizon $H=40$:

- output row $i$ targets control timestep $s+i$;
- `served_history[t]` is the exact action row returned by the server for target $t$;
- `timeline[t]` is an unserved action row for target $t$;
- the prior prefix is the longest contiguous sequence beginning at $s$ that can be assembled
  from served history plus the current timeline;
- `frozen_steps` is the conservative predicted inference delay;
- `overlap_steps` is the native soft-overlap length and satisfies
  `frozen_steps <= overlap_steps <= H - min_new_tail_steps`; and
- `trained_prefix` uses exactly `frozen_steps`; it does not use the native soft overlap.

Prefix re-encoding disables percentile clipping explicitly. The prefix consists of actions
previously decoded from the same checkpoint; clipping it to `[-1, 1]` during re-normalization
would silently change an otherwise valid physical prior. Non-finite, malformed, or
non-invertible prefixes still fail closed, and the decoded hard-prefix invariant remains the
admission proof.

History is bounded to the last `H` target steps and cleared on register, close, idle expiry,
generation replacement, shutdown, or inference error that invalidates continuity.

If continuity is lost and the remaining timeline drains, the server holds. It does not create
an unconditioned replacement inside the same generation. Recovery requires an explicit session
re-registration, which clears RTC state and permits one new `off` bootstrap inference.

The server defines actual delay as the number of source-relative rows that became committed
before result admission, `max(1, completion_timestep - source_timestep + 1)`. All predictor,
underestimate, prefix, and training-range checks use this same inclusive definition.

### Result admission invariants

A result may replace the future timeline only when all of the following hold:

1. session ID and generation still match;
2. the snapshot was valid when launched and its state/images are the immutable values used
   for inference;
3. actual delay does not exceed the frozen prediction;
4. enough unexpired output remains for the configured minimum new tail;
5. the decoded hard prefix matches its physical prior actions within configured numerical
   tolerances for position, SO(3) angle, and gripper values;
6. every output is finite and has the exact semihumanoid shape; and
7. action and seam diagnostics are present in the inference result.

Failure rejects the new result atomically. It never partially merges a failed chunk.

### Checkpoint compatibility

- `off`: always allowed.
- `native`: allowed only for the N1.7 action head with a valid 40-step saved processor and a
  finite positive ramp rate.
- `trained_prefix`: allowed only when the checkpoint config records a positive
  `rtc_training_max_prefix_steps` and the requested delay is within that bound.
- Checkpoints lacking the new config field load with a default of zero and remain compatible
  with `off` and `native`.
- Server metadata reports the selected RTC mode, trained maximum prefix, delay window,
  margin, overlap/tail bounds, ramp rate, and checkpoint hashes. The existing required async
  capabilities and `progress=0.0` remain unchanged.

## Phased implementation

### Phase 0 — Capture the baseline without commanding the robot

- [x] Record the repository revision, checkpoint/config contract, GPU inventory/occupancy,
  free storage, source subset inventory, converted-ledger counts, and relevant host processes
  read-only. Do not stop, restart, reconfigure, or send commands to any workload.
- [x] Initialize only the pinned `packages/industrialnext_rpc` submodule and verify that `uv
  sync --locked` can resolve the workspace without changing the lockfile.
- [x] Run the focused existing CPU baseline before edits and record any pre-existing failure.
- [x] Preserve `outputs/gr00t/semihumanoid_20260819_080107` and the current converted root as
  immutable baseline evidence; do not append, rewrite statistics, or train into either path.
- [x] Defer checkpoint latency, seam measurement, and RTC-parameter selection until Phase 4
  creates the reproducible benchmark harness. Do not use an ad-hoc one-off benchmark as the
  training-bound source of truth.

**Evidence (2026-08-19):** initialized the pinned RPC revision
`c3dc583ee36310581ad1ec154559698051988b9f`; `uv sync --locked` and
`uv sync --all-extras --locked` preserved the existing `uv.lock` SHA-256; the focused
pre-change suite passed 55 tests. No checkpoint, converted corpus, service, or robot state
was changed.

**Gate:** the dependency workspace and CPU baseline are healthy before model changes. No RTC
parameter is called robot-ready from an unrecorded warmup or one rollout.

### Phase 1 — Add tokenwise flow conditioning without changing the default model

- [x] Add `rtc_training_max_prefix_steps: int = 0` to `Gr00tN1d7Config`. Validate
  `0 <= value < action_horizon`; zero preserves the existing objective.
- [x] Add the corresponding `FinetuneConfig` argument and propagate it through
  `launch_finetune.py` so the value is saved in `config.json` and the experiment config.
- [x] Extend `MultiEmbodimentActionEncoder.forward()` to accept either `(B,)` or `(B,T)`
  timesteps. Keep the current scalar expansion branch numerically unchanged.
- [x] Extend `TimestepEncoder`, `AdaLayerNorm`, `DiT`, and `AlternateVLDiT` so timestep
  embeddings can be `(B,D)` or `(B,T,D)`. Tokenwise conditioning covers the state token and
  every action token; the state token retains the ordinary sampled flow time.
- [x] In training, sample an integer prefix independently for every batch item from the
  configured inclusive range `0..rtc_training_max_prefix_steps`.
- [x] Set prefix action tokens to clean normalized ground truth with flow time 1.0, use the
  existing noisy trajectory for the postfix, and multiply the existing action mask by a
  postfix mask before reducing loss. Return the effective mask, sampled prefix lengths, and
  postfix valid-element count as non-loss diagnostics.
- [x] Keep the original noise distribution, optimizer, trainable modules, state dropout,
  image augmentation, and action statistics unchanged.
- [x] Add a trained-prefix sampler that restores the normalized prefix before every model
  evaluation, supplies flow time 1.0 for prefix tokens and the current integration time for
  postfix/state tokens, and restores the hard prefix once more before return. Keep the
  existing native ramp sampler as its own branch.
- [x] Reject malformed mode/option combinations at construction or call entry: negative or
  oversized prefix, frozen greater than overlap, missing prefix tensor, non-finite ramp, or
  `trained_prefix` on a checkpoint trained with zero maximum prefix.

**Implementation evidence (2026-08-19):** the config, action head, tokenwise DiT
conditioning, zero-default compatibility path, trained-prefix sampler, and focused model
tests are implemented. Prefix sampling is per item and uniform over the inclusive range;
the hard prefix is restored before every model evaluation and after the final integration
step.

**Acceptance:** with a fixed seed and `rtc_training_max_prefix_steps=0`, the refactored
forward and sampler match the prior scalar path. With a positive setting, prefix tokens are
clean, prefix loss is zero, postfix loss remains finite, and hard-clamped inference returns
the prefix exactly.

### Phase 2 — Wire physical prefixes through `Gr00tPolicy`

- [x] Replace the unused `options` behavior with one validated RTC request contract. Keep
  the public `BasePolicy.get_action(observation, options)` signature compatible.
- [x] Add strict action-prefix validation using the checkpoint's action keys, dimensions,
  batch size, horizon, dtype, and finiteness rules.
- [x] Add a policy-side helper that takes an absolute physical prefix plus the new
  observation state, converts it to relative action, and normalizes it at new prefix rows
  `0..O-1`.
- [x] Avoid the horizon-statistics trap by processing a full 40-row scratch action, reading
  only its correctly normalized prefix, and packing that normalized prefix into the exact
  native or trained sampler layout after normalization.
- [x] Extend `StateActionProcessor.apply_action()` with a call-local `clip_outliers` override
  and use `False` only for this generated-prefix path. Do not mutate processor-global state;
  ordinary observation/training processing retains the saved clipping behavior.
- [x] Pass the resulting normalized action tensor and validated mode options to
  `self.model.get_action()`.
- [x] Decode the entire generated chunk with the same new observation state used to encode
  the prefix.
- [x] Add the inverse of `map_action_chunk()`: wire action rows in Industrial Next rot6d
  convention to batched GR00T physical action arrays. Reuse the central rot6d conversion
  utilities.
- [x] Add round-trip and prefix-invariance checks for both EEFs and grippers. Orientation
  comparisons use reconstructed rotation matrices/geodesic angle, not componentwise rot6d
  distance.

**Implementation evidence (2026-08-19):** policy options now use one strict RTC contract;
physical Ube prefixes are re-anchored through a full-horizon scratch action without clipping,
packed in the sampler-specific layout, decoded against the same state, and checked through
both EEF/gripper round trips. The old checkpoint completed the native checkpoint-backed
benchmark through this path.

**Acceptance:** an arbitrary finite, non-degenerate absolute physical prefix round-trips through
re-anchor/normalize/pack/sample/decode without a horizon-row shift. Frozen native and
trained prefixes reproduce their input actions to numerical tolerance.

### Phase 3 — Make the async server an RTC scheduler

- [x] Extend the serving config and CLI with explicit `rtc_mode`, delay bootstrap, rolling
  delay window, safety margin, maximum prefix, native overlap, minimum new tail, ramp rate,
  and hard-prefix tolerances. The safe default remains `rtc_mode=off`; production commands
  must opt in explicitly.
- [x] Retain a bounded `served_history` alongside the future timeline. Store the exact row
  returned before removing it from the timeline.
- [x] Split inference data into an observation snapshot and a launch-time immutable RTC
  request. Prefix assembly happens only in `_launch_inference()` on the event-loop thread.
- [x] Assemble a prefix only from contiguous rows at the new source timestep. Never borrow
  rows across session generations, missing targets, rejected results, or stale observations.
- [x] Keep only one inference in flight and one latest pending observation. After a result is
  admitted or rejected, rebuild the pending request using the now-current timeline before
  launching it.
- [x] Predict frozen delay from the rolling maximum plus margin. Measure actual delay from
  accepted server control steps with the inclusive committed-prefix definition, not
  wall-clock latency divided and rounded after the fact.
- [x] For `native`, choose overlap from the available contiguous prefix and configured bounds,
  then arrange the normalized prefix tail expected by the checked-in GR00T primitive.
- [x] For `trained_prefix`, supply exactly the predicted hard prefix and require it to be at
  most the checkpoint's trained maximum.
- [x] On completion, reject underestimated-delay, insufficient-tail, non-contiguous-prefix,
  and hard-prefix-mismatch results atomically. Continue the prior timeline when possible.
- [x] Replace the remaining future timeline only after admission; drop rows that expired
  during inference and retain the absolute target-timestep contract.
- [x] Clear RTC state on registration replacement, close, idle expiry, shutdown, and any
  continuity-invalidating failure.
- [x] Add monitoring for mode, predicted/actual delay, overlap, available prefix, new tail,
  prefix position/angle/gripper error, first admitted seam, inference timings, result
  rejections, null reasons, and position/orientation finite differences.
- [x] Keep `handle_request()` non-blocking, the WebSocket response schema compatible, and
  `progress=0.0` with a fresh monitoring timestep.

**Implementation evidence (2026-08-19):** the scheduler uses immutable launch requests,
bounded served/future timelines, rolling accepted-step delay, atomic admission, hard-prefix
and optional dynamics gates, and mandatory re-registration after continuity loss. Focused
server/protocol tests pass for native/trained rollover, underestimated delay, prefix
mismatch, replacement/reconnect, and dynamics rejection.

**Acceptance:** deterministic fake-policy scenarios cover first inference, normal RTC
rollover, pending-snapshot ordering, delay underestimate, latency spike, missing prefix,
generation replacement, reconnect, inference exception, and shutdown. No scenario admits a
partial or cross-session chunk.

### Phase 4 — Add reproducible training and evaluation controls

- [x] Extend the zdata training config with `rtc_training_max_prefix_steps` and propagate it
  to the finetune command.
- [x] Add trainer logging for the reduced sampled-prefix histogram and postfix valid-element
  count. DDP ranks must contribute to one global diagnostic; logging must not alter the loss.
- [x] Before launch, write a run manifest under the new output directory containing the
  resolved dataset paths, action horizon/FPS, trainable-start count, effective epochs,
  pipeline config hash, modality module hash, ledger hashes, dataset metadata/statistics
  hashes, SHA-256 inventory of every converted parquet/video artifact, base model and resolved
  revision, repository revision and dirty diff hash, command, and selected prefix
  distribution. Write atomically, then verify it before spawning `torchrun`.
- [x] Refuse a fresh training launch if any conversion transaction is incomplete, required
  statistics are missing, the requested prefix exceeds the model horizon/tail bound, or the
  output directory already exists.
- [x] Add `benchmark_industrialnext_rtc.py` with two bounded workloads:
  - checkpoint latency on synthetic semihumanoid observations, including preprocessing and
    decode; and
  - sequential held-out episode replay with a supplied or measured delay trace.
- [x] Report the same deterministic metrics for `off`, `native`, and `trained_prefix`:
  target-timestep coverage, rejection/hold rate, first-executable-row action error,
  cross-chunk position and SO(3) seam, gripper seam, per-step velocity, and second finite
  difference. Use the same fixed noise-seed sequence and delay trace for every mode, repeat
  enough seeds to expose stochastic strategy changes, and save JSON plus a concise text
  summary.
- [x] Add checkpoint provenance to server metadata and benchmark outputs, including whether
  training-time prefix conditioning is present and its maximum delay.
- [x] With the old checkpoint, record warmup separately from at least 100 steady-state
  checkpoint-latency calls on an allocated GPU. Report p50/p95/p99/max for preprocessing,
  generation, decode, and total time. Convert latency to a sizing proxy with
  `max(1, ceil(total_ms * control_hz / 1000))`; runtime admission continues to use accepted
  control timesteps, not this proxy.
- [x] Replay held-out Ube episodes through `off` and `native` with fixed seeds/delay traces to
  establish baseline seam, velocity, second-difference, coverage, and hold metrics. Mark
  `trained_prefix` unsupported for the old checkpoint rather than fabricating a result.
- [x] Select provisional `initial_frozen_steps`, rolling-window size, safety margin,
  `rtc_training_max_prefix_steps`, native overlap, and minimum new tail from the recorded
  distributions. Require at least 16 independently generated tail steps, so both the trained
  maximum and native overlap may not exceed 24 for horizon 40.

**Implementation and measurement evidence (2026-08-19):**
`artifacts/semihumanoid_rtc_baseline_20260819/` contains the checkpoint/shard provenance,
five warmups plus 100 steady calls per supported mode, JSON/text timing output, and the
reviewed sizing decision. Measured total p99 was 62.80 ms (`off`) and 70.42 ms (`native`),
both a 4-step sizing proxy at 50 Hz. The selected training maximum is 12, leaving 28 postfix
steps; provisional runtime values are initial 4, window 20, margin 2, maximum/overlap 12,
and minimum new tail 16. Deployment shadow measurements may tighten these runtime values but
cannot exceed the checkpoint's trained maximum.

`artifacts/semihumanoid_rtc_baseline_replay_20260819/` records fixed-seed/fixed-delay replay
for trajectories 0, 1, and 2 from every Ube validation release. `off` and `native` are
measured and the old checkpoint reports `trained_prefix` unsupported. Native replay also
records the observed hard-prefix errors and fail-closed rejections; some trajectories exceed
the 0.1 mm position tolerance after bf16 normalization/decode, so native remains a baseline,
not an assumed production winner.

**Acceptance:** the manifest is sufficient to reconstruct the exact corpus and command,
the old-checkpoint baseline and sizing decision are saved, and the benchmark refuses an
unsupported mode/checkpoint combination rather than silently falling back.

### Phase 5 — Freeze the corpus and run the fresh finetune

This phase waits until the Ube data-collection campaign reaches an explicit cutoff. Do not
read an actively written `episode.h5` as a completed source episode, and do not stop a
recorder or serving process merely to begin training. The old broad converted root is not
reused or modified.

- [x] Record the agreed source cutoff timestamp and pin the included source list to
  `flexiv_ube_v1`, `flexiv_ube_v2`, `flexiv_ube_v3`, and `flexiv_ube_v4`. Confirm no included
  file is open for writing and no included episode changed across two inventory passes.
- [x] Bind producer sign-off for each included release: absolute commanded-action semantics,
  both-arm/gripper ownership, fixed head camera, task catalog, and QC authority. Exclude any
  release missing that binding.
- [x] Make `require_valid_for_training: true` reject a missing admission field as well as an
  explicit false value. A warning plus implicit acceptance is not sufficient for this frozen
  training release.
- [x] Change the YAML to a new cutoff-named converted output root, the explicit Ube source
  list, and `continuity.split_on_gap_ms: 40`. Never point the fresh run at the historical
  broad output root.
- [x] Preview new work:

  ```bash
  uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
    sync --config configs/embodiments/semihumanoid.yaml --dry-run
  ```

  **Cutoff evidence (2026-08-19):**
  `artifacts/semihumanoid_ube_rtc_cutoff_20260819_033143/source_audit.json` binds the
  03:31:43 UTC cutoff, two identical 1,339-file stat inventories, zero open source files,
  explicit admission on every episode, the producer/release contract, and the disposition of
  every observed gap/short-segment warning. The full dry run accepted all 1,339 records.

- [x] After recording has stopped cleanly for the included episodes, convert the frozen Ube
  corpus into the new root, generate statistics, and run the full converted-output check:

  ```bash
  uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
    sync --config configs/embodiments/semihumanoid.yaml --workers 4
  uv run python scripts/lerobot_conversion/run_zdata_pipeline.py \
    stats --config configs/embodiments/semihumanoid.yaml --jobs 4
  uv run python scripts/lerobot_conversion/run_zdata_pipeline.py \
    check --config configs/embodiments/semihumanoid.yaml --full
  ```

- [x] Verify every >40 ms boundary became a segment break, no action window crosses a
  segment, every kept segment has at least 40 rows, all source records have explicit
  `valid_for_training=true`, and all configured RGB freshness/QC warnings are either zero or
  individually dispositioned before admission.
- [x] Create and verify the content-bound corpus manifest, then make the converted root
  immutable to the pipeline for the duration of training. A later source addition requires a
  new output root and manifest, not an append to this run.

  ```bash
  uv run python scripts/lerobot_conversion/run_zdata_pipeline.py \
    freeze --config configs/embodiments/semihumanoid.yaml
  ```

- [x] Re-run the checkpoint latency benchmark on the intended training/inference GPU and set
  `rtc_training_max_prefix_steps` from the measured p99 steps plus the reviewed safety
  margin, bounded so at least 16 postfix steps remain. Record the final value in the YAML and
  run manifest.

  **Frozen-corpus evidence (2026-08-19):** all eight train/validation datasets passed the
  exhaustive check. The frozen manifest binds 1,339 source guards and 5,820 converted
  artifacts and has SHA-256
  `3158713bfc4851e382007d2bb69f54a2532610a324d5a6cc9ef90fbd72899977`. The four train
  datasets contain 1,367 gap-safe segments, 460,109 frames, and 406,796 trainable 40-step
  starts; the four validation datasets contain 73 segments and 24,286 frames. At batch 256,
  2.2 effective epochs derive to 3,496 optimizer steps.
- [x] Run a bounded one-GPU training smoke through the pipeline's smoke-launch option against
  the full four-dataset Ube mixture. Require
  finite loss, nonzero postfix loss, varying sampled prefix lengths including zero, no NaN,
  and a saved config that contains the requested RTC training field.
- [x] Confirm all intended training GPUs are free or explicitly allocated. The existing data
  collection/inference workload is not to be killed or preempted implicitly.

  **Training-smoke evidence (2026-08-19):** the first smoke at
  `semihumanoid_smoke_20260819_225426` was rejected because checkpoint construction dropped
  the requested RTC training field and exercised only prefix length zero. The model setup was
  repaired to pass the finetune override into checkpoint construction. The replacement smoke
  at `semihumanoid_smoke_20260819_225723` passed the automated acceptance gate: its persisted
  model configs record maximum prefix 12, losses and gradient norms are finite, postfix-valid
  counts are positive, every prefix bin 0 through 12 is nonzero, and the checkpoint was saved.
  Immediately before the full launch, all four 49,140 MiB RTX 4090 GPUs reported 4 MiB used
  and zero utilization.
- [x] Launch a genuinely fresh run from `nvidia/GR00T-N1.7-3B`, not from the current
  semihumanoid checkpoint:

  ```bash
  uv run python scripts/lerobot_conversion/run_zdata_pipeline.py \
    train --config configs/embodiments/semihumanoid.yaml --jobs 4
  ```

- [x] Monitor loss, gradient norm, throughput, GPU memory, sampled prefix histogram, and
  postfix valid-element count. Stop on non-finite values, collapsed postfix coverage, or
  repeated data-loader failures.
- [x] Validate every saved candidate structurally, then run `gr00t/eval/open_loop_eval.py`
  and RTC sequential replay on every held-out Ube `_val` dataset with fixed bounded trajectory
  IDs recorded in the evaluation manifest.
- [x] Select the checkpoint from held-out action accuracy plus RTC continuity and rejection
  metrics. Do not select by training loss or recency alone.

  **Training and selection evidence (2026-08-20):** the fresh four-GPU run
  `semihumanoid_20260819_230043` completed all 3,496 optimizer steps in 6,088.57 seconds at
  146.99 samples/s. Loss and gradient norm remained finite, postfix coverage remained
  positive, and all sampled-prefix bins 0 through 12 were populated. Checkpoints 1000, 2000,
  3000, and 3496 each contain their processor, prefix-enabled config, two model shards,
  scheduler, trainer state, four RNG states, and four optimizer shards. Every candidate was
  evaluated on trajectories 0, 1, and 2 of all four held-out Ube releases. The accepted
  corrected replay matrix is `artifacts/semihumanoid_rtc_candidates_20260820_v2`; the initial
  matrix is retained as rejected diagnostic evidence because it exposed the BF16 committed-
  prefix decode defect fixed before selection.

  `checkpoint-3496` was selected by the recorded multi-metric decision in
  `artifacts/semihumanoid_rtc_candidate_selection_20260820/selection.json`: it has the best
  aggregate open-loop MSE/MAE, trained-prefix replay MAE, and off replay MAE. `checkpoint-3000`
  is marginally better only in native replay. All corrected modes have 0.96 coverage, zero
  holds, and zero rejections; native and trained-prefix position/gripper prefix errors are
  zero and orientation error is at most 4.22e-8 rad. Trained-prefix nevertheless has
  materially worse replay accuracy, so `native` is the deployment-shadow frontrunner and no
  motion mode is selected on the GPU machine.

**Gate:** a new checkpoint is not deployment-ready merely because training completed. It
must load from its own saved processor, report training-time prefix support, generate finite
actions in all declared modes, and pass held-out evaluation.

### Phase 6 — GPU-machine validation and deployment handoff

- [x] Run the focused CPU checks:

  ```bash
  uv run pytest \
    tests/gr00t/model/test_action_head.py \
    tests/gr00t/policy/test_gr00t_policy.py \
    tests/gr00t/policy/test_industrialnext_adapter.py \
    tests/gr00t/policy/test_industrialnext_async_server.py \
    tests/gr00t/policy/test_industrialnext_protocol.py \
    -m "not gpu" -q
  ```

- [x] Run Ruff/format checks on changed Python files, `git diff --check`, and the repository's
  pre-commit suite before publication.
- [x] With an explicitly allocated GPU, run the checkpoint-backed smoke test for the selected
  checkpoint in `off`, `native`, and `trained_prefix` modes. Verify unsupported-mode gates
  separately with the old checkpoint.
- [x] Launch the selected server on loopback with explicit RTC arguments and no robot command
  path. Run the command once per mode on unused ports:

  ```bash
  uv run python gr00t/eval/run_gr00t_industrialnext_server.py \
    --model-path outputs/gr00t/semihumanoid_20260819_230043/checkpoint-3496 \
    --task-catalog-path artifacts/semihumanoid_rtc_loopback_inputs_20260820/task_catalog.yaml \
    --embodiment-tag new_embodiment \
    --device cuda \
    --host 127.0.0.1 \
    --port 11120 \
    --control-hz 50 \
    --rtc-mode MODE \
    --rtc-initial-frozen-steps 4 \
    --rtc-delay-window-size 20 \
    --rtc-delay-margin-steps 2 \
    --rtc-max-prefix-steps 12 \
    --rtc-native-overlap-steps 12 \
    --rtc-min-new-tail-steps 16 \
    --min-usable-action-steps 16
  ```

  Use an alternate loopback port for standalone smoke tests if `10012` is occupied. Do not
  stop or replace the current `industrialnext_ai` service merely to claim this gate. Port
  `10012` is used for the real ROS shadow only in an explicitly coordinated service window.

- [x] Build `artifacts/semihumanoid_rtc_handoff_20260820_3496/` without modifying a deployment
  checkout. Include the selected checkpoint or an explicit copy command, SHA256 inventory,
  exact source revision and dirty diff, final training YAML, frozen-corpus and run manifests,
  held-out/replay/latency reports, server invocation, task catalog, runtime parameter
  template, rollback command, and operator checklist.
- [x] Verify every handoff hash after copying the bundle to a staging directory. The
  deployment machine must repeat the same verification after transfer; a path existing is
  not evidence that the artifact is complete.

  **GPU validation and handoff evidence (2026-08-20):** 166 focused CPU tests passed with
  one unrelated test deselected; changed-file Ruff format/checks, `git diff --check`, the
  locked dependency check, and all pre-commit hooks passed. The selected checkpoint produced
  finite outputs in all three modes. Its isolated p99 latencies were 60.38 ms (`off`),
  69.34 ms (`native`), and 71.13 ms (`trained_prefix`), preserving the conservative maximum
  prefix 12. Three paced 50 Hz loopback runs each returned 56 finite action responses after
  four startup responses with zero protocol errors. The bundle contains the exact source
  patch, checkpoint and bundle SHA-256 inventories, selected reports, runtime parameters,
  copy/server instructions, rollback template, and operator checklist. A staged copy passed
  every bundle hash and every selected-checkpoint hash.

**Gate:** GPU work is complete only when the bundle can reproduce checkpoint-backed
`off`, `native`, and `trained_prefix` loopback results and contains everything needed for a
deployment operator to verify provenance without access to mutable training directories.

### Phase 7 — Deployment-machine no-motion shadow

This phase is a deployment-machine handoff. It is not executed from the GPU machine.

GPU evidence makes `native` the initial shadow frontrunner; this is not a motion-mode
selection. The deployment machine must still reproduce and compare all modes against the
same no-motion observation stream.

- [ ] Inspect the deployment checkout, related worktrees, active services, and effective
  configuration before changing anything. Verify the handoff inventory, source revision,
  checkpoint hash, task catalog, and processor artifacts locally.
- [ ] Run the checkpoint-backed loopback smoke on the deployment machine in `off`, `native`,
  and `trained_prefix` modes. Unsupported old-checkpoint/mode combinations must fail closed.
- [ ] Coordinate a service window, then run the real ROS policy client only with
  `inference_mode=true`. Confirm from effective runtime parameters and observed topics that
  no arm or gripper command is published.
- [ ] Shadow each candidate mode against the same observation/task sequence. Record at least
  several hundred steady-state inferences and retain server plus client logs.
- [ ] Require zero protocol/reconnect errors, zero non-finite actions, zero prefix invariant
  failures, zero cross-session rows, bounded image staleness, and delay-estimator coverage of
  the observed maximum. Compare seam/velocity metrics against the expert data distribution
  and the current `off` baseline.
- [ ] Select the motion candidate from the recorded evidence. `trained_prefix` is preferred
  only if it preserves held-out action accuracy and materially improves continuity without
  increasing holds or rejections. Otherwise select `native` if it passes, or stop with no
  motion if neither does.

**Gate:** the deployment operator signs the no-motion report and records the exact effective
runtime configuration. This does not authorize command publication.

### Phase 8 — Separately authorized deployment-machine real-robot trial

No step in this phase is authorized by implementation or shadow-test approval. Obtain a
fresh explicit go from the operator immediately before command publication.

- [ ] Confirm the data-collection run is stopped or isolated, the robot workspace is clear,
  manual stop/E-stop is staffed, the intended task and arm set are correct, robot-state trust
  is fresh, cameras match the checkpoint, and only one policy session is active.
- [ ] Reconcile the user's existing Flexiv Ube config changes, then enable the existing
  `action_delta_clamp_enabled` deployment setting with reviewed small deltas. Regenerate
  deployment parameters and restart only the required units with explicit authorization.
- [ ] Verify effective runtime parameters: 50 Hz, async node, selected task UUID/text,
  checkpoint hash, RTC mode, trained maximum prefix, delay settings, clamp enabled,
  `inference_mode=false`, and `progress=0.0` semantics.
- [ ] Begin with a one- to two-second limited rollout in a low-risk pose region. Manually
  stop, inspect command/state traces, clamp activation, server rejection/hold events,
  position/orientation speed, and the first two chunk transitions.
- [ ] Increase duration in small reviewed increments only after clean traces. Keep manual
  termination; do not wait for progress-based completion.
- [ ] Stop immediately on repeated clamp activation, any unexpected fast motion, prefix or
  delay rejection, stale state/camera, timeline gap, reconnect, model exception, or operator
  concern.
- [ ] Preserve the exact logs and mark the trial mode/checkpoint/config so results can be
  attributed. Restore the prior deployment setting after the trial if the operator requests
  it.

## Validation matrix

| Layer | Required evidence | Motion allowed? |
|---|---|---|
| Model math | Scalar regression, tokenwise timestep shapes, prefix loss mask, exact hard prefix | No |
| Processor | Absolute-to-relative re-anchor and correct new-horizon normalization round-trip | No |
| Server | Deterministic timeline/delay/session scenarios and loopback protocol | No |
| Checkpoint | Finite GPU inference, saved config/processor, held-out open-loop metrics | No |
| RTC replay | Same delay trace across modes; seam, velocity, acceleration, holds, accuracy | No |
| ROS shadow | Effective `inference_mode=true`, real observations, no command publications | No |
| Limited rollout | Explicit operator approval, clamp enabled, staffed stop, short duration | Yes, supervised |
| Extended rollout | Review of limited-rollout traces and renewed operator go | Yes, supervised |

## Failure policy

| Failure | Required behavior |
|---|---|
| Missing/invalid prefix | Do not launch an RTC refresh; keep current timeline or hold |
| Predicted delay exceeds checkpoint training range | Fail the request/configuration; do not extrapolate silently |
| Actual delay exceeds prediction | Reject new chunk, update estimator, retain prior future rows |
| Frozen prefix mismatch after decode | Reject entire chunk and record per-field error |
| Insufficient new tail | Reject entire chunk and keep prior timeline if available |
| Inference exception/non-finite output | No new action rows; expose error and hold |
| Session replacement/reconnect | Clear history, timeline, delay state, pending request, and first-chunk RTC state |
| Continuity lost with no usable future timeline | Hold and require explicit session re-registration before new first-chunk `off` generation |
| Frequent ROS clamp activation | Stop rollout and treat as model/trajectory failure |
| Progress remains zero | Continue only under operator supervision; manual stop ends rollout |

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Native GR00T RTC is not full gradient-guided RTC | Results may differ from the paper | Name it `native`, document its actual mechanism, benchmark it independently |
| Tokenwise AdaNorm refactor changes old behavior | Base/old checkpoints regress | Default-zero scalar compatibility path and fixed-seed regression checks |
| Relative-action horizon statistics are misindexed | Frozen physical actions shift after decode | Re-anchor absolute actions at the new state and normalize only as new prefix rows |
| Model stochasticity switches strategies | Smooth prefix but poor postfix/task behavior | Held-out sequential replay and short supervised trials; do not rely on seam alone |
| Delay spike exceeds trained prefix | First executable new action is unconstrained | Conservative rolling maximum, margin, hard rejection, and minimum tail bound |
| ROS safety clamp changes the action the server assumes | RTC prior differs from actual command | Treat any clamp hit as a rollout failure; compare served command and observed state traces |
| Mixed producer action ownership enters training | Inactive Matcha coordinates become false targets | Pin only signed-off Ube releases for this run; add source-specific coordinate masks before any future mixed run |
| A manifest lists paths but not file content | Mutated parquet/video bytes can pass provenance checks | Hash every admitted converted artifact and verify the inventory immediately before launch |
| Active data collection produces a moving corpus | Irreproducible or partial training input | Explicit cutoff, complete-episode sync, full check, content manifest, and a new root per release |
| Fresh data changes task/subset balance | Model quality regresses despite more episodes | Report per-task/subset counts and use held-out per-task evaluation |
| GPU contention with the current serving/data job | Latency distortion, OOM, or operational disruption | Read-only occupancy check and explicit allocation; never kill/preempt implicitly |
| Progress never completes the task | Rollout runs until max frames | Keep progress zero by design and require manual stop for every trial |

## Implementation and publication order

1. Land Phases 1-4 with focused tests and no live-system mutation.
2. Re-run the baseline benchmark and select the training prefix bound.
3. Freeze/validate the corpus and launch the fresh base-model finetune.
4. Select a checkpoint through held-out and RTC replay evidence.
5. Validate the GPU implementation and produce the content-verified deployment handoff
   bundle. Commit or publish only when separately requested.
6. Transfer the bundle and repeat its hash/checkpoint/loopback verification on the deployment
   machine.
7. Run the deployment-machine no-motion ROS shadow and select or reject a motion candidate.
8. Make the separate, reviewed Flexiv Ube clamp/config change on the deployment machine.
9. Obtain explicit motion approval and conduct the limited robot trial there.

## Completion criteria

This plan is complete only when:

1. Old checkpoints retain working `off` and `native` paths, and the new checkpoint advertises
   and runs `trained_prefix` explicitly.
2. Physical action-prefix conversion is correct across relative EEF normalization, rot6d
   conventions, horizon-indexed statistics, and decode.
3. The server predicts and verifies delay, preserves a contiguous action timeline, rejects
   unsafe results atomically, and remains non-blocking at 50 Hz.
4. A fresh base-model finetune is tied to a frozen, checked corpus and complete provenance.
5. The selected checkpoint passes held-out open-loop evaluation, sequential RTC replay, and
   checkpoint-backed GPU inference; its content-verified bundle is accepted on the
   deployment machine.
6. Real ROS no-motion shadow passes on the deployment machine and produces an operator-signed
   evidence report before any motion request.
7. Any robot motion occurs only under separate explicit authorization with the existing
   clamp, trust gates, staffed manual stop, and short-to-long rollout progression.
8. Trial logs identify the exact repository revisions, checkpoint hashes, data manifest,
   server settings, effective ROS parameters, and observed safety events.

## References

- [NVIDIA GR00T real-world deployment guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/real_world_deployment.md)
- [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
- [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964)
