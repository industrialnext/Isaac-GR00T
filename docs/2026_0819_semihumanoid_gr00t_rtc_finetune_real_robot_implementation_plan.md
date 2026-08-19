# Semihumanoid GR00T RTC, Fresh Finetune, and Real-Robot Implementation Plan

**Date:** 2026-08-19

**Status:** Planning

**Primary repository:** `/home/ubuntu/Isaac-GR00T`

**Related serving plan:**
[`2026_0819_semihumanoid_gr00t_real_robot_serving_implementation_plan.md`](2026_0819_semihumanoid_gr00t_real_robot_serving_implementation_plan.md)

## Outcome

Implement the complete action-prefix path needed by the Industrial Next async GR00T server,
train a fresh semihumanoid checkpoint with training-time action-prefix conditioning, compare
the available RTC modes offline and in a no-motion shadow run, and only then perform a
separately authorized, supervised robot rollout.

The production candidate will keep the existing 50 Hz ROS policy client and WebSocket
protocol. GR00T inference remains asynchronous and produces a 40-step, 0.8-second action
chunk. The server owns chunk alignment, delay prediction, action-prefix construction, and
RTC diagnostics. The ROS side continues to own command publication, robot-state trust,
manual stop, and the last-resort per-step action clamp.

Progress remains `0.0`. These trials are manually stopped; this work does not add an
automatic completion signal.

## Decisions

### 1. Implement two RTC modes, but run only one at a time

The repository and the research literature contain two different techniques:

| Mode | Checkpoint requirement | Prefix behavior | Intended use |
|---|---|---|---|
| `off` | Any compatible checkpoint | No prior action is supplied | Regression baseline and first-chunk generation |
| `native` | Any compatible GR00T N1.7 checkpoint | Use the checked-in GR00T prefix seeding, frozen prefix, and exponential velocity ramp | Immediate compatibility with the current checkpoint and an A/B baseline |
| `trained_prefix` | Checkpoint explicitly trained with action-prefix conditioning | Clamp a hard prefix and use tokenwise flow timesteps learned during finetuning | Primary candidate for the freshly trained checkpoint |

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

## Verified current baseline

- The current checkpoint is
  `/home/ubuntu/Isaac-GR00T/outputs/gr00t/semihumanoid_20260819_080107`.
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
  `nvidia/GR00T-N1.7-3B` checkpoint.
- The ROS async client already has `inference_mode`, robot-state trust, manual stop, and an
  optional per-step action delta clamp. The Flexiv Ube deployment currently has the clamp
  configured but disabled.
- `/home/ubuntu/industrialnext_ros2` and `/home/ubuntu/industrialnext_ai` currently contain
  unrelated local modifications. They must be preserved. No implementation phase may use
  broad staging or cleanup commands.

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
  absolute/relative action normalization and inverse conversion; extend through its public
  behavior rather than duplicating pose math.
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
- `/home/ubuntu/industrialnext_ros2/src/industrialnext_operator_ros2/industrialnext_operator_policy_client/industrialnext_operator_policy_client/policy_control_node_async.py:230-363,2633-2661,2713-2745`
  — existing action-clamp configuration and publication-time application.
- `/home/ubuntu/industrialnext_ros2/src/industrialnext_deployments/robots/flexiv_ube/task_config.yaml:264-290`
  — current 50 Hz async policy deployment and disabled clamp.

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
| `gr00t/policy/gr00t_policy.py` | Validate RTC options, preprocess absolute prefixes correctly, and pass action/options to the model |
| `gr00t/policy/industrialnext/adapter.py` | Add the inverse wire-row-to-model-action mapping and physical prefix validation |
| `gr00t/policy/industrialnext/async_server.py` | Add served history, delay estimator, launch-time prefix assembly, RTC result admission, and metrics |
| `gr00t/eval/run_gr00t_industrialnext_server.py` | Add explicit RTC CLI/config/provenance and startup compatibility gates |
| `gr00t/eval/benchmark_industrialnext_rtc.py` | Add repeatable checkpoint latency and offline chunk-continuity evaluation |
| `scripts/lerobot_conversion/zdata_pipeline/config.py` | Add semihumanoid training-time prefix configuration |
| `scripts/lerobot_conversion/zdata_pipeline/check.py` | Propagate the setting and save a corpus/training manifest before launch |
| `configs/embodiments/semihumanoid.yaml` | Record the measured training prefix distribution for the new run |
| `getting_started/real_world_deployment.md` | Replace the "unwired" status after the implementation and document supported modes |
| Focused existing tests under `tests/gr00t/model/` and `tests/gr00t/policy/` | Cover scalar compatibility, prefix math, normalization, timing, and protocol behavior |

### ROS2 files

No ROS Python or RPC protocol change is required. Before the first commanded trial, make
one reviewed deployment-only edit to:

`/home/ubuntu/industrialnext_ros2/src/industrialnext_deployments/robots/flexiv_ube/task_config.yaml`

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

History is bounded to the last `H` target steps and cleared on register, close, idle expiry,
generation replacement, shutdown, or inference error that invalidates continuity.

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

- [ ] Confirm the current robot/data-collection processes and GPU ownership read-only. Do
  not stop, restart, reconfigure, or send commands to them.
- [ ] Run a checkpoint-backed synthetic-observation benchmark only when an unused GPU is
  available. Record warmup separately from at least 100 steady-state calls.
- [ ] Measure p50/p95/p99/max for preprocessing, model generation, decode, total inference,
  and event-loop `step` latency at 50 Hz.
- [ ] Capture the preliminary chunk seam metrics for the current `off` server: position
  distance, SO(3) angle, gripper delta, first finite difference, and second finite difference
  at every chunk replacement.
- [ ] Select provisional `initial_frozen_steps`, rolling-window size, safety margin,
  `max_prefix_steps`, native overlap, and minimum new tail from measurements. Require at
  least 16 independently generated tail steps; therefore the trained/frozen maximum may not
  exceed 24 with the current horizon.

**Gate:** no parameter is called robot-ready from one warmup or one rollout. Save the
hardware, software revision, checkpoint hash, sample count, and timing distribution.

### Phase 1 — Add tokenwise flow conditioning without changing the default model

- [ ] Add `rtc_training_max_prefix_steps: int = 0` to `Gr00tN1d7Config`. Validate
  `0 <= value < action_horizon`; zero preserves the existing objective.
- [ ] Add the corresponding `FinetuneConfig` argument and propagate it through
  `launch_finetune.py` so the value is saved in `config.json` and the experiment config.
- [ ] Extend `MultiEmbodimentActionEncoder.forward()` to accept either `(B,)` or `(B,T)`
  timesteps. Keep the current scalar expansion branch numerically unchanged.
- [ ] Extend `TimestepEncoder`, `AdaLayerNorm`, `DiT`, and `AlternateVLDiT` so timestep
  embeddings can be `(B,D)` or `(B,T,D)`. Tokenwise conditioning covers the state token and
  every action token; the state token retains the ordinary sampled flow time.
- [ ] In training, sample an integer prefix independently for every batch item from the
  configured inclusive range `0..rtc_training_max_prefix_steps`.
- [ ] Set prefix action tokens to clean normalized ground truth with flow time 1.0, use the
  existing noisy trajectory for the postfix, and multiply the existing action mask by a
  postfix mask before reducing loss.
- [ ] Keep the original noise distribution, optimizer, trainable modules, state dropout,
  image augmentation, and action statistics unchanged.
- [ ] Add a trained-prefix sampler that hard-clamps the normalized prefix and supplies the
  tokenwise timestep mask during every denoising step. Keep the existing native ramp sampler
  as its own branch.
- [ ] Reject malformed mode/option combinations at construction or call entry: negative or
  oversized prefix, frozen greater than overlap, missing prefix tensor, non-finite ramp, or
  `trained_prefix` on a checkpoint trained with zero maximum prefix.

**Acceptance:** with a fixed seed and `rtc_training_max_prefix_steps=0`, the refactored
forward and sampler match the prior scalar path. With a positive setting, prefix tokens are
clean, prefix loss is zero, postfix loss remains finite, and hard-clamped inference returns
the prefix exactly.

### Phase 2 — Wire physical prefixes through `Gr00tPolicy`

- [ ] Replace the unused `options` behavior with one validated RTC request contract. Keep
  the public `BasePolicy.get_action(observation, options)` signature compatible.
- [ ] Add strict action-prefix validation using the checkpoint's action keys, dimensions,
  batch size, horizon, dtype, and finiteness rules.
- [ ] Add a policy-side helper that takes an absolute physical prefix plus the new
  observation state, converts it to relative action, and normalizes it at new prefix rows
  `0..O-1`.
- [ ] Avoid the horizon-statistics trap by processing a full 40-row scratch action, reading
  only its correctly normalized prefix, and packing that normalized prefix into the exact
  native or trained sampler layout after normalization.
- [ ] Pass the resulting normalized action tensor and validated mode options to
  `self.model.get_action()`.
- [ ] Decode the entire generated chunk with the same new observation state used to encode
  the prefix.
- [ ] Add the inverse of `map_action_chunk()`: wire action rows in Industrial Next rot6d
  convention to batched GR00T physical action arrays. Reuse the central rot6d conversion
  utilities.
- [ ] Add round-trip and prefix-invariance checks for both EEFs and grippers. Orientation
  comparisons use reconstructed rotation matrices/geodesic angle, not componentwise rot6d
  distance.

**Acceptance:** an arbitrary valid absolute physical prefix round-trips through
re-anchor/normalize/pack/sample/decode without a horizon-row shift. Frozen native and
trained prefixes reproduce their input actions to numerical tolerance.

### Phase 3 — Make the async server an RTC scheduler

- [ ] Extend the serving config and CLI with explicit `rtc_mode`, delay bootstrap, rolling
  delay window, safety margin, maximum prefix, native overlap, minimum new tail, ramp rate,
  and hard-prefix tolerances. The safe default remains `rtc_mode=off`; production commands
  must opt in explicitly.
- [ ] Retain a bounded `served_history` alongside the future timeline. Store the exact row
  returned before removing it from the timeline.
- [ ] Split inference data into an observation snapshot and a launch-time immutable RTC
  request. Prefix assembly happens only in `_launch_inference()` on the event-loop thread.
- [ ] Assemble a prefix only from contiguous rows at the new source timestep. Never borrow
  rows across session generations, missing targets, rejected results, or stale observations.
- [ ] Keep only one inference in flight and one latest pending observation. After a result is
  admitted or rejected, rebuild the pending request using the now-current timeline before
  launching it.
- [ ] Predict frozen delay from the rolling maximum plus margin. Measure actual delay from
  accepted server control steps with the inclusive committed-prefix definition, not
  wall-clock latency divided and rounded after the fact.
- [ ] For `native`, choose overlap from the available contiguous prefix and configured bounds,
  then arrange the normalized prefix tail expected by the checked-in GR00T primitive.
- [ ] For `trained_prefix`, supply exactly the predicted hard prefix and require it to be at
  most the checkpoint's trained maximum.
- [ ] On completion, reject underestimated-delay, insufficient-tail, non-contiguous-prefix,
  and hard-prefix-mismatch results atomically. Continue the prior timeline when possible.
- [ ] Replace the remaining future timeline only after admission; drop rows that expired
  during inference and retain the absolute target-timestep contract.
- [ ] Clear RTC state on registration replacement, close, idle expiry, shutdown, and any
  continuity-invalidating failure.
- [ ] Add monitoring for mode, predicted/actual delay, overlap, available prefix, new tail,
  prefix position/angle/gripper error, first admitted seam, inference timings, result
  rejections, null reasons, and position/orientation finite differences.
- [ ] Keep `handle_request()` non-blocking, the WebSocket response schema compatible, and
  `progress=0.0` with a fresh monitoring timestep.

**Acceptance:** deterministic fake-policy scenarios cover first inference, normal RTC
rollover, pending-snapshot ordering, delay underestimate, latency spike, missing prefix,
generation replacement, reconnect, inference exception, and shutdown. No scenario admits a
partial or cross-session chunk.

### Phase 4 — Add reproducible training and evaluation controls

- [ ] Extend the zdata training config with `rtc_training_max_prefix_steps` and propagate it
  to the finetune command.
- [ ] Before launch, write a run manifest under the new output directory containing the
  resolved dataset paths, action horizon/FPS, trainable-start count, effective epochs,
  pipeline config hash, modality module hash, ledger hashes, dataset metadata/statistics
  hashes, base model, repository revision, command, and selected prefix distribution.
- [ ] Refuse a fresh training launch if any conversion transaction is incomplete, required
  statistics are missing, the requested prefix exceeds the model horizon/tail bound, or the
  output directory already exists.
- [ ] Add `benchmark_industrialnext_rtc.py` with two bounded workloads:
  - checkpoint latency on synthetic semihumanoid observations, including preprocessing and
    decode; and
  - sequential held-out episode replay with a supplied or measured delay trace.
- [ ] Report the same deterministic metrics for `off`, `native`, and `trained_prefix`:
  target-timestep coverage, rejection/hold rate, first-executable-row action error,
  cross-chunk position and SO(3) seam, gripper seam, per-step velocity, and second finite
  difference. Use the same fixed noise-seed sequence and delay trace for every mode, repeat
  enough seeds to expose stochastic strategy changes, and save JSON plus a concise text
  summary.
- [ ] Add checkpoint provenance to server metadata and benchmark outputs, including whether
  training-time prefix conditioning is present and its maximum delay.

**Acceptance:** the manifest is sufficient to reconstruct the exact corpus and command,
and the benchmark refuses an unsupported mode/checkpoint combination rather than silently
falling back.

### Phase 5 — Freeze the corpus and run the fresh finetune

This phase waits until the current data-collection campaign reaches an explicit cutoff. Do
not read an actively written `episode.h5` as a completed source episode, and do not stop the
recorder or the current AI server merely to begin training.

- [ ] Record the agreed source cutoff timestamp and the included `flexiv_*` subsets.
- [ ] Preview new work:

  ```bash
  uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
    sync --config configs/embodiments/semihumanoid.yaml --dry-run
  ```

- [ ] After recording has stopped cleanly for the included episodes, append them, regenerate
  statistics, and run the full converted-output check:

  ```bash
  uv run --no-sync --with h5py python scripts/lerobot_conversion/run_zdata_pipeline.py \
    sync --config configs/embodiments/semihumanoid.yaml --workers 4
  uv run python scripts/lerobot_conversion/run_zdata_pipeline.py \
    stats --config configs/embodiments/semihumanoid.yaml --jobs 4
  uv run python scripts/lerobot_conversion/run_zdata_pipeline.py \
    check --config configs/embodiments/semihumanoid.yaml --full
  ```

- [ ] Re-run the checkpoint latency benchmark on the intended training/inference GPU and set
  `rtc_training_max_prefix_steps` from the measured p99 steps plus the reviewed safety
  margin, bounded so at least 16 postfix steps remain. Record the final value in the YAML and
  run manifest.
- [ ] Run a short one-GPU training smoke against the full multi-dataset mixture. Require
  finite loss, nonzero postfix loss, varying sampled prefix lengths including zero, no NaN,
  and a saved config that contains the requested RTC training field.
- [ ] Confirm all intended training GPUs are free or explicitly allocated. The existing data
  collection/inference workload is not to be killed or preempted implicitly.
- [ ] Launch a genuinely fresh run from `nvidia/GR00T-N1.7-3B`, not from the current
  semihumanoid checkpoint:

  ```bash
  uv run python scripts/lerobot_conversion/run_zdata_pipeline.py \
    train --config configs/embodiments/semihumanoid.yaml --jobs 4
  ```

- [ ] Monitor loss, gradient norm, throughput, GPU memory, sampled prefix histogram, and
  postfix valid-element count. Stop on non-finite values, collapsed postfix coverage, or
  repeated data-loader failures.
- [ ] Validate every saved candidate structurally, then run ordinary open-loop evaluation
  and RTC sequential replay on the held-out `_val` datasets.
- [ ] Select the checkpoint from held-out action accuracy plus RTC continuity and rejection
  metrics. Do not select by training loss or recency alone.

**Gate:** a new checkpoint is not deployment-ready merely because training completed. It
must load from its own saved processor, report training-time prefix support, generate finite
actions in all declared modes, and pass held-out evaluation.

### Phase 6 — Software validation and no-motion shadow serving

- [ ] Run the focused CPU checks:

  ```bash
  uv run pytest \
    tests/gr00t/model/test_action_head.py \
    tests/gr00t/policy/test_gr00t_policy.py \
    tests/gr00t/policy/test_industrialnext_adapter.py \
    tests/gr00t/policy/test_industrialnext_async_server.py \
    tests/gr00t/policy/test_industrialnext_protocol.py \
    -m "not gpu" -q
  ```

- [ ] Run Ruff/format checks on changed Python files, `git diff --check`, and the repository's
  pre-commit suite before publication.
- [ ] With an explicitly allocated GPU, run the checkpoint-backed smoke test for the selected
  checkpoint in `off`, `native`, and `trained_prefix` modes. Verify unsupported-mode gates
  separately with the old checkpoint.
- [ ] Launch the selected server on loopback with explicit RTC arguments and no robot command
  path. The exact values come from the recorded benchmark; an illustrative command is:

  ```bash
  uv run python gr00t/eval/run_gr00t_industrialnext_server.py \
    --model-path /absolute/path/to/selected-checkpoint \
    --task-catalog-path /absolute/path/to/task_catalog.yaml \
    --embodiment-tag new_embodiment \
    --device cuda \
    --host 127.0.0.1 \
    --port 10012 \
    --control-hz 50 \
    --rtc-mode trained_prefix \
    --rtc-initial-frozen-steps MEASURED \
    --rtc-delay-margin-steps REVIEWED \
    --rtc-max-prefix-steps TRAINED_MAX \
    --rtc-min-new-tail-steps REVIEWED \
    --min-usable-action-steps REVIEWED
  ```

  Use an alternate loopback port for standalone smoke tests if `10012` is occupied. Do not
  stop or replace the current `industrialnext_ai` service merely to claim this gate. Port
  `10012` is used for the real ROS shadow only in an explicitly coordinated service window.

- [ ] Run the real ROS policy client only with `inference_mode=true`. Confirm from effective
  runtime parameters and observed topics that no arm or gripper command is published.
- [ ] Shadow each candidate mode against the same observation/task sequence. Record at least
  several hundred steady-state inferences and retain server plus client logs.
- [ ] Require zero protocol/reconnect errors, zero non-finite actions, zero prefix invariant
  failures, zero cross-session rows, bounded image staleness, and delay-estimator coverage of
  the observed maximum. Compare seam/velocity metrics against the expert data distribution
  and the current `off` baseline.

**Gate:** `trained_prefix` becomes the motion candidate only if it preserves held-out action
accuracy and materially improves continuity without increasing holds/rejections. Otherwise
select `native` if it passes, or stop with no motion if neither does.

### Phase 7 — Separately authorized real-robot trial

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
| Active data collection produces a moving corpus | Irreproducible or partial training input | Explicit cutoff, complete-episode sync, full check, and corpus manifest |
| Fresh data changes task/subset balance | Model quality regresses despite more episodes | Report per-task/subset counts and use held-out per-task evaluation |
| GPU contention with the current serving/data job | Latency distortion, OOM, or operational disruption | Read-only occupancy check and explicit allocation; never kill/preempt implicitly |
| Progress never completes the task | Rollout runs until max frames | Keep progress zero by design and require manual stop for every trial |

## Implementation and publication order

1. Land Phases 1-4 with focused tests and no live-system mutation.
2. Re-run the baseline benchmark and select the training prefix bound.
3. Freeze/validate the corpus and launch the fresh base-model finetune.
4. Select a checkpoint through held-out and RTC replay evidence.
5. Publish the Isaac-GR00T changes with exact-file staging and verified remote state.
6. Run the no-motion ROS shadow against the published revision.
7. Make the separate, reviewed Flexiv Ube clamp/config change.
8. Obtain explicit motion approval and conduct the limited robot trial.

## Completion criteria

This plan is complete only when:

1. Old checkpoints retain working `off` and `native` paths, and the new checkpoint advertises
   and runs `trained_prefix` explicitly.
2. Physical action-prefix conversion is correct across relative EEF normalization, rot6d
   conventions, horizon-indexed statistics, and decode.
3. The server predicts and verifies delay, preserves a contiguous action timeline, rejects
   unsafe results atomically, and remains non-blocking at 50 Hz.
4. A fresh base-model finetune is tied to a frozen, checked corpus and complete provenance.
5. The selected checkpoint passes held-out open-loop evaluation, sequential RTC replay,
   checkpoint-backed inference, and real ROS no-motion shadow.
6. Any robot motion occurs only under separate explicit authorization with the existing
   clamp, trust gates, staffed manual stop, and short-to-long rollout progression.
7. Trial logs identify the exact repository revisions, checkpoint hashes, data manifest,
   server settings, effective ROS parameters, and observed safety events.

## References

- [NVIDIA GR00T real-world deployment guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/real_world_deployment.md)
- [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
- [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964)
