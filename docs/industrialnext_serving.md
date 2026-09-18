# GR00T N1.7 serving through Industrial Next WebSocket

This reference connects a checkpoint from the
[internal-data training pipeline](industrialnext_training.md) to the existing Industrial
Next async policy client. Use the same reviewed embodiment layout for offline and online
data. Model loading and action decoding remain in `Gr00tPolicy`; the adapter owns wire
mapping and action timing, while ROS/controllers own command publication and robot safety.

```text
ROS async client → DirectClient / WebSocket → IndustrialNextAsyncServer
  → profile mapping → Gr00tPolicy + saved processor → physical action chunk
  → target-timestep timeline → one wire action or null per client step
```

This implements the core Industrial Next async contract, not every DEFT serving feature.
It does not import DEFT, ROS, or robot publishers. The integration remains experimental;
software checks and synthetic inference do not establish deployment or motion readiness.

## Prepare the checkpoint and profile

Use the environment setup in the [training guide](industrialnext_training.md#environment-and-entry-points).
The RPC package is the pinned `packages/industrialnext_rpc` submodule and the
`industrialnext` extra. Keep the GR00T GPU environment separate from the ROS environment.
Commands below use dGPU `uv run`; use the activated platform interpreter on Jetson/Spark.

Set `serving.model_path` in the embodiment YAML to the chosen checkpoint, or override it
with `--model-path`. Use the override when the training YAML is bound by a frozen manifest.
The model needs complete weights/config and its saved processor, normalization, embodiment
mapping, and referenced tokenizer/image assets. `Gr00tPolicy` loads `processor/` when the
root lacks `processor_config.json`; do not supply a separately invented serving normalizer.

The [profile loader](../gr00t/policy/industrialnext/profile_config.py) derives:

| Config | Online contract |
|---|---|
| `cameras` / `serving.cameras` | Model video key → source camera name; optional serving override supplies wire names with identical model keys/order |
| `state` | Ordered wire fields assembled into model state keys |
| `action.keys` | Decoded model keys split into ordered wire action fields |
| `action.horizon`, `action.observation_offset` | Chunk length and target-timestep offset |
| `tasks.text_overrides` | Accepted task UUIDs and exact instruction text |
| `serving.field_lengths` | Exact widths covering every state/action wire field |
| `serving.image_size` | Required decoded `[height, width]` |
| `serving.ignored_observation_keys` | Explicitly ignored extra fields, including listed depth streams |
| `serving.field_units`, `serving.eef_frame` | Descriptive contract metadata, not unit/frame conversion |
| Other `serving` fields | Device/address, selected/supported RTC modes and gripper monitoring fields |

`eef_frame: deployment_configured` does not transform coordinates or prove frame agreement.
Match the recording's frame, units, gripper direction/range, hand ordering, camera identity,
and image crop to the live producer. Relative paths in the YAML resolve from the repository;
`~` expands normally. A CLI model-path override resolves from the current working directory.

The current async server enforces **50 Hz**, even though a control-rate option is exposed.
The supplied profiles use horizon 40, one current state/image observation, and RGB only.
The semihumanoid profile supports all three RTC modes; the xArm profile permits only `off`.
Adding a mode to a profile is not evidence that its robot/action semantics have been validated.

## Start and smoke-test

From the repository root, after reviewing the YAML paths:

```bash
CFG=configs/embodiments/semihumanoid.yaml
MODEL=/absolute/path/to/selected/checkpoint

CUDA_VISIBLE_DEVICES=0 uv run --no-sync --extra industrialnext python \
  gr00t/eval/run_gr00t_industrialnext_server.py \
  --config "$CFG" --model-path "$MODEL" --rtc-mode off
```

The [entry point](../gr00t/eval/run_gr00t_industrialnext_server.py) loads the model with
`strict=True`, asserts the saved modalities/action representations against the profile,
and runs finite synthetic inference before binding. It uses one dedicated model worker.
The example address is `127.0.0.1:10012`; select a free alternate port for standalone tests
and pass the same `--port` to the client.

With the server running in another terminal:

```bash
uv run --no-sync --extra industrialnext python \
  gr00t/eval/smoke_industrialnext_loopback.py --config "$CFG"
```

This uses the real transport with synthetic observations and writes a unique JSON report
under `/tmp` (`--output-json-path` overrides it). It registers a session: run it against
an isolated test server, because valid registration replaces the active session.

For recorded xArm inputs, use the embodiment-specific example:

```bash
uv run --no-sync --extra industrialnext python \
  examples/xarm_psyonic_mori_bracket/industrialnext_client.py \
  --config configs/embodiments/xarm_psyonic_mori_bracket.yaml \
  --dataset-path /absolute/path/to/converted/xarm_psyonic_val
```

This example's recorded-data mapping is xArm-specific. It is not a generic replay client
for every YAML. Neither client publishes ROS commands.

The DirectServer protocol uses **Python pickle over binary WebSocket**, not JSON or
MsgPack. Bind to loopback by default. Non-loopback binding requires
`--allow-unsafe-non-loopback` and a trusted network boundary; the override adds no
authentication or encryption. Do not expose an unpickling endpoint to untrusted clients.

## Client lifecycle and wire payloads

Use `industrialnext_rpc.direct.client.DirectClient` or the compatible ROS client.
Its metadata request validates the declared request/response schemas.

| Request | Required payload | Response |
|---|---|---|
| `get_metadata` | `type` | Server, service and request/response metadata |
| `register_session` | `type`, `control_hz`, `task_uuid`, `task_text` | `session_id`, `metadata`, `error` |
| `step` | `type`, `session_id`, `observation` | One `action` or `None`, timeline/queue/inference/monitoring fields, `error` |
| `close_session` | `type`, `session_id` | `ok`, `error` |

Registration requires the exact configured UUID/text pair and 50 Hz. Metadata advertises
async protocol version 2, `error_envelope_v2`, `monitoring_in_step`, and
`server_owned_gripper_snap`. Gripper snap is explicitly disabled; native hand/gripper coordinates blend continuously.
Successful responses omit `error`; clients must use `response.get("error")`.
Compare the advertised task catalog, dimensions, fields, and capabilities with the actual
deployment client before connecting its control loop.

There is one active session. A valid replacement registration clears prior image caches,
timeline, history and task state. Old requests return `session_not_found`; late worker
results cannot cross session generations. The idle timeout defaults to 300 seconds.
A second diagnostic client can displace the first client, so use one session owner.

Every `step.observation` supplies all configured state fields as finite numeric sequences
of the exact declared lengths. Optional repeated `task_uuid`/`task_text` must match the
registered task. Send each updated RGB view as JPEG bytes plus matching metadata:

```python
observation[wire_camera_name] = jpeg_bytes
observation.setdefault("images_meta", {})[wire_camera_name] = {
    "format": "jpeg",
    "dtype": "uint8",
    "channels": 3,
    "height": configured_height,
    "width": configured_width,
    "quality": 90,  # optional, must be 1..100 if supplied
}
response = client.request({
    "type": "step",
    "session_id": session_id,
    "observation": observation,
})
if response.get("error"):
    raise RuntimeError(response["error"])
action = response["action"]  # None is a supported response, not a zero command.
```

The snippet assumes an already connected/registered client and a complete state observation.
The [recorded client](../examples/xarm_psyonic_mori_bracket/industrialnext_client.py)
contains the complete connect/register/paced-step/close sequence and RGB-to-JPEG encoding.

Omit unchanged images and their metadata to reuse the per-session cache; do not send null
image payloads. All required cameras must have arrived and remain fresh before inference.
The default age limit is five accepted control steps (100 ms at 50 Hz), measured since
receipt, not sensor capture time. Live producer synchronization/freshness remains important.
Only explicitly configured ignored fields are accepted as extras; arbitrary depth names
are not automatically ignored.

JPEG decoding occurs in the model worker, converts OpenCV BGR to RGB `uint8`, and requires
the configured dimensions. The adapter does not add a deployment crop or resize; the saved
processor still performs its own model preprocessing.

## Representation and action timeline

Model video arrays are `(1, 1, H, W, 3)`, states are `(1, 1, D)`, and language is
`{"annotation.human.task_description": [[task_text]]}`. The profile assembles state
fields and converts source-column rot6d to GR00T-row rot6d using the same
[helpers](../gr00t/data/state_action/rot6d.py) as offline conversion.

`Gr00tPolicy` returns decoded physical action arrays `(1, horizon, D)`.
For the relative EEF examples, these are already absolute poses: **do not add the live
state again**. The adapter converts rotations back to source columns and returns flat
wire-field lists. F/T inputs and gripper/hand values are not automatically rescaled into
different units. No scalar-gripper clipping or snapping is added. The default ensemble and transition
blend native hand coordinates continuously in physical units.

Keep the training target offset `D = action.observation_offset` separate from serving
lookahead `K = --action-offset`. Discard original model rows `i < K`, then schedule:

```text
execution_tick = source_tick + D + i - K
first executable original row = K + max(0, last_consumed_tick + 1 - source_tick - D)
```

For source tick 100, consumed tick 104, D=0 and K=2, model row 7 contributes to tick
105. Offset applies to the first chunk too. The 40-row horizon leaves 38 selectable
rows before latency trimming. It never changes the dataset's target offset or playback
speed; speed remains 1× and output remains 50 Hz.

The output order is physical decoding, offset/age admission, same-tick ensemble,
transition smoothing, final command checks, then wire output and served history.
Defaults are `temporal_exponential`, coefficient 0.1, newest three contributions per
tick, and four transition frames. Temporal weights are proportional to
`exp(-coefficient * (newest_source_receipt - source_receipt) * 50)`. Positions and
native hand coordinates use weighted means; rotations use a quaternion scatter mean.
Transitions use SO(3) interpolation and cubic smoothstep weights
0.15625, 0.5, 0.84375, 1.0 from the previously planned command. This is chunk-transition
smoothing, not DEFT's startup/human-handover ramp. No output EMA is applied.

`--ensemble-strategy latest_only` disables ensemble; `--chunk-transition-frames 0`
disables transition smoothing independently. Profile values use the corresponding
`serving` keys; CLI values take precedence. Old profiles omit these keys and receive
the new ensemble/smoothing defaults, while their execution offset stays 0. The Taro
launch below explicitly selects offset 2 without changing frozen training YAML.

One inference worker and one replaceable pending observation keep work bounded.
Expiry occurs before admission and emission. Image freshness uses both accepted ticks
and server-monotonic receipt time (five steps / 0.1 s by default); zero permitted reuse
requires a same-tick image and allows receipt within one control period. This is not a
sensor capture-time guarantee. Each contribution expires at its mapped due time plus
`--max-action-lateness-s` (0.04 s). Transition anchors inherit the earliest source
deadline; they cannot keep expired predictions executable.

Pace requests at 50 Hz. `--max-control-clock-drift-s` defaults to 0.1 s relative to the
first valid request; exceeding it ends the session instead of renumbering ticks.
These timing values are engineering defaults requiring deployment jitter measurement.
A temporary null can hold the prior ROS command; after emission begins, a command gap
beyond the next expected period plus the lateness allowance becomes terminal.

Irrecoverable failures return `error: "session_unusable"`, a bounded `reason`, and no
action. The paired ROS client stops the matching run under its run-generation lock,
clears held commands/slowdown state and retires the connection. A delayed old-run error
cannot stop a new run. Explicit operator new-run recovery is required; this error does
not trigger automatic re-registration. Transport loss is a separate client lifecycle.

Monitoring separates raw-chunk dynamics from actual emitted seam/dynamics, and includes
contributor source ticks, original rows, weights, transition anchors, image ages,
source-to-result latency and usable tail. Optional command limits apply to the actual
postprocessed sequence as well as raw chunk admission. This is a served-command check,
not a measured robot-pose or actuator-acknowledgement guarantee.

Monitoring reports `progress=0.0`, classification `unknown`, confidence zero and a fresh
monitoring timestep. `scene_valid=true` in this compatibility response is not a learned
scene assessment. GR00T provides no task-completion signal here; use the deployment's
manual-stop lifecycle rather than waiting for progress to reach a threshold.

## RTC modes and timing

| Mode | Requirement | Behavior |
|---|---|---|
| `off` | Compatible saved model/profile | Offset, ensemble and transition recipe; default |
| `native` | Compatible N1.7 model and profile support | Prior-prefix seeding, frozen prefix, velocity ramp; no prefix-training requirement |
| `trained_prefix` | Saved positive `rtc_training_max_prefix_steps` and profile support | Hard prefix with training-time tokenwise conditioning |

RTC requires `--action-offset 0 --ensemble-strategy latest_only
--chunk-transition-frames 0`. Other combinations are rejected before model loading.
Existing RTC launch commands must add these flags explicitly. The Taro profiles advertise
only `off`; native RTC needs a separately qualified serving profile. Current Taro
checkpoints have no trained-prefix support.

The modes are alternatives, not stacked stages. Native RTC is the repository's prefix/ramp
mechanism, not the full gradient-guided RTC algorithm. The first inference after registration
bootstraps with `off`; subsequent RTC requests require contiguous prior actions.

Prefixes come from exact served history plus the future timeline. The policy re-anchors
physical prefixes against the new observation and normalizes at the new chunk's horizon
indices without percentile clipping. Hard physical prefix rows are restored after decoding
to avoid BF16 drift. New chunks must satisfy prefix, delay, shape, finite-value, and usable
tail checks before admission.

The runtime measures committed steps as
`max(0, completion_timestep - (source_timestep + offset) + 1)`. It predicts a frozen
length from the rolling maximum plus margin, bounded below by the bootstrap setting.
**It does not clamp an excessive prediction down to the allowed maximum.** An out-of-range
requirement or underestimated delay rejects the refresh. If continuity drains, RTC requires
explicit re-registration instead of silently switching to unconditioned generation.

Important CLI defaults (the entry point, not every internal dataclass):

| Option | Default |
|---|---:|
| `--max-image-staleness-steps` | 5 |
| `--min-usable-action-steps` | 1 |
| `--rtc-initial-frozen-steps` | 1 |
| `--rtc-delay-window-size` | 20 |
| `--rtc-delay-margin-steps` | 1 |
| `--rtc-max-prefix-steps` | 12 |
| `--rtc-native-overlap-steps` | 12 |
| `--rtc-min-new-tail-steps` | 16 |
| `--rtc-ramp-rate` | 6.0 |

Select delay/tail settings from representative deployment latency, not historical GPU
measurements. The one-row minimum is an experimental default, not a qualified motion budget.
Runtime maximum prefix must fit the checkpoint's trained maximum for `trained_prefix`, and
prefix/overlap bounds must leave the configured new tail. Optional position, orientation,
gripper step and second-difference limits reject chunks; they do not modify predictions.

The benchmark is profile-driven and uses the production server for sequential replay:

```bash
.venv/bin/python gr00t/eval/benchmark_industrialnext_rtc.py \
  --profile-config configs/embodiments/taro_exp_100.yaml \
  --model-path "$MODEL" --output-dir outputs/gr00t/serving_validation/UNIQUE_RUN \
  --dataset-path data/training_data/gr00t/taro_exp_full --modes off
```

The full conversion root selects its three explicit `*_val` datasets, totaling 81
shared holdout episodes in the frozen preparation. Use this same root for both model
variants. All episodes are selected by default; `--trajectory-ids 0 1 2` limits each
subset to those indices for a shorter check. A direct LeRobot dataset root is also accepted.
Each selected episode replays its first 100 data frames by default; set `--replay-steps`
to cover the desired evaluation window. The delay trace uses integer 50 Hz periods.
Use a new output directory, a completed checkpoint, and an available GPU. Saved checkpoint
processors/statistics remain authoritative; do not refit on holdout. Defaults run six
matched recipes: unprocessed control, offset only, ensemble only, transition only,
postprocessing with offset 0, and the default offset 2 recipe. `--no-run-ablations`
selects only the explicit settings. RTC experiments additionally require the disable
flags above and a profile/checkpoint advertising that mode.

Reports retain masks and distinguish EEF metres, geodesic orientation radians and native
hand error. Nominal lookahead-target error and same-time error are separate. Per-step output
also compares the final command with each contributor's selected original dataset
target, preserving source-to-data indices across masked observation gaps. Output
records identify original rows, contributors, transitions, coverage, nulls and terminal
failures. Sequential replay uses a virtual delay trace; it does not measure deployment
latency or closed-loop success. Fixed-input model latency is reported separately.

The transport smoke can also load the actual ROS RPC client, without ROS node imports:

```bash
.venv/bin/python gr00t/eval/smoke_industrialnext_loopback.py --config "$CFG" \
  --ros-client-source-dir /home/azureuser/industrialnext_ros2/src/industrialnext_operator_ros2/industrialnext_operator_policy_client \
  --output-json-path outputs/gr00t/serving_validation/UNIQUE_ROS_SMOKE.json
```

It records request RTT and server monitoring for metadata/register/step/close exchanges.
It does not qualify the ROS node's publication lifecycle.

## Deployment validation and handoff

1. Check source-to-converted samples, held-out accuracy, complete checkpoint/processor
   loading, saved modality agreement, and finite predictions.
2. Run isolated WebSocket loopback and recorded-input checks. Measure warm and steady
   inference latency separately from step round-trip latency. At 50 Hz the step budget
   is 20 ms; inference must leave a useful portion of the 40-step/0.8-second horizon.
3. On the deployment machine, verify the actual ROS client's metadata/catalog checks,
   reconnect behavior, null-action handling, required fields, image crop/identity,
   frames/units, and checkpoint/RTC settings. Run with command publication disabled and
   verify the effective parameters and observed topics, not just the intended YAML.
4. Use the established controller/operator process for supervised motion only after the
   no-motion evidence passes. Workspace/collision/velocity limits, state trust, emergency
   stop, command ownership and the final hold remain outside this model server.

Record the exact checkpoint and processor identity, repository/RPC revisions, YAML and CLI
overrides, evaluation reports, target hardware/load, and effective ROS settings. The normal
server advertises lightweight model-path provenance; it does not generate or verify a full
deployment bundle or checkpoint hash inventory. Package/verify those separately when needed.

For rollback, stop the active policy through its existing lifecycle, verify command
publication has stopped, then restore the recorded prior server/config. Do not switch model
processes beneath an active rollout or reuse a historical DEFT command without checking the
current deployment. A disconnect can cause a controller hold and reconnect; it is not an
automatic successful task termination.

Historical evidence remains under [candidate selection](../artifacts/semihumanoid_rtc_candidate_selection_20260820/README.md)
and [RTC handoff](../artifacts/semihumanoid_rtc_handoff_20260820_3496/).
Those records include older invocations and source patches; use current commands above.
The historical work reached GPU/deployment loopback checks but left real ROS shadow and
motion unqualified. Re-measure on the intended deployment rather than treating those
records as current approval or availability evidence.

## Model-only debugging and focused tests

The stock [run_gr00t_server.py](../gr00t/eval/run_gr00t_server.py) uses ZeroMQ with
MsgPack/NumPy and returns whole chunks. It has no Industrial Next session/timeline API.
Use it for model-only debugging, or call `Gr00tPolicy` in process; the
[policy guide](../getting_started/policy.md) describes that interface. Model serving loads
the saved modality config; the stock server's `--modality-config-path` applies to replay.

For changes to the integration, run the existing CPU contract tests in a prepared dev
environment:

```bash
uv run --no-sync --extra industrialnext python -m pytest \
  tests/gr00t/policy/test_industrialnext_profile_config.py \
  tests/gr00t/policy/test_industrialnext_adapter.py \
  tests/gr00t/policy/test_industrialnext_async_server.py \
  tests/gr00t/policy/test_industrialnext_protocol.py -q
```

Real-checkpoint tests and ROS trials are separate from these fake-policy CPU checks.

| Symptom | Check |
|---|---|
| Connection fails before readiness | Model path, processor/cache assets, saved profile contract, free port |
| Registration fails | Exact task UUID/text and 50 Hz; compare metadata with deployment catalog |
| `session_not_found` | Replacement/expiry; reconnect and register, do not reuse the old ID |
| Missing/stale RGB or repeated null actions | Required image updates, JPEG metadata/dimensions, age limits and worker errors |
| Incorrect orientation or large EEF jump | Rot6d convention, frame/unit agreement, accidental double composition |
| RTC `prefix_out_of_range` or delay rejection | Deployment latency/contention, measured prefix bound and checkpoint capability |
| RTC requires re-registration | Continuity was lost; inspect the failure before restarting the session |
| No automatic task completion | Expected: progress is fixed at zero |

Implementation owners: [profile_config.py](../gr00t/policy/industrialnext/profile_config.py),
[async_server.py](../gr00t/policy/industrialnext/async_server.py),
[Gr00tPolicy](../gr00t/policy/gr00t_policy.py), and the
[server entry point](../gr00t/eval/run_gr00t_industrialnext_server.py).

## Serving the Taro ablation checkpoints

Use the same variant YAML used for [training](industrialnext_training.md#taro-rgb-v5-fixed-compute-ablation)
and pass the exact run checkpoint explicitly. There is no shared `latest` pointer.
The small/full configs default to separate ports 10012/10013. Both use head and two
fisheye wrist RGB streams; `serving.cameras` maps these live names onto the same
three model keys used by conversion. External RGB/depth are explicitly ignored.

```bash
source .venv/bin/activate
export HF_HUB_CACHE="$PWD/outputs/gr00t/huggingface/hub"
: "${MIN_USABLE_ACTION_STEPS:?Set from measured p99 source-to-result latency}"
: "${SERVING_GPU:?Set a GPU available for serving}"
CUDA_VISIBLE_DEVICES="$SERVING_GPU" python gr00t/eval/run_gr00t_industrialnext_server.py \
  --config configs/embodiments/taro_exp_100.yaml \
  --model-path outputs/gr00t/EXACT_TARO_100_RUN/checkpoint-40000 \
  --action-offset 2 --rtc-mode off \
  --ensemble-strategy temporal_exponential --ensemble-coeff 0.1 \
  --max-ensemble-chunks 3 --chunk-transition-frames 4 \
  --min-usable-action-steps "$MIN_USABLE_ACTION_STEPS"
```

The required low-dimensional observations are left/right EEF position (3) and
source-column rot6d (6), plus the native 20-coordinate `right_hand`, in the order
bound by `outputs/gr00t/preparation/taro_split.json`. Responses contain right EEF
and right-hand commands only. Gripper snapping and RTC are disabled. The server
expects 256×256 JPEG RGB with matching metadata at the 50 Hz control contract.
This recipe enables DEFT-style temporal ensemble and chunk transitions, with execution
offset 2 and fixed 1× speed. Size the usable-tail requirement to at least
`ceil(p99_source_to_result_seconds * 50) + 1` and demonstrate that the remaining horizon
covers it. Do not infer this budget from training throughput.

The paired ROS client derives commanded components from `action_fields`, while retaining
required bilateral observations. Cold-start hand initialization uses the same declared
command scope. Use `inference_mode=true` for a publication-disabled
shadow run. For a strict GR00T profile, set its ROS parameter `expected_eef_frame` to the
locally verified semantic frame contract (`per_arm_flexiv_base` for these profiles), and
configure real per-arm TF base frames; measured pose frame IDs must match those bases.
This parameter is an operator declaration, not an automatic transform or calibration.
Pass it in the existing ROS parameter YAML (`config_file`); no new launch wrapper is needed.

Validate native hand joint order and camera ROI/crop against the checkpoint-linked
preparation and actual robot configuration. Width 20 alone does not prove joint order.
The checked-in ROS 10-DOF/bottom-camera Taro configuration is incompatible with this
20-coordinate/fisheye contract; do not alias it into compatibility. Record the exact
paired client revision and validation reports before motion. Frozen training configs,
normalizers and data remain unchanged.

Rollback: explicitly stop, then select offset 0, `latest_only`, zero transition frames,
RTC off and the prior qualified checkpoint/client. Clear sessions and cached inputs.
