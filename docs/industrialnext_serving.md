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
`server_owned_gripper_snap`. Gripper snap is explicitly disabled; values pass through.
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
if response["error"]:
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
different units. No server-side gripper clipping or smoothing is added.

For observation timestep `s`, row `i` targets:

```text
target_timestep = s + action.observation_offset + i
```

One background inference and at most one replaceable pending snapshot keep work bounded.
`step` continues responding while inference runs. Pending observations are rechecked for
freshness before launch. Valid new chunks replace the older unserved future timeline;
expired rows are dropped rather than replayed from row zero. Each accepted step advances
the session clock and returns only that tick's action. Pace requests at 50 Hz: the timeline
is indexed by accepted steps, not by an independent wall-clock robot scheduler.

Missing/stale images prevent a new inference snapshot; previously admitted, unexpired
timeline actions can still be returned. Malformed observations return errors. Inference
or output failures admit no new chunk; existing valid future rows may remain. An empty
timeline returns `None`. ROS owns the resulting no-publication/hold behavior; the server
does not fabricate a hold command or loop an old trajectory.

Monitoring reports `progress=0.0`, classification `unknown`, confidence zero and a fresh
monitoring timestep. `scene_valid=true` in this compatibility response is not a learned
scene assessment. GR00T provides no task-completion signal here; use the deployment's
manual-stop lifecycle rather than waiting for progress to reach a threshold.

## RTC modes and timing

| Mode | Requirement | Behavior |
|---|---|---|
| `off` | Compatible saved model/profile | Independent chunks with age-correct replacement |
| `native` | Compatible N1.7 model and profile support | Prior-prefix seeding, frozen prefix, velocity ramp; no prefix-training requirement |
| `trained_prefix` | Saved positive `rtc_training_max_prefix_steps` and profile support | Hard prefix with training-time tokenwise conditioning |

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

For the original semihumanoid contract, a checkpoint benchmark supports latency and
held-out sequential replay:

```bash
uv run --no-sync --extra industrialnext python \
  gr00t/eval/benchmark_industrialnext_rtc.py \
  --model-path "$MODEL" --output-dir /tmp/gr00t-rtc-benchmark \
  --dataset-path /absolute/path/to/converted/ube_val \
  --trajectory-ids 0 1 2 --modes off native trained_prefix
```

Choose a new output directory for each benchmark run; an existing directory is rejected.
This benchmark uses semihumanoid helpers and is **not config-driven for arbitrary
embodiments**. Unsupported checkpoint modes are reported. Compare the same seeds/delay
trace across modes and inspect accuracy, coverage, holds/rejections, pose seams and finite
differences. Profile-driven loopback remains the generic transport smoke. Startup warmup
uses ordinary inference; it does not validate all configured RTC refresh behavior.

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
python gr00t/eval/run_gr00t_industrialnext_server.py \
  --config configs/embodiments/taro_exp_100.yaml \
  --model-path outputs/gr00t/EXACT_TARO_100_RUN/checkpoint-40000
```

The required low-dimensional observations are left/right EEF position (3) and
source-column rot6d (6), plus the native 20-coordinate `right_hand`, in the order
bound by `outputs/gr00t/preparation/taro_split.json`. Responses contain right EEF
and right-hand commands only. Gripper snapping and RTC are disabled. The server
expects 256×256 JPEG RGB with matching metadata at the 50 Hz control contract.
Do not copy DEFT's temporal ensemble, fixed action selection offset, or playback
speed into this baseline. Validate the real deployment's field names, hand order,
pose frames, image freshness, and latency before robot use.
