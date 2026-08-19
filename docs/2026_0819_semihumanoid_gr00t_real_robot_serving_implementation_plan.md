# Semihumanoid GR00T N1.7 Real-Robot Serving Implementation Plan

**Date:** 2026-08-19

**Status:** Software implementation complete — live-robot rollout gates deferred while the
robot is in active data collection

**Selected direction:** Add an Industrial Next-compatible async server to Isaac-GR00T and
keep the existing ROS2 `policy_control_node_async.py` unchanged.

**Related design:**
[`2026_0819_semihumanoid_gr00t_real_robot_serving_design.md`](2026_0819_semihumanoid_gr00t_real_robot_serving_design.md)

## Motivation

The finetuned semihumanoid GR00T N1.7 model is now fully downloaded, but the stock GR00T
ZeroMQ server does not implement the session, sparse-image, asynchronous inference, action
timing, task-catalog, or monitoring contract consumed by the robot's existing async policy
client.

The target is a new Isaac-GR00T entry point that:

- constructs the repository's unchanged `Gr00tPolicy(strict=True)`;
- serves the binary WebSocket protocol through the shared `industrialnext_rpc` direct
  server;
- accepts the current named robot observation without changes to the ROS client;
- performs the semihumanoid field, image, and rot6d conversions next to the model;
- runs GPU inference outside the WebSocket request path;
- age-corrects each returned 40-step trajectory and returns at most one current absolute
  action per 50 Hz `step` request;
- reports fresh monitoring with `progress = 0.0`; and
- leaves manual stop, command publication, final hold, and robot safety gates on the ROS2
  side.

### Verified baseline

- `outputs/gr00t/semihumanoid_20260819_080107/` now contains the root final model, its two
  indexed safetensor shards, and `processor/{processor_config,statistics,embodiment_id}.json`.
- `checkpoint-5251/` also contains its two indexed shards and checkpoint-local processor
  metadata. Both model indexes parse and reference present, non-empty shard files.
- The saved `new_embodiment` contract still has three video keys, six state keys, four
  action keys, one observation step, and action deltas `0..39`.
- `industrialnext_ai` currently pins `packages/industrialnext_rpc` at
  `c3dc583ee36310581ad1ec154559698051988b9f` and exposes it as a `uv` workspace source.
- The ROS client requires async protocol version 2 or newer plus
  `error_envelope_v2`, `monitoring_in_step`, and `server_owned_gripper_snap`.

The checkpoint files were checked structurally while preparing this plan. The later
implementation record documents the completed `Gr00tPolicy` construction, warmup, and
finite checkpoint-backed inference validation.

### Required scope

All product-code changes belong in `/home/ubuntu/Isaac-GR00T`. The implementation will use
the other repositories as follows:

- `/home/ubuntu/industrialnext_ai`: reference implementation for the direct metadata,
  session lifecycle, non-blocking request path, sparse image handling, background
  inference, and shutdown behavior; no runtime import and no source modification.
- `/home/ubuntu/industrialnext_ros2`: unchanged protocol consumer and later shadow/motion
  validation target; no new policy client and no required Python source modification.

The first server is manually launched. Supervisor/autostart integration is deferred until
the protocol, latency, and real observation path have passed shadow validation.

### Non-goals

- Modifying or wrapping the stock `PolicyServer`/`PolicyClient` ZeroMQ protocol.
- Importing `industrialnext_ai` or copying its DEFT model, QP, PTE, temporal ensemble,
  microbatch, upsampling, EMA, or smoothing implementations.
- Reimplementing GR00T preprocessing, relative-action composition, or unnormalization.
- Adding a second ROS policy-control node.
- Consuming depth images, joint positions, or a state history longer than one frame.
- Fabricating automatic completion; progress remains zero and manual stop ends the trial.
- Multiple simultaneous robot sessions in the first implementation.
- Exposing the pickle protocol beyond the trusted loopback boundary.
- Choosing a best training checkpoint. The root final model is the first integration
  candidate; offline policy quality comparison remains separate.

## Implementation decisions

| Concern | Decision |
|---|---|
| RPC dependency | Add `industrialnext_rpc` as `packages/industrialnext_rpc`, using the same URL and pinned commit as `industrialnext_ai` |
| Packaging | Add an `industrialnext` optional dependency and `uv` workspace source, including the AI server's proven `websockets>=16.1.1,<17.0` range; update `uv.lock` |
| Wire transport | Import `Metadata` from `industrialnext_rpc.direct.metadata` and `DirectServer`/`Server` from `industrialnext_rpc.direct.server`; the package-level `industrialnext_rpc.direct` module does not re-export them |
| Protocol profile | Advertise async protocol version 2 and only the three capabilities the existing ROS client requires |
| Bind address | Default to `127.0.0.1:10012`; do not permit an implicit public bind |
| Model path | Require an explicit `--model-path`; use the root final model for the first integration run |
| Model execution | Construct, warm up, and call the policy on one dedicated executor thread; never call inference from `handle_request` |
| Session policy | One active session; a new valid registration atomically invalidates and replaces the prior generation so reconnect does not wait for idle expiry |
| Idle cleanup | Default to the existing AI server's 300-second positive idle-session timeout; replacement remains immediate on a new valid registration |
| Pending work | At most one inference in flight and one replaceable latest pending observation |
| Action timing | Key decoded rows by `source_timestep + delta_index`; drop expired rows and replace the older future timeline with the newest valid chunk |
| Sparse RGB | Cache immutable JPEG bytes and metadata by required view; decode in the worker, not in `step` |
| RGB staleness | Start with a configurable default of 5 control steps (100 ms at 50 Hz); shadow logs must confirm or tighten it |
| Minimum usable tail | Use 1 row only for initial shadow measurement; require an explicit measured value before commands are enabled |
| Grippers | Pass through the decoded absolute GR00T output and advertise snap `enabled: false`; do not silently clip |
| Progress | Return `0.0` with a monotonically fresh `monitoring_timestep` on every successful step |
| Error behavior | Build every handler success and error through one schema-complete response helper; error responses retain safe sentinel values for all declared fields so `DirectServer` preserves the exact `session_not_found` value expected by ROS |
| Shutdown | Stop admission, invalidate the active generation, discard late results, wait for the in-flight call, and close the executor |

## Existing shared modules to reuse

- `gr00t/policy/gr00t_policy.py:70-482` — authoritative model construction, strict nested
  observation/action validation, saved processor invocation, and decoding back to absolute
  EEF trajectories. The new server calls this class rather than duplicating it.
- `gr00t/eval/run_gr00t_server.py:1-180` — command-line conventions for model path,
  embodiment tag, device, host, port, logging, and `tyro` configuration.
- `examples/semihumanoid/semihumanoid_config.py:29-84` — source form of the exact saved
  `new_embodiment` contract used for startup assertions and test fixtures.
- `scripts/lerobot_conversion/zdata_pipeline/source.py:51-76` — already-tested forward
  conversion from Industrial Next first-two-columns rot6d to GR00T first-two-rows rot6d.
  Promote this math into an importable core helper and add the inverse instead of creating
  a second deployment-only implementation.
- `industrialnext_ai/.gitmodules:1-3` and `industrialnext_ai/pyproject.toml:17,105-109` —
  exact submodule URL and `uv` workspace pattern to reproduce in Isaac-GR00T.
- `industrialnext_ai/packages/industrialnext_rpc/python/industrialnext_rpc/direct/server/websocket.py:19-94`
  — the WebSocket context manager, pickle framing, request/response metadata validation,
  and exception envelope. The new handler must remain synchronous and fast because
  `DirectServer` calls it on the asyncio event-loop thread.
- `industrialnext_ai/src/industrialnext_ai/serve/deft/async_server.py:236-448` — direct
  metadata shape and the adaptation from an application response to the unwrapped
  direct-RPC response.
- `industrialnext_ai/src/industrialnext_ai/serve/deft/async_server.py:452-818` — session
  registration, fast `step`, monitoring fields, and close-session response behavior to
  match at the protocol surface.
- `industrialnext_ai/src/industrialnext_ai/serve/deft/async_server.py:988-1306` — reference
  background-inference trigger, immutable snapshot, result admission, and error/timing
  accounting. Reuse the concurrency invariants, not the DEFT-specific queue or model code.
- `industrialnext_ros2/.../async_protocol.py:5-59` — the minimum protocol version and
  required capability names.
- `industrialnext_ros2/.../rpc_client.py:14-157` — the exact metadata and direct-pickle
  validation path used by `RobustDirectClient`.
- `industrialnext_ros2/.../policy_control_node_async.py:1237-1490` — metadata gates,
  gripper-snap validation, exact task-catalog comparison, registration request, reconnect,
  and close behavior.
- `industrialnext_ros2/.../policy_control_node_async.py:1515-1605` — current null-action
  hold behavior and the only response status fields read by the control loop.
- `industrialnext_ros2/.../policy_control_node_async.py:1819-1920` — actual flat state,
  sparse RGB/depth, task, and image-metadata request shape.

The shortened `industrialnext_ros2/...` paths above share this prefix:
`/home/ubuntu/industrialnext_ros2/src/industrialnext_operator_ros2/industrialnext_operator_policy_client/industrialnext_operator_policy_client/`.

## Planned file scope

| Repository | Path | Planned change |
|---|---|---|
| Isaac-GR00T | `.gitmodules` | Register the `industrialnext_rpc` submodule |
| Isaac-GR00T | `packages/industrialnext_rpc` | Gitlink pinned to the verified compatibility commit |
| Isaac-GR00T | `pyproject.toml` | Add the optional serving dependency and workspace source |
| Isaac-GR00T | `uv.lock` | Lock the submodule package and WebSocket dependency |
| Isaac-GR00T | `gr00t/data/state_action/rot6d.py` | Shared row/column convention conversion helpers |
| Isaac-GR00T | `scripts/lerobot_conversion/zdata_pipeline/source.py` | Import the shared forward conversion without behavior change |
| Isaac-GR00T | `gr00t/policy/industrialnext/task_catalog.py` | Strict standalone parser and UUID/text lookup |
| Isaac-GR00T | `gr00t/policy/industrialnext/adapter.py` | Sparse image cache, strict observation mapping, output validation, and robot action mapping |
| Isaac-GR00T | `gr00t/policy/industrialnext/async_server.py` | Direct-RPC metadata, sessions, inference worker, action timeline, monitoring, stats, and shutdown |
| Isaac-GR00T | `gr00t/policy/industrialnext/__init__.py` | Narrow public exports for the integration |
| Isaac-GR00T | `gr00t/eval/run_gr00t_industrialnext_server.py` | Production entry point and startup/warmup flow |
| Isaac-GR00T | `tests/gr00t/data/test_rot6d_conventions.py` | Convention and inverse round-trip coverage |
| Isaac-GR00T | `tests/gr00t/policy/test_industrialnext_adapter.py` | CPU-safe mapping, JPEG, catalog, and output tests |
| Isaac-GR00T | `tests/gr00t/policy/test_industrialnext_async_server.py` | Deterministic session/timeline/failure tests with a fake policy |
| Isaac-GR00T | `tests/gr00t/policy/test_industrialnext_protocol.py` | Real `DirectServer`/`DirectClient` loopback conformance test |
| Isaac-GR00T | `tests/gr00t/policy/test_industrialnext_gr00t_policy_gpu.py` | Opt-in actual-checkpoint construction and one-inference smoke test |

No `industrialnext_ai` file is copied or edited. No ROS Python file is expected to change.
Any later edit to the currently dirty `flexiv_ube/task_config.yaml` must be a separate,
explicitly reviewed deployment change.

## Phased breakdown

### Phase 1: Pin the RPC dependency and centralize rotation conversion

- **Problem:** Isaac-GR00T has no WebSocket/direct-RPC dependency, and the only tested
  Industrial Next-to-GR00T rot6d conversion lives under a conversion script with no inverse.
- **Solution:** Mirror the submodule/workspace pattern from `industrialnext_ai`, keep it in
  an optional serving extra, and promote the existing rotation math into a core module with
  both directions.
- **Impact:** Production code can import the exact shared direct server, and training-time
  conversion and deployment use one tested definition of the convention boundary.

### Phase 2: Implement the pure semihumanoid wire/model adapter

- **Problem:** The ROS request is a flat named state plus sparse JPEG bytes, while
  `Gr00tPolicy` consumes nested batched arrays and returns four absolute trajectories.
- **Solution:** Add a strict task-catalog loader, immutable sparse-image snapshots, exact
  observation mapping, JPEG RGB decoding, saved-contract assertions, decoded-output
  validation, and row-to-column action conversion.
- **Impact:** Model-specific transformations are testable without WebSockets, CUDA, ROS, or
  robot hardware.

### Phase 3: Implement the non-blocking direct-RPC server and action timeline

- **Problem:** A synchronous model call inside `DirectServer.handle_request` would block
  every 50 Hz `step`, and serving action index zero after inference would replay stale
  commands.
- **Solution:** Keep all mutable session state on the asyncio thread, run one model call in
  a dedicated executor, retain only the newest pending snapshot, generation-tag results,
  and schedule rows by absolute target timestep.
- **Impact:** The current ROS client receives a quick action-or-null response on every tick,
  reconnects cleanly, and never receives an expired or cross-session action.

### Phase 4: Add the production entry point and checkpoint-backed validation

- **Problem:** The protocol core still needs a safe startup sequence, CLI, model provenance,
  saved-contract gate, GPU warmup, and clean shutdown.
- **Solution:** Add an explicit Industrial Next server entry point that builds and warms
  `Gr00tPolicy` on the inference worker before binding `DirectServer`, then publishes honest
  metadata and runtime timing.
- **Impact:** A server is only discoverable after the selected checkpoint has loaded,
  passed its contract checks, and completed one finite synthetic inference.

### Phase 5: Prove compatibility and stage the real-robot trial

- **Problem:** Unit tests cannot establish that the real ROS client, live camera identities,
  model latency, action ranges, or safety configuration are suitable for motion.
- **Solution:** Run loopback protocol checks, the ROS catalog checker, an inference-only live
  shadow rollout, and only then a separately authorized supervised low-speed rollout.
- **Impact:** Robot motion remains gated on observed end-to-end evidence rather than model
  load or network connectivity alone.

## Detailed checklist

### Phase 1: Dependency and shared rotation utilities

- [x] **1.1 Add and pin the RPC submodule** (`.gitmodules`, `packages/industrialnext_rpc`).
  - Use `git@github.com:industrialnext/industrialnext_rpc.git`.
  - Initially pin `c3dc583ee36310581ad1ec154559698051988b9f`, matching the currently
    reviewed `industrialnext_ai` server and the ROS client's deployed package behavior.
  - Record the pin in review output; do not track the submodule's `main` branch implicitly.
  - Acceptance: a fresh `git submodule update --init --recursive` checks out the exact pin.

- [x] **1.2 Add the optional dependency surface** (`pyproject.toml`, `uv.lock`).
  - Add an `industrialnext` extra containing `industrialnext_rpc>=0.1.0`, an explicit
    PyYAML dependency for the task catalog, and the same `websockets>=16.1.1,<17.0`
    compatibility range as the reviewed AI server.
  - Add `industrialnext_rpc = { workspace = true }` under `[tool.uv.sources]` and
    `members = ["packages/industrialnext_rpc"]` under `[tool.uv.workspace]`, following the
    AI repository's proven layout.
  - Regenerate the lockfile and confirm that core GR00T imports still work without importing
    the optional server module.
  - Acceptance: the extra imports `industrialnext_rpc.direct.server.DirectServer` from the
    submodule, not an installed stale copy.

- [x] **1.3 Promote and complete rot6d convention helpers**
  (`gr00t/data/state_action/rot6d.py`,
  `scripts/lerobot_conversion/zdata_pipeline/source.py`).
  - Move the existing Gram-Schmidt-safe source-columns-to-GR00T-rows behavior into the core
    helper and add GR00T-rows-to-source-columns.
  - Both functions must accept vectorized arrays, reject wrong width, non-finite inputs, and
    degenerate axes, and reconstruct a proper rotation matrix. Preserve the converter's
    existing float64 result; cast to `float32` only when the adapter builds model tensors.
  - Keep the converter's output byte-for-byte/numerically equivalent by importing the shared
    forward function.
  - Acceptance: identity, axis rotations, 100 deterministic random rotations, batched
    values, and both directions round-trip by matrix/angular error; the existing converter
    tests continue to pass.

### Phase 2: Catalog and semihumanoid adapter

- [x] **2.1 Add a strict, model-independent task catalog**
  (`gr00t/policy/industrialnext/task_catalog.py`).
  - Parse the existing schema fields `schema_version`, `task_family`, optional
    `catalog_version`, and non-empty `tasks` entries with unique `task_uuid`, exact
    `task_text`, and `display_name`.
  - Reject missing/unknown top-level keys, duplicate/empty UUIDs, empty text, and UUID/text
    mismatches at registration.
  - Provide protocol-safe ordered catalog entries plus `task_uuid_to_text`; do not import
    `industrialnext_ai.datasets.task_catalog`.
  - Acceptance: the real Ube catalog loads and produces exactly the three current UUID/text
    pairs; malformed fixtures fail deterministically.

- [x] **2.2 Assert the saved GR00T contract at startup** (`adapter.py`).
  - Compare `policy.get_modality_config()` against the exact saved keys and deltas rather
    than relying only on the source example config.
  - Require `video={head,left_wrist,right_wrist}`, the six ordered state keys, the four
    ordered action keys, observation delta `[0]`, language key
    `annotation.human.task_description`, and action deltas `0..39`.
  - Require relative XYZ_ROT6D for both EEF outputs and absolute semantics for both grippers.
  - Fail before the WebSocket binds on any mismatch.

- [x] **2.3 Implement strict sparse observation admission and snapshots** (`adapter.py`).
  - Require `left_arm_pose_pos` (3), `left_arm_pose_rot` (6), `left_gripper` (1),
    `left_ft` (6), and the corresponding four `right_*` fields on every step; validate
    numeric type, exact length, and finiteness before mutating the session.
  - Accept only `head_rgb`, `eoat_left_bottom_rgb`, and `eoat_right_bottom_rgb` as model RGB
    views; count and ignore `*_depth` fields. Reject unexpected `*_rgb` or other image
    payload fields, without treating declared state, task, or `images_meta` fields as images.
  - Cache immutable JPEG bytes, their metadata, and last-update timestep per session. Omitted
    unchanged RGB fields reuse the cache; a present null payload is an error.
  - Require `images_meta` to be a mapping when present and require a matching metadata entry
    for each present RGB payload update. Ignore metadata paired with ignored depth fields and
    reject orphan non-depth image metadata. Validate JPEG `format`, `dtype`, channels, height,
    and width on admission. Require all three views and enforce the configured age before
    scheduling an inference.
  - Snapshot the current state, cached bytes, cache ages, pinned task text, source timestep,
    and session generation. Do not share mutable dictionaries or arrays with the worker.

- [x] **2.4 Convert a snapshot into the exact GR00T observation** (`adapter.py`).
  - Decode JPEG with OpenCV in the worker; convert BGR decode output to RGB `uint8`; reject
    decode failure or any shape other than `256x256x3`.
  - Map physical camera names to `video.head`, `video.left_wrist`, and
    `video.right_wrist`, adding batch/time axes for `(1,1,H,W,C)`.
  - Form left/right EEF state from position plus converted row-rot6d, and form gripper/F/T
    arrays as `(1,1,D)` `float32`.
  - Use only the session-pinned catalog text as `[[task_text]]`; if task fields are repeated
    in the observation, require them to match the session.
  - Do not crop, resize, normalize, or convert F/T units in the adapter; the ROS node already
    applies deployment cropping and the saved processor owns model preprocessing.

- [x] **2.5 Validate and map decoded trajectories back to the robot schema** (`adapter.py`).
  - Require exactly the four configured action keys with finite numeric shapes
    `(1,40,9)`, `(1,40,1)`, `(1,40,9)`, and `(1,40,1)`.
  - Treat EEF trajectories as already decoded absolute poses. Never compose the live state a
    second time.
  - Convert only the selected EEF rotation from GR00T row-rot6d to the ROS column convention
    and emit `left_arm_pose_pos`, `left_arm_pose_rot`, `left_gripper`,
    `right_arm_pose_pos`, `right_arm_pose_rot`, and `right_gripper` as flat lists with
    lengths `3,6,1,3,6,1`.
  - Pass grippers through without snap or clipping and expose observed min/max in shadow
    statistics.

### Phase 3: Async server, sessions, and timeline

- [x] **3.1 Implement the direct metadata contract** (`async_server.py`).
  - Implement the `industrialnext_rpc.direct.server.interface.Server` methods
    `get_metadata()` and `handle_request()`; use `DirectServer` for framing and transport.
  - Declare only `register_session`, `step`, and `close_session`. Require
    `{type, control_hz, task_uuid, task_text}`, `{type, session_id, observation}`, and
    `{type, session_id}` respectively. Mirror the reference response shapes:
    `{session_id, metadata, error}` for registration; the reference action, timestep, queue,
    inference, uptime, count, server-step, monitoring, and error fields for `step`; and
    `{ok, error}` for close.
  - Advertise `async_serving: true`, `async_protocol_version: 2`, and the three required
    strings in `async_capabilities`. Use the client-consumed names
    `expert_camera_height: 256`, `expert_camera_width: 256`,
    `effective_gripper_snap_config` with `enabled: false`, both gripper action fields, and
    null snap/signal ranges, and `task_conditioning` with `mode: text`,
    `prompt_field_name: task_text`, catalog version, ordered catalog, and UUID-to-text map.
    Also publish absolute/rot6d action semantics, server instance ID, model path/fingerprint,
    embodiment tag, modality keys, horizon, effective minimum usable tail, and implementation
    revision.
  - Define one schema-complete response builder for success and failure. Because
    `DirectServer` validates declared success fields even when `error` is present, an error
    response must include safe values for every declared field plus the exact error string;
    it must not return an error-only dictionary. For `step`, use `action: null`,
    `timestep: -1`, zero counts/latencies, `queue_len: 0`, `inference_status: error`, empty
    monitoring/gripper dictionaries, and `monitoring_timestep: -1`. Registration errors use
    an empty session ID and metadata; close errors use `ok: false`.
  - Add a test that an unknown session arrives at the client as the exact
    `session_not_found` value rather than a rewritten validation error. Apply the same
    schema-complete rule to registration and close failures.

- [x] **3.2 Implement generation-safe session lifecycle** (`async_server.py`).
  - Require finite `control_hz == 50.0` plus exact catalog UUID/text on registration.
  - Only after registration fields pass validation, atomically replace any prior session,
    increment a process-scoped generation, reset RGB cache/timeline/stats, and return the
    effective service metadata. Log the old and new session IDs without task or state data;
    subsequent requests from the displaced session receive `session_not_found`.
  - On close, invalidate the generation and clear executable actions immediately. Make
    duplicate close safe.
  - Add a configurable positive idle timeout, defaulting to 300 seconds like the reference
    AI server, for abandoned sessions. A late worker result from a closed, timed-out, or
    replaced session must be discarded.

- [x] **3.3 Keep `step` deterministic and non-blocking** (`async_server.py`).
  - Perform only request validation, immutable byte caching, timestep advance, current
    action lookup, snapshot replacement, inference trigger, monitoring/stat construction,
    and response assembly on the event-loop thread.
  - Return `action: null` before the first usable result and during gaps. Do not repeat the
    last row, wrap the horizon, or synthesize a hold action; the ROS client owns its absolute
    hold behavior.
  - Populate the reference top-level queue, inference, uptime, count, latency, and monitoring
    fields on every successful step. Put latest inference error, action/source ages,
    usable-tail count, image ages, and null reason inside `monitoring` so diagnostics do not
    expand the established top-level wire contract.
  - Increment `monitoring_timestep` on every accepted step and always return progress `0.0`,
    classification `unknown`, `scene_valid: true`, and confidence `0.0`.

- [x] **3.4 Implement the single-worker latest-observation scheduler** (`async_server.py`).
  - Run at most one `policy.get_action()` in the dedicated executor.
  - While it runs, overwrite one pending snapshot with the newest complete observation;
    never queue an unbounded sequence.
  - Marshal completion back to the event-loop thread before touching sessions or timelines.
  - Record inference latency and source timestep. On error, publish no new chunk, clear any
    partial result, retain no executable data from the failed inference, and continue serving
    null/previously valid future rows only if they remain within their original timeline.
  - After completion, inspect the newest pending snapshot. Revalidate its generation and
    recompute every cached view's age against the session's current timestep; launch it
    immediately only if both checks pass, otherwise discard it. Valid-at-capture is not
    sufficient after a long inference.

- [x] **3.5 Implement the absolute-timestep action timeline** (`async_server.py`).
  - Associate action delta `i` with target `source_timestep + i`.
  - At result admission, discard targets that the event loop has already processed and
    reject the whole result when fewer than the configured minimum usable rows remain.
  - Validate the configured minimum usable tail as an integer in `[1, 40]`; advertise and
    log the effective value so a commanded rollout cannot silently inherit the shadow value.
  - Replace all older unserved future rows with the new chunk; do not append it.
  - On each `step`, return only the row matching that step's target timestep and remove it.
    Expire gaps and past rows explicitly.
  - Cover inference completion both immediately before and immediately after a step so an
    off-by-one race cannot replay target `t` at `t+1`.

- [x] **3.6 Implement cleanup and bounded observability** (`async_server.py`).
  - Gracefully stop admission and invalidate sessions on cancellation/SIGTERM.
  - Wait for or safely drain the single in-flight future, cancel pending observations, and
    shut down the executor without leaving a CUDA worker alive.
  - Log periodic aggregate latency, stale/missing RGB counts, inference failures, expired
    rows, usable tail, null-action reasons, and gripper output ranges. Do not log full image
    bytes or full state/action values by default.

### Phase 4: Entry point and actual checkpoint

- [x] **4.1 Add the server CLI** (`gr00t/eval/run_gr00t_industrialnext_server.py`).
  - Follow the existing `tyro` dataclass pattern and require `model_path` and
    `task_catalog_path`.
  - Default embodiment to `new_embodiment`, device to CUDA, host to `127.0.0.1`, port to
    `10012`, control rate to 50 Hz, RGB staleness to 5 steps, message size to 10 MiB, and
    idle-session timeout to 300 seconds; enforce one active session.
  - Expose an explicit minimum usable action-tail setting; shadow measurement must choose
    the commanded-trial value rather than hiding it in code. Reject values outside `[1, 40]`
    before model construction and advertise the effective value in metadata.
  - Reject a non-loopback host unless a separate explicit unsafe override is supplied and
    logged, because the protocol unpickles network input.

- [x] **4.2 Load and warm up on the dedicated worker before binding** (`run_...py`,
  `async_server.py`).
  - Construct `Gr00tPolicy(model_path=..., embodiment_tag=..., strict=True)` in the same
    single-thread executor used for inference.
  - Assert the saved contract and build one synthetic valid observation using identity
    EEF rotations, finite zero F/T, valid grippers, three `256x256` RGB frames, and the first
    catalog task.
  - Run one inference, validate the four decoded trajectories, discard its actions, then
    enter the `DirectServer` context. Do not announce readiness before this completes.
  - Store lightweight provenance: resolved model path, index and processor-config SHA-256,
    expected shard names/sizes, GR00T revision, and RPC submodule revision. Do not hash the
    multi-gigabyte shards at every startup.

- [x] **4.3 Add an opt-in actual-checkpoint GPU test**
  (`test_industrialnext_gr00t_policy_gpu.py`).
  - Read the model path from `GR00T_SEMIHUMANOID_MODEL_PATH`; skip with a clear reason when
    unset and mark the test `gpu`.
  - Construct the real policy, assert the saved contract, run the synthetic adapter
    observation, and verify finite absolute output shapes and conversion to one robot row.
  - Keep this test independent of ROS and do not send a command.

### Phase 5: End-to-end validation and rollout gates

- [x] **5.1 Run CPU and loopback protocol conformance.**
  - Use a deterministic fake policy whose inference future can be released by the test.
  - Start the real `industrialnext_rpc.DirectServer` on a loopback ephemeral port and use its
    real `DirectClient` for metadata/register/step/close flows.
  - Prove `step` remains responsive while inference is blocked; prove sparse JPEG reuse,
    exact task registration, null startup behavior, constant progress, close/re-register,
    generation discard, exact session-not-found recovery, and clean shutdown. Use the
    stricter stock `DirectClient` validator here: an exact error surviving it proves the
    response also satisfies the server-side schema before ROS's error-first client sees it.

- [ ] **5.2 Start the checkpoint-backed server manually and run the ROS catalog gate.**
  - Implementation-session evidence: the real checkpoint server loaded, warmed up, bound
    only to alternate loopback port `19172`, returned protocol v2 metadata with the real
    three-task Ube catalog through the stock `DirectClient`, and shut down cleanly. The ROS
    catalog executable and port `10012` remain intentionally untouched during active robot
    data collection.
  - Launch with the root final model and the Ube deployment's real `task_catalog.yaml`.
  - From the built ROS workspace, run `check_policy_server_task_catalog` against
    `127.0.0.1:10012` and require all catalog tasks to match.
  - Verify protocol version, required capabilities, image dimensions, gripper-snap metadata,
    model provenance, horizon, and effective minimum usable tail. The full
    `RobustDirectClient` register/step/close/reconnect path is exercised by the unmodified
    ROS node in Phase 5.4; do not add a second ad-hoc client probe here.

- [ ] **5.3 Measure timeline and latency with the real model before ROS shadow.**
  - The synthetic startup inference completed successfully, but it is not a replacement for
    the representative 50 Hz p50/p95/p99 study required here. Keep this gate open.
  - Run at 50 Hz with representative JPEG sizes and state ranges; exclude the declared
    warmup calls from steady-state statistics.
  - Record p50/p95/p99 inference latency, WebSocket step latency, decoded image latency,
    expired prefix length, remaining usable rows, inference errors, and null-action ratio.
  - Set the minimum usable tail from these measurements. No commanded trial proceeds if p99
    inference can consume the 40-step/0.8-second horizon or if `step` p99 exceeds the 20 ms
    control period.

- [ ] **5.4 Run the unmodified ROS client in inference-only shadow mode.**
  - Deferred: no ROS process, parameter, topic, or robot-side orchestration was changed or
    invoked while the robot was collecting data.
  - Treat the current dirty `flexiv_ube/task_config.yaml` as user-owned. Review its diff and
    make any shadow-mode edit as a separate explicit change; do not overwrite or stage
    unrelated robot configuration.
  - Set the existing policy path to `inference_mode: true`, regenerate effective parameters,
    and restart only the relevant units after explicit operational authorization.
  - Confirm all three camera identities and RGB channel order, task text, EEF frames, F/T
    units/ranges, input/output finiteness, gripper output distribution, action deltas,
    latency/tail metrics, cache age, null-action behavior, and `progress == 0.0`.
  - Manually stop the rollout; confirm it is recorded as a successful `manual_stop` and that
    no progress-threshold or completion-slowdown transition is triggered.

- [ ] **5.5 Gate and execute the first commanded trial separately.**
  - Deferred and still requires explicit motion authorization after shadow evidence.
  - Review shadow traces for action frame correctness, reasonable absolute target jumps, and
    whether learned grippers can remain pass-through. Add a server-owned hysteretic snap only
    as a separately reviewed change if evidence requires it.
  - Enable the existing ROS per-step action delta clamp with reviewed position/rotation
    limits before setting `inference_mode: false`; do not silently add server-side clipping.
  - Use supervised low-speed operation, an operator at the emergency stop, a bounded rollout,
    and manual stop as the expected completion path.
  - Stop on stale images, repeated inference errors, exhausted timelines, unexpected
    gripper range, frame mismatch, discontinuous chunk transitions, or loss of client/server
    timing budget.

- [ ] **5.6 Verify the operational rollback path before commanding motion.**
  - Deferred: no process on production port `10012` was started, stopped, or replaced.
  - Record the effective ROS policy parameters and currently selected server command before
    the trial. Never switch serving processes during an active rollout: issue manual stop,
    confirm the policy is inactive and no new policy targets are being published, and only
    then stop the GR00T server.
  - Test unexpected server loss in inference-only mode. The current ROS client should stop
    receiving/publishing fresh policy actions after its request timeout and enter reconnect;
    it does not automatically terminate the task, and the controller may hold its last
    accepted absolute target. Treat this as an operator-stop condition, not an automatic
    rollout-completion path.
  - The rollback is operational, not a code migration: stop the GR00T entry point, restore
    the previously reviewed generated policy parameters, confirm port 10012 is free, and
    restart the prior server with:

    ```bash
    cd /home/ubuntu/industrialnext_ai
    uv run python scripts/serve.py \
      --config config/manipulation/semihumanoid/deft_m_universal_rgb_v4_finetune.yaml \
      --host 127.0.0.1 \
      --port 10012
    ```

  - Exercise this handoff in inference-only mode first. Do not restart ROS units, switch the
    serving process, or move hardware without explicit operational authorization.

## Implementation record

The software phases were completed on 2026-08-19. The implemented server uses the pinned
`industrialnext_rpc` gitlink at `c3dc583ee36310581ad1ec154559698051988b9f`. CPU tests
cover strict catalogs, transactional sparse image caching, both rot6d convention directions,
saved model/output contracts, generation replacement, stale pending work, absolute action
timing, inference failures, idle expiry, and real WebSocket framing. The opt-in GPU test loaded
`outputs/gr00t/semihumanoid_20260819_080107`, loaded the cached
`nvidia/Cosmos-Reason2-2B` backbone with real weights, passed the saved-contract assertion,
and produced one finite 40-step trajectory.

Final validation passed 95 scoped CPU tests and the repository-wide CPU set with 731 passed,
9 skipped, and 50 intentionally deselected GPU/edge-device tests. Ruff lint/format, scoped
pre-commit hooks, `uv lock --locked`, `git diff --check`, and `uv build` also passed. The
repository's broader `-m "not gpu"` expression additionally selects an unrelated
`edge_device` download test; on this host that single test requires `HF_TOKEN`, so the true
CPU regression command excludes both markers.

The production process was also smoke-tested without ROS or robot traffic. It completed model
load and warmup before binding `127.0.0.1:19172`; a metadata-only stock client confirmed the
protocol version, required capabilities, three real catalog tasks, horizon, model provenance,
minimum usable tail, and RPC revision. The process then handled SIGINT and released the port.
Neither `10012` nor any robot command path was touched.

The initial gated-backbone probe exposed a test-cache distinction: repository pytest redirects
Hugging Face lookups to an isolated shared cache, while this workstation's complete Cosmos
snapshot is in `/home/ubuntu/.cache/huggingface`. The opt-in GPU test therefore accepts
`GR00T_SEMIHUMANOID_HF_HOME` to select an already populated cache, and disables pytest's
test-only no-weight model shortcut for this smoke. Production loading uses the normal process
cache and does not use that test shortcut.

Phases 5.2 through 5.6 remain operational gates, not incomplete product code. They require an
available robot window and, where noted, explicit authorization for ROS restarts or motion.

## Risk assessment

| Risk | Impact | Mitigation |
|---|---|---|
| RPC submodule drifts from AI/ROS deployment | Metadata or framing behavior changes silently | Pin the reviewed commit, record both revisions in metadata, and run the actual ROS client conformance gate |
| `handle_request` performs decode/inference | 50 Hz ROS callback times out and reconnects | Cache bytes only on the event loop; decode and infer on the single worker; latency-test blocked fake inference |
| `DirectServer` rewrites an error during response validation | ROS never sees exact `session_not_found` and can loop on a dead session | Make error responses satisfy the declared response schema and assert the exact client-visible sentinel |
| Old inference completes after reconnect | A new rollout receives actions conditioned on the old task/state | Process-scoped generation tags and result admission on the event-loop thread |
| A second local client preempts the active rollout | The first client suddenly loses its session | Accept replacement only after full registration validation, bind to trusted loopback, log both session IDs, invalidate the old generation, and document single-launcher operation |
| Inference latency shifts the trajectory | Every chunk starts with expired actions and causes discontinuities | Absolute target timesteps, expired-prefix removal, newest-chunk replacement, and minimum-tail gate |
| A pending snapshot becomes stale while another inference runs | Worker time is wasted and a chunk may be conditioned on a stalled camera view | Revalidate source and per-view age at launch time against the current session timestep; discard stale pending work |
| The server disappears during a commanded rollout | ROS reconnects while the controller may hold its last accepted target; the rollout is not automatically terminated | Treat disconnect as an operator-stop condition, verify it in shadow, and never switch serving processes until manual stop has made the policy inactive |
| Rot6d convention is passed through or permuted | Large, systematic EEF orientation error | One shared matrix-reconstruction conversion with forward/inverse random-rotation tests |
| Relative EEF result is composed twice | Targets jump away from the observed pose | Treat `Gr00tPolicy` output as decoded absolute and test with known synthetic relative deltas |
| Sparse RGB cache accepts wrong/stale camera | Policy conditions on missing or misidentified scene views | Fixed physical-to-model map, strict metadata/shape checks, per-view age, and shadow visualization/logging |
| Model output contains NaN, wrong shape, or out-of-range gripper | Unsafe or rejected robot command | Reject the entire chunk; log ranges in shadow; preserve ROS strict validation; no silent clipping |
| New registration cannot replace a disconnected session | ROS reconnect stalls until idle timeout | Atomically invalidate the previous single session on valid registration and discard its late results |
| Pickle server is remotely reachable | Arbitrary code execution from untrusted input | Loopback default, explicit non-loopback refusal/override, no supervisor exposure |
| Checkpoint exists but saved processor is inconsistent | Server binds with the wrong embodiment contract | Strict policy load, saved-modality assertions, synthetic warmup, and GPU smoke before bind |
| ROS deployment file has unrelated local edits | Trial preparation overwrites user work | Keep implementation in Isaac-GR00T; inspect and isolate any later inference-mode/clamp edit |

## Validation commands

Run dependency and source validation from `/home/ubuntu/Isaac-GR00T`:

```bash
git submodule status packages/industrialnext_rpc
test "$(git -C packages/industrialnext_rpc rev-parse HEAD)" = \
  c3dc583ee36310581ad1ec154559698051988b9f

uv lock --locked
uv sync --extra dev --extra industrialnext
uv run --extra industrialnext python -c \
  'from industrialnext_rpc.direct.metadata import Metadata; from industrialnext_rpc.direct.server import DirectServer, Server; print(Metadata, DirectServer, Server)'

uv run ruff check \
  gr00t/data/state_action/rot6d.py \
  gr00t/policy/industrialnext \
  gr00t/eval/run_gr00t_industrialnext_server.py \
  tests/gr00t/data/test_rot6d_conventions.py \
  tests/gr00t/policy/test_industrialnext_adapter.py \
  tests/gr00t/policy/test_industrialnext_async_server.py \
  tests/gr00t/policy/test_industrialnext_protocol.py \
  tests/gr00t/policy/test_industrialnext_gr00t_policy_gpu.py

uv run ruff format --check \
  gr00t/data/state_action/rot6d.py \
  gr00t/policy/industrialnext \
  gr00t/eval/run_gr00t_industrialnext_server.py \
  tests/gr00t/data/test_rot6d_conventions.py \
  tests/gr00t/policy/test_industrialnext_*.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
uv run --extra industrialnext --with h5py pytest -p pytest_timeout \
  tests/gr00t/data/test_rot6d_conventions.py \
  scripts/lerobot_conversion/test_zdata_pipeline.py \
  tests/gr00t/policy/test_industrialnext_adapter.py \
  tests/gr00t/policy/test_industrialnext_async_server.py \
  tests/gr00t/policy/test_industrialnext_protocol.py \
  -m "not gpu" -v --timeout=300
```

Run the opt-in real checkpoint smoke without ROS or robot commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
GR00T_SEMIHUMANOID_MODEL_PATH=\
outputs/gr00t/semihumanoid_20260819_080107 \
GR00T_SEMIHUMANOID_HF_HOME=/home/ubuntu/.cache/huggingface \
uv run --extra industrialnext pytest -p pytest_timeout \
  tests/gr00t/policy/test_industrialnext_gr00t_policy_gpu.py \
  -m gpu -v --timeout=300
```

Start the first manual server from `/home/ubuntu/Isaac-GR00T`:

```bash
GROOT_HF_LOCAL_FIRST=1 GROOT_PATCH_MISTRAL=1 \
uv run --extra industrialnext python \
  gr00t/eval/run_gr00t_industrialnext_server.py \
  --model-path outputs/gr00t/semihumanoid_20260819_080107 \
  --embodiment-tag new_embodiment \
  --task-catalog-path \
    /home/ubuntu/industrialnext_ros2/src/industrialnext_deployments/robots/flexiv_ube/task_catalog.yaml \
  --host 127.0.0.1 \
  --port 10012 \
  --control-hz 50 \
  --min-usable-action-steps 1
```

The one-row tail in this command is for inference-only measurement. Replace it with the
value selected from the latency study before any commanded rollout.

With the server running, validate the catalog from the built ROS workspace:

```bash
cd /home/ubuntu/industrialnext_ros2
source install/setup.bash
ros2 run industrialnext_operator_policy_client check_policy_server_task_catalog \
  --task-catalog \
    src/industrialnext_deployments/robots/flexiv_ube/task_catalog.yaml \
  --server-host 127.0.0.1 \
  --server-port 10012
```

Before committing the implementation, also run the repository-required checks:

```bash
uv run pre-commit run --all-files
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
uv run python -m pytest -p pytest_timeout \
  tests/ -m "not gpu and not edge_device" -v --timeout=300
uv build
uv lock --locked
git diff --check
```

Use `uv run python` instead of bare `python` in this checkout. The host currently does not
provide a `python` executable outside the managed environment. Disabling third-party pytest
plugin autoload also prevents the host ROS Humble Python 3.10 launch-testing plugin from
being imported into this Python 3.12 environment; the required timeout plugin is enabled
explicitly.

## Completion criteria

The implementation is complete only when all of the following are true:

1. A fresh clone plus recursive submodule initialization resolves and imports the pinned
   `industrialnext_rpc` direct server through the `industrialnext` extra.
2. CPU tests prove exact field/image/rotation mappings, session generation safety,
   non-blocking steps, action age correction, horizon expiry, and protocol behavior.
3. The completed root checkpoint loads strictly, passes the saved-contract gate, warms up,
   and produces a finite decoded 40-step trajectory before the server binds.
4. The unmodified ROS `RobustDirectClient` accepts metadata, validates all task strings,
   registers, steps, closes, reconnects, and sees constant fresh progress zero.
5. Real-model latency leaves the agreed usable tail while WebSocket `step` p99 stays under
   the 20 ms control period.
6. A real-robot inference-only shadow run confirms camera identity, frames, units, action
   ranges, cache age, and manual-stop completion without publishing commands.
7. A commanded trial remains a separate explicitly authorized action with the existing ROS
   delta clamp enabled and all hardware/operator safety gates active.
8. The inference-only rollback drill returns the deployment to the prior DEFT serving path,
   and the exact commands and effective ROS parameters needed for that handoff are recorded.
