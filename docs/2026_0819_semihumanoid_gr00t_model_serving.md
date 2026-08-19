# Semihumanoid GR00T N1.7 Model Serving Guide

**Date:** 2026-08-19

**Status:** Ready for deployment integration

**Trained model:** `outputs/gr00t/semihumanoid_20260819_080107`

**Related:**
[`2026_0818_semihumanoid_gr00t_conversion_plan.md`](2026_0818_semihumanoid_gr00t_conversion_plan.md),
[`2026_0819_zdata_pipeline_generalization_design.md`](2026_0819_zdata_pipeline_generalization_design.md)

## Purpose and path convention

This guide explains how to load the completed semihumanoid finetune, expose it through the
repository's ZeroMQ policy server, construct requests from a robot process, interpret the
returned action trajectory, and bring the policy onto hardware safely.

Run every command from the Isaac-GR00T repository root. This repository is expected to have
the same portable symlinks on every machine:

```text
data     -> external dataset storage
outputs  -> external model-output storage
```

All paths below therefore use `data/...` or `outputs/...`; no machine-specific absolute
path is required.

## What is served

Use the final model directory:

```text
outputs/gr00t/semihumanoid_20260819_080107
```

It contains the final step-5,251 model weights and a `processor/` directory containing the
semihumanoid modality configuration, embodiment mapping, and normalization statistics.
`Gr00tPolicy` automatically falls back to that processor directory when the model root does
not contain `processor_config.json`. The server therefore needs no separate modality config.

The output also contains `checkpoint-2000` through `checkpoint-5251`. Any complete
checkpoint directory can be supplied to `--model-path` for an offline comparison, but the
final model root is the normal serving target. Training did not select a "best" checkpoint
from a validation metric, so checkpoint choice should ultimately be based on held-out
open-loop evaluation and supervised robot trials rather than step number alone.

On a new machine, prepare the repository environment with `uv sync --all-extras`. The
checkpoint processor refers to `nvidia/Cosmos-Reason2-2B` for tokenizer/image-processor
assets, so those assets must already be in the Hugging Face cache or be downloadable when
the policy first loads.

## Serving architecture

```text
Robot adapter / control process
  RGB images + current state + instruction
                  |
                  | ZeroMQ request on TCP port 5555
                  v
PolicyServer -> Gr00tPolicy -> saved processor -> GR00T N1.7 model
                  |
                  | decoded physical-unit action dictionary
                  v
Safety filter -> short action prefix -> robot controller -> re-observe
```

The server uses a ZeroMQ request/reply loop and processes requests serially. A single robot
control client is the simplest operating model. The protocol supports a batch dimension,
but normal closed-loop robot serving uses batch size 1.

## Start the model server

For a client on the same machine:

```bash
uv run --no-sync python gr00t/eval/run_gr00t_server.py \
  --model-path outputs/gr00t/semihumanoid_20260819_080107 \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 5555
```

The process loads the model in BF16 on the selected GPU, switches it to evaluation mode,
binds the ZeroMQ socket, and prints both of these readiness messages before accepting work:

```text
✓ Server ready — listening on 127.0.0.1:5555
Server is ready and listening on tcp://127.0.0.1:5555
```

The server runs in the foreground. For persistent manual operation, run it in a dedicated
tmux session so logs remain visible:

```bash
tmux new-session -s gr00t-serve
# Run the server command above inside the new session.
```

Keep server strict validation enabled; it defaults to `True`. Do not pass
`--modality-config-path` for model serving. That flag belongs to the replay-policy branch of
`run_gr00t_server.py`; a real model always loads the modality config saved in its processor.

### Remote clients and security

Use `--host 0.0.0.0` only when a client must connect from another trusted machine. Restrict
port 5555 with the host firewall, a private LAN/VPN, or an SSH tunnel. The CLI does not
expose `PolicyServer`'s optional API token, and its unauthenticated endpoints include
`kill`. Never expose this server directly to a public or untrusted network.

## Embedded semihumanoid contract

The saved processor defines one current observation and a 40-step action prediction:

| Modality | Key | Request shape | Meaning |
|---|---|---:|---|
| Video | `head` | `(B, 1, H, W, 3)` | `head_rgb` camera |
| Video | `left_wrist` | `(B, 1, H, W, 3)` | `eoat_left_bottom_rgb` camera |
| Video | `right_wrist` | `(B, 1, H, W, 3)` | `eoat_right_bottom_rgb` camera |
| State | `left_eef` | `(B, 1, 9)` | XYZ + row-convention rot6d |
| State | `left_gripper` | `(B, 1, 1)` | Left gripper state |
| State | `left_ft` | `(B, 1, 6)` | Left TCP force/torque |
| State | `right_eef` | `(B, 1, 9)` | XYZ + row-convention rot6d |
| State | `right_gripper` | `(B, 1, 1)` | Right gripper state |
| State | `right_ft` | `(B, 1, 6)` | Right TCP force/torque |
| Language | `annotation.human.task_description` | nested list `(B, 1)` | Task instruction |

Video arrays must be NumPy `uint8` in RGB channel order. State arrays must be NumPy
`float32`. The policy accepts arbitrary image height and width structurally, but deployment
should provide the trained 256×256 view and camera geometry. A common OpenCV mistake is to
send BGR without converting it to RGB.

The state/action adapter must preserve the dataset's units and conventions. In particular:

- EEF translation must use the same frame and units as the training data.
- EEF rotation must be GR00T's row convention: flatten the first two rows of the rotation
  matrix, `R[:2, :].reshape(6)`.
- Gripper values must use the same scale and open/closed direction as the recorded data.
- `left_ft` and `right_ft` must use the recorded TCP-frame wrench convention; the source
  pipeline is expected to provide force in N and torque in Nm.

Do not silently clamp or renormalize inputs before checking that the robot-side convention
matches the training corpus. GR00T's saved processor performs the learned statistical
normalization itself.

## Minimal client

The client sends batched NumPy arrays. The following skeleton assumes that the robot
adapter has already produced correctly framed and scaled values:

```python
import numpy as np

from gr00t.policy.server_client import PolicyClient


def make_observation(
    head_rgb: np.ndarray,
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    left_eef: np.ndarray,
    left_gripper: np.ndarray,
    left_ft: np.ndarray,
    right_eef: np.ndarray,
    right_gripper: np.ndarray,
    right_ft: np.ndarray,
    instruction: str,
) -> dict:
    return {
        "video": {
            "head": np.asarray(head_rgb, dtype=np.uint8)[None, None],
            "left_wrist": np.asarray(left_wrist_rgb, dtype=np.uint8)[None, None],
            "right_wrist": np.asarray(right_wrist_rgb, dtype=np.uint8)[None, None],
        },
        "state": {
            "left_eef": np.asarray(left_eef, dtype=np.float32).reshape(1, 1, 9),
            "left_gripper": np.asarray(left_gripper, dtype=np.float32).reshape(1, 1, 1),
            "left_ft": np.asarray(left_ft, dtype=np.float32).reshape(1, 1, 6),
            "right_eef": np.asarray(right_eef, dtype=np.float32).reshape(1, 1, 9),
            "right_gripper": np.asarray(right_gripper, dtype=np.float32).reshape(1, 1, 1),
            "right_ft": np.asarray(right_ft, dtype=np.float32).reshape(1, 1, 6),
        },
        "language": {
            "annotation.human.task_description": [[instruction]],
        },
    }


observation = make_observation(
    head_rgb=head_rgb,
    left_wrist_rgb=left_wrist_rgb,
    right_wrist_rgb=right_wrist_rgb,
    left_eef=left_eef,
    left_gripper=left_gripper,
    left_ft=left_ft,
    right_eef=right_eef,
    right_gripper=right_gripper,
    right_ft=right_ft,
    instruction="Pick the grounded target object and hold it securely in the gripper.",
)

with PolicyClient(
    host="127.0.0.1",
    port=5555,
    timeout_ms=120_000,
    strict=False,
) as policy:
    if not policy.ping():
        raise RuntimeError("GR00T policy server is not responding")

    modality_config = policy.get_modality_config()
    assert modality_config["video"].modality_keys == [
        "head",
        "left_wrist",
        "right_wrist",
    ]
    assert modality_config["action"].delta_indices == list(range(40))

    action, info = policy.get_action(observation)
    for key, value in action.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite action returned for {key}")
```

Keep `PolicyClient(strict=False)`. Its client-side `check_observation` and `check_action`
methods are intentionally unimplemented; server-side `Gr00tPolicy(strict=True)` performs
the real validation. A generous initial timeout accommodates first-inference warm-up; it
can be reduced after measuring the target machine.

Available protocol endpoints are `ping`, `get_modality_config`, `get_action`, `reset`, and
`kill`. Use `kill` only for intentional shutdown from a trusted client; normal process
management can stop the foreground server with `Ctrl-C`.

## Returned action and controller conversion

For batch size 1, the response contains:

| Key | Shape | Saved action representation | Returned value |
|---|---:|---|---|
| `left_eef` | `(1, 40, 9)` | Relative EEF, XYZ + rot6d | Decoded absolute EEF trajectory |
| `left_gripper` | `(1, 40, 1)` | Absolute non-EEF | Absolute gripper trajectory |
| `right_eef` | `(1, 40, 9)` | Relative EEF, XYZ + rot6d | Decoded absolute EEF trajectory |
| `right_gripper` | `(1, 40, 1)` | Absolute non-EEF | Absolute gripper trajectory |

The model predicts normalized relative EEF actions internally. The saved processor
unnormalizes them and composes them with the current EEF state before the policy returns,
so the EEF arrays received by the client are absolute poses. They are still represented in
GR00T's XYZ + row-convention rot6d format.

Before sending a pose to a controller that uses the source pipeline's column-convention
rot6d, rebuild its rotation matrix with GR00T's rot6d decoder and encode it as:

```python
controller_rot6d = np.concatenate([rotation_matrix[:, 0], rotation_matrix[:, 1]])
```

If the controller accepts rotation matrices or quaternions directly, convert the GR00T
rot6d to that native representation instead. Omitting this convention conversion can
transpose/invert the commanded orientation while leaving all shapes numerically valid.

## Closed-loop execution and safety

Forty predictions at 50 Hz represent 0.8 seconds. Treat them as a receding-horizon proposal,
not as an unchecked open-loop command:

1. Capture synchronized current images and state.
2. Request one 40-step trajectory.
3. Convert action representations into the controller's native convention.
4. Apply workspace, joint, velocity, acceleration, collision, gripper, and force/torque
   limits.
5. Execute a short prefix, initially 4–8 steps (80–160 ms at 50 Hz).
6. Re-observe and request a new trajectory.

Keep the robot's emergency stop and existing low-level safety controller outside the model
server. Network timeout, non-finite output, stale observations, missing cameras, limit
violations, or controller disagreement should stop motion or enter the robot's established
safe state; they should not reuse an old action chunk indefinitely.

Bring-up should progress through recorded-observation inference, held-out open-loop
evaluation, shadow mode with no commands sent, low-speed supervised trials, and only then
normal closed-loop operation.

## Startup and operating checks

Before each serving session:

1. Confirm the intended model path and GPU.
2. Confirm port 5555 is not already in use.
3. Start the server and wait for the explicit readiness messages.
4. Call `ping()` and `get_modality_config()` before the first action request.
5. Send one recorded, known-good observation and verify keys, shapes, `float32` output, and
   finite values.
6. Measure warm and steady-state inference latency before selecting the execution prefix and
   timeout.

During operation, log at least the model path, instruction, observation timestamp, response
latency, selected action prefix, safety-filter result, and any dropped or stale input. Avoid
logging full-rate images indefinitely unless storage and retention are deliberately managed.

## Direct in-process use

When the robot adapter and policy deliberately share one Python process, the same checkpoint
can be loaded without ZeroMQ:

```python
from gr00t.policy import Gr00tPolicy

policy = Gr00tPolicy(
    model_path="outputs/gr00t/semihumanoid_20260819_080107",
    embodiment_tag="NEW_EMBODIMENT",
    device="cuda:0",
    strict=True,
)

action, info = policy.get_action(observation)
```

The in-process path removes serialization and network overhead. The server/client path is
preferable when GPU inference and robot control need separate processes, failure boundaries,
or machines.

## Common failures

| Symptom | Likely cause | Action |
|---|---|---|
| Model path not found | Command was not run from the repo root, or the `outputs` symlink is missing | Restore the standard symlink and retry from the repo root |
| Processor loading tries the network | `nvidia/Cosmos-Reason2-2B` processor assets are not cached | Provide Hugging Face access once or pre-populate the cache on the serving machine |
| Missing modality-key assertion | Robot request keys do not match the saved processor | Use the exact keys in the embedded-contract table |
| Video dtype/shape assertion | Image is not RGB `uint8` `(B, 1, H, W, 3)` | Convert channel order/dtype and add batch/time axes |
| State dtype/shape assertion | State is float64 or missing batch/time axes | Cast to `np.float32` and reshape to `(B, 1, D)` |
| Client raises `NotImplementedError` before sending | Client strict mode was enabled | Use `PolicyClient(strict=False)`; retain server strict mode |
| First request times out | GPU warm-up or timeout is too short | Verify the server log and retry with a larger `timeout_ms`; the client recreates its socket after a timeout |
| Poses translate correctly but rotate incorrectly | Row/column rot6d convention mismatch | Apply the controller conversion described above |
| Connection works remotely but is unsafe | Port is reachable from an untrusted network | Stop the server and restrict it to loopback, VPN, SSH tunnel, or a firewall-controlled LAN |

The authoritative serving entry points are
[`gr00t/eval/run_gr00t_server.py`](../gr00t/eval/run_gr00t_server.py),
[`gr00t/policy/gr00t_policy.py`](../gr00t/policy/gr00t_policy.py), and
[`gr00t/policy/server_client.py`](../gr00t/policy/server_client.py).
