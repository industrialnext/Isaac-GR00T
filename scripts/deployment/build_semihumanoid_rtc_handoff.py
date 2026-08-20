#!/usr/bin/env python3
"""Build a content-verified semihumanoid RTC deployment handoff bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = Path("outputs/gr00t/semihumanoid_20260819_230043")
DEFAULT_CHECKPOINT = DEFAULT_RUN / "checkpoint-3496"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _source_patch() -> tuple[bytes, list[str]]:
    tracked_patch = _run_git(
        "diff",
        "--binary",
        "HEAD",
        "--",
        ".",
        ":(exclude)artifacts/**",
    ).stdout
    untracked = _run_git("ls-files", "--others", "--exclude-standard", "--").stdout
    untracked_paths = [
        line.decode().strip()
        for line in untracked.splitlines()
        if line and not line.decode().startswith("artifacts/")
    ]
    chunks = [tracked_patch]
    for relative_path in sorted(untracked_paths):
        result = _run_git(
            "diff",
            "--no-index",
            "--binary",
            "--",
            "/dev/null",
            relative_path,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode())
        chunks.append(result.stdout)
    return b"".join(chunks), sorted(untracked_paths)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_reports(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".txt"}:
            _copy(path, destination / path.relative_to(source))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _checkpoint_inventory(checkpoint: Path) -> list[dict[str, object]]:
    required = {
        "config.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "statistics.json",
    }
    files = sorted(
        path
        for path in checkpoint.iterdir()
        if path.is_file()
        and (
            path.name in required
            or path.name.startswith("model-")
            and path.suffix == ".safetensors"
        )
    )
    missing = required - {path.name for path in files}
    if missing:
        raise FileNotFoundError(f"checkpoint is missing required files: {sorted(missing)}")
    return [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in files
    ]


def _readme() -> str:
    return """# Semihumanoid GR00T RTC deployment handoff

This bundle transfers the selected `checkpoint-3496` candidate and the evidence needed for
deployment-machine no-motion shadow. It does not authorize a ROS command path or robot motion.

## Transfer and verify

From the GPU checkout, copy the model next to this handoff directory:

```bash
RTC_TRANSFER_DEST=<transfer-root>
rsync -a --info=progress2 \\
  outputs/gr00t/semihumanoid_20260819_230043/checkpoint-3496/ \\
  "$RTC_TRANSFER_DEST/checkpoint-3496/"
```

After transferring both directories, verify from the transfer root:

```bash
(cd semihumanoid_rtc_handoff_20260820_3496 && sha256sum -c SHA256SUMS)
sha256sum -c semihumanoid_rtc_handoff_20260820_3496/checkpoint_SHA256SUMS
```

The second command expects `checkpoint-3496/` beside the handoff directory.

## Selected evidence

- Selected checkpoint: `checkpoint-3496`.
- Runtime: 50 Hz, initial frozen steps 4, delay window 20, delay margin 2, trained maximum
  prefix 12, native overlap 12, minimum new tail 16, minimum usable action rows 16.
- Isolated p99 inference latency: off 60.38 ms, native 69.34 ms, trained-prefix 71.13 ms.
- All three GPU loopback modes produced 56 finite actions after four startup responses, with
  zero protocol errors.
- `native` is the shadow frontrunner. `trained_prefix` is valid but had materially worse
  replay accuracy, so it must not be assumed to be the motion default.

Use `server_invocations.md` for loopback reproduction and `operator_checklist.md` for the
deployment sequence. Record the deployment checkout, service state, task, observation
sequence, and exact rollback command before any shadow or motion work.
"""


def _server_invocations() -> str:
    return """# Server invocations

Set `RTC_CHECKPOINT_DIR` and run one mode at a time on an unused loopback port. These commands
do not start a robot or ROS client.

```bash
RTC_CHECKPOINT_DIR=<checkpoint-3496-directory>
RTC_TASK_CATALOG=semihumanoid_rtc_handoff_20260820_3496/task_catalog.yaml
RTC_MODE=off  # repeat with native and trained_prefix

uv run python gr00t/eval/run_gr00t_industrialnext_server.py \\
  --model-path "$RTC_CHECKPOINT_DIR" \\
  --task-catalog-path "$RTC_TASK_CATALOG" \\
  --embodiment-tag new_embodiment \\
  --device cuda \\
  --host 127.0.0.1 \\
  --port 11120 \\
  --control-hz 50 \\
  --rtc-mode "$RTC_MODE" \\
  --rtc-initial-frozen-steps 4 \\
  --rtc-delay-window-size 20 \\
  --rtc-delay-margin-steps 2 \\
  --rtc-max-prefix-steps 12 \\
  --rtc-native-overlap-steps 12 \\
  --rtc-min-new-tail-steps 16 \\
  --min-usable-action-steps 16
```

In a second shell, reproduce the synthetic paced client:

```bash
uv run python gr00t/eval/smoke_industrialnext_loopback.py \\
  --host 127.0.0.1 \\
  --port 11120 \\
  --steps 60 \\
  --control-hz 50 \\
  --image-refresh-steps 4 \\
  --output-json-path loopback-result.json
```

Use distinct ports for concurrent diagnostic servers. Do not bind beyond loopback and do not
connect a ROS command client during this reproduction gate.
"""


def _operator_checklist() -> str:
    return """# Deployment operator checklist

- [ ] Verify `SHA256SUMS` and `checkpoint_SHA256SUMS` after transfer.
- [ ] Inspect the deployment revision, dirty worktrees, services, ports, and effective config.
- [ ] Record the exact previous launch and rollback commands in `runtime_parameters.yaml`.
- [ ] Reproduce off, native, and trained-prefix checkpoint-backed loopback with no ROS client.
- [ ] Confirm 50 Hz, task UUID/text, observation layout, action layout, and image freshness.
- [ ] Run all three modes in no-motion shadow against the same observation/task sequence.
- [ ] Require zero protocol errors, reconnect errors, non-finite actions, prefix violations,
      unexplained holds, and re-registration loops.
- [ ] Compare coverage, delay, seam, position, orientation, gripper, and continuity evidence.
- [ ] Select and record a motion candidate; do not assume trained-prefix is the default.
- [ ] Complete a reviewed rollback rehearsal before enabling any command path.
- [ ] Only then follow the plan's limited-duration real-robot gates with an operator and E-stop.
"""


def _runtime_parameters() -> str:
    return """schema_version: 1
checkpoint: checkpoint-3496
embodiment_tag: new_embodiment
control_hz: 50
rtc:
  initial_frozen_steps: 4
  delay_window_size: 20
  delay_margin_steps: 2
  max_prefix_steps: 12
  native_overlap_steps: 12
  min_new_tail_steps: 16
  min_usable_action_steps: 16
  ramp_rate: 6.0
tolerances:
  position_m: 0.0001
  orientation_rad: 0.001
  gripper: 0.0001
deployment_fill_before_shadow:
  checkout_revision: null
  service_name: null
  previous_launch_command: null
  rollback_command: null
  task_uuid: null
  task_text: null
motion_authorized: false
"""


def _rollback_template() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

: "${RTC_PREVIOUS_LAUNCH_COMMAND:?Record and export the previous launch command first}"
exec bash -lc "$RTC_PREVIOUS_LAUNCH_COMMAND"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    training_run = (REPO_ROOT / args.training_run).resolve()
    checkpoint = (REPO_ROOT / args.checkpoint).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing handoff: {output}")
    output.mkdir(parents=True)

    run_manifest_path = training_run / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    frozen_manifest = Path(run_manifest["frozen_corpus_manifest"]["path"])
    expected_frozen_hash = run_manifest["frozen_corpus_manifest"]["sha256"]
    if _sha256(frozen_manifest) != expected_frozen_hash:
        raise RuntimeError("frozen corpus manifest does not match the training run manifest")

    source_patch, untracked_source_paths = _source_patch()
    (output / "repository.patch").write_bytes(source_patch)
    checkpoint_inventory = _checkpoint_inventory(checkpoint)
    revision = _run_git("rev-parse", "HEAD").stdout.decode().strip()
    rpc_revision = _run_git("-C", "packages/industrialnext_rpc", "rev-parse", "HEAD").stdout
    source_inventory = {
        "schema_version": 1,
        "repository_revision": revision,
        "repository_patch_sha256": hashlib.sha256(source_patch).hexdigest(),
        "training_launch_repository_revision": run_manifest["repository_revision"],
        "training_launch_dirty_diff_sha256": run_manifest["repository_dirty_diff_sha256"],
        "industrialnext_rpc_revision": rpc_revision.decode().strip(),
        "untracked_source_paths_in_patch": untracked_source_paths,
        "artifacts_excluded_from_source_patch": True,
    }
    _write_json(output / "source_inventory.json", source_inventory)
    _write_json(output / "checkpoint_inventory.json", checkpoint_inventory)

    copies = {
        REPO_ROOT / "configs/embodiments/semihumanoid.yaml": output / "training_config.yaml",
        run_manifest_path: output / "run_manifest.json",
        frozen_manifest: output / "frozen_corpus_manifest.json",
        REPO_ROOT
        / "artifacts/semihumanoid_ube_rtc_cutoff_20260819_033143/source_audit.json": output
        / "source_audit.json",
        REPO_ROOT / "artifacts/semihumanoid_rtc_candidate_selection_20260820/selection.json": output
        / "selection.json",
        REPO_ROOT / "artifacts/semihumanoid_rtc_candidate_selection_20260820/README.md": output
        / "selection.md",
        REPO_ROOT / "artifacts/semihumanoid_rtc_loopback_inputs_20260820/task_catalog.yaml": output
        / "task_catalog.yaml",
    }
    for source, destination in copies.items():
        _copy(source, destination)

    _copy_reports(
        REPO_ROOT / "artifacts/semihumanoid_rtc_candidates_20260820",
        output / "candidate_reports/diagnostic_initial",
    )
    _copy_reports(
        REPO_ROOT / "artifacts/semihumanoid_rtc_candidates_20260820_v2",
        output / "candidate_reports/corrected_replay",
    )
    _copy_reports(
        REPO_ROOT / "artifacts/semihumanoid_rtc_selected_latency_20260820",
        output / "selected_latency",
    )
    _copy_reports(
        REPO_ROOT / "artifacts/semihumanoid_rtc_loopback_20260820",
        output / "loopback",
    )

    (output / "README.md").write_text(_readme())
    (output / "server_invocations.md").write_text(_server_invocations())
    (output / "operator_checklist.md").write_text(_operator_checklist())
    (output / "runtime_parameters.yaml").write_text(_runtime_parameters())
    rollback = output / "rollback_command.sh.template"
    rollback.write_text(_rollback_template())
    rollback.chmod(0o755)

    checkpoint_sums = "".join(
        f"{item['sha256']}  checkpoint-3496/{item['path']}\n" for item in checkpoint_inventory
    )
    (output / "checkpoint_SHA256SUMS").write_text(checkpoint_sums)

    bundle_files = sorted(path for path in output.rglob("*") if path.is_file())
    bundle_sums = "".join(f"{_sha256(path)}  {path.relative_to(output)}\n" for path in bundle_files)
    (output / "SHA256SUMS").write_text(bundle_sums)
    print(output)


if __name__ == "__main__":
    main()
