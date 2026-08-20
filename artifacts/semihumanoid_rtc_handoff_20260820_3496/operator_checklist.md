# Deployment operator checklist

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
