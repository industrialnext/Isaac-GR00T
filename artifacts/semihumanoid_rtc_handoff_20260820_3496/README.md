# Semihumanoid GR00T RTC deployment handoff

This bundle transfers the selected `checkpoint-3496` candidate and the evidence needed for
deployment-machine no-motion shadow. It does not authorize a ROS command path or robot motion.

## Transfer and verify

From the GPU checkout, copy the model next to this handoff directory:

```bash
RTC_TRANSFER_DEST=<transfer-root>
rsync -a --info=progress2 \
  outputs/gr00t/semihumanoid_20260819_230043/checkpoint-3496/ \
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
