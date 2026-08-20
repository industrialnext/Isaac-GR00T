# Server invocations

Set `RTC_CHECKPOINT_DIR` and run one mode at a time on an unused loopback port. These commands
do not start a robot or ROS client.

```bash
RTC_CHECKPOINT_DIR=<checkpoint-3496-directory>
RTC_TASK_CATALOG=semihumanoid_rtc_handoff_20260820_3496/task_catalog.yaml
RTC_MODE=off  # repeat with native and trained_prefix

uv run python gr00t/eval/run_gr00t_industrialnext_server.py \
  --model-path "$RTC_CHECKPOINT_DIR" \
  --task-catalog-path "$RTC_TASK_CATALOG" \
  --embodiment-tag new_embodiment \
  --device cuda \
  --host 127.0.0.1 \
  --port 11120 \
  --control-hz 50 \
  --rtc-mode "$RTC_MODE" \
  --rtc-initial-frozen-steps 4 \
  --rtc-delay-window-size 20 \
  --rtc-delay-margin-steps 2 \
  --rtc-max-prefix-steps 12 \
  --rtc-native-overlap-steps 12 \
  --rtc-min-new-tail-steps 16 \
  --min-usable-action-steps 16
```

In a second shell, reproduce the synthetic paced client:

```bash
uv run python gr00t/eval/smoke_industrialnext_loopback.py \
  --host 127.0.0.1 \
  --port 11120 \
  --steps 60 \
  --control-hz 50 \
  --image-refresh-steps 4 \
  --output-json-path loopback-result.json
```

Use distinct ports for concurrent diagnostic servers. Do not bind beyond loopback and do not
connect a ROS command client during this reproduction gate.
