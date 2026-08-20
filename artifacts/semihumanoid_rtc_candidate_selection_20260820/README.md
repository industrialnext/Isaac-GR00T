# Semihumanoid RTC checkpoint selection

`checkpoint-3496` is the GPU-machine candidate selected for deployment-machine shadow.
Across twelve held-out trajectories, it has the best aggregate open-loop MSE and MAE, the
best trained-prefix replay MAE, and the best off replay MAE. `checkpoint-3000` is only
marginally better in native replay.

All corrected replay modes achieved 0.96 target-timestep coverage, zero holds, and zero
rejections. Native and trained-prefix physical hard-prefix errors were zero for position and
gripper and at most 4.22e-8 rad for orientation.

The first candidate matrix is retained as diagnostic evidence because it exposed a BF16
decode round-trip defect in committed physical prefixes. The corrected matrix in
`artifacts/semihumanoid_rtc_candidates_20260820_v2` is the accepted replay evidence.

`trained_prefix` is not the current motion-mode recommendation: its replay errors are
materially higher than native and off. `native` is the no-motion shadow frontrunner, but the
deployment machine must compare all three modes before selecting any motion candidate.
