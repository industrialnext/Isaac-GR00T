#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Real finetune: GR00T N1.7 on the converted IndustrialNext semihumanoid corpus.
#
# Every setting below is either measured on this machine or inherited from the base
# checkpoint. Override any of them with env vars, e.g.:
#   BATCH=256 LR=2.8e-4 STEPS=3750 bash train_semihumanoid.sh
#   RESUME=1 bash train_semihumanoid.sh        # continue after an interruption
#
# Measured batch scaling (4x RTX 4090, 3 cameras, real corpus, 100-step probes):
#   batch  32 ( 8/GPU): 1.630 it/s =  52.2 samples/s, 17.6 GB/GPU (37%)
#   batch  64 (16/GPU): 1.030 it/s =  65.9 samples/s, 19.1 GB/GPU (40%)
#   batch 128 (32/GPU): 0.787 it/s = 100.8 samples/s, 22.7 GB/GPU (47%)
#   batch 160 (40/GPU): 0.694 it/s = 111.1 samples/s, 24.7 GB/GPU (51%)
#   batch 200 (50/GPU): 0.610 it/s = 122.0 samples/s, 28.8 GB/GPU (60%)
#   batch 256 (64/GPU): 0.526 it/s = 134.7 samples/s, 33.0 GB/GPU (69%)   <- default
# Throughput rises ~+10% per step up with no knee in the probed range, so 256 is the
# fastest measured point that still leaves ~31% VRAM headroom. The trade is optimizer
# steps: a fixed 960k-sample budget gives 3750 updates at 256 versus 7500 at 128. No
# quality measurement distinguishes them yet -- settle it with open-loop MSE on the
# held-out _val datasets before treating 256 as validated rather than merely fastest.
# Batch 32 was PCIe-all-reduce bound (56% DDP scaling efficiency); feeding more compute
# per step nearly doubles throughput at 128 while leaving half the VRAM free.
set -uo pipefail

cd /home/yskim/ws/Isaac-GR00T

# --- knobs ---------------------------------------------------------------------------
BATCH=${BATCH:-256}          # global, pre-accumulation; per-GPU = BATCH/GPUS
STEPS=${STEPS:-3750}         # 3750 x 256 = 960k samples = 2.2 epochs over 440,280 starts
LR=${LR:-2.8e-4}             # sqrt-scaled from the repo's 1e-4 @ batch 32 (sqrt(256/32)=2.83)
GPUS=${GPUS:-4}
WORKERS=${WORKERS:-8}        # was 4; the aborted run showed 80 ongoing shard-wait stalls (250s)
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_LIMIT=${SAVE_LIMIT:-5}
RESUME=${RESUME:-0}
USE_WANDB=${USE_WANDB:-0}    # wandb is NOT logged in on this host; `wandb login` first
RUN_NAME=${RUN_NAME:-semihumanoid_260819_b256}

OUT_BASE=$HOME/ml_data/outputs/gr00t
RUN_DIR=$OUT_BASE/$RUN_NAME
CORPUS=$HOME/ml_data/data/training_data/gr00t/semihumanoid_260818
LOG=$RUN_DIR/train_console.log

mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG") 2>&1          # summary + marker land in the log, not just the pane

export NO_ALBUMENTATIONS_UPDATE=1     # skip an outbound version check at import

echo "=========================================================================="
echo " GR00T N1.7 finetune -- semihumanoid"
date '+ start: %Y-%m-%d %H:%M:%S'
echo " run dir: $RUN_DIR"
echo " batch=$BATCH ($((BATCH/GPUS))/GPU)  steps=$STEPS  lr=$LR  gpus=$GPUS"
echo " samples to be seen: $((BATCH*STEPS))"
echo "=========================================================================="

# --- preflight -----------------------------------------------------------------------
FAIL=0
PATHS=$(uv run --no-sync python scripts/lerobot_conversion/semihumanoid_datasets.py \
          --out-root "$CORPUS" --print train 2>/dev/null | tail -1)
N=$(echo "$PATHS" | tr ':' '\n' | wc -l)
echo "preflight: $N train datasets"
[ "$N" -ge 1 ] || { echo "  FAIL: no train datasets"; FAIL=1; }

# stats must exist, or rank 0 regenerates them serially before step 1 (~40 min, GPUs idle)
for ds in $(echo "$PATHS" | tr ':' ' '); do
  for f in stats.json relative_stats.json; do
    [ -f "$ds/meta/$f" ] || { echo "  FAIL: missing $(basename $ds)/meta/$f -- run gr00t/data/stats.py first"; FAIL=1; }
  done
done
[ "$FAIL" -eq 0 ] && echo "preflight: all datasets have stats.json + relative_stats.json"

# val sets must NOT be in the training path
echo "$PATHS" | tr ':' '\n' | grep -q '_val$' && { echo "  FAIL: a _val dataset is in --dataset-path"; FAIL=1; }
[ "$FAIL" -eq 0 ] && echo "preflight: no _val dataset in the training path"

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' \
  '{ if ($2+0 > 1000) { print "  WARN: gpu"$1" already holds "$2" MiB"; } }'
AVAIL=$(df -BG --output=avail "$OUT_BASE" | tail -1 | tr -dc '0-9')
echo "preflight: ${AVAIL}G free at $OUT_BASE (need ~200G for $SAVE_LIMIT resumable checkpoints)"
[ "$AVAIL" -lt 250 ] && { echo "  FAIL: not enough free space"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "PREFLIGHT FAILED -- not launching"; exit 1; }
echo "preflight: OK"
echo

EXTRA=()
[ "$RESUME" = "1" ] && EXTRA+=(--resume-from-checkpoint) && echo "RESUMING from latest checkpoint in $RUN_DIR"
[ "$USE_WANDB" = "1" ] && EXTRA+=(--use-wandb --wandb-project gr00t-semihumanoid)

# --- VRAM sampler --------------------------------------------------------------------
( while true; do
    printf '%s,' "$(date +%s)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd, -
    sleep 30
  done ) > "$RUN_DIR/vram.csv" &
VP=$!
trap 'kill $VP 2>/dev/null' EXIT

# --- train ---------------------------------------------------------------------------
uv run --no-sync torchrun --nproc_per_node="$GPUS" --master_port=29517 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$PATHS" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/semihumanoid/semihumanoid_config.py \
  --num-gpus "$GPUS" \
  --output-dir "$RUN_DIR" \
  --max-steps "$STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_LIMIT" \
  --global-batch-size "$BATCH" \
  --dataloader-num-workers "$WORKERS" \
  --learning-rate "$LR" \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --state-dropout-prob 0.2 \
  --shortest-image-edge 256 --crop-fraction 0.95 \
  --color-jitter-params brightness 0.15 contrast 0.15 saturation 0.2 hue 0.1 \
  "${EXTRA[@]}"
RC=$?
kill $VP 2>/dev/null

# --- summary -------------------------------------------------------------------------
date '+ end:   %Y-%m-%d %H:%M:%S'
echo
echo "==================== TRAIN RESULT ===================="
echo "exit code: $RC"
awk -F, 'NR>1{for(i=2;i<=NF;i++) if($i+0>m[i]) m[i]=$i+0}
         END{printf "peak VRAM: "; for(i=2;i<=5;i++) printf "gpu%d=%dMiB(%.0f%%) ", i-2, m[i], m[i]*100/49140; print ""}' \
    "$RUN_DIR/vram.csv" 2>/dev/null
echo "checkpoints:"; ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sed 's/^/  /' || echo "  none"
du -sh "$RUN_DIR" 2>/dev/null | sed 's/^/run dir size: /'
echo "TRAIN_DONE rc=$RC"
