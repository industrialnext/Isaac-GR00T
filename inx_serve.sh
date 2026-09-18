#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {100|full} [server options...]" >&2
  exit 2
fi

case "$1" in
  100) run=taro_exp_100_20260917_233103 ;;
  full) run=taro_exp_full_20260917_233111 ;;
  *)
    echo "Usage: $0 {100|full} [server options...]" >&2
    exit 2
    ;;
esac
variant="$1"
shift

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -f packages/industrialnext_rpc/pyproject.toml ]]; then
  git submodule update --init --recursive packages/industrialnext_rpc
fi

export HF_HUB_CACHE="$PWD/outputs/gr00t/huggingface/hub"
# These switches are read when gr00t is first imported.
export GROOT_HF_LOCAL_FIRST=1
export GROOT_PATCH_MISTRAL=1
export CUDA_VISIBLE_DEVICES="${SERVING_GPU:-0}"

exec uv run --locked --extra industrialnext python \
  gr00t/eval/run_gr00t_industrialnext_server.py \
  --config "configs/embodiments/taro_exp_${variant}.yaml" \
  --model-path "outputs/gr00t/${run}/checkpoint-40000" \
  --port 10012 \
  --rtc-mode off \
  "$@"
