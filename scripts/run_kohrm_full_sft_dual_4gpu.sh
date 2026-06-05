#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-/home/work/.data/hrm_text_checkpoints}"
LOG_ROOT="${LOG_ROOT:-/home/work/.data/hrm_text_logs}"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/hrm_text_prepared}"

RESUME_FROM="${RESUME_FROM:-$CKPT_ROOT/KoHRM-Text-1.4B-stage4d-korean-tool-finance-repeat3-gbs180}"
LFM_DATA="${LFM_DATA:-$DATA_ROOT/kohrm_sft_lfm25_terminal_toolbench_full_v1}"
TOP2_DATA="${TOP2_DATA:-$DATA_ROOT/kohrm_sft_top2_terminal_tool_raw8192_v1}"

EPOCHS="${EPOCHS:-1}"
GBS="${GBS:-90112}"
LR="${LR:-2.0e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
KEEP_LAST="${KEEP_LAST:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
WEIGHTS_ONLY_RESUME_FROM_EMA="${WEIGHTS_ONLY_RESUME_FROM_EMA:-true}"

LFM_OUT="${LFM_OUT:-$CKPT_ROOT/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-gbs90k-4gpu}"
TOP2_OUT="${TOP2_OUT:-$CKPT_ROOT/KoHRM-Text-1.4B-fullsft-top2-terminal-tool-merge-gbs90k-4gpu}"

require_prepared() {
  local path="$1"
  for rel in metadata.json tokens.npy epoch_0/inst_start.npy epoch_0/inst_len.npy epoch_0/resp_start.npy epoch_0/resp_len.npy; do
    if [[ ! -e "$path/$rel" ]]; then
      echo "missing prepared dataset file: $path/$rel" >&2
      exit 1
    fi
  done
}

run_one() {
  local name="$1"
  local data_path="$2"
  local out_path="$3"
  local gpus="$4"
  local port="$5"
  local log_path="$LOG_ROOT/${name}.log"

  require_prepared "$data_path"
  mkdir -p "$out_path" "$LOG_ROOT"
  (
    cd "$ROOT"
    export CUDA_VISIBLE_DEVICES="$gpus"
    export PYTHONUNBUFFERED=1
    export HYDRA_FULL_ERROR=1
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="${WANDB_DIR:-/home/work/.data/wandb}"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) $name gpus=$gpus data=$data_path out=$out_path ====="
    exec torchrun \
      --nnodes=1 \
      --nproc_per_node=4 \
      --master_addr=127.0.0.1 \
      --master_port="$port" \
      pretrain.py \
      arch/size@arch=XL \
      "data.path=$data_path" \
      "resume_from=$RESUME_FROM" \
      "+checkpoint_path=$out_path" \
      "+project_name=KoHRM-Text" \
      "+run_name=$name" \
      "epochs=$EPOCHS" \
      "global_batch_size=$GBS" \
      "lr=$LR" \
      "lr_warmup_steps=$LR_WARMUP_STEPS" \
      "weights_only_resume_from_ema=$WEIGHTS_ONLY_RESUME_FROM_EMA" \
      "+log_interval=$LOG_INTERVAL" \
      "checkpoint_step_interval=$SAVE_STEPS" \
      "checkpoint_keep_last=$KEEP_LAST" \
      "checkpoint_interval=1"
  ) >"$log_path" 2>&1
}

main() {
  if [[ ! -d "$RESUME_FROM" ]]; then
    echo "missing RESUME_FROM checkpoint directory: $RESUME_FROM" >&2
    exit 1
  fi
  require_prepared "$LFM_DATA"
  require_prepared "$TOP2_DATA"

  case "${1:-both}" in
    lfm)
      run_one "KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench" "$LFM_DATA" "$LFM_OUT" "${LFM_GPUS:-0,1,2,3}" "${LFM_PORT:-29610}"
      ;;
    top2)
      run_one "KoHRM-Text-1.4B-fullsft-top2-terminal-tool-merge" "$TOP2_DATA" "$TOP2_OUT" "${TOP2_GPUS:-4,5,6,7}" "${TOP2_PORT:-29620}"
      ;;
    both)
      run_one "KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench" "$LFM_DATA" "$LFM_OUT" "${LFM_GPUS:-0,1,2,3}" "${LFM_PORT:-29610}" &
      pid_lfm=$!
      run_one "KoHRM-Text-1.4B-fullsft-top2-terminal-tool-merge" "$TOP2_DATA" "$TOP2_OUT" "${TOP2_GPUS:-4,5,6,7}" "${TOP2_PORT:-29620}" &
      pid_top2=$!
      echo "started lfm pid=$pid_lfm top2 pid=$pid_top2"
      wait "$pid_lfm"
      wait "$pid_top2"
      ;;
    *)
      echo "usage: $0 {both|lfm|top2}" >&2
      exit 2
      ;;
  esac
}

main "$@"
