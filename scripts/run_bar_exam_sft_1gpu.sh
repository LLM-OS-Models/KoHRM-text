#!/usr/bin/env bash
# Single-H200 full-SFT runner for KoHRM-Text-1.4B bar-exam experiments.
#
# Defaults target GPU index 7. Override with BAR_GPU, BAR_RESUME_FROM,
# BAR_PREPARED, BAR_OUT, BAR_EPOCHS, BAR_GBS, BAR_LR as needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BAR_GPU="${BAR_GPU:-7}"
BAR_RESUME_FROM="${BAR_RESUME_FROM:-/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage4d-korean-tool-finance-repeat3-gbs180}"
BAR_PREPARED="${BAR_PREPARED:?set BAR_PREPARED=/path/to/prepared_sft_data}"
BAR_OUT="${BAR_OUT:?set BAR_OUT=/path/to/output_ckpt_dir}"
BAR_RUN_NAME="${BAR_RUN_NAME:-bar_exam_sft}"
BAR_EPOCHS="${BAR_EPOCHS:-2}"
BAR_GBS="${BAR_GBS:-8192}"
BAR_LR="${BAR_LR:-3.0e-5}"
BAR_LR_WARMUP_STEPS="${BAR_LR_WARMUP_STEPS:-20}"
BAR_SAVE_STEPS="${BAR_SAVE_STEPS:-200}"
BAR_KEEP_LAST="${BAR_KEEP_LAST:-2}"
BAR_LOG_INTERVAL="${BAR_LOG_INTERVAL:-5}"
BAR_PORT="${BAR_PORT:-29677}"
BAR_LOG_DIR="${BAR_LOG_DIR:-/home/work/.data/hrm_text_logs}"
BAR_USE_EMA="${BAR_USE_EMA:-true}"
BAR_WEIGHTS_ONLY_RESUME_FROM_EMA="${BAR_WEIGHTS_ONLY_RESUME_FROM_EMA:-true}"

LOG_PATH="$BAR_LOG_DIR/${BAR_RUN_NAME}.log"

require_prepared() {
  local p="$1"
  for rel in metadata.json tokens.npy epoch_0/inst_start.npy epoch_0/inst_len.npy epoch_0/resp_start.npy epoch_0/resp_len.npy; do
    [[ -e "$p/$rel" ]] || { echo "missing $p/$rel" >&2; exit 1; }
  done
}

if [[ ! -d "$BAR_RESUME_FROM" ]]; then
  echo "missing BAR_RESUME_FROM=$BAR_RESUME_FROM" >&2
  exit 1
fi
require_prepared "$BAR_PREPARED"
mkdir -p "$BAR_OUT" "$BAR_LOG_DIR"

export CUDA_VISIBLE_DEVICES="$BAR_GPU"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-/home/work/.data/wandb}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NCCL_DEBUG=WARN

echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) $BAR_RUN_NAME gpu=$BAR_GPU resume=$BAR_RESUME_FROM prepared=$BAR_PREPARED out=$BAR_OUT epochs=$BAR_EPOCHS gbs=$BAR_GBS lr=$BAR_LR ====="

exec torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port="$BAR_PORT" \
  pretrain.py \
  --config-name=cfg_sft \
  arch/size@arch=XL \
  "data.path=$BAR_PREPARED" \
  "resume_from=$BAR_RESUME_FROM" \
  "+checkpoint_path=$BAR_OUT" \
  "+project_name=KoHRM-Text-BarExam" \
  "+run_name=$BAR_RUN_NAME" \
  "epochs=$BAR_EPOCHS" \
  "global_batch_size=$BAR_GBS" \
  "lr=$BAR_LR" \
  "lr_warmup_steps=$BAR_LR_WARMUP_STEPS" \
  "weights_only_resume_from_ema=$BAR_WEIGHTS_ONLY_RESUME_FROM_EMA" \
  "+log_interval=$BAR_LOG_INTERVAL" \
  "+checkpoint_step_interval=$BAR_SAVE_STEPS" \
  "+checkpoint_keep_last=$BAR_KEEP_LAST" \
  "checkpoint_interval=1" \
  >"$LOG_PATH" 2>&1
