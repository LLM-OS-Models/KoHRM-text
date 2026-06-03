#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/hrm_text_prepared}"
CKPT_ROOT="${CKPT_ROOT:-/home/work/.data/hrm_text_checkpoints}"
OUT_ROOT="${OUT_ROOT:-/home/work/.data/hrm_text_lora}"
RESUME_FROM="${RESUME_FROM:?set RESUME_FROM=/path/to/KoHRM full checkpoint directory}"
NPROC="${NPROC:-8}"
GBS="${GBS:-32768}"
RANK="${LORA_RANK:-16}"
ALPHA="${LORA_ALPHA:-32.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LR="${LR:-8.0e-5}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
KEEP_LAST="${KEEP_LAST:-2}"

run_lora() {
  local name="$1"
  local data="$2"
  local epochs="$3"
  local out="$OUT_ROOT/$name"

  mkdir -p "$out"
  cd "$ROOT"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  WANDB_MODE="${WANDB_MODE:-offline}" \
  WANDB_DIR="${WANDB_DIR:-/home/work/.data/wandb}" \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
  torchrun --standalone --nproc_per_node="$NPROC" train_lora.py \
    arch/size@arch=XL \
    "data.path=$data" \
    "resume_from=$RESUME_FROM" \
    "checkpoint_path=$out" \
    "run_name=$name" \
    "global_batch_size=$GBS" \
    "epochs=$epochs" \
    "lr=$LR" \
    "checkpoint_step_interval=$SAVE_STEPS" \
    "checkpoint_keep_last=$KEEP_LAST" \
    "lora.rank=$RANK" \
    "lora.alpha=$ALPHA" \
    "lora.dropout=$LORA_DROPOUT"
}

case "${1:-all}" in
  behavior-mini)
    run_lora "KoHRM-Text-1.4B-lora-behavior-mini-v1" "$DATA_ROOT/kohrm_sft_behavior_mini_v1" "${EPOCHS:-1}"
    ;;
  terminal-tool)
    run_lora "KoHRM-Text-1.4B-lora-terminal-tool-core-v1" "$DATA_ROOT/kohrm_sft_terminal_tool_core_v1" "${EPOCHS:-1}"
    ;;
  korean-domain)
    run_lora "KoHRM-Text-1.4B-lora-korean-domain-core-v1" "$DATA_ROOT/kohrm_sft_korean_domain_core_v1" "${EPOCHS:-1}"
    ;;
  behavior-core)
    run_lora "KoHRM-Text-1.4B-lora-behavior-core-v1" "$DATA_ROOT/kohrm_sft_behavior_core_v1" "${EPOCHS:-1}"
    ;;
  text2sql-core)
    run_lora "KoHRM-Text-1.4B-lora-text2sql-core-v1" "$DATA_ROOT/kohrm_sft_text2sql_core_v1" "${EPOCHS:-1}"
    ;;
  text2sql-core-clean)
    run_lora "KoHRM-Text-1.4B-lora-text2sql-core-clean-v1" "$DATA_ROOT/kohrm_sft_text2sql_core_clean_v1" "${EPOCHS:-1}"
    ;;
  text2sql-duckdb)
    run_lora "KoHRM-Text-1.4B-lora-text2sql-duckdb-v1" "$DATA_ROOT/kohrm_sft_text2sql_duckdb_v1" "${EPOCHS:-1}"
    ;;
  text2sql-core-clean-duckdb)
    run_lora "KoHRM-Text-1.4B-lora-text2sql-core-clean-duckdb-v1" "$DATA_ROOT/kohrm_sft_text2sql_core_clean_duckdb_v1" "${EPOCHS:-1}"
    ;;
  text2sql-core-duckdb)
    run_lora "KoHRM-Text-1.4B-lora-text2sql-core-duckdb-v1" "$DATA_ROOT/kohrm_sft_text2sql_core_duckdb_v1" "${EPOCHS:-1}"
    ;;
  text2sql-large)
    run_lora "KoHRM-Text-1.4B-lora-text2sql-large-v1" "$DATA_ROOT/kohrm_sft_text2sql_large_v1" "${EPOCHS:-1}"
    ;;
  text2sql-all)
    run_lora "KoHRM-Text-1.4B-lora-text2sql-all-v1" "$DATA_ROOT/kohrm_sft_text2sql_all_v1" "${EPOCHS:-1}"
    ;;
  all)
    "${BASH_SOURCE[0]}" behavior-mini
    "${BASH_SOURCE[0]}" korean-domain
    "${BASH_SOURCE[0]}" terminal-tool
    "${BASH_SOURCE[0]}" behavior-core
    ;;
  phase1)
    "${BASH_SOURCE[0]}" behavior-mini
    "${BASH_SOURCE[0]}" korean-domain
    "${BASH_SOURCE[0]}" terminal-tool
    ;;
  *)
    echo "usage: $0 {behavior-mini|terminal-tool|korean-domain|behavior-core|text2sql-core-clean|text2sql-core-clean-duckdb|text2sql-core|text2sql-duckdb|text2sql-core-duckdb|text2sql-large|text2sql-all|phase1|all}" >&2
    exit 2
    ;;
esac
