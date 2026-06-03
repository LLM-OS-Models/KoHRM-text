#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/home/work/.data/hrm_text_raw/text2sql}"
PREP_ROOT="${PREP_ROOT:-/home/work/.data/hrm_text_prepared}"
HF_HOME="${HF_HOME:-/home/work/.data/hf_cache}"
TOKENIZER="${TOKENIZER:-/home/work/.data/hrm_text_prepared/koterm_korean_tool_finance_mix_v1/tokenizer.json}"
EPOCHS="${EPOCHS:-2}"
CONTEXT_SIZE="${CONTEXT_SIZE:-4097}"
SEED="${SEED:-20260603}"
STREAMING="${STREAMING:-false}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10000}"
MAX_SCHEMA_CHARS="${MAX_SCHEMA_CHARS:-24000}"

mkdir -p "$RAW_ROOT" "$PREP_ROOT" "$HF_HOME"
export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

build_jsonl() {
  local profile="$1"
  local out="$2"
  shift 2
  local streaming_flag="--no-streaming"
  if [[ "$STREAMING" == "1" || "$STREAMING" == "true" || "$STREAMING" == "yes" ]]; then
    streaming_flag="--streaming"
  fi
  python "$ROOT/scripts/build_text2sql_sft_data.py" \
    --profile "$profile" \
    --output "$out" \
    --cache-dir "$HF_HOME" \
    --progress-interval "$PROGRESS_INTERVAL" \
    --max-schema-chars "$MAX_SCHEMA_CHARS" \
    "$streaming_flag" \
    "$@"
}

prepare_v1() {
  local input="$1"
  local out="$2"
  python "$ROOT/scripts/prepare_sft_data.py" \
    --train "$input" \
    --tokenizer "$TOKENIZER" \
    --output "$out" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --context-size "$CONTEXT_SIZE" \
    --overflow-policy truncate-instruction-middle \
    --truncate-head-tokens 1024 \
    --condition-override direct \
    --progress-interval "$PROGRESS_INTERVAL"
}

MODE="${1:-core-duckdb}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$MODE" in
  core)
    build_jsonl core "$RAW_ROOT/text2sql_core_sft.jsonl"
    prepare_v1 "$RAW_ROOT/text2sql_core_sft.jsonl" "$PREP_ROOT/kohrm_sft_text2sql_core_v1"
    ;;
  core-clean)
    build_jsonl core_clean "$RAW_ROOT/text2sql_core_clean_sft.jsonl"
    prepare_v1 "$RAW_ROOT/text2sql_core_clean_sft.jsonl" "$PREP_ROOT/kohrm_sft_text2sql_core_clean_v1"
    ;;
  duckdb)
    build_jsonl duckdb "$RAW_ROOT/text2sql_duckdb_sft.jsonl"
    prepare_v1 "$RAW_ROOT/text2sql_duckdb_sft.jsonl" "$PREP_ROOT/kohrm_sft_text2sql_duckdb_v1"
    ;;
  schema-heavy)
    build_jsonl schema_heavy "$RAW_ROOT/text2sql_schema_heavy_sft.jsonl" "$@"
    prepare_v1 "$RAW_ROOT/text2sql_schema_heavy_sft.jsonl" "$PREP_ROOT/kohrm_sft_text2sql_schema_heavy_v1"
    ;;
  mix-v2)
    build_jsonl mix_v2 "$RAW_ROOT/text2sql_mix_v2_sft.jsonl" "$@"
    prepare_v1 "$RAW_ROOT/text2sql_mix_v2_sft.jsonl" "$PREP_ROOT/kohrm_sft_text2sql_mix_v2_v1"
    ;;
  core-clean-duckdb)
    "$BASH_SOURCE" core-clean
    "$BASH_SOURCE" duckdb
    python "$ROOT/scripts/merge_prepared_sft_data.py" \
      --inputs \
        "$PREP_ROOT/kohrm_sft_text2sql_core_clean_v1" \
        "$PREP_ROOT/kohrm_sft_text2sql_duckdb_v1" \
      --output "$PREP_ROOT/kohrm_sft_text2sql_core_clean_duckdb_v1" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --copy-tokenizer
    ;;
  core-duckdb)
    "$BASH_SOURCE" core
    "$BASH_SOURCE" duckdb
    python "$ROOT/scripts/merge_prepared_sft_data.py" \
      --inputs \
        "$PREP_ROOT/kohrm_sft_text2sql_core_v1" \
        "$PREP_ROOT/kohrm_sft_text2sql_duckdb_v1" \
      --output "$PREP_ROOT/kohrm_sft_text2sql_core_duckdb_v1" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --copy-tokenizer
    ;;
  large)
    build_jsonl large "$RAW_ROOT/text2sql_large_sft.jsonl" "$@"
    prepare_v1 "$RAW_ROOT/text2sql_large_sft.jsonl" "$PREP_ROOT/kohrm_sft_text2sql_large_v1"
    ;;
  all)
    "$BASH_SOURCE" core-duckdb
    "$BASH_SOURCE" large
    python "$ROOT/scripts/merge_prepared_sft_data.py" \
      --inputs \
        "$PREP_ROOT/kohrm_sft_text2sql_core_duckdb_v1" \
        "$PREP_ROOT/kohrm_sft_text2sql_large_v1" \
      --output "$PREP_ROOT/kohrm_sft_text2sql_all_v1" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --copy-tokenizer
    ;;
  jsonl-only)
    build_jsonl core "$RAW_ROOT/text2sql_core_sft.jsonl"
    build_jsonl duckdb "$RAW_ROOT/text2sql_duckdb_sft.jsonl"
    ;;
  *)
    echo "usage: $0 {core-clean|core-clean-duckdb|core|duckdb|schema-heavy|mix-v2|core-duckdb|large|all|jsonl-only}" >&2
    echo "env: RAW_ROOT PREP_ROOT HF_HOME TOKENIZER EPOCHS CONTEXT_SIZE STREAMING" >&2
    exit 2
    ;;
esac
