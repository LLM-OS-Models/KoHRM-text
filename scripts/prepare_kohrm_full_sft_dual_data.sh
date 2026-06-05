#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/home/work/.data/hrm_text_raw_sft}"
PREP_ROOT="${PREP_ROOT:-/home/work/.data/hrm_text_prepared}"
TOKENIZER="${TOKENIZER:-/home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1/tokenizer.json}"
LFM_DATASET="${LFM_DATASET:-/home/work/.data/liquid_cli_sft/datasets/lfm25_8b_a1b_terminal_full_toolbench_full_conversations_v1}"

EPOCHS="${EPOCHS:-1}"
CONTEXT_SIZE="${CONTEXT_SIZE:-8193}"
SEED="${SEED:-20260605}"
OVERWRITE="${OVERWRITE:-0}"

LFM_RAW="${LFM_RAW:-$RAW_ROOT/lfm25_terminal_toolbench_hrm_turns_v1.jsonl}"
LFM_PREP="${LFM_PREP:-$PREP_ROOT/kohrm_sft_lfm25_terminal_toolbench_full_v1}"
TOP2_PREP="${TOP2_PREP:-$PREP_ROOT/kohrm_sft_top2_terminal_tool_raw8192_v1}"

TOP1_INPUT="${TOP1_INPUT:-$PREP_ROOT/kohrm_sft_terminal_tool_core_v1}"
TOP2_INPUT="${TOP2_INPUT:-$PREP_ROOT/kohrm_sft_comp_terminal_80m_v1}"
TOOLBENCH_RAW="${TOOLBENCH_RAW:-/home/work/.data/huggingface/hrm_text_extra/sft/toolbench_actions_512m.jsonl}"
LOCAL_TERMINAL_RAW="${LOCAL_TERMINAL_RAW:-/home/work/.data/huggingface/hrm_text_extra/sft/local_terminal_conversations_ctx9k_resp6k_20260524.jsonl}"
TOP2_TARGET_TOKENS="${TOP2_TARGET_TOKENS:-245010000}"
TOP2_MODE="${TOP2_MODE:-raw8192}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing file: $path" >&2
    exit 1
  fi
}

require_prepared() {
  local path="$1"
  for rel in metadata.json tokens.npy epoch_0/inst_start.npy epoch_0/inst_len.npy epoch_0/resp_start.npy epoch_0/resp_len.npy; do
    if [[ ! -e "$path/$rel" ]]; then
      echo "missing prepared dataset file: $path/$rel" >&2
      exit 1
    fi
  done
}

maybe_remove_dir() {
  local path="$1"
  if [[ -e "$path" && "$OVERWRITE" == "1" ]]; then
    rm -rf "$path"
  fi
}

mkdir -p "$RAW_ROOT" "$PREP_ROOT"
require_file "$TOKENIZER"
require_prepared "$TOP1_INPUT"
require_prepared "$TOP2_INPUT"
require_file "$TOOLBENCH_RAW"
require_file "$LOCAL_TERMINAL_RAW"

cd "$ROOT"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-256}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-64}"

if [[ ! -s "$LFM_RAW" || "$OVERWRITE" == "1" ]]; then
  rm -f "$LFM_RAW" "$LFM_RAW.stats.json"
  python scripts/build_lfm25_terminal_toolbench_hrm_sft_data.py \
    --dataset-path "$LFM_DATASET" \
    --output "$LFM_RAW" \
    --source-filter all \
    --tool-call-format action \
    --turn-selection "${TURN_SELECTION:-final-assistant}" \
    --condition direct \
    --max-history-messages "${MAX_HISTORY_MESSAGES:-10}" \
    --max-instruction-chars "${MAX_INSTRUCTION_CHARS:-48000}" \
    --max-response-chars "${MAX_RESPONSE_CHARS:-24000}" \
    --min-response-chars "${MIN_RESPONSE_CHARS:-1}" \
    --progress-interval "${PROGRESS_INTERVAL:-10000}"
else
  echo "raw LFM HRM JSONL already exists: $LFM_RAW"
fi

if [[ ! -e "$LFM_PREP/metadata.json" || "$OVERWRITE" == "1" ]]; then
  maybe_remove_dir "$LFM_PREP"
  python scripts/prepare_sft_data.py \
    --train "$LFM_RAW" \
    --tokenizer "$TOKENIZER" \
    --output "$LFM_PREP" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --context-size "$CONTEXT_SIZE" \
    --overflow-policy truncate-instruction-middle \
    --truncate-head-tokens "${TRUNCATE_HEAD_TOKENS:-1536}" \
    --strip-think-blocks \
    --condition-override direct \
    --progress-interval "${PREP_PROGRESS_INTERVAL:-5000}"
else
  echo "prepared LFM HRM dataset already exists: $LFM_PREP"
fi

if [[ ! -e "$TOP2_PREP/metadata.json" || "$OVERWRITE" == "1" ]]; then
  maybe_remove_dir "$TOP2_PREP"
  if [[ "$TOP2_MODE" == "merge4096" ]]; then
    python scripts/merge_prepared_sft_data.py \
      --inputs "$TOP1_INPUT" "$TOP2_INPUT" \
      --output "$TOP2_PREP" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --copy-tokenizer
  elif [[ "$TOP2_MODE" == "raw8192" ]]; then
    python scripts/prepare_sft_data.py \
      --train "$TOOLBENCH_RAW" "$LOCAL_TERMINAL_RAW" \
      --tokenizer "$TOKENIZER" \
      --output "$TOP2_PREP" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --context-size "$CONTEXT_SIZE" \
      --overflow-policy truncate-instruction-middle \
      --truncate-head-tokens "${TRUNCATE_HEAD_TOKENS:-1536}" \
      --strip-think-blocks \
      --condition-override direct \
      --target-tokens "$TOP2_TARGET_TOKENS" \
      --progress-interval "${PREP_PROGRESS_INTERVAL:-5000}"
  else
    echo "unknown TOP2_MODE=$TOP2_MODE; expected raw8192 or merge4096" >&2
    exit 2
  fi
else
  echo "prepared top2 merge dataset already exists: $TOP2_PREP"
fi

LFM_PREP="$LFM_PREP" TOP2_PREP="$TOP2_PREP" python - <<'PY'
from pathlib import Path
import json
import os
for path in [
    Path(os.environ["LFM_PREP"]),
    Path(os.environ["TOP2_PREP"]),
]:
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        continue
    meta = json.loads(meta_path.read_text())
    print(f"{path}: total_length={meta.get('total_length'):,} max_seq_len={meta.get('max_seq_len')}")
PY
