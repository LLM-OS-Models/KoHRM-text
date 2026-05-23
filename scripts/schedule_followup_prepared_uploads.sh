#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text"
DATA_IO_ROOT="/home/work/.projects/LLM-OS-Models/Terminal/data_io"
TOKENIZER="/home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1/tokenizer.json"
LOG_DIR="/home/work/.data/hrm_text_logs"
EXTRA_DIR="/home/work/.data/huggingface/hrm_text_extra/sft"
PREP_DIR="/home/work/.data/hrm_text_prepared"
STAGE_ROOT="/home/work/.data/hrm_text_hf_upload_stage"
DATASET_REPO="LLM-OS-Models/KoHRM-Text-1.4B-prepared-data"
TOKENIZED_ROOT="/home/work/.data/hrm_text_tokenized/koterm_hrm_cleaned_fastcap_v1"

mkdir -p "$LOG_DIR" "$EXTRA_DIR" "$PREP_DIR" "$STAGE_ROOT"
LOG="$LOG_DIR/followup_prepared_uploads_20260524.log"
exec >>"$LOG" 2>&1

log() {
  printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

wait_for_current_dataset_upload() {
  while pgrep -af "upload_folder_to_hf.py .*${DATASET_REPO}" >/dev/null 2>&1; do
    log "waiting for existing ${DATASET_REPO} upload to finish"
    sleep 120
  done
}

upload_stage_folder() {
  local folder="$1"
  local label="$2"
  wait_for_current_dataset_upload
  log "uploading ${label}: ${folder}"
  python scripts/upload_folder_to_hf.py \
    --folder "$folder" \
    --repo-id "$DATASET_REPO" \
    --repo-type dataset \
    --large \
    --num-workers 4
  log "uploaded ${label}"
}

cd "$ROOT"
log "follow-up prepared upload scheduler started"

LEGAL_JSONL="$EXTRA_DIR/korean_legal_tasks_full_20260524.jsonl"
LEGAL_PREP="$PREP_DIR/korean_legal_tasks_full_v1"
LEGAL_STAGE="$STAGE_ROOT/LLM-OS-Models__KoHRM-Text-1.4B-prepared-data-legal-full"

log "building uncapped full Korean legal task JSONL"
rm -f "$LEGAL_JSONL"
python scripts/build_korean_legal_sft_data.py \
  --output "$LEGAL_JSONL" \
  --max-output-mib 0 \
  --seed 42 \
  --root law=/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text/legalize-kr \
  --root ordinance=/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text/ordinance-kr \
  --root admrule=/home/work/.projects/LLM-OS-Models/Terminal/admrule-kr \
  --root precedent=/home/work/.projects/LLM-OS-Models/Terminal/precedent-kr

log "preparing full Korean legal task V1Dataset"
rm -rf "$LEGAL_PREP"
python scripts/prepare_sft_data.py \
  --train "$LEGAL_JSONL" \
  --tokenizer "$TOKENIZER" \
  --output "$LEGAL_PREP" \
  --epochs 1 \
  --context-size 4097 \
  --overflow-policy truncate-instruction-middle \
  --truncate-head-tokens 768 \
  --strip-think-blocks \
  --progress-interval 20000

log "staging full Korean legal task upload"
rm -rf "$LEGAL_STAGE"
mkdir -p "$LEGAL_STAGE/raw_jsonl"
cp -al "$LEGAL_PREP" "$LEGAL_STAGE/korean_legal_tasks_full_v1"
cp -al "$LEGAL_JSONL" "$LEGAL_STAGE/raw_jsonl/korean_legal_tasks_full_20260524.jsonl"
cat > "$LEGAL_STAGE/LEGAL_FULL_TASKS_README.md" <<'EOF'
# KoHRM Full Korean Legal Task Data

This upload adds the uncapped Korean legal/admin task dataset generated from:

- legalize-kr/legalize-kr
- legalize-kr/ordinance-kr
- local admrule-kr
- local precedent-kr

It complements the raw full Korean legal/admin datasets already present in this repository.
EOF

upload_stage_folder "$LEGAL_STAGE" "full Korean legal task data"

HRM_FULL_PREP="$PREP_DIR/koterm_hrm_cleaned_full_nocap_v1"
HRM_FULL_STAGE="$STAGE_ROOT/LLM-OS-Models__KoHRM-Text-1.4B-prepared-data-hrm-full-nocap"

log "waiting for HRM full/no-cap tokenizer process to finish"
while pgrep -af "target/release/tokenizer .*koterm_hrm_cleaned_fastcap_v1" >/dev/null 2>&1; do
  sleep 300
done
log "HRM tokenizer process finished; waiting for filesystem to settle"
sleep 180

if [ -d "$DATA_IO_ROOT" ] && [ -f "$DATA_IO_ROOT/sample_tokenized.py" ]; then
  log "sampling full/no-cap HRM tokenized corpus into V1Dataset"
  rm -rf "$HRM_FULL_PREP"
  python "$DATA_IO_ROOT/sample_tokenized.py" \
    tokenized_path="$TOKENIZED_ROOT" \
    output_path="$HRM_FULL_PREP" \
    prefix_config_path="$DATA_IO_ROOT/prefix_config.yaml" \
    epochs=1 \
    context_size=4097

  log "staging full/no-cap HRM V1Dataset upload"
  rm -rf "$HRM_FULL_STAGE"
  mkdir -p "$HRM_FULL_STAGE"
  cp -al "$HRM_FULL_PREP" "$HRM_FULL_STAGE/koterm_hrm_cleaned_full_nocap_v1"
  cat > "$HRM_FULL_STAGE/HRM_FULL_NOCAP_README.md" <<'EOF'
# KoHRM Full No-Cap HRM Cleaned Prepared Dataset

This upload adds the full/no-cap retokenized HRM cleaned corpus prepared with the KoHRM 131K tokenizer.

The source is the upstream `sapientinc/HRM-Text-data-io-cleaned-20260515` corpus, retokenized for KoHRM-Text.
EOF

  upload_stage_folder "$HRM_FULL_STAGE" "full/no-cap HRM prepared dataset"
else
  log "data_io sample_tokenized.py not found; skipped HRM full/no-cap V1Dataset packing"
fi

log "follow-up prepared upload scheduler finished"
