#!/usr/bin/env bash
set -euo pipefail

TRAIN_LOG="${1:-/home/work/.data/hrm_text_logs/KoHRM-Text-1.4B-stage0-available-mix-gbs172.log}"
SCHED_LOG="${2:-/home/work/.data/hrm_text_logs/KoHRM-Text-1.4B-stage-chain-scheduler.log}"
TOKENIZED_ROOT="${3:-/home/work/.data/hrm_text_tokenized/koterm_hrm_cleaned_fastcap_v1}"

echo "===== KoHRM status $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
echo "--- processes"
pgrep -af "schedule_kohrm|schedule_kohm|watch_and_upload|torchrun|pretrain.py|target/release/tokenizer" || true
echo "--- training log"
tail -100 "$TRAIN_LOG" 2>/dev/null || true
echo "--- scheduler log"
tail -80 "$SCHED_LOG" 2>/dev/null || true
echo "--- gpu"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits || true
echo "--- preprocessing"
find "$TOKENIZED_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
find "$TOKENIZED_ROOT" -mindepth 2 -maxdepth 2 -name metadata.json 2>/dev/null | wc -l
du -sh "$TOKENIZED_ROOT" 2>/dev/null || true
echo "--- checkpoints"
find /home/work/.data/hrm_text_checkpoints -maxdepth 2 -type d -name 'fsdp2_epoch_*' -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -20
echo "--- disk"
df -h /home/work/.data
