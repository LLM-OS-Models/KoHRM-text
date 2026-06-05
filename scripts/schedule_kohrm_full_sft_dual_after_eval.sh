#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-/home/work/.data/hrm_text_logs}"
PREP_SCRIPT="${PREP_SCRIPT:-$ROOT/scripts/prepare_kohrm_full_sft_dual_data.sh}"
RUN_SCRIPT="${RUN_SCRIPT:-$ROOT/scripts/run_kohrm_full_sft_dual_4gpu.sh}"

LFM_GPUS="${LFM_GPUS:-1,2,3,7}"
TOP2_GPUS="${TOP2_GPUS:-0,4,5,6}"
MEM_FREE_THRESHOLD_MB="${MEM_FREE_THRESHOLD_MB:-20000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

SCHEDULE_LOG="${SCHEDULE_LOG:-$LOG_ROOT/kohrm_full_sft_dual_scheduler_$(date -u +%Y%m%dT%H%M%SZ).log}"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$SCHEDULE_LOG"
}

gpu_used_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="$gpu" | tr -d ' '
}

gpu_set_free() {
  local gpus_csv="$1"
  IFS=',' read -ra gpus <<<"$gpus_csv"
  local gpu used
  for gpu in "${gpus[@]}"; do
    used="$(gpu_used_mb "$gpu")"
    if [[ "$used" -gt "$MEM_FREE_THRESHOLD_MB" ]]; then
      return 1
    fi
  done
  return 0
}

gpu_set_status() {
  local gpus_csv="$1"
  IFS=',' read -ra gpus <<<"$gpus_csv"
  local parts=()
  local gpu used
  for gpu in "${gpus[@]}"; do
    used="$(gpu_used_mb "$gpu")"
    parts+=("gpu${gpu}=${used}MiB")
  done
  printf '%s ' "${parts[@]}"
}

wait_gpu_set() {
  local name="$1"
  local gpus="$2"
  log "waiting for $name GPUs $gpus below ${MEM_FREE_THRESHOLD_MB}MiB"
  while ! gpu_set_free "$gpus"; do
    log "$name still busy: $(gpu_set_status "$gpus")"
    sleep "$POLL_SECONDS"
  done
  log "$name GPUs are free: $(gpu_set_status "$gpus")"
}

main() {
  mkdir -p "$LOG_ROOT"
  log "scheduler started; LFM first on $LFM_GPUS, top2 second on $TOP2_GPUS"
  log "preparing datasets via $PREP_SCRIPT"
  bash "$PREP_SCRIPT" >>"$SCHEDULE_LOG" 2>&1
  log "dataset preparation finished"

  wait_gpu_set "lfm25-fullsft" "$LFM_GPUS"
  (
    export LFM_GPUS
    bash "$RUN_SCRIPT" lfm
  ) >>"$SCHEDULE_LOG" 2>&1 &
  pid_lfm=$!
  log "started lfm25-fullsft pid=$pid_lfm"

  wait_gpu_set "top2-merge-fullsft" "$TOP2_GPUS"
  (
    export TOP2_GPUS
    bash "$RUN_SCRIPT" top2
  ) >>"$SCHEDULE_LOG" 2>&1 &
  pid_top2=$!
  log "started top2-merge-fullsft pid=$pid_top2"

  wait "$pid_lfm"
  log "lfm25-fullsft finished"
  wait "$pid_top2"
  log "top2-merge-fullsft finished"
}

main "$@"
