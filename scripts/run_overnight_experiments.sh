#!/usr/bin/env bash
# Overnight 12h+ experiment chain. Tries every approach to beat 51.4%.
set -u

ROOT=/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text
GEMMA_BASE=/home/work/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445
CKPTS=/home/work/.data/bar_exam_sft/ckpts
RES=/home/work/.data/bar_exam_sft/results
RAW=/home/work/.data/bar_exam_sft/raw
LOG=/home/work/.data/bar_exam_sft/overnight_v2.log

echo "===== overnight v2 START $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$LOG"

run() {
  local name="$1"; shift
  echo "----- [$name] START $(date -u +%H:%M:%SZ) -----" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "----- [$name] DONE rc=$rc $(date -u +%H:%M:%SZ) -----" | tee -a "$LOG"
  # Print result if summary exists
  return $rc
}

cd "$ROOT"

# ===== PHASE 1: Quick eval experiments (no training needed) =====

# 1a. Temperature 0.3 (sampling, single shot)
run "temp03_eval" env CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval_bar_exam_round15_fewshot.py \
  --base-model "$GEMMA_BASE" \
  --adapter "$CKPTS/gemma4_31b_cot_mix_qlora" \
  --train-jsonl "$RAW/bar_exam_train_1_14.jsonl" \
  --output "$RES/gemma31b_cot_temp03_pred.jsonl" \
  --summary-out "$RES/gemma31b_cot_temp03_summary.json" \
  --n-shots 0 --temperature 0.3 --max-new-tokens 64 || true

# 1b. Temperature 0.1
run "temp01_eval" env CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval_bar_exam_round15_fewshot.py \
  --base-model "$GEMMA_BASE" \
  --adapter "$CKPTS/gemma4_31b_cot_mix_qlora" \
  --train-jsonl "$RAW/bar_exam_train_1_14.jsonl" \
  --output "$RES/gemma31b_cot_temp01_pred.jsonl" \
  --summary-out "$RES/gemma31b_cot_temp01_summary.json" \
  --n-shots 0 --temperature 0.1 --max-new-tokens 64 || true

# 1c. Few-shot 1 (1 example per subject)
run "fewshot1_eval" env CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval_bar_exam_round15_fewshot.py \
  --base-model "$GEMMA_BASE" \
  --adapter "$CKPTS/gemma4_31b_cot_mix_qlora" \
  --train-jsonl "$RAW/bar_exam_train_1_14.jsonl" \
  --output "$RES/gemma31b_cot_fewshot1_pred.jsonl" \
  --summary-out "$RES/gemma31b_cot_fewshot1_summary.json" \
  --n-shots 1 --max-new-tokens 64 || true

# 1d. Few-shot 5
run "fewshot5_eval" env CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval_bar_exam_round15_fewshot.py \
  --base-model "$GEMMA_BASE" \
  --adapter "$CKPTS/gemma4_31b_cot_mix_qlora" \
  --train-jsonl "$RAW/bar_exam_train_1_14.jsonl" \
  --output "$RES/gemma31b_cot_fewshot5_pred.jsonl" \
  --summary-out "$RES/gemma31b_cot_fewshot5_summary.json" \
  --n-shots 5 --max-new-tokens 64 || true

# 1e. mix_all model greedy (비교용)
run "mixall_greedy" env CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval_bar_exam_round15_full150.py \
  --base-model "$GEMMA_BASE" \
  --adapter "$CKPTS/gemma4_31b_mix_all_qlora" \
  --output "$RES/gemma31b_mixall_150_pred.jsonl" \
  --summary-out "$RES/gemma31b_mixall_150_summary.json" \
  --model-class gemma4 --max-new-tokens 256 || true

# ===== PHASE 2: New SFT experiments =====

# 2a. 1-14 answer-only + current-law answer-only (no hard, no CoT)
cat "$RAW/bar_exam_train_1_14.jsonl" "$RAW/bar_exam_current_law_1000_answer_only.jsonl" \
  > "$RAW/bar_exam_mix_14_curr.jsonl"
mkdir -p "$CKPTS/gemma4_31b_mix14curr_qlora"
run "mix14curr_sft" env CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u scripts/qlora_gemma31b_bar_exam.py \
  --model "$GEMMA_BASE" \
  --train-jsonl "$RAW/bar_exam_mix_14_curr.jsonl" \
  --output "$CKPTS/gemma4_31b_mix14curr_qlora" \
  --epochs 2 --lr 1e-4 \
  --per-device-batch 8 --grad-accum 2 \
  --max-len 1536 --warmup-steps 15 --logging-steps 20 \
  --lora-r 64 --lora-alpha 128 || true

run "mix14curr_eval" env CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval_bar_exam_round15_full150.py \
  --base-model "$GEMMA_BASE" \
  --adapter "$CKPTS/gemma4_31b_mix14curr_qlora" \
  --output "$RES/gemma31b_mix14curr_pred.jsonl" \
  --summary-out "$RES/gemma31b_mix14curr_summary.json" \
  --model-class gemma4 --max-new-tokens 256 || true

# 2b. LoRA r=128, alpha=256 (CoT mix)
mkdir -p "$CKPTS/gemma4_31b_cot_r128"
run "cot_r128_sft" env CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u scripts/qlora_gemma31b_bar_exam.py \
  --model "$GEMMA_BASE" \
  --train-jsonl "$RAW/bar_exam_cot_mix.jsonl" \
  --output "$CKPTS/gemma4_31b_cot_r128" \
  --epochs 2 --lr 1e-4 \
  --per-device-batch 8 --grad-accum 2 \
  --max-len 1536 --warmup-steps 15 --logging-steps 20 \
  --lora-r 128 --lora-alpha 256 || true

run "cot_r128_eval" env CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval_bar_exam_round15_full150.py \
  --base-model "$GEMMA_BASE" \
  --adapter "$CKPTS/gemma4_31b_cot_r128" \
  --output "$RES/gemma31b_cot_r128_pred.jsonl" \
  --summary-out "$RES/gemma31b_cot_r128_summary.json" \
  --model-class gemma4 --max-new-tokens 256 || true

# 2c. CoT mix + current-law 2x weighted (more law knowledge)
python3 -c "
import json
rows = []
with open('$RAW/bar_exam_cot_mix.jsonl') as f:
    for l in f: rows.append(json.loads(l))
# Double current-law CoT samples
curr_cot = [r for r in rows if 'current_law' in r.get('source','')]
rows.extend(curr_cot)
with open('$RAW/bar_exam_cot_mix_curr2x.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f'curr2x: {len(rows)} rows')
" >> "$LOG" 2>&1

mkdir -p "$CKPTS/gemma4_31b_cot_curr2x"
run "cot_curr2x_sft" env CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u scripts/qlora_gemma31b_bar_exam.py \
  --model "$GEMMA_BASE" \
  --train-jsonl "$RAW/bar_exam_cot_mix_curr2x.jsonl" \
  --output "$CKPTS/gemma4_31b_cot_curr2x" \
  --epochs 2 --lr 1e-4 \
  --per-device-batch 8 --grad-accum 2 \
  --max-len 1536 --warmup-steps 15 --logging-steps 20 \
  --lora-r 64 --lora-alpha 128 || true

run "cot_curr2x_eval" env CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval_bar_exam_round15_full150.py \
  --base-model "$GEMMA_BASE" \
  --adapter "$CKPTS/gemma4_31b_cot_curr2x" \
  --output "$RES/gemma31b_cot_curr2x_pred.jsonl" \
  --summary-out "$RES/gemma31b_cot_curr2x_summary.json" \
  --model-class gemma4 --max-new-tokens 256 || true

# 2d. CoT mix with LR 5e-5 (lower LR, same data)
mkdir -p "$CKPTS/gemma4_31b_cot_lr5e5"
run "cot_lr5e5_sft" env CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u scripts/qlora_gemma31b_bar_exam.py \
  --model "$GEMMA_BASE" \
  --train-jsonl "$RAW/bar_exam_cot_mix.jsonl" \
  --output "$CKPTS/gemma4_31b_cot_lr5e5" \
  --epochs 2 --lr 5e-5 \
  --per-device-batch 8 --grad-accum 2 \
  --max-len 1536 --warmup-steps 15 --logging-steps 20 \
  --lora-r 64 --lora-alpha 128 || true

run "cot_lr5e5_eval" env CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval_bar_exam_round15_full150.py \
  --base-model "$GEMMA_BASE" \
  --adapter "$CKPTS/gemma4_31b_cot_lr5e5" \
  --output "$RES/gemma31b_cot_lr5e5_pred.jsonl" \
  --summary-out "$RES/gemma31b_cot_lr5e5_summary.json" \
  --model-class gemma4 --max-new-tokens 256 || true

# 2e. CoT mix with LR 2e-4 (higher LR)
mkdir -p "$CKPTS/gemma4_31b_cot_lr2e4"
run "cot_lr2e4_sft" env CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u scripts/qlora_gemma31b_bar_exam.py \
  --model "$GEMMA_BASE" \
  --train-jsonl "$RAW/bar_exam_cot_mix.jsonl" \
  --output "$CKPTS/gemma4_31b_cot_lr2e4" \
  --epochs 2 --lr 2e-4 \
  --per-device-batch 8 --grad-accum 2 \
  --max-len 1536 --warmup-steps 15 --logging-steps 20 \
  --lora-r 64 --lora-alpha 128 || true

run "cot_lr2e4_eval" env CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval_bar_exam_round15_full150.py \
  --base-model "$GEMMA_BASE" \
  --adapter "$CKPTS/gemma4_31b_cot_lr2e4" \
  --output "$RES/gemma31b_cot_lr2e4_pred.jsonl" \
  --summary-out "$RES/gemma31b_cot_lr2e4_summary.json" \
  --model-class gemma4 --max-new-tokens 256 || true

# ===== PHASE 3: Ensemble vote =====
# Collect predictions from top models and majority vote
run "ensemble_vote" python3 -c "
import json
from collections import Counter
from pathlib import Path

models = [
    '$RES/gemma31b_cot_mix_150_pred.jsonl',
    '$RES/gemma31b_mixall_150_pred.jsonl',
    '$RES/gemma31b_r256_pred.jsonl',
]
# Load preds per question
all_preds = {}  # idx -> list of preds
for mf in models:
    p = Path(mf)
    if not p.exists(): continue
    for line in p.open(encoding='utf-8'):
        o = json.loads(line)
        idx = o.get('idx', o.get('qno', 0))
        pred = o.get('pred')
        if pred is not None:
            all_preds.setdefault(idx, []).append(pred)

# Load gold
import csv
gold = {}
with open('$RAW/data/questions.csv', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        if r['round'] == '15':
            a = r['answer'].strip()
            if a.isdigit() and 1 <= int(a) <= 5:
                gold[r['question_no']] = a
            elif a.isdigit() and len(a)==2 and a.endswith('0') and 1<=int(a[0])<=5:
                gold[r['question_no']] = a[0]

nc = nt = 0
for qno, preds in all_preds.items():
    vote = Counter(preds).most_common(1)[0][0]
    g = gold.get(qno)
    ok = g is not None and vote == g
    nt += 1; nc += int(ok)
print(f'Ensemble: {nc}/{nt} = {nc/nt:.3f}' if nt else 'no preds')
" || true

# ===== SUMMARY =====
echo "===== SUMMARY $(date -u +%H:%M:%SZ) =====" | tee -a "$LOG"
for f in "$RES"/*summary*.json; do
    name=$(basename "$f" .json)
    acc=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d.get('accuracy',0):.3f}\")" 2>/dev/null)
    echo "  $name: $acc" | tee -a "$LOG"
done
echo "===== overnight v2 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$LOG"
