#!/usr/bin/env bash
# Overnight chain: runs all remaining experiments sequentially on GPU 7.
# Each step is self-contained; failure of one step logs and moves on.
set -u

ROOT=/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text
GEMMA_BASE=/home/work/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445
QWEN_BASE=/home/work/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
CKPTS=/home/work/.data/bar_exam_sft/ckpts
RES=/home/work/.data/bar_exam_sft/results
LOG=/home/work/.data/bar_exam_sft/overnight_chain.log
RAW=/home/work/.data/bar_exam_sft/raw
HRM_BASE=/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage4d-korean-tool-finance-repeat3-gbs180
TOKENIZER=/home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1/tokenizer.json

echo "===== overnight chain START $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$LOG"

run() {
  local name="$1"; shift
  echo "----- [$name] START $(date -u +%H:%M:%SZ) -----" | tee -a "$LOG"
  if "$@" >> "$LOG" 2>&1; then
    echo "----- [$name] DONE $(date -u +%H:%M:%SZ) -----" | tee -a "$LOG"
    return 0
  else
    echo "----- [$name] FAILED $(date -u +%H:%M:%SZ) -----" | tee -a "$LOG"
    tail -20 "$LOG"
    return 1
  fi
}

# Step 0: wait for current r=256 SFT to finish
echo "waiting for r=256 SFT to finish..." | tee -a "$LOG"
until grep -q "QLoRA SFT DONE\|Traceback" "$RAW/../gemma4_31b_r256.log" 2>/dev/null; do
  sleep 60
done
echo "r=256 SFT done, proceeding" | tee -a "$LOG"

# Step 1: eval Gemma-4 r=256 (CoT, max_new 256)
run "gemma_r256_eval" python3 -u "$ROOT/scripts/eval_bar_exam_round15_full.py" \
  --model "$GEMMA_BASE" \
  --output "$RES/gemma31b_r256_pred.jsonl" \
  --summary-out "$RES/gemma31b_r256_summary.json" \
  --max-new-tokens 256 \
  --load-4bit || true
# Actually the r=256 is a LoRA adapter, need adapter eval
python3 -u - <<'PY' >> "$LOG" 2>&1 || echo "adapter eval inline failed"
import torch, json, csv, re, time
from pathlib import Path
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma4ForConditionalGeneration
from peft import PeftModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
BASE="/home/work/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
ADAPTER="/home/work/.data/bar_exam_sft/ckpts/gemma4_31b_cot_mix_r256"
QCSV="/home/work/.data/bar_exam_sft/raw/data/questions.csv"
OUT="/home/work/.data/bar_exam_sft/results/gemma31b_r256_pred.jsonl"
SUMMARY="/home/work/.data/bar_exam_sft/results/gemma31b_r256_summary.json"
ANS_RE=re.compile(r"정답\s*[:：]\s*([1-5])")
def build_user(r):
    return ("다음 변호사시험 선택형 문제를 읽고, 가장 옳은 정답 번호만 답하시오. "
            "출력은 '정답: <번호>' 형식 한 줄로만 하시오.\n\n"
            f"[회차] 제{r['round']}회\n[과목] {r['subject']}\n[문항번호] {r['question_no']}\n"
            f"[문제]\n{r['question_text'].strip()}")
def parse_pred(t):
    if not t: return None
    m=ANS_RE.search(t[:120])
    if m: return m.group(1)
    for ch in t[:60]:
        if ch in "12345": return ch
    return None
def norm_gold(a):
    a=(a or "").strip()
    if a in {"","정답없음"} or "," in a: return None
    if a.isdigit() and 1<=int(a)<=5: return a
    if len(a)==2 and a.endswith("0") and 1<=int(a[0])<=5: return a[0]
    return None
rows=[]
with open(QCSV, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        try:
            if int(r["round"])!=15: continue
        except: continue
        if norm_gold(r.get("answer","")) is None: continue
        rows.append(r)
print(f"rows: {len(rows)}", flush=True)
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
print("loading 4-bit base...", flush=True)
model=Gemma4ForConditionalGeneration.from_pretrained(BASE, quantization_config=bnb,
    trust_remote_code=True, attn_implementation="sdpa").eval()
model.config.use_cache=False
for name, mod in list(model.named_modules()):
    if isinstance(mod, Gemma4ClippableLinear):
        pn,_,cn=name.rpartition('.')
        parent=model.get_submodule(pn) if pn else model
        setattr(parent, cn, mod.linear)
print("loading adapter...", flush=True)
model=PeftModel.from_pretrained(model, ADAPTER).eval()
nc=nt=np=0; bs={}
t0=time.time()
with open(OUT,"w",encoding="utf-8") as fpo:
    for idx,r in enumerate(rows):
        msgs=[{"role":"user","content":build_user(r)}]
        try: prompt=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except: prompt=msgs[0]["content"]
        inputs=tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.inference_mode():
            out=model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen=tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gold=norm_gold(r["answer"]); pred=parse_pred(gen); ok=pred is not None and pred==gold
        nt+=1; np+=int(pred is not None); nc+=int(ok)
        s=r.get("subject","?"); b=bs.setdefault(s,{"c":0,"t":0,"p":0})
        b["c"]+=int(ok); b["t"]+=1; b["p"]+=int(pred is not None)
        fpo.write(json.dumps({"idx":idx,"gold":gold,"pred":pred,"correct":ok,"raw":gen[:500]}, ensure_ascii=False)+"\n")
        fpo.flush()
        if (idx+1)%20==0 or (idx+1)==len(rows):
            el=time.time()-t0
            print(f"  {idx+1}/{len(rows)} acc={nc/nt:.3f} parsed={np/nt:.3f} el={el:.0f}s", flush=True)
summary={"model":BASE,"adapter":ADAPTER,"n_total":nt,"n_parsed":np,"n_correct":nc,
    "accuracy":nc/nt if nt else 0, "parse_rate":np/nt if nt else 0,
    "by_subject":{s:{"accuracy":v["c"]/v["t"],"correct":v["c"],"total":v["t"],"parsed":v["p"]} for s,v in sorted(bs.items())}}
Path(SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

# Step 2: HRM + current-law CoT SFT
# Build CoT JSONL for HRM (KoHRM uses instruction/response, system prefix in instruction)
python3 -u - <<'PY' >> "$LOG" 2>&1
import json, re
from pathlib import Path
SRC_CURR="/home/work/.data/bar_exam_sft/raw_hard2/sft/train.jsonl"
SRC_HARD="/home/work/.data/bar_exam_sft/raw_hard/sft/train.jsonl"
OUT="/home/work/.data/bar_exam_sft/raw/bar_exam_hrm_cot_mix.jsonl"
TRAIN_14="/home/work/.data/bar_exam_sft/raw/bar_exam_train_1_14.jsonl"
SYSTEM="대한민국 현행 법령을 기준으로 변호사시험 선택형 문제를 풀이하는 법률 학습 도우미이다."

def build_inst(question, subject):
    user=(f"다음 변호사시험 선택형 문제를 읽고, 가장 옳은 정답 번호와 해설을 제시하시오. "
          f"첫 줄은 '정답: <번호>' 형식이어야 한다.\n\n[과목] {subject}\n[문제]\n{question.strip()}")
    return f"{SYSTEM}\n\n{user}"

def clean_q(raw):
    return re.sub(r"^[^\n]*정답 번호와 해설[^\n]*\n+", "", raw.strip())

with open(OUT,"w",encoding="utf-8") as out:
    # 1-14 회: 정답만 (해설 없음)
    with open(TRAIN_14, encoding="utf-8") as f:
        for line in f: out.write(line)
    # current-law + hard: CoT 응답 그대로
    for src_path, src_tag in [(SRC_CURR,"current_law"),(SRC_HARD,"hard_precedent")]:
        with open(src_path, encoding="utf-8") as f:
            for line in f:
                o=json.loads(line)
                msgs={m.get("role"):(m.get("content") or "") for m in (o.get("messages") or [])}
                user_raw=msgs.get("user",""); asst=msgs.get("assistant","").strip()
                if not asst: continue
                q=clean_q(user_raw)
                inst=build_inst(q, o.get("subject",""))
                out.write(json.dumps({
                    "instruction":inst, "response":asst,
                    "condition":"cot",
                    "source":f"hrm_bar_exam_{src_tag}_cot",
                    "id":o.get("id",""),
                    "subject":o.get("subject",""),
                }, ensure_ascii=False)+"\n")
print("HRM CoT mix built")
PY

# Prepare V1Dataset for HRM CoT
run "hrm_cot_prepare" python3 -u "$ROOT/scripts/prepare_sft_data.py" \
  --train "$RAW/bar_exam_hrm_cot_mix.jsonl" \
  --tokenizer "$TOKENIZER" \
  --output "$CKPTS/bar_exam_hrm_cot_prepared" \
  --epochs 2 --seed 20260616 --context-size 4097 \
  --overflow-policy truncate-instruction-middle \
  --truncate-head-tokens 1024 \
  --strip-think-blocks || true

# HRM CoT SFT (KoHRM runtime)
run "hrm_cot_sft" bash -c '
cd "$ROOT"
BAR_RUN_NAME=bar_exam_hrm_cot_ep2 \
BAR_RESUME_FROM="'"$HRM_BASE"'" \
BAR_PREPARED="'"$CKPTS"'/bar_exam_hrm_cot_prepared" \
BAR_OUT="'"$CKPTS"'/bar_exam_hrm_cot_ep2" \
BAR_EPOCHS=2 BAR_GBS=4096 BAR_LR=2.0e-5 \
BAR_LR_WARMUP_STEPS=10 BAR_SAVE_STEPS=200 BAR_KEEP_LAST=2 \
BAR_GPU=7 BAR_PORT=29690 \
bash scripts/run_bar_exam_sft_1gpu.sh' || true

# HRM CoT eval (KoHRM runtime)
run "hrm_cot_eval" bash -c '
cd "$ROOT"
KOHRM_DISABLE_INFERENCE_COMPILE=1 CUDA_VISIBLE_DEVICES=7 \
python3 -u scripts/eval_bar_exam_round15_simple.py \
  --ckpt-path "'"$CKPTS"'"/bar_exam_hrm_cot_ep2 \
  --ckpt-epoch 2 --no-ema \
  --questions-csv "'"$RAW"'"/data/questions.csv \
  --round 15 \
  --output "'"$RES"'"/hrm_cot_pred.jsonl \
  --summary-out "'"$RES"'"/hrm_cot_summary.json \
  --max-tokens 2048 --max-generation 256 --temp 0.0 --condition cot' || true

# Step 3: Qwen3.6 + CoT mix SFT
run "qwen36_cot_sft" bash -c '
cd "$ROOT"
mkdir -p "'"$CKPTS"'"/qwen36_27b_cot_mix_qlora
CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u scripts/qlora_qwen27b_bar_exam.py \
  --model "'"$QWEN_BASE"'" \
  --train-jsonl "'"$RAW"'"/bar_exam_cot_mix.jsonl \
  --output "'"$CKPTS"'"/qwen36_27b_cot_mix_qlora \
  --epochs 2 --lr 1e-4 \
  --per-device-batch 4 --grad-accum 4 \
  --max-len 1536 --warmup-steps 15 --logging-steps 10 \
  --lora-r 64 --lora-alpha 128' || true

run "qwen36_cot_eval" bash -c '
cd "$ROOT"
CUDA_VISIBLE_DEVICES=7 python3 -u scripts/eval_bar_exam_round15_qwen36_mm.py \
  --base-model "'"$QWEN_BASE"'" \
  --adapter "'"$CKPTS"'"/qwen36_27b_cot_mix_qlora \
  --questions-csv "'"$RAW"'"/data/questions.csv --round 15 \
  --output "'"$RES"'"/qwen36_cot_pred.jsonl \
  --summary-out "'"$RES"'"/qwen36_cot_summary.json \
  --max-new-tokens 256' || true

# Step 4: Gemma-4 r=256 epoch 3 (continue)
run "gemma_r256_ep3" bash -c '
cd "$ROOT"
mkdir -p "'"$CKPTS"'"/gemma4_31b_cot_mix_r256_ep3
CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u scripts/qlora_gemma31b_bar_exam.py \
  --model "'"$GEMMA_BASE"'" \
  --train-jsonl "'"$RAW"'"/bar_exam_cot_mix.jsonl \
  --output "'"$CKPTS"'"/gemma4_31b_cot_mix_r256_ep3" \
  --epochs 3 --lr 5e-5 \
  --per-device-batch 4 --grad-accum 4 \
  --max-len 1536 --warmup-steps 15 --logging-steps 10 \
  --lora-r 256 --lora-alpha 512' || true

echo "===== overnight chain DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$LOG"
