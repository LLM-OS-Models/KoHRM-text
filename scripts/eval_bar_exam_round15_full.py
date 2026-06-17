"""Evaluate a full fine-tuned Gemma-4 checkpoint on round 15.

Loads the fine-tuned checkpoint (saved as bf16 safetensors) directly without
4-bit quantization, runs greedy inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


ANSWER_LINE_RE = re.compile(r"정답\s*[:：]\s*([1-5])")


def build_user(row: dict) -> str:
    return (
        "다음 변호사시험 선택형 문제를 읽고, 가장 옳은 정답 번호만 답하시오. "
        "출력은 '정답: <번호>' 형식 한 줄로만 하시오.\n\n"
        f"[회차] 제{row['round']}회\n"
        f"[과목] {row['subject']}\n"
        f"[문항번호] {row['question_no']}\n"
        f"[문제]\n{row['question_text'].strip()}"
    )


def parse_pred(text: str) -> str | None:
    if not text: return None
    m = ANSWER_LINE_RE.search(text[:120])
    if m: return m.group(1)
    for ch in text[:60]:
        if ch in "12345": return ch
    return None


def norm_gold(a):
    a = (a or "").strip()
    if a in {"", "정답없음"} or "," in a: return None
    if a.isdigit() and 1 <= int(a) <= 5: return a
    if len(a) == 2 and a.endswith("0") and 1 <= int(a[0]) <= 5: return a[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions-csv", default="/home/work/.data/bar_exam_sft/raw/data/questions.csv")
    ap.add_argument("--round", type=int, default=15)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    rows = []
    with open(args.questions_csv, encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            try:
                if int(row["round"]) != args.round: continue
            except: continue
            if norm_gold(row.get("answer", "")) is None: continue
            rows.append(row)
    print(f"rows: {len(rows)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    if args.load_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
            trust_remote_code=True, attn_implementation="sdpa").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa").eval().to("cuda")

    # Unwrap Gemma4ClippableLinear if present
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        for name, module in list(model.named_modules()):
            if isinstance(module, Gemma4ClippableLinear):
                pn, _, cn = name.rpartition('.')
                parent = model.get_submodule(pn) if pn else model
                setattr(parent, cn, module.linear)
    except Exception: pass

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_correct = n_total = n_parsed = 0
    by_subj = {}
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fpo:
        for idx, row in enumerate(rows):
            msgs = [{"role":"user","content":build_user(row)}]
            try: prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except: prompt = msgs[0]["content"]
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            gold = norm_gold(row["answer"])
            pred = parse_pred(gen)
            ok = pred is not None and pred == gold
            n_total += 1
            n_parsed += int(pred is not None)
            n_correct += int(ok)
            s = row.get("subject","?")
            b = by_subj.setdefault(s, {"c":0,"t":0,"p":0})
            b["c"] += int(ok); b["t"] += 1; b["p"] += int(pred is not None)
            fpo.write(json.dumps({"idx":idx,"gold":gold,"pred":pred,"correct":ok,"raw":gen[:300]}, ensure_ascii=False)+"\n")
            fpo.flush()
            if (idx+1) % 20 == 0 or (idx+1) == len(rows):
                el = time.time() - t0
                print(f"  {idx+1}/{len(rows)} acc={n_correct/n_total:.3f} parsed={n_parsed/n_total:.3f} elapsed={el:.0f}s", flush=True)

    summary = {"model":args.model,"n_total":n_total,"n_parsed":n_parsed,"n_correct":n_correct,
        "accuracy":n_correct/n_total if n_total else 0, "parse_rate":n_parsed/n_total if n_total else 0,
        "by_subject":{s:{"accuracy":v["c"]/v["t"],"correct":v["c"],"total":v["t"],"parsed":v["p"]} for s,v in sorted(by_subj.items())}}
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
