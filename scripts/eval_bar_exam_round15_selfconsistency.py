"""Self-consistency eval: sample N times with temperature>0, take majority vote.

Boosts accuracy by reducing variance in greedy single-shot predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma4ForConditionalGeneration
from peft import PeftModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
from bitsandbytes.nn import Linear4bit


ANSWER_LINE_RE = re.compile(r"정답\s*[:：]\s*([1-9])")


def build_user(row):
    return ("다음 변호사시험 선택형 문제를 읽고, 가장 옳은 정답 번호만 답하시오. "
            "출력은 '정답: <번호>' 형식 한 줄로만 하시오.\n\n"
            f"[회차] 제{row['round']}회\n[과목] {row['subject']}\n[문항번호] {row['question_no']}\n"
            f"[문제]\n{row['question_text'].strip()}")


def parse_pred(t):
    if not t: return None
    m = ANSWER_LINE_RE.search(t[:120])
    if m: return m.group(1)
    for ch in t[:60]:
        if ch in "12345": return ch
    return None


def norm_gold(a):
    a = (a or "").strip()
    if a in {"", "정답없음"} or "," in a: return None
    if a.isdigit():
        n = int(a)
        if 1 <= n <= 9: return str(n)
        if len(a) == 2 and a.endswith("0"):
            head = int(a[0])
            if 1 <= head <= 9: return str(head)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--questions-csv", default="/home/work/.data/bar_exam_sft/raw/data/questions.csv")
    ap.add_argument("--round", type=int, default=15)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    args = ap.parse_args()

    rows = []
    with open(args.questions_csv, encoding="utf-8-sig") as fp:
        for r in csv.DictReader(fp):
            try:
                if int(r["round"]) != args.round: continue
            except: continue
            if norm_gold(r.get("answer","")) is None: continue
            rows.append(r)
    print(f"rows: {len(rows)}, n_samples: {args.n_samples}, temp: {args.temperature}", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    print("loading 4-bit base...", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(args.base_model, quantization_config=bnb,
        trust_remote_code=True, attn_implementation="sdpa").eval()
    model.config.use_cache = True
    # Unwrap ClippableLinear (bnb 4-bit only)
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Gemma4ClippableLinear) and isinstance(mod.linear, Linear4bit):
            pn, _, cn = name.rpartition('.')
            parent = model.get_submodule(pn) if pn else model
            setattr(parent, cn, mod.linear)
    print("loading adapter...", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    nc = nt = 0; bs = {}
    t0 = time.time()
    with out.open("w", encoding="utf-8") as fpo:
        for idx, r in enumerate(rows):
            gold = norm_gold(r["answer"])
            msgs = [{"role":"user","content":build_user(r)}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(model.device)
            votes = []
            sample_texts = []
            for _ in range(args.n_samples):
                with torch.inference_mode():
                    out_ids = model.generate(**inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
                gen = tok.decode(out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred = parse_pred(gen)
                if pred is not None: votes.append(pred)
                sample_texts.append(gen[:200])
            # Majority vote (fallback to None if no votes)
            if votes:
                final = Counter(votes).most_common(1)[0][0]
                vote_dist = dict(Counter(votes))
            else:
                final = None
                vote_dist = {}
            ok = final is not None and final == gold
            nt += 1; nc += int(ok)
            s = r.get("subject","?")
            b = bs.setdefault(s, {"c":0,"t":0})
            b["c"] += int(ok); b["t"] += 1
            fpo.write(json.dumps({"idx":idx,"qno":r.get("question_no"),"subject":s,
                "gold":gold,"votes":vote_dist,"final":final,"correct":ok,
                "n_votes":len(votes),"sample_first":sample_texts[0] if sample_texts else ""},
                ensure_ascii=False)+"\n")
            fpo.flush()
            if (idx+1) % 20 == 0 or (idx+1) == len(rows):
                el = time.time() - t0
                print(f"  {idx+1}/{len(rows)} acc={nc/nt:.3f} el={el:.0f}s", flush=True)

    summary = {"base_model":args.base_model,"adapter":args.adapter,
        "n_total":nt,"n_correct":nc,"accuracy":nc/nt if nt else 0,
        "n_samples":args.n_samples,"temperature":args.temperature,"top_p":args.top_p,
        "by_subject":{s:{"accuracy":v["c"]/v["t"],"correct":v["c"],"total":v["t"]} for s,v in sorted(bs.items())}}
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
