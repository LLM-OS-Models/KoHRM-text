"""Few-shot eval: inject 2-3 similar Q&A examples from rounds 1-14 before the
round-15 question. Uses SFT CoT model (adapter). Greedy, no RAG.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
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
    a=(a or "").strip()
    if a in {"","정답없음"} or "," in a: return None
    if a.isdigit():
        n=int(a)
        if 1<=n<=9: return str(n)
        if len(a)==2 and a.endswith("0") and 1<=int(a[0])<=9: return str(a[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--questions-csv", default="/home/work/.data/bar_exam_sft/raw/data/questions.csv")
    ap.add_argument("--train-jsonl", default="/home/work/.data/bar_exam_sft/raw/bar_exam_train_1_14.jsonl")
    ap.add_argument("--round", type=int, default=15)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-input-tokens", type=int, default=3072)
    ap.add_argument("--n-shots", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    # Load few-shot examples from train data (same subject)
    train_by_subj = {}
    with open(args.train_jsonl, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            subj = o.get("subject","")
            if subj:
                train_by_subj.setdefault(subj, []).append(o)

    rows = []
    with open(args.questions_csv, encoding="utf-8-sig") as fp:
        for r in csv.DictReader(fp):
            try:
                if int(r["round"]) != args.round: continue
            except: continue
            if norm_gold(r.get("answer","")) is None: continue
            rows.append(r)
    print(f"rows: {len(rows)}, n_shots: {args.n_shots}, temp: {args.temperature}", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    print("loading 4-bit base...", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(args.base_model, quantization_config=bnb,
        trust_remote_code=True, attn_implementation="sdpa").eval()
    model.config.use_cache = True
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Gemma4ClippableLinear) and isinstance(mod.linear, Linear4bit):
            pn,_,cn = name.rpartition('.')
            parent = model.get_submodule(pn) if pn else model
            setattr(parent, cn, mod.linear)
    print("loading adapter...", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    nc=nt=np=0; bs={}; t0=time.time()
    with out.open("w", encoding="utf-8") as fpo:
        for idx, r in enumerate(rows):
            gold = norm_gold(r["answer"])
            subj = r["subject"]
            # Build few-shot messages: n_shots examples + target question
            messages = []
            examples = train_by_subj.get(subj, [])
            for ex in examples[:args.n_shots]:
                messages.append({"role":"user","content":ex["instruction"][:1500]})
                messages.append({"role":"assistant","content":ex["response"]})
            messages.append({"role":"user","content":build_user(r)})

            try: prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except: prompt = "\n\n".join(m["content"] for m in messages)
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(model.device)
            with torch.inference_mode():
                kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
                if args.temperature > 0:
                    kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.95)
                out_ids = model.generate(**inputs, **kwargs)
            gen = tok.decode(out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = parse_pred(gen)
            ok = pred is not None and pred == gold
            nt += 1; np += int(pred is not None); nc += int(ok)
            s = subj
            b = bs.setdefault(s,{"c":0,"t":0,"p":0})
            b["c"]+=int(ok); b["t"]+=1; b["p"]+=int(pred is not None)
            fpo.write(json.dumps({"idx":idx,"qno":r.get("question_no"),"gold":gold,
                "pred":pred,"correct":ok,"raw":gen[:300]},ensure_ascii=False)+"\n")
            fpo.flush()
            if (idx+1)%20==0 or (idx+1)==len(rows):
                el=time.time()-t0
                print(f"  {idx+1}/{len(rows)} acc={nc/nt:.3f} parsed={np/nt:.3f} el={el:.0f}s",flush=True)

    summary={"mode":"few-shot","n_shots":args.n_shots,"temperature":args.temperature,
        "n_total":nt,"n_parsed":np,"n_correct":nc,
        "accuracy":nc/nt if nt else 0,"parse_rate":np/nt if nt else 0,
        "by_subject":{s:{"accuracy":v["c"]/v["t"],"correct":v["c"],"total":v["t"],"parsed":v["p"]} for s,v in sorted(bs.items())}}
    Path(args.summary_out).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
