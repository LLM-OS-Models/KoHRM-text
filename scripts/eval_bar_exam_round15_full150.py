"""Eval Gemma-4 multimodal QLoRA on round 15 — full 150 questions including
composite-answer items.

Composite answers like '30','40','70' are normalized to single-digit choice
numbers (30->3, 70->7, etc.) and compared against model predictions parsed
the same way.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


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
    """Normalize gold answer to single-digit 1-9.

    '3' -> '3'
    '30' -> '3' (composite-answer encoding; 30 means choice 3)
    '70' -> '7'
    """
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
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--model-class", default="gemma4",
                    choices=["gemma4", "qwen35"])
    args = ap.parse_args()

    rows = []
    dropped = []
    with open(args.questions_csv, encoding="utf-8-sig") as fp:
        for r in csv.DictReader(fp):
            try:
                if int(r["round"]) != args.round: continue
            except: continue
            g = norm_gold(r.get("answer",""))
            if g is None:
                dropped.append((r.get("question_no"), r.get("answer")))
                continue
            rows.append(r)
    print(f"round {args.round}: {len(rows)} questions (dropped {len(dropped)})", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    print(f"loading 4-bit base {args.base_model}...", flush=True)

    if args.model_class == "gemma4":
        try:
            from transformers.models.gemma4.modeling_gemma4 import Gemma4UnifiedForConditionalGeneration as MC
        except Exception:
            from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration as MC
        try:
            from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
            has_clip = True
        except Exception:
            has_clip = False
        from bitsandbytes.nn import Linear4bit
    else:
        from transformers import Qwen3_5ForConditionalGeneration as MC
        has_clip = False
        Linear4bit = None

    model = MC.from_pretrained(args.base_model, quantization_config=bnb,
        trust_remote_code=True, attn_implementation="sdpa").eval()
    model.config.use_cache = False
    if hasattr(model.config, "language_model_only"):
        model.config.language_model_only = True
    if has_clip:
        cnt = 0
        for name, mod in list(model.named_modules()):
            if isinstance(mod, Gemma4ClippableLinear) and isinstance(mod.linear, Linear4bit):
                pn,_,cn = name.rpartition('.')
                parent = model.get_submodule(pn) if pn else model
                setattr(parent, cn, mod.linear); cnt += 1
        print(f"unwrapped {cnt} ClippableLinear", flush=True)
    print(f"loading adapter {args.adapter}...", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    nc = nt = np_ = 0; bs = {}
    t0 = time.time()
    composite_total = composite_correct = 0
    with out_path.open("w", encoding="utf-8") as fpo:
        for idx, r in enumerate(rows):
            gold_raw = r["answer"].strip()
            is_composite = gold_raw.isdigit() and len(gold_raw) == 2
            gold = norm_gold(gold_raw)
            msgs = [{"role":"user","content":build_user(r)}]
            try: prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except: prompt = msgs[0]["content"]
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = parse_pred(gen)
            ok = pred is not None and pred == gold
            nt += 1
            np_ += int(pred is not None)
            nc += int(ok)
            if is_composite:
                composite_total += 1
                composite_correct += int(ok)
            s = r.get("subject","?")
            b = bs.setdefault(s, {"c":0,"t":0,"p":0,"comp_t":0,"comp_c":0})
            b["c"] += int(ok); b["t"] += 1; b["p"] += int(pred is not None)
            if is_composite:
                b["comp_t"] += 1; b["comp_c"] += int(ok)
            fpo.write(json.dumps({"idx":idx,"qno":r.get("question_no"),"subject":s,
                "gold_raw":gold_raw,"gold":gold,"pred":pred,"correct":ok,
                "composite":is_composite,"raw":gen[:500]}, ensure_ascii=False)+"\n")
            fpo.flush()
            if (idx+1) % 20 == 0 or (idx+1) == len(rows):
                el = time.time()-t0
                print(f"  {idx+1}/{len(rows)} acc={nc/nt:.3f} parsed={np_/nt:.3f} el={el:.0f}s", flush=True)
    summary = {
        "base_model": args.base_model, "adapter": args.adapter, "round": args.round,
        "n_total": nt, "n_parsed": np_, "n_correct": nc,
        "accuracy": nc/nt if nt else 0,
        "parse_rate": np_/nt if nt else 0,
        "n_composite": composite_total, "n_composite_correct": composite_correct,
        "composite_accuracy": composite_correct/composite_total if composite_total else 0,
        "elapsed_seconds": time.time()-t0,
        "by_subject": {s: {"accuracy": v["c"]/v["t"] if v["t"] else 0,
                          "correct": v["c"], "total": v["t"], "parsed": v["p"],
                          "composite_total": v["comp_t"], "composite_correct": v["comp_c"]}
                       for s,v in sorted(bs.items())},
        "dropped": dropped,
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
