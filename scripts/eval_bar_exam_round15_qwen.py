"""Evaluate a Hugging Face chat model (Qwen3.5 etc) on round 15 of Korean Bar Exam.

Uses transformers AutoModelForCausalLM + apply_chat_template. Different
runtime from the KoHRM PrefixLM script. Greedy decode, parse '정답: X'.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    if not text:
        return None
    m = ANSWER_LINE_RE.search(text[:120])
    if m:
        return m.group(1)
    for ch in text[:60]:
        if ch in "12345":
            return ch
    return None


def norm_gold(answer: str) -> str | None:
    a = (answer or "").strip()
    if a in {"", "정답없음"} or "," in a:
        return None
    if a.isdigit() and 1 <= int(a) <= 5:
        return a
    if len(a) == 2 and a.endswith("0") and 1 <= int(a[0]) <= 5:
        return a[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions-csv", default="/home/work/.data/bar_exam_sft/raw/data/questions.csv")
    ap.add_argument("--round", type=int, default=15)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    args = ap.parse_args()

    rows = []
    with open(args.questions_csv, encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            try:
                if int(row["round"]) != args.round:
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            if norm_gold(row.get("answer", "")) is None:
                continue
            rows.append(row)
    print(f"round {args.round}: {len(rows)} questions loaded", flush=True)

    print(f"loading model {args.model} ...", flush=True)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device, trust_remote_code=True,
    )
    model.eval()
    print(f"loaded; device={next(model.parameters()).device}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_correct = n_total = n_parsed = 0
    by_subject: dict[str, dict[str, int]] = {}
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fpo:
        for idx, row in enumerate(rows):
            messages = [
                {"role": "system", "content": "대한민국 현행 법령을 기준으로 변호사시험 선택형 문제를 푸는 법률 학습 도우미이다."},
                {"role": "user", "content": build_user(row)},
            ]
            try:
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(model.device)
            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)

            gold = norm_gold(row["answer"])
            pred = parse_pred(gen_text)
            ok = (pred is not None and pred == gold)
            n_total += 1
            n_parsed += int(pred is not None)
            n_correct += int(ok)
            subj = row.get("subject", "?")
            bucket = by_subject.setdefault(subj, {"correct": 0, "total": 0, "parsed": 0})
            bucket["correct"] += int(ok)
            bucket["total"] += 1
            bucket["parsed"] += int(pred is not None)

            fpo.write(json.dumps({
                "idx": idx,
                "id": row.get("id", ""),
                "subject": subj,
                "question_no": row.get("question_no", ""),
                "gold": gold,
                "pred": pred,
                "correct": ok,
                "raw": gen_text[:300],
            }, ensure_ascii=False) + "\n")
            fpo.flush()
            if (idx + 1) % 10 == 0 or (idx + 1) == len(rows):
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(rows) - idx - 1) / rate if rate > 0 else 0
                print(f"  {idx+1}/{len(rows)} acc={n_correct/n_total:.3f} parsed={n_parsed/n_total:.3f} "
                      f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    summary = {
        "model": args.model,
        "round": args.round,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "parse_rate": n_parsed / n_total if n_total else 0.0,
        "elapsed_seconds": time.time() - t0,
        "by_subject": {
            s: {
                "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
                "correct": v["correct"], "total": v["total"], "parsed": v["parsed"],
            } for s, v in sorted(by_subject.items())
        },
        "output": str(out_path),
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
