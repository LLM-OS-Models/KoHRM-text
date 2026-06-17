"""Simpler eval loop: run inference_generate over round-15 questions, batch=1.

This avoids the multi-batch stopped/cache bookkeeping complexity of
inference_generate when batch_size > 1 by always running one prompt at a
time. Slower but more robust for diagnosing per-question generation.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_inference_engine import inference_load_checkpoint, inference_generate  # noqa: E402


ANSWER_LINE_RE = re.compile(r"정답\s*[:：]\s*([1-5])")


def build_eval_instruction(row: dict) -> str:
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
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--ckpt-epoch", type=int, default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--questions-csv", default="/home/work/.data/bar_exam_sft/raw/data/questions.csv")
    ap.add_argument("--round", type=int, default=15)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-generation", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--condition", default="direct")
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

    print(f"loading checkpoint {args.ckpt_path} ...", flush=True)
    ckpt = inference_load_checkpoint(
        ckpt_path=args.ckpt_path,
        ckpt_epoch=args.ckpt_epoch,
        ckpt_use_ema=not args.no_ema,
        device="cuda",
    )
    print("model loaded", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_correct = n_total = n_parsed = 0
    by_subject: dict[str, dict[str, int]] = {}
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fpo:
        for idx, row in enumerate(rows):
            inst = build_eval_instruction(row)
            items = [(0, (args.condition, inst))]
            gen_text = ""
            for _idx, gen in inference_generate(
                ckpt, iter(items),
                max_tokens=args.max_tokens,
                max_generation=args.max_generation,
                batch_size=1,
                temp=args.temp,
            ):
                gen_text = gen
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
                "raw": gen_text[:200],
            }, ensure_ascii=False) + "\n")
            fpo.flush()
            if (idx + 1) % 10 == 0 or (idx + 1) == len(rows):
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(rows) - idx - 1) / rate if rate > 0 else 0
                print(f"  {idx+1}/{len(rows)} acc={n_correct/n_total:.3f} parsed={n_parsed/n_total:.3f} "
                      f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    summary = {
        "ckpt_path": args.ckpt_path,
        "ckpt_epoch": args.ckpt_epoch,
        "use_ema": not args.no_ema,
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
                "correct": v["correct"],
                "total": v["total"],
                "parsed": v["parsed"],
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
