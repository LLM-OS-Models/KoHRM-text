"""Build KoHRM SFT JSONL from Korean Bar Exam (MOJ) multiple-choice CSV.

Output is plain instruction/response JSONL that `scripts/prepare_sft_data.py`
consumes directly.

Train rows use rounds 1..N (default 1..14). Eval rows use the last round
(default 15) and the response field is left empty for inference-time fill-in.

Instruction construction mirrors the format recommended by the source dataset
README (gyung/korean-bar-exam-moj-multiple-choice), adapted to KoHRM's plain
instruction/response layout.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


MULTI_DIGIT_RE = re.compile(r"^[1-7]0$")
SINGLE_DIGIT_RE = re.compile(r"^[1-5]$")


def is_valid_answer(value: str) -> bool:
    v = value.strip()
    if v in {"정답없음", ""}:
        return False
    if "," in v:
        return False
    return bool(SINGLE_DIGIT_RE.match(v) or MULTI_DIGIT_RE.match(v))


def build_instruction(row: dict) -> str:
    return (
        "다음 변호사시험 선택형 문제를 읽고, 가장 옳은 정답 번호만 답하시오. "
        "출력은 '정답: <번호>' 형식 한 줄로만 하시오.\n\n"
        f"[회차] 제{row['round']}회\n"
        f"[과목] {row['subject']}\n"
        f"[문항번호] {row['question_no']}\n"
        f"[문제]\n{row['question_text'].strip()}"
    )


def build_response(row: dict) -> str:
    return f"정답: {row['answer'].strip()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="questions.csv path")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument(
        "--rounds",
        default="1-14",
        help="inclusive range of rounds to include (e.g. 1-14 or 15-15)",
    )
    ap.add_argument(
        "--include-invalid-answer",
        action="store_true",
        help="keep rows whose answer is not in 1..5 or 10..70",
    )
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    if "-" in args.rounds:
        lo, hi = (int(x) for x in args.rounds.split("-", 1))
        keep_rounds = set(range(lo, hi + 1))
    else:
        keep_rounds = {int(args.rounds)}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = kept = skipped = 0
    by_round: dict[int, int] = {}
    with out_path.open("w", encoding="utf-8") as out, \
            open(args.input, encoding="utf-8-sig") as fp:
        rdr = csv.DictReader(fp)
        for row in rdr:
            seen += 1
            try:
                rnd = int(row["round"])
            except (KeyError, ValueError, TypeError):
                skipped += 1
                continue
            if rnd not in keep_rounds:
                continue
            if not args.include_invalid_answer and not is_valid_answer(row.get("answer", "")):
                skipped += 1
                continue
            inst = build_instruction(row)
            resp = build_response(row)
            out.write(json.dumps(
                {"instruction": inst, "response": resp, "condition": "direct",
                 "source": "moj_bar_exam_multiple_choice",
                 "round": rnd, "subject": row.get("subject", ""),
                 "question_no": row.get("question_no", ""),
                 "id": row.get("id", "")},
                ensure_ascii=False,
            ) + "\n")
            kept += 1
            by_round[rnd] = by_round.get(rnd, 0) + 1

    stats = {
        "input": args.input,
        "output": str(out_path),
        "rounds": sorted(keep_rounds),
        "rows_seen": seen,
        "rows_kept": kept,
        "rows_skipped": skipped,
        "by_round": {str(k): v for k, v in sorted(by_round.items())},
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.stats_out:
        Path(args.stats_out).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
