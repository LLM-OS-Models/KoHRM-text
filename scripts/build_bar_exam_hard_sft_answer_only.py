"""Build hard-1000 SFT data in the SAME format as the 1-14 round SFT data.

Only the answer number is kept as the response; the long 해설/참고 법령
assistant body is dropped. This keeps the train distribution consistent with
the 1-14 multiple-choice SFT set so the model does not collapse to a single
answer class when continued from a parent checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ANSWER_RE = re.compile(r"정답\s*[:：]\s*([1-5])")


def extract_answer(assistant: str) -> str | None:
    m = ANSWER_RE.search(assistant or "")
    return m.group(1) if m else None


def build_user_prompt(row: dict) -> str:
    """Rebuild the user instruction using the 1-14 template (answer-only).

    The hard-1000 user message already contains the question. We strip the
    '해설 및 참고 법령을 제시하시오' framing and ask only for an answer number.
    """
    raw = row.get("user", "").strip()
    # Drop the leading '정답 번호와 해설 및 참고 법령을 제시하시오' line.
    cleaned = re.sub(r"^[^\n]*정답 번호와 해설[^\n]*\n+", "", raw)
    return (
        "다음 변호사시험 선택형 문제를 읽고, 가장 옳은 정답 번호만 답하시오. "
        "출력은 '정답: <번호>' 형식 한 줄로만 하시오.\n\n"
        f"[과목] {row.get('subject', '')}\n"
        f"[문제]\n{cleaned}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = kept = skipped = 0
    by_subject: dict[str, int] = {}
    by_answer: dict[str, int] = {}

    with out.open("w", encoding="utf-8") as fpo, open(args.input, encoding="utf-8") as fp:
        for line in fp:
            seen += 1
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = {m.get("role"): (m.get("content") or "") for m in (obj.get("messages") or [])}
            ans = extract_answer(msgs.get("assistant", ""))
            if ans is None:
                skipped += 1
                continue
            user = build_user_prompt({**obj, "user": msgs.get("user", "")})
            response = f"정답: {ans}"
            fpo.write(json.dumps({
                "instruction": user,
                "response": response,
                "condition": "direct",
                "source": "gyung_korean_bar_exam_hard_current_law_precedent_sft_1000_answer_only",
                "id": obj.get("id", ""),
                "subject": obj.get("subject", ""),
                "law_title": obj.get("law_title", ""),
                "article": obj.get("article", ""),
                "gold_answer": ans,
            }, ensure_ascii=False) + "\n")
            kept += 1
            by_subject[obj.get("subject", "?")] = by_subject.get(obj.get("subject", "?"), 0) + 1
            by_answer[ans] = by_answer.get(ans, 0) + 1

    stats = {
        "input": args.input,
        "output": str(out),
        "rows_seen": seen,
        "rows_kept": kept,
        "rows_skipped": skipped,
        "by_subject": dict(sorted(by_subject.items())),
        "by_answer": dict(sorted(by_answer.items())),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.stats_out:
        Path(args.stats_out).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
