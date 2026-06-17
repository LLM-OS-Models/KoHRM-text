"""Build CoT (정답 + 해설) mix SFT data for Gemma-4 LoRA.

- 1-14 회: 정답만 (해설 데이터 없음)
- current-law-1000: assistant 원본 (정답 + 해설 + 참고 법령) 그대로
- hard-1000: assistant 원본 그대로

User template은 평가와 동일 (정답만 답하시오) → 모델이 학습 시 해설까지 생성하도록.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SYSTEM = "대한민국 현행 법령을 기준으로 변호사시험 선택형 문제를 푸는 법률 학습 도우미이다."


def build_user_cot(question_text: str, subject: str) -> str:
    return (
        "다음 변호사시험 선택형 문제를 읽고, 가장 옳은 정답 번호와 간단한 해설을 제시하시오. "
        "첫 줄은 반드시 '정답: <번호>' 형식이어야 한다.\n\n"
        f"[과목] {subject}\n"
        f"[문제]\n{question_text.strip()}"
    )


def extract_question_from_user(raw_user: str) -> str:
    cleaned = re.sub(r"^[^\n]*정답 번호와 해설[^\n]*\n+", "", raw_user.strip())
    return cleaned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-law-input", required=True)
    ap.add_argument("--hard-input", required=True)
    ap.add_argument("--train-1-14", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0

    with out.open("w", encoding="utf-8") as fpo:
        # 1-14 회: 정답만 (해설 없음)
        with open(args.train_1_14, encoding="utf-8") as fp:
            for line in fp:
                o = json.loads(line)
                fpo.write(json.dumps(o, ensure_ascii=False) + "\n")
                kept += 1

        # current-law-1000: assistant 원본 (해설 포함)
        with open(args.current_law_input, encoding="utf-8") as fp:
            for line in fp:
                o = json.loads(line)
                msgs = {m.get("role"): (m.get("content") or "") for m in (o.get("messages") or [])}
                user_raw = msgs.get("user", "")
                assistant = msgs.get("assistant", "").strip()
                if not assistant: continue
                question = extract_question_from_user(user_raw)
                user = build_user_cot(question, o.get("subject", ""))
                inst = f"{SYSTEM}\n\n{user}"
                fpo.write(json.dumps({
                    "instruction": inst, "response": assistant,
                    "condition": "cot",
                    "source": "gyung_korean_current_law_bar_exam_sft_1000_cot",
                    "id": o.get("id", ""),
                    "subject": o.get("subject", ""),
                }, ensure_ascii=False) + "\n")
                kept += 1

        # hard-1000: assistant 원본
        with open(args.hard_input, encoding="utf-8") as fp:
            for line in fp:
                o = json.loads(line)
                msgs = {m.get("role"): (m.get("content") or "") for m in (o.get("messages") or [])}
                user_raw = msgs.get("user", "")
                assistant = msgs.get("assistant", "").strip()
                if not assistant: continue
                question = extract_question_from_user(user_raw)
                user = build_user_cot(question, o.get("subject", ""))
                inst = f"{SYSTEM}\n\n{user}"
                fpo.write(json.dumps({
                    "instruction": inst, "response": assistant,
                    "condition": "cot",
                    "source": "gyung_korean_bar_exam_hard_current_law_precedent_sft_1000_cot",
                    "id": o.get("id", ""),
                    "subject": o.get("subject", ""),
                }, ensure_ascii=False) + "\n")
                kept += 1

    stats = {"output": str(out), "rows": kept}
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
