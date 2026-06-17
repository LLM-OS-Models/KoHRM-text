"""Convert gyung/korean-bar-exam-hard-current-law-precedent-sft-1000 to KoHRM SFT JSONL.

The source dataset stores each sample as a ChatML-style messages list
(system / user / assistant). KoHRM SFT data is plain instruction/response, so:
- system content is prepended to the instruction (KoHRM has no chat template)
- user content becomes the instruction verbatim
- assistant content becomes the response verbatim

The assistant response follows the order: 정답 -> 해설 -> 참고 법령, so the
answer token is the first generation token (good for eval parsing).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten_messages(messages: list[dict]) -> tuple[str, str]:
    sys_parts, user_parts, asst_parts = [], [], []
    for m in messages or []:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            sys_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            asst_parts.append(content)
    instruction = "\n\n".join(user_parts)
    if sys_parts:
        instruction = "\n\n".join(sys_parts + [instruction])
    return instruction, "\n\n".join(asst_parts)


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

    with out.open("w", encoding="utf-8") as fpo, \
            open(args.input, encoding="utf-8") as fp:
        for line in fp:
            seen += 1
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages") or []
            instruction, response = flatten_messages(msgs)
            if not instruction or not response:
                skipped += 1
                continue
            row = {
                "instruction": instruction,
                "response": response,
                "condition": "cot",
                "source": "gyung_korean_bar_exam_hard_current_law_precedent_sft_1000",
                "id": obj.get("id", ""),
                "subject": obj.get("subject", ""),
                "law_title": obj.get("law_title", ""),
                "article": obj.get("article", ""),
            }
            fpo.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            by_subject[row["subject"]] = by_subject.get(row["subject"], 0) + 1

    stats = {
        "input": args.input,
        "output": str(out),
        "rows_seen": seen,
        "rows_kept": kept,
        "rows_skipped": skipped,
        "by_subject": dict(sorted(by_subject.items())),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.stats_out:
        Path(args.stats_out).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
