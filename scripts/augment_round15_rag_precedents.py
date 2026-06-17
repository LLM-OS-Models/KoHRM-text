"""Augment round15 RAG md files with relevant precedent snippets via grep.

Uses question stem keywords to grep the precedent + tasks jsonl files (1.7GB +
4.1GB). Adds top hits to each md file in-place.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


# Subject→broad keyword list to expand grep queries
SUBJ_KEYWORDS = {
    "공법": ["헌법", "헌법재판소", "대법원", "행정처분", "행정소송", "행정심판", "국가배상", "국회", "대통령"],
    "민사법": ["민법", "대리", "계약", "소유권", "채권", "물권", "상속", "민사소송", "상법", "주식", "회사"],
    "형사법": ["형법", "형사소송", "판례", "무죄", "유죄", "공소", "피고인", "압수", "구속"],
}


def grep_jsonl(jsonl_path: Path, patterns: list[str], max_hits: int = 3, timeout: int = 120) -> list[dict]:
    """Run grep with extended regex; collect first matches."""
    if not patterns: return []
    pattern_str = "|".join(re.escape(p) for p in patterns if p)
    if not pattern_str: return []
    try:
        result = subprocess.run(
            ["grep", "-m", str(max_hits * 5), "-E", pattern_str, str(jsonl_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception as e:
        print(f"  grep error: {e}", flush=True)
        return []
    hits = []
    for line in result.stdout.splitlines():
        try: o = json.loads(line)
        except: continue
        score = sum(1 for p in patterns if p in (o.get("instruction") or "") + (o.get("response") or ""))
        hits.append({"score": score, "inst": (o.get("instruction") or "")[:300],
                     "resp": (o.get("response") or "")[:600],
                     "path": o.get("path", ""), "source": o.get("source", "")})
    hits.sort(key=lambda x: -x["score"])
    return hits[:max_hits]


def extract_question_keywords(stem: str, subject: str) -> list[str]:
    """Pick 3-5 keywords from stem."""
    keywords = []
    # Specific terms first
    specific = re.findall(r"[가-힣]{2,8}(?:죄|권|법|소|제도|원칙|이론|조항|행위|관계)", stem)
    keywords.extend(specific[:5])
    # Fallback to subject broad keywords
    if len(keywords) < 3:
        keywords.extend(SUBJ_KEYWORDS.get(subject, [])[:3])
    return list(dict.fromkeys(keywords))[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-csv", default="/home/work/.data/bar_exam_sft/raw/data/questions.csv")
    ap.add_argument("--round", type=int, default=15)
    ap.add_argument("--precedent-jsonl", default="/home/work/.data/huggingface/hrm_text_extra/sft/korean_admrule_precedent_raw_full_20260524.jsonl")
    ap.add_argument("--tasks-jsonl", default="/home/work/.data/huggingface/hrm_text_extra/sft/korean_legal_tasks_full_20260524.jsonl")
    ap.add_argument("--rag-dir", default="/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text/data/bar_exam/round15_rag")
    args = ap.parse_args()

    rag_dir = Path(args.rag_dir)
    prec_jsonl = Path(args.precedent_jsonl)
    tasks_jsonl = Path(args.tasks_jsonl)

    rows = []
    with open(args.questions_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if int(r["round"]) == args.round: rows.append(r)
    print(f"rows: {len(rows)}", flush=True)

    n_added = 0
    for i, r in enumerate(rows, start=1):
        qno = int(r["question_no"])
        subj = r["subject"]
        stem = r["stem"]
        keywords = extract_question_keywords(stem, subj)
        if not keywords: continue

        prec_hits = grep_jsonl(prec_jsonl, keywords, max_hits=2, timeout=30)
        task_hits = grep_jsonl(tasks_jsonl, keywords, max_hits=2, timeout=30)
        if not prec_hits and not task_hits: continue

        # find matching md file
        md_path = rag_dir / f"q{qno:03d}_{subj}.md"
        if not md_path.exists(): continue
        md_text = md_path.read_text(encoding="utf-8")

        # Replace the "판례/행정문서 검색 (별도 수동)" section
        precedent_section = ["", "## 관련 판례/행정문서 (precedent/tasks jsonl)"]
        precedent_section.append(f"**검색 키워드**: {', '.join(keywords)}")
        if prec_hits:
            precedent_section.append("")
            precedent_section.append("### 판례 (korean_admrule_precedent_raw_full)")
            for h in prec_hits:
                precedent_section.append(f"- 점수 {h['score']} | path: `{h['path']}`")
                precedent_section.append("  ```")
                precedent_section.append("  " + h["inst"][:300].replace("\n", "\n  "))
                precedent_section.append("  ---")
                precedent_section.append("  " + h["resp"][:500].replace("\n", "\n  "))
                precedent_section.append("  ```")
        if task_hits:
            precedent_section.append("")
            precedent_section.append("### 법령 과제 (korean_legal_tasks_full)")
            for h in task_hits:
                precedent_section.append(f"- 점수 {h['score']} | path: `{h['path']}`")
                precedent_section.append("  ```")
                precedent_section.append("  " + h["inst"][:300].replace("\n", "\n  "))
                precedent_section.append("  ```")
        precedent_section.append("")
        new_section = "\n".join(precedent_section)

        # Insert before "## 추가 참고 URL"
        if "## 추가 참고 URL" in md_text:
            md_text = md_text.replace("## 추가 참고 URL", new_section + "\n## 추가 참고 URL")
        else:
            md_text = md_text + "\n" + new_section
        md_path.write_text(md_text, encoding="utf-8")
        n_added += 1

        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} (precedent added: {n_added})", flush=True)

    print(f"\nDone. {n_added}/{len(rows)} questions have precedent matches.", flush=True)


if __name__ == "__main__":
    main()
