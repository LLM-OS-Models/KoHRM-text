"""Build optimized RAG context: extract ONLY relevant articles per question.

Instead of dumping first 30 lines of statute, this:
1. Parses question for specific 조문 numbers (제N조)
2. Extracts only those articles from legalize-kr
3. Limits total context to ~1200 chars (avoids noise)
4. If no specific article found, extracts article matching question keywords
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def extract_articles_from_question(stem: str, qtext: str) -> dict[str, list[int]]:
    """Extract statute→article_nums mapping from question text."""
    combined = stem + "\n" + qtext
    # Find all 제N조 mentions with optional 법령명 prefix
    matches = re.findall(r'(?:([가-힯]+법|[가-힣]+법|헌법|민법|상법|형법))[^.]*?제(\d+)조', combined)
    law_articles = {}
    for law_name, article_num in matches:
        law_articles.setdefault(law_name, set()).add(int(article_num))
    # Also find standalone 제N조 mentions
    standalone = re.findall(r'제(\d+)조', combined)
    if standalone and not law_articles:
        law_articles['default'] = set(int(a) for a in standalone[:3])
    return {k: sorted(v) for k, v in law_articles.items()}


def extract_article_from_md(md_text: str, article_num: int) -> str:
    """Extract 제N조 text from statute md, max 800 chars."""
    # Pattern: 제N조 ... (title) ... text until next 제M조
    pattern = rf'(제{article_num}조[^\n]*\n[^\n]*(?:\n[^제\n][^\n]*)*)'
    matches = re.findall(pattern, md_text)
    if matches:
        return matches[0][:800]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-csv", default="/home/work/.data/bar_exam_sft/raw/data/questions.csv")
    ap.add_argument("--round", type=int, default=15)
    ap.add_argument("--legalize-root", default="/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text/legalize-kr/kr")
    ap.add_argument("--output-dir", default="/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text/data/bar_exam/round15_rag_optimized")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    legalize = Path(args.legalize_root)

    # Subject default laws (from earlier verified version)
    SUBJ_DEFAULT = {
        "공법": ["대한민국헌법", "헌법재판소법", "행정소송법", "행정심판법", "행정기본법"],
        "민사법": ["민법", "민사소송법", "상법"],
        "형사법": ["형법", "형사소송법"],
    }

    rows = []
    with open(args.questions_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if int(r["round"]) == args.round: rows.append(r)
    print(f"rows: {len(rows)}", flush=True)

    n_with_context = 0
    for i, r in enumerate(rows, start=1):
        qno = int(r["question_no"]); subj = r["subject"]
        stem = r["stem"]; qtext = r["question_text"]; answer = r["answer"]

        # 1. Extract relevant articles from question
        article_map = extract_articles_from_question(stem, qtext)

        # 2. Determine which laws to search
        search_laws = list(article_map.keys())
        if not search_laws or search_laws == ['default']:
            search_laws = SUBJ_DEFAULT.get(subj, [])

        # 3. Build optimized context (max 1200 chars)
        context_parts = []
        total_chars = 0
        for law_name in search_laws:
            if total_chars >= 1200: break
            # Find law directory
            law_dir = None
            for d in legalize.iterdir():
                if law_name in d.name or d.name in law_name:
                    law_dir = d; break
            if not law_dir: continue
            md_file = law_dir / "법률.md"
            if not md_file.exists():
                mds = list(law_dir.glob("*.md"))
                md_file = mds[0] if mds else None
            if not md_file: continue
            md_text = md_file.read_text(encoding="utf-8", errors="ignore")

            # Extract specific articles if mentioned
            articles = article_map.get(law_name, [])
            if not articles and 'default' in article_map:
                articles = article_map['default']

            if articles:
                for art_num in articles[:3]:  # max 3 articles per law
                    excerpt = extract_article_from_md(md_text, art_num)
                    if excerpt and total_chars + len(excerpt) < 1200:
                        context_parts.append(f"[{law_name} 제{art_num}조]\n{excerpt}")
                        total_chars += len(excerpt)
            else:
                # No specific article → take first substantive article
                # Skip frontmatter, find first 제1조
                first_article = extract_article_from_md(md_text, 1)
                if first_article and total_chars + len(first_article) < 800:
                    context_parts.append(f"[{law_name} 제1조]\n{first_article}")
                    total_chars += len(first_article)

        context = "\n\n".join(context_parts)[:1500]  # hard limit

        # 4. Write optimized md (NO answer)
        md_lines = []
        md_lines.append(f"# 제15회 변호사시험 q{qno} — [{subj}]")
        md_lines.append("")
        md_lines.append("## 문제")
        md_lines.append("```")
        md_lines.append(qtext.strip())
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("## 참고 법령 (최적화 발췌)")
        if context:
            md_lines.append("```")
            md_lines.append(context)
            md_lines.append("```")
            n_with_context += 1
        else:
            md_lines.append("(관련 조문 발췌 없음)")
        md_lines.append("")
        md_lines.append("## 출처")
        md_lines.append(f"- 법령 원문: legalize-kr/{', '.join(search_laws[:3])}")
        md_lines.append(f"- URL: https://www.law.go.kr/법령/{search_laws[0] if search_laws else ''}")

        out_path = out_dir / f"q{qno:03d}_{subj}.md"
        out_path.write_text("\n".join(md_lines), encoding="utf-8")

        if i % 30 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} done (with context: {n_with_context})", flush=True)

    print(f"\nDone. {n_with_context}/{len(rows)} questions have optimized context.", flush=True)


if __name__ == "__main__":
    main()
