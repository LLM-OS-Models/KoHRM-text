"""Build per-question RAG reference markdown files for round 15.

For each of the 150 questions:
  1. Parse question_text + answer
  2. Extract candidate statute names + article numbers from question/choices
  3. Look up statute text in legalize-kr/kr/<statute>/법률.md
  4. Look up precedent snippets in hrm_text_extra jsonl
  5. Emit /data/bar_exam/round15_rag/qXXX_<subj>.md with citations + URLs
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


# Common Korean statute names that appear in bar-exam questions (15회 공법/민사법/형사법)
LAW_NAMES = [
    "대한민국헌법", "헌법", "국가공무원법", "국회법", "정당법", "공직선거법",
    "감사원법", "헌법재판소법", "정부조직법", "국회규칙", "국회사무처법",
    "대법원규칙", "법원조직법", "행정절차법", "행정심판법", "행정소송법",
    "행정기본법", "국가배상법", "국가재정법", "국가를당사자로하는계약에관한법률",
    "민법", "민사소송법", "상법", "가족관계의등록등에관한법률", "국제사법",
    "가등기담보등에관한법률", "가족관계의등록", "집합건물의소유및관리에관한법률",
    "주택임대차보호법", "상가건물임대차보호법", "전세권", "저작권법", "특허법",
    "형법", "형사소송법", "특정범죄가중처벌등에관한법률", "성폭력범죄의처벌등에관한특례법",
    "아동및청소년의성보호에관한법률", "교통사고특례법",
    "채무자회생및파산에관한법률", "회생및파산에관한법률",
    "국가보훈처법", "병역법", "군인사법",
    "사법양성", "변호사법", "법무사법", "변리사법",
    "국가인권위원회법", "공익신고자보호법", "부패방지및국민권익위원회의설치와운영에관한법률",
    "국가재난관리기본법", "재난및안전관리기본법",
    "정보통신망법", "개인정보보호법", "전자서명법", "전자문서및전자거래기본법",
]


def find_statute_dirs(root: Path) -> dict[str, Path]:
    """Map statute name (case-insensitive) to directory path."""
    out = {}
    if not root.exists(): return out
    for d in root.iterdir():
        if d.is_dir():
            out[d.name] = d
    return out


def extract_article_text(md_text: str, article_num: int) -> str:
    """Extract 제N조 text from a statute md file."""
    pattern = re.compile(rf"(제{article_num}조[^[]{0,80}\])\s*\n([^\[]+?)(?=\n제\d+조|\Z)", re.DOTALL)
    m = pattern.search(md_text)
    if m:
        return m.group(0).strip()[:1500]
    return ""


def extract_keywords(text: str, subject: str = "") -> dict:
    """Extract statute names + article numbers mentioned in the text.

    Falls back to subject→statute mapping when no explicit statute name found.
    """
    found_laws = []
    text_norm = text.replace(" ", "").replace("\n", "")
    for law in LAW_NAMES:
        law_norm = law.replace(" ", "")
        if law_norm in text_norm:
            found_laws.append(law)
    # Subject-based fallback mapping
    SUBJ_DEFAULT = {
        "공법": ["대한민국헌법", "헌법재판소법", "행정소송법", "행정심판법", "행정기본법", "국가배상법", "국회법", "정부조직법"],
        "민사법": ["민법", "민사소송법", "상법", "가족관계의등록등에관한법률", "주택임대차보호법"],
        "형사법": ["형법", "형사소송법", "특정범죄가중처벌등에관한법률"],
    }
    if not found_laws and subject in SUBJ_DEFAULT:
        found_laws = SUBJ_DEFAULT[subject]
    # article numbers: 제N조
    articles = re.findall(r"제(\d+)조(?:의\d+)?\s*\(([^)]+)\)?", text)
    return {"laws": sorted(set(found_laws)), "articles": articles[:10]}


def search_precedent(jsonl_path: Path, query_terms: list[str], max_hits: int = 2) -> list[dict]:
    """Search precedent jsonl for matching terms."""
    if not jsonl_path.exists() or not query_terms: return []
    hits = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            inst = (o.get("instruction") or "") + " " + (o.get("response") or "")
            score = sum(1 for t in query_terms if t and t in inst)
            if score >= 2:
                hits.append({"score": score, "inst": o.get("instruction","")[:300],
                             "resp": o.get("response","")[:600],
                             "path": o.get("path","")})
                if len(hits) >= max_hits * 3: break
    hits.sort(key=lambda x: -x["score"])
    return hits[:max_hits]


def law_url(name: str) -> str:
    return f"https://www.law.go.kr/법령/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-csv", default="/home/work/.data/bar_exam_sft/raw/data/questions.csv")
    ap.add_argument("--round", type=int, default=15)
    ap.add_argument("--legalize-root", default="/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text/legalize-kr/kr")
    ap.add_argument("--precedent-jsonl", default="/home/work/.data/huggingface/hrm_text_extra/sft/korean_admrule_precedent_raw_full_20260524.jsonl")
    ap.add_argument("--tasks-jsonl", default="/home/work/.data/huggingface/hrm_text_extra/sft/korean_legal_tasks_full_20260524.jsonl")
    ap.add_argument("--output-dir", default="/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text/data/bar_exam/round15_rag")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    statute_dirs = find_statute_dirs(Path(args.legalize_root))
    print(f"statute dirs: {len(statute_dirs)}", flush=True)
    print(f"output dir: {out_dir}", flush=True)

    # Load questions
    rows = []
    with open(args.questions_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if int(r["round"]) == args.round: rows.append(r)
    print(f"round {args.round}: {len(rows)} questions", flush=True)

    prec_jsonl = Path(args.precedent_jsonl)
    tasks_jsonl = Path(args.tasks_jsonl)

    n_with_law = 0
    for i, r in enumerate(rows, start=1):
        qno = r["question_no"]
        subj = r["subject"]
        qtext = r["question_text"].strip()
        stem = r["stem"].strip()
        answer = r["answer"].strip()
        kw = extract_keywords(stem + "\n" + qtext, subj)

        md_lines = []
        md_lines.append(f"# 제15회 변호사시험 q{qno} — [{subj}]")
        md_lines.append("")
        md_lines.append("## 문제")
        md_lines.append("```")
        md_lines.append(qtext)
        md_lines.append("```")
        md_lines.append("")
        md_lines.append(f"## 정답")
        md_lines.append(f"`{answer}`")
        md_lines.append("")

        md_lines.append("## 추출된 키워드")
        md_lines.append(f"- 언급 법령: {', '.join(kw['laws']) if kw['laws'] else '(명시 없음)'}")
        md_lines.append(f"- 언급 조문: {', '.join(f'제{n}조' for n,_ in kw['articles']) if kw['articles'] else '(없음)'}")
        md_lines.append("")

        # Statute text excerpts
        md_lines.append("## 관련 법령 (legalize-kr)")
        law_added = False
        for law in kw["laws"]:
            # Find matching dir (exact or fuzzy)
            dir_path = None
            for dname, dp in statute_dirs.items():
                if dname == law or law in dname or dname in law:
                    dir_path = dp; break
            if dir_path is None: continue
            md_path = dir_path / "법률.md"
            if not md_path.exists():
                md_path = dir_path / f"{law}.md"
            if not md_path.exists():
                # try first md
                mds = list(dir_path.glob("*.md"))
                if mds: md_path = mds[0]
                else: continue
            try:
                md_text = md_path.read_text(encoding="utf-8", errors="ignore")
            except Exception: continue
            md_lines.append(f"### {law}")
            md_lines.append(f"- 출처 파일: `{md_path}`")
            md_lines.append(f"- URL: {law_url(law)}")
            # Try article excerpts
            for article_num_str, article_title in kw["articles"]:
                if article_num_str.isdigit():
                    excerpt = extract_article_text(md_text, int(article_num_str))
                    if excerpt:
                        md_lines.append(f"#### 제{article_num_str}조 {article_title}")
                        md_lines.append("```")
                        md_lines.append(excerpt[:1000])
                        md_lines.append("```")
            # Always include a short head excerpt
            head = md_text.strip().split("\n")
            head_text = "\n".join(head[:30])
            md_lines.append("#### 본문 일부 (처음 30줄)")
            md_lines.append("```")
            md_lines.append(head_text[:1500])
            md_lines.append("```")
            md_lines.append("")
            law_added = True
        if not law_added:
            md_lines.append("(직접 매칭되는 법령 없음 — 부분 키워드로 수동 검색 필요)")
            md_lines.append("")
        if kw["laws"]: n_with_law += 1

        # Precedent search
        md_lines.append("## 관련 판례/행정문서 (korean_admrule_precedent_raw_full / korean_legal_tasks_full)")
        search_terms = kw["laws"] + [t for _, t in kw["articles"] if t][:3]
        prec_hits = search_precedent(prec_jsonl, search_terms, max_hits=2)
        task_hits = search_precedent(tasks_jsonl, search_terms, max_hits=2)
        if prec_hits or task_hits:
            md_lines.append("")
            md_lines.append("### 판례 (precedent jsonl)")
            for h in prec_hits:
                md_lines.append(f"- 점수 {h['score']}, path: `{h['path']}`")
                md_lines.append("  ```")
                md_lines.append("  " + h["inst"][:300])
                md_lines.append("  ---")
                md_lines.append("  " + h["resp"][:600].replace("\n", "\n  "))
                md_lines.append("  ```")
            md_lines.append("")
            md_lines.append("### 법령 과제 (legal tasks jsonl)")
            for h in task_hits:
                md_lines.append(f"- 점수 {h['score']}, path: `{h['path']}`")
                md_lines.append("  ```")
                md_lines.append("  " + h["inst"][:300])
                md_lines.append("  ```")
        else:
            md_lines.append("(자동 매칭 판례 없음 — 부분 키워드로 수동 검색 필요)")
        md_lines.append("")

        md_lines.append("## 추가 참고 URL (수동 검색 권장)")
        md_lines.append("- 법령: https://www.law.go.kr/법령/")
        md_lines.append("- 판례: https://www.law.go.kr/precInfoP.do")
        md_lines.append("- 헌법재판소: https://www.ccourt.go.kr/")
        md_lines.append("- 대법원: https://www.scourt.go.kr/")
        md_lines.append("")

        out_path = out_dir / f"q{int(qno):03d}_{subj}.md"
        out_path.write_text("\n".join(md_lines), encoding="utf-8")
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} done (with law: {n_with_law})", flush=True)

    print(f"\nDone. {n_with_law}/{len(rows)} questions have statute matches.", flush=True)


if __name__ == "__main__":
    main()
