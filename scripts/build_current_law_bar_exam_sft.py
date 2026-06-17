"""Build synthetic Korean bar-exam-style SFT data from current Korean statutes.

The output is a Hugging Face dataset folder with:
- data/questions.csv: flat preview table
- sft/train.jsonl: messages-format SFT rows
- metadata/qa_report.json: generation and duplication checks
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path


ARTICLE_RE = re.compile(r"(?m)^#####\s+(.+?)\s*$")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
KOREAN_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ROUND15_SIM_THRESHOLD = 0.72

CURRENT_COMMIT_OVERRIDES = {
    # The working tree contains promulgated future-effective versions for these
    # core bar-exam laws. Use the latest checked commit that is safely effective
    # on the dataset 기준일 instead of accidentally training on future law.
    "kr/형법/법률.md": "70512b7f",
    "kr/형사소송법/법률.md": "6b116b13",
    "kr/민사소송법/법률.md": "4cc47519",
}


@dataclass(frozen=True)
class LawSpec:
    subject: str
    group: str
    rel_path: str
    weight: int = 1


@dataclass
class LawDoc:
    subject: str
    group: str
    title: str
    rel_path: str
    body: str
    source_url: str
    effective_date: str
    promulgation_date: str
    promulgation_number: str
    source_commit: str


@dataclass
class Article:
    doc: LawDoc
    heading: str
    article_no: str
    article_title: str
    content: str
    snippets: list[str]


LAW_SPECS = [
    LawSpec("공법", "헌법", "kr/대한민국헌법/헌법.md", 5),
    LawSpec("공법", "헌법", "kr/헌법재판소법/법률(법률).md", 3),
    LawSpec("공법", "헌법", "kr/국회법/법률.md", 2),
    LawSpec("공법", "행정법", "kr/행정기본법/법률.md", 4),
    LawSpec("공법", "행정법", "kr/행정절차법/법률.md", 4),
    LawSpec("공법", "행정법", "kr/행정심판법/법률.md", 3),
    LawSpec("공법", "행정법", "kr/행정소송법/법률.md", 4),
    LawSpec("공법", "행정법", "kr/국가배상법/법률.md", 2),
    LawSpec("공법", "행정법", "kr/개인정보보호법/법률.md", 2),
    LawSpec("민사법", "민법", "kr/민법/법률.md", 8),
    LawSpec("민사법", "상법", "kr/상법/법률.md", 5),
    LawSpec("민사법", "민사소송법", "kr/민사소송법/법률.md", 5),
    LawSpec("민사법", "민사집행법", "kr/민사집행법/법률.md", 3),
    LawSpec("민사법", "부동산등기법", "kr/부동산등기법/법률.md", 2),
    LawSpec("민사법", "채무자회생법", "kr/채무자회생및파산에관한법률/법률.md", 2),
    LawSpec("민사법", "가족관계", "kr/가족관계의등록등에관한법률/법률.md", 1),
    LawSpec("형사법", "형법", "kr/형법/법률.md", 7),
    LawSpec("형사법", "형사소송법", "kr/형사소송법/법률.md", 7),
    LawSpec("형사법", "형사특별법", "kr/폭력행위등처벌에관한법률/법률.md", 2),
    LawSpec("형사법", "형사특별법", "kr/특정범죄가중처벌등에관한법률/법률.md", 2),
    LawSpec("형사법", "형사특별법", "kr/성폭력범죄의처벌등에관한특례법/법률.md", 2),
    LawSpec("형사법", "형사특별법", "kr/아동ㆍ청소년의성보호에관한법률/법률.md", 1),
    LawSpec("형사법", "형사특별법", "kr/형의실효등에관한법률/법률.md", 1),
]


STOP_TERMS = {
    "이하",
    "경우",
    "다음",
    "각호",
    "각호의",
    "어느",
    "하나",
    "사람",
    "사항",
    "법률",
    "대하여",
    "관하여",
    "필요한",
    "정하는",
    "대통령령",
    "총리령",
    "부령",
    "제항",
}


def run_git(root: Path, args: list[str]) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL)


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    meta: dict[str, str] = {}
    current_key = ""
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.startswith(" ") or line.startswith("-"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip()
            meta[current_key] = value.strip().strip("'\"")
    return meta, text[m.end() :]


def parse_date(raw: str) -> date | None:
    raw = (raw or "").strip().strip("'\"")
    if not KOREAN_DATE_RE.match(raw):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def read_current_text(legal_root: Path, rel_path: str, as_of: date) -> tuple[str, str]:
    path = legal_root / rel_path
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="ignore")
        meta, _ = split_frontmatter(text)
        effective = parse_date(meta.get("시행일자", ""))
        if effective and effective <= as_of:
            try:
                commit = run_git(legal_root, ["rev-parse", "HEAD"]).strip()
            except Exception:
                commit = "working-tree"
            return text, commit

    override = CURRENT_COMMIT_OVERRIDES.get(rel_path)
    if override:
        try:
            text = run_git(legal_root, ["show", f"{override}:{rel_path}"])
        except subprocess.CalledProcessError:
            pass
        else:
            meta, _ = split_frontmatter(text)
            effective = parse_date(meta.get("시행일자", ""))
            status = meta.get("상태", "")
            if effective and effective <= as_of and status == "시행":
                return text, override

    raise FileNotFoundError(f"No current version as of {as_of}: {rel_path}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\*\*([①-⑳])\*\*", r"\1", text)
    text = re.sub(r"<[^>\n]{1,80}>", "", text)
    text = re.sub(r"\[[^\]\n]{1,120}\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,])", r"\1", text)
    return text.strip()


def article_name(heading: str) -> tuple[str, str]:
    m = re.match(r"(제\d+조(?:의\d+)?)(?:\s*\(([^)]+)\))?", heading)
    if not m:
        return heading.strip(), ""
    return m.group(1), (m.group(2) or "").strip()


def split_articles(doc: LawDoc) -> list[Article]:
    matches = list(ARTICLE_RE.finditer(doc.body))
    articles: list[Article] = []
    for idx, match in enumerate(matches):
        heading = match.group(1).strip()
        if not heading.startswith("제") or "삭제" in heading:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(doc.body)
        content = clean_text(doc.body[start:end])
        snippets = extract_snippets(content)
        if snippets:
            article_no, title = article_name(heading)
            articles.append(
                Article(
                    doc=doc,
                    heading=heading,
                    article_no=article_no,
                    article_title=title,
                    content=content,
                    snippets=snippets,
                )
            )
    return articles


def sentence_split(text: str) -> list[str]:
    text = compact(text)
    parts = re.split(r"(?<=[.])\s+|(?<=다\.)\s+|(?<=한다\.)\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_snippets(content: str) -> list[str]:
    content = clean_text(content)
    if "삭제" in content[:20] or len(content) < 15:
        return []
    parts = re.split(r"(?=[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", content)
    raw_snippets: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first = sentence_split(part)
        if not first:
            continue
        snippet = first[0]
        snippet = re.sub(r"^[①-⑳]\s*", "", snippet)
        snippet = re.sub(r"^\d+\.\s*", "", snippet)
        snippet = compact(snippet)
        if not is_self_contained_snippet(snippet):
            continue
        if 28 <= len(snippet) <= 240 and "삭제" not in snippet:
            raw_snippets.append(snippet)
    seen = set()
    snippets = []
    for snippet in raw_snippets:
        key = re.sub(r"\W+", "", snippet)
        if key not in seen:
            seen.add(key)
            snippets.append(snippet)
    return snippets[:6]


def is_self_contained_snippet(snippet: str) -> bool:
    if snippet.count("(") != snippet.count(")"):
        return False
    if snippet.count("「") != snippet.count("」"):
        return False
    if snippet.count('"') % 2 != 0:
        return False
    weak_patterns = [
        "다음 각 호",
        "다음 각 목",
        "다음의 각 호",
        "다음의 각 목",
        "각 호의 사항",
        "각 목의 사항",
        "각 호와 같다",
        "각 목과 같다",
    ]
    if any(pattern in snippet for pattern in weak_patterns):
        return False
    if snippet.endswith(("각 호와 같다.", "각 목과 같다.", "포함되어야 한다.")):
        return False
    return True


def load_law_docs(legal_root: Path, as_of: date) -> list[LawDoc]:
    docs: list[LawDoc] = []
    for spec in LAW_SPECS:
        try:
            text, commit = read_current_text(legal_root, spec.rel_path, as_of)
        except Exception as exc:
            print(f"[warn] skip {spec.rel_path}: {exc}")
            continue
        meta, body = split_frontmatter(text)
        title = meta.get("제목") or re.search(r"^#\s+(.+)$", body, re.M).group(1)
        for _ in range(spec.weight):
            docs.append(
                LawDoc(
                    subject=spec.subject,
                    group=spec.group,
                    title=title,
                    rel_path=spec.rel_path,
                    body=body,
                    source_url=meta.get("출처", ""),
                    effective_date=meta.get("시행일자", ""),
                    promulgation_date=meta.get("공포일자", ""),
                    promulgation_number=meta.get("공포번호", "").strip("'\""),
                    source_commit=commit,
                )
            )
    return docs


def make_reference(article: Article) -> dict[str, str]:
    doc = article.doc
    return {
        "title": doc.title,
        "article": article.heading,
        "source_url": doc.source_url,
        "source_path": f"legalize-kr/{doc.rel_path}",
        "source_commit": doc.source_commit,
        "effective_date": doc.effective_date,
        "promulgation_date": doc.promulgation_date,
        "promulgation_number": doc.promulgation_number,
        "excerpt": article.snippets[0],
    }


def reference_label(article: Article) -> str:
    doc = article.doc
    number = f" 제{doc.promulgation_number}호" if doc.promulgation_number else ""
    return f"{doc.title} {article.heading} [시행 {doc.effective_date}, {doc.promulgation_date}{number}]"


def option_prefix(i: int) -> str:
    return ["①", "②", "③", "④", "⑤"][i]


def question_body(stem: str, choices: list[str]) -> str:
    return stem + "\n" + "\n".join(f"{option_prefix(i)} {choice}" for i, choice in enumerate(choices))


def choose_answer_position(rng: random.Random) -> int:
    return rng.randrange(5)


def sample_distractors(
    rng: random.Random,
    pool: list[tuple[Article, str]],
    target: Article,
    count: int,
    same_subject: str | None = None,
) -> list[tuple[Article, str]]:
    candidates = [
        item
        for item in pool
        if item[0] is not target
        and item[0].heading != target.heading
        and item[1] not in target.snippets
        and (same_subject is None or item[0].doc.subject == same_subject)
    ]
    rng.shuffle(candidates)
    picked: list[tuple[Article, str]] = []
    seen = set()
    for article, snippet in candidates:
        key = re.sub(r"\W+", "", snippet)
        if key and key not in seen:
            seen.add(key)
            picked.append((article, snippet))
        if len(picked) == count:
            break
    if len(picked) < count:
        raise ValueError("not enough distractors")
    return picked


def distort_statement(text: str) -> str | None:
    candidates: list[str] = []
    m = re.search(r"(\d+)(년|월|일|세|명|인|개|분의\s*\d+|분)", text)
    if m:
        num = int(m.group(1))
        replacement = str(num + 1 if num not in {0, 1} else num + 2)
        candidates.append(text[: m.start(1)] + replacement + text[m.end(1) :])
    replacements = [
        ("하여야 한다", "할 수 있다"),
        ("해야 한다", "할 수 있다"),
        ("할 수 있다", "하여야 한다"),
        ("아니한다", "한다"),
        ("없다", "있다"),
        ("있다", "없다"),
        ("대통령", "국무총리"),
        ("국회", "정부"),
        ("법원", "검사"),
        ("검사", "사법경찰관"),
        ("피고인", "피해자"),
        ("채권자", "채무자"),
        ("채무자", "채권자"),
        ("주주", "이사"),
    ]
    for old, new in replacements:
        if old in text and new not in text:
            candidates.append(text.replace(old, new, 1))
    for candidate in candidates:
        if candidate != text and 20 <= len(candidate) <= 260:
            return candidate
    return None


def extract_terms(text: str, article: Article) -> list[str]:
    terms: list[str] = []
    if article.article_title and 2 <= len(article.article_title) <= 14:
        terms.append(article.article_title)
    terms.extend(re.findall(r"「([^」]{2,18})」", text))
    terms.extend(re.findall(r'"([^"]{2,18})"', text))
    for word in re.findall(r"[가-힣ㆍ]{2,14}", text):
        if word in STOP_TERMS:
            continue
        if word.endswith(("권", "죄", "법", "청구", "심판", "소송", "처분", "계약", "대리", "채권", "채무", "형벌", "증거", "재판", "의결", "허가", "취소", "등기")):
            terms.append(word)
    out: list[str] = []
    seen = set()
    for term in terms:
        term = term.strip()
        if term in STOP_TERMS or len(term) < 2:
            continue
        if term not in seen and term in text:
            seen.add(term)
            out.append(term)
    return out


def make_article_match(
    rng: random.Random,
    qid: int,
    article: Article,
    snippet: str,
    pool: list[tuple[Article, str]],
    as_of: date,
) -> dict:
    answer_pos = choose_answer_position(rng)
    distractors = sample_distractors(rng, pool, article, 4, same_subject=article.doc.subject)
    choices = [item[1] for item in distractors]
    choices.insert(answer_pos, snippet)
    stem = (
        f"다음 중 {article.doc.title} {article.heading}의 내용으로 직접 확인되는 것은? "
        f"(기준일: {as_of.isoformat()}, 다툼이 있는 경우 현행 법령에 의함)"
    )
    explanation = (
        f"정답은 {answer_pos + 1}번이다. {reference_label(article)}에서 \"{snippet}\"라고 확인된다. "
        "나머지 선택지는 같은 과목의 다른 조문 내용이거나 대상 조문에 직접 규정된 내용이 아니므로 이 문항의 정답이 아니다."
    )
    return build_row(qid, "article_match", article, stem, choices, answer_pos, explanation, [make_reference(article)])


def make_negative(
    rng: random.Random,
    qid: int,
    article: Article,
    snippet: str,
    pool: list[tuple[Article, str]],
    as_of: date,
) -> dict | None:
    wrong = distort_statement(snippet)
    if not wrong:
        return None
    same_law = [item for item in pool if item[0].doc.title == article.doc.title and item[0] is not article]
    if len(same_law) < 4:
        return None
    rng.shuffle(same_law)
    true_choices = []
    seen = set()
    for other_article, other_snippet in same_law:
        key = re.sub(r"\W+", "", other_snippet)
        if key not in seen:
            seen.add(key)
            true_choices.append(other_snippet)
        if len(true_choices) == 4:
            break
    if len(true_choices) < 4:
        return None
    answer_pos = choose_answer_position(rng)
    choices = true_choices[:]
    choices.insert(answer_pos, wrong)
    stem = (
        f"{article.doc.title}의 조문 내용에 관한 설명으로 옳지 않은 것은? "
        f"(기준일: {as_of.isoformat()}, 다툼이 있는 경우 현행 법령에 의함)"
    )
    explanation = (
        f"정답은 {answer_pos + 1}번이다. {reference_label(article)}의 실제 문언은 \"{snippet}\"이다. "
        f"선택지 {answer_pos + 1}은 이를 \"{wrong}\"라고 바꾸어 서술하므로 현행 조문과 맞지 않는다."
    )
    return build_row(qid, "negative_statement", article, stem, choices, answer_pos, explanation, [make_reference(article)])


def make_blank(
    rng: random.Random,
    qid: int,
    article: Article,
    snippet: str,
    all_terms: list[str],
    as_of: date,
) -> dict | None:
    terms = extract_terms(snippet, article)
    if not terms:
        return None
    answer = terms[0]
    if len(answer) < 2 or snippet.count(answer) == 0:
        return None
    blanked = snippet.replace(answer, "㉠", 1)
    distractors = [t for t in all_terms if t != answer and 2 <= len(t) <= 18]
    rng.shuffle(distractors)
    choices = []
    seen = {answer}
    for term in distractors:
        if term not in seen:
            seen.add(term)
            choices.append(term)
        if len(choices) == 4:
            break
    if len(choices) < 4:
        return None
    answer_pos = choose_answer_position(rng)
    choices.insert(answer_pos, answer)
    stem = (
        f"{article.doc.title} {article.heading}의 다음 문언 중 ㉠에 들어갈 말로 옳은 것은? "
        f"(기준일: {as_of.isoformat()})\n\n{blanked}"
    )
    explanation = (
        f"정답은 {answer_pos + 1}번이다. {reference_label(article)}는 해당 부분을 \"{snippet}\"라고 규정하므로 "
        f"㉠에는 '{answer}'이 들어간다."
    )
    return build_row(qid, "blank_term", article, stem, choices, answer_pos, explanation, [make_reference(article)])


def make_reference_select(
    rng: random.Random,
    qid: int,
    article: Article,
    snippet: str,
    articles: list[Article],
    as_of: date,
) -> dict:
    answer_pos = choose_answer_position(rng)
    candidates = [
        a
        for a in articles
        if a.doc.subject == article.doc.subject
        and not (a.doc.title == article.doc.title and a.heading == article.heading)
    ]
    rng.shuffle(candidates)
    distractors = []
    seen = {f"{article.doc.title} {article.heading}"}
    for a in candidates:
        label = f"{a.doc.title} {a.heading}"
        if label not in seen:
            seen.add(label)
            distractors.append(label)
        if len(distractors) == 4:
            break
    if len(distractors) < 4:
        raise ValueError("not enough article labels")
    choices = distractors[:]
    choices.insert(answer_pos, f"{article.doc.title} {article.heading}")
    stem = (
        "다음 문언의 직접적인 근거 조문으로 가장 적절한 것은? "
        f"(기준일: {as_of.isoformat()}, 현행 법령 기준)\n\n"
        f"\"{snippet}\""
    )
    explanation = (
        f"정답은 {answer_pos + 1}번이다. 제시 문언은 {reference_label(article)}에서 확인되는 문언이다."
    )
    return build_row(qid, "reference_select", article, stem, choices, answer_pos, explanation, [make_reference(article)])


def build_row(
    qid: int,
    question_type: str,
    article: Article,
    stem: str,
    choices: list[str],
    answer_pos: int,
    explanation: str,
    references: list[dict[str, str]],
) -> dict:
    question_text = question_body(stem, choices)
    answer = str(answer_pos + 1)
    assistant = (
        f"정답: {answer}\n\n"
        f"해설: {explanation}\n\n"
        "참고 법령: "
        + "; ".join(f"{r['title']} {r['article']}({r['source_url']})" for r in references)
    )
    prompt = (
        "다음 변호사시험 선택형 스타일 문제를 풀고, 정답 번호와 해설 및 참고 법령을 제시하시오.\n\n"
        f"{question_text}"
    )
    messages = [
        {
            "role": "system",
            "content": "대한민국 현행 법령을 기준으로 변호사시험 선택형 문제를 풀이하는 법률 학습 도우미이다.",
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant},
    ]
    stable = f"{question_type}|{article.doc.title}|{article.heading}|{question_text}"
    return {
        "id": f"ko_current_law_bar_sft_{qid:04d}",
        "dataset": "korean-current-law-bar-exam-sft-1000",
        "language": "ko",
        "exam_style": "bar_exam_multiple_choice_synthetic",
        "subject": article.doc.subject,
        "law_group": article.doc.group,
        "question_type": question_type,
        "law_title": article.doc.title,
        "article": article.heading,
        "question_text": question_text,
        "stem": stem,
        "choices_json": json.dumps(choices, ensure_ascii=False),
        "answer": answer,
        "answer_text": choices[answer_pos],
        "explanation": explanation,
        "references_json": json.dumps(references, ensure_ascii=False),
        "prompt": prompt,
        "response": assistant,
        "messages_json": json.dumps(messages, ensure_ascii=False),
        "source_url": article.doc.source_url,
        "source_path": f"legalize-kr/{article.doc.rel_path}",
        "source_commit": article.doc.source_commit,
        "effective_date": article.doc.effective_date,
        "promulgation_date": article.doc.promulgation_date,
        "promulgation_number": article.doc.promulgation_number,
        "synthetic": "true",
        "generation_method": "statute_template_generation_with_round15_similarity_filter",
        "fingerprint": hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16],
    }


def normalize_for_similarity(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", "", text)
    return text


def similarity_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
        if token not in STOP_TERMS
    }


def load_round15_texts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        items: list[dict[str, object]] = []
        for row in reader:
            if str(row.get("round")) != "15":
                continue
            text = row.get("question_text") or row.get("stem") or ""
            norm = normalize_for_similarity(text)
            items.append({"norm": norm, "tokens": similarity_tokens(text)})
        return items


def max_round15_similarity(text: str, round15_texts: list[dict[str, object]]) -> float:
    norm = normalize_for_similarity(text)
    if not norm or not round15_texts:
        return 0.0
    tokens = similarity_tokens(text)
    max_score = 0.0
    probe = norm[:900]
    for item in round15_texts:
        old = str(item["norm"])
        if not old:
            continue
        if norm == old or (len(norm) > 120 and norm[:120] in old):
            return 1.0
        length_ratio = len(norm) / max(len(old), 1)
        if not 0.25 <= length_ratio <= 3.5:
            continue
        old_tokens = item["tokens"]
        if isinstance(old_tokens, set):
            denom = max(1, min(len(tokens), len(old_tokens)))
            overlap = len(tokens & old_tokens) / denom
            if overlap < 0.18:
                continue
        score = SequenceMatcher(None, probe, old[:900]).ratio()
        if score > max_score:
            max_score = score
    return max_score


def build_rows(
    articles: list[Article],
    target_count: int,
    rng: random.Random,
    as_of: date,
    round15_texts: list[dict[str, object]],
) -> tuple[list[dict], dict]:
    snippet_pool = [(article, snippet) for article in articles for snippet in article.snippets]
    all_terms = []
    for article, snippet in snippet_pool:
        all_terms.extend(extract_terms(snippet, article))
    all_terms = sorted(set(all_terms), key=lambda item: (len(item), item))

    rows: list[dict] = []
    fingerprints: set[str] = set()
    rejected_round15 = 0
    rejected_duplicate = 0
    attempts = 0
    max_similarity = 0.0
    weighted_articles = []
    for article in articles:
        weighted_articles.extend([article] * max(1, article.doc.subject.count(article.doc.subject)))

    makers = ["article_match", "negative", "blank", "reference_select"]
    subject_targets = {"공법": 270, "민사법": 460, "형사법": 270}
    subject_counts = {"공법": 0, "민사법": 0, "형사법": 0}

    while len(rows) < target_count and attempts < target_count * 80:
        attempts += 1
        remaining_subjects = [
            subject
            for subject, target in subject_targets.items()
            if subject_counts.get(subject, 0) < target
        ]
        if remaining_subjects:
            subject = rng.choice(remaining_subjects)
            candidates = [a for a in articles if a.doc.subject == subject]
        else:
            candidates = articles
        article = rng.choice(candidates)
        snippet = rng.choice(article.snippets)
        kind = rng.choice(makers)
        qid = len(rows) + 1
        try:
            if kind == "article_match":
                row = make_article_match(rng, qid, article, snippet, snippet_pool, as_of)
            elif kind == "negative":
                row = make_negative(rng, qid, article, snippet, snippet_pool, as_of)
            elif kind == "blank":
                row = make_blank(rng, qid, article, snippet, all_terms, as_of)
            else:
                row = make_reference_select(rng, qid, article, snippet, articles, as_of)
        except Exception:
            continue
        if row is None:
            continue
        sim = max_round15_similarity(row["question_text"], round15_texts)
        max_similarity = max(max_similarity, sim)
        if sim >= ROUND15_SIM_THRESHOLD:
            rejected_round15 += 1
            continue
        fp = row["fingerprint"]
        if fp in fingerprints:
            rejected_duplicate += 1
            continue
        fingerprints.add(fp)
        row["round15_max_similarity"] = f"{sim:.4f}"
        rows.append(row)
        subject_counts[row["subject"]] = subject_counts.get(row["subject"], 0) + 1

    if len(rows) != target_count:
        raise RuntimeError(f"Generated {len(rows)} rows after {attempts} attempts, expected {target_count}")

    for idx, row in enumerate(rows, start=1):
        row["id"] = f"ko_current_law_bar_sft_{idx:04d}"

    qa = {
        "target_count": target_count,
        "row_count": len(rows),
        "attempts": attempts,
        "subject_counts": subject_counts,
        "question_type_counts": count_by(rows, "question_type"),
        "law_counts_top20": dict(sorted(count_by(rows, "law_title").items(), key=lambda kv: (-kv[1], kv[0]))[:20]),
        "round15_similarity_threshold": ROUND15_SIM_THRESHOLD,
        "round15_max_similarity": round(max(float(r["round15_max_similarity"]) for r in rows), 4),
        "rejected_round15_similarity": rejected_round15,
        "rejected_duplicate_fingerprint": rejected_duplicate,
        "duplicate_fingerprints": len(rows) - len({row["fingerprint"] for row in rows}),
    }
    return rows, qa


def count_by(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "dataset",
        "language",
        "exam_style",
        "subject",
        "law_group",
        "question_type",
        "law_title",
        "article",
        "question_text",
        "stem",
        "choices_json",
        "answer",
        "answer_text",
        "explanation",
        "references_json",
        "prompt",
        "response",
        "messages_json",
        "source_url",
        "source_path",
        "source_commit",
        "effective_date",
        "promulgation_date",
        "promulgation_number",
        "synthetic",
        "generation_method",
        "round15_max_similarity",
        "fingerprint",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_sft_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            item = {
                "id": row["id"],
                "subject": row["subject"],
                "law_title": row["law_title"],
                "article": row["article"],
                "messages": json.loads(row["messages_json"]),
                "references": json.loads(row["references_json"]),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_readme(path: Path, repo_id: str, as_of: date, qa: dict) -> None:
    text = f"""---
license: other
language:
- ko
pretty_name: Korean Current-Law Bar Exam SFT 1000
task_categories:
- question-answering
- text-classification
tags:
- law
- korean-law
- bar-exam
- sft
- synthetic
- public-sector
configs:
- config_name: questions
  data_files:
  - split: train
    path: data/questions.csv
---

# Korean Current-Law Bar Exam SFT 1000

대한민국 현행 법령을 기준으로 만든 변호사시험 선택형 스타일 SFT 데이터 1,000문항입니다.

이 데이터셋은 법무부 기출문제를 복제하지 않습니다. 기존 `gyung/korean-bar-exam-moj-multiple-choice`의 `data/questions.csv`는 난도와 과목 분포 참고 및 제15회 중복 방지 기준으로만 사용했습니다.

## Files

- `data/questions.csv`: Hugging Face preview용 메인 CSV입니다.
- `sft/train.jsonl`: `messages` 형식 SFT용 JSONL입니다.
- `metadata/qa_report.json`: 생성 수량, 과목 분포, 제15회 유사도 QA 결과입니다.

## Columns

- `question_text`: 문제와 5개 선택지
- `answer`: 정답 번호, `1`부터 `5`
- `answer_text`: 정답 선택지 원문
- `explanation`: 정답 근거와 오답 판단 이유
- `references_json`: 참고 법령, 조문, 출처 URL, 로컬 경로, git commit, 시행일자
- `prompt`, `response`, `messages_json`: SFT 변환용 텍스트

## Generation Policy

- 기준일: `{as_of.isoformat()}`
- 법령 원천: `legalize-kr/legalize-kr` 로컬 스냅샷 및 git 이력
- 미래 시행본 제외: `시행일자 <= {as_of.isoformat()}`인 버전만 사용
- 중복 방지: 제15회 변호사시험 `question_text`와 문자열 유사도 `{ROUND15_SIM_THRESHOLD}` 이상인 생성 문항은 제외
- 문항 유형: 조문 직접 확인형, 옳지 않은 설명형, 빈칸형, 근거 조문 선택형

## QA Summary

```json
{json.dumps(qa, ensure_ascii=False, indent=2)}
```

## Sources

- Existing bar-exam reference dataset: https://huggingface.co/datasets/gyung/korean-bar-exam-moj-multiple-choice
- Ministry of Justice, 2026 제15회 변호사시험 기출문제/정답 공지
  - https://www.moj.go.kr/bbs/moj/150/602397/artclView.do
  - https://www.moj.go.kr/bbs/moj/150/602398/artclView.do
  - https://www.moj.go.kr/bbs/moj/150/602399/artclView.do
  - https://www.moj.go.kr/bbs/moj/151/603464/artclView.do
- Korean statutes: https://github.com/legalize-kr/legalize-kr and https://www.law.go.kr
- Related local legal corpora consulted for project context: `ordinance-kr`, `/home/work/.data/huggingface/hrm_text_extra/sft/korean_legal_tasks_full_20260524.jsonl`, `/home/work/.data/huggingface/hrm_text_extra/sft/korean_admrule_precedent_raw_full_20260524.jsonl`

## License

법령 원문은 대한민국 정부 공공저작물입니다. 데이터셋의 패키징, 생성 코드, 메타데이터는 프로젝트 생성물입니다.

이 데이터는 학습/평가용 법률 교육 데이터이며 법률 자문이 아닙니다.

HF repo: https://huggingface.co/datasets/{repo_id}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legal-root", default="legalize-kr")
    ap.add_argument("--bar-questions", default="data/bar_exam/hf_multiple_choice/data/questions.csv")
    ap.add_argument("--out-dir", default="data/current_law_bar_exam_sft_1000/hf_dataset")
    ap.add_argument("--repo-id", default="gyung/korean-current-law-bar-exam-sft-1000")
    ap.add_argument("--as-of", default="2026-06-13")
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260613)
    args = ap.parse_args()

    as_of = date.fromisoformat(args.as_of)
    rng = random.Random(args.seed)
    legal_root = Path(args.legal_root)
    out_dir = Path(args.out_dir)

    docs = load_law_docs(legal_root, as_of)
    base_docs = []
    seen_doc_keys = set()
    for doc in docs:
        key = (doc.title, doc.rel_path, doc.source_commit)
        if key not in seen_doc_keys:
            seen_doc_keys.add(key)
            base_docs.append(doc)
    articles = []
    for doc in base_docs:
        articles.extend(split_articles(doc))
    if len(articles) < 100:
        raise RuntimeError(f"Too few parsed articles: {len(articles)}")

    # Re-apply weighting at article level.
    weighted_articles: list[Article] = []
    for spec in LAW_SPECS:
        for article in articles:
            if article.doc.rel_path == spec.rel_path:
                weighted_articles.extend([article] * spec.weight)

    round15_texts = load_round15_texts(Path(args.bar_questions))
    rows, qa = build_rows(weighted_articles, args.count, rng, as_of, round15_texts)
    qa.update(
        {
            "as_of": as_of.isoformat(),
            "law_doc_count": len(base_docs),
            "article_count": len(articles),
            "weighted_article_count": len(weighted_articles),
            "round15_reference_count": len(round15_texts),
            "repo_id": args.repo_id,
            "bar_questions_path": args.bar_questions,
        }
    )

    if out_dir.exists():
        for dirpath, dirnames, filenames in os.walk(out_dir, topdown=False):
            for filename in filenames:
                Path(dirpath, filename).unlink()
            for dirname in dirnames:
                Path(dirpath, dirname).rmdir()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "data" / "questions.csv", rows)
    write_sft_jsonl(out_dir / "sft" / "train.jsonl", rows)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata" / "qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(out_dir / "README.md", args.repo_id, as_of, qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
