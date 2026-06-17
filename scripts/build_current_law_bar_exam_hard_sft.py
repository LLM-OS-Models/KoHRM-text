"""Build harder Korean bar-exam-style SFT data from current statutes.

This generator replaces direct statute lookup with multi-statement and
case-framed questions so the surface form is closer to recent Korean bar exam
multiple-choice questions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

import build_current_law_bar_exam_sft as base


LETTERS = ["ㄱ", "ㄴ", "ㄷ", "ㄹ"]
OPTION_SETS = [
    frozenset(["ㄱ"]),
    frozenset(["ㄴ"]),
    frozenset(["ㄷ"]),
    frozenset(["ㄹ"]),
    frozenset(["ㄱ", "ㄴ"]),
    frozenset(["ㄱ", "ㄷ"]),
    frozenset(["ㄱ", "ㄹ"]),
    frozenset(["ㄴ", "ㄷ"]),
    frozenset(["ㄴ", "ㄹ"]),
    frozenset(["ㄷ", "ㄹ"]),
    frozenset(["ㄱ", "ㄴ", "ㄷ"]),
    frozenset(["ㄱ", "ㄴ", "ㄹ"]),
    frozenset(["ㄱ", "ㄷ", "ㄹ"]),
    frozenset(["ㄴ", "ㄷ", "ㄹ"]),
    frozenset(["ㄱ", "ㄴ", "ㄷ", "ㄹ"]),
]

PRECEDENT_PATH = "/home/work/.data/huggingface/hrm_text_extra/sft/korean_legal_tasks_full_20260524.jsonl"


def label_combo(items: frozenset[str]) -> str:
    return ", ".join(letter for letter in LETTERS if letter in items)


def clean_statement(text: str) -> str:
    text = base.compact(text)
    text = re.sub(r"^제\d+항에\s*따른\s*", "", text)
    text = re.sub(r"^전항의\s*", "", text)
    text = re.sub(r"^제\d+항의\s*", "", text)
    return text


def is_good_statement(text: str) -> bool:
    if not base.is_self_contained_snippet(text):
        return False
    if len(text) < 45 or len(text) > 230:
        return False
    if text.count("제") > 10:
        return False
    if text.endswith(("있다.", "한다.", "아니하다.", "수 있다.", "하여야 한다.", "못한다.", "된다.")):
        return True
    return False


def statement_pool(articles: list[base.Article]) -> list[dict]:
    rows = []
    seen = set()
    for article in articles:
        for snippet in article.snippets:
            text = clean_statement(snippet)
            key = re.sub(r"\W+", "", f"{article.doc.title}|{article.heading}|{text}")
            if key in seen or not is_good_statement(text):
                continue
            seen.add(key)
            rows.append({"kind": "statute", "article": article, "text": text, "truth": True, "actual": text})
    return rows


def precedent_subject(path: str, text: str) -> str | None:
    joined = f"{path}\n{text}"
    if any(key in joined for key in ["/형사/", "형법", "형사소송법", "공소", "피고인", "무죄", "유죄", "압수", "구속"]):
        return "형사법"
    if any(key in joined for key in ["/민사/", "민법", "상법", "민사소송법", "채권", "채무", "손해배상", "계약", "소유권", "주주", "회사"]):
        return "민사법"
    if any(key in joined for key in ["/행정/", "/세무/", "행정소송", "처분", "취소소송", "헌법", "위헌", "기본권", "국가배상"]):
        return "공법"
    return None


def simplify_holding(text: str) -> str:
    text = base.compact(text)
    text = re.sub(r"\[[^\]]{1,100}\]", "", text)
    text = re.sub(r"\([^)]{1,80}\)", "", text)
    sentences = base.sentence_split(text)
    for sent in sentences:
        sent = base.compact(sent)
        if 70 <= len(sent) <= 260 and base.is_self_contained_snippet(sent):
            return sent
    return ""


def load_precedent_pool(path: str = PRECEDENT_PATH, max_items: int = 5000) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    seen = set()
    with p.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("source") != "korean_legal_precedent":
                continue
            try:
                data = json.loads(obj.get("response", ""))
            except Exception:
                continue
            title = str(data.get("사건명", "")).strip()
            court = str(data.get("법원", "")).strip()
            decided = str(data.get("선고일자", "")).strip()
            holding = simplify_holding(str(data.get("판결요지", "")).strip())
            url = str(data.get("출처", "")).strip()
            rel_path = str(obj.get("path", "")).strip()
            if not title or not holding or len(holding) < 70:
                continue
            subject = precedent_subject(rel_path, f"{title}\n{holding}")
            if not subject:
                continue
            key = re.sub(r"\W+", "", f"{title}|{court}|{decided}|{holding[:80]}")
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "kind": "precedent",
                    "subject": subject,
                    "text": holding,
                    "truth": True,
                    "actual": holding,
                    "precedent": {
                        "case_name": title,
                        "court": court,
                        "decision_date": decided,
                        "source_url": url,
                        "source_path": rel_path,
                        "holding": holding,
                    },
                }
            )
            if len(rows) >= max_items:
                break
    return rows


def item_subject(item: dict) -> str:
    if item["kind"] == "statute":
        return item["article"].doc.subject
    return item["subject"]


def item_title(item: dict) -> str:
    if item["kind"] == "statute":
        return item["article"].doc.title
    return item["precedent"]["case_name"]


def pick_related(rng: random.Random, pool: list[dict], item: dict, n: int) -> list[dict]:
    subject = item_subject(item)
    title = item_title(item)
    same_law = [s for s in pool if item_title(s) == title and s is not item]
    same_subject = [s for s in pool if item_subject(s) == subject and s is not item]
    candidates = same_law + same_subject
    rng.shuffle(candidates)
    picked = []
    seen = set()
    for cand in candidates:
        key = cand["text"]
        if key in seen:
            continue
        seen.add(key)
        picked.append(cand)
        if len(picked) == n:
            return picked
    raise ValueError("not enough related statements")


def make_false(text: str) -> str | None:
    false_text = base.distort_statement(text)
    if false_text and base.is_self_contained_snippet(false_text) and 40 <= len(false_text) <= 330:
        return false_text
    replacements = [
        ("60일", "30일"),
        ("30일", "60일"),
        ("3년", "5년"),
        ("5년", "3년"),
        ("과반수", "3분의 2 이상"),
        ("3분의 2 이상", "과반수"),
        ("서면", "구술"),
        ("허가", "신고"),
        ("신고", "허가"),
        ("취소", "무효"),
        ("무효", "취소"),
    ]
    for old, new in replacements:
        if old in text:
            candidate = text.replace(old, new, 1)
            if candidate != text and base.is_self_contained_snippet(candidate) and 40 <= len(candidate) <= 330:
                return candidate
    return None


def case_intro(article: base.Article, rng: random.Random) -> str:
    group = article.doc.group
    title = article.doc.title
    topic = article.article_title or title
    year = rng.choice([2022, 2023, 2024, 2025])
    money = rng.choice(["500만 원", "1,200만 원", "3억 원", "10억 원"])
    if article.doc.subject == "민사법":
        if group == "상법":
            return (
                f"A주식회사의 이사 甲은 {year}. 3. 10. 이사회 결의와 주주총회 절차가 문제되는 거래를 추진하였다. "
                f"주주 乙은 {topic} 관련 쟁점에 관하여 회사와 甲의 책임을 다투고 있고, 丙은 별도의 {money} 채권을 주장한다. "
                "제1심에서 일부 청구가 인용되자 당사자들은 항소 여부와 절차상 하자의 효과까지 함께 다투고 있다."
            )
        if "소송" in title or "집행" in title or "파산" in title:
            return (
                f"甲은 {year}. 5. 1. 乙을 상대로 {money} 지급을 구하는 소를 제기하였다. "
                f"소송 계속 중 丙이 이해관계를 주장하고, 법원은 {topic} 관련 절차상 조치를 검토하고 있다. "
                "변론 과정에서 소송요건, 신청의 취하, 판결의 효력 및 권리보전 조치가 함께 문제되었다."
            )
        return (
            f"甲은 {year}. 2. 15. 乙과 계약을 체결하였고, 이후 丙에게 관련 권리가 이전되었다. "
            f"甲과 乙은 {topic} 관련 효력 및 권리행사 범위를 두고 다투고 있다. "
            "계약 체결 당시의 의사표시, 이행 지체, 제3자에 대한 대항요건 및 손해배상 범위가 순차적으로 문제된다."
        )
    if article.doc.subject == "형사법":
        if "소송" in title:
            return (
                f"사법경찰관 P는 {year}. 4. 20. 甲의 범죄 혐의를 수사하면서 압수ㆍ수색 및 피의자신문 절차를 진행하였다. "
                f"검사 S는 甲을 기소하였고, 법원은 {topic} 관련 절차 위반 여부를 심리하고 있다. "
                "공판에서는 증거능력, 방어권 보장, 영장주의 위반 여부 및 상소심 판단 범위가 함께 다투어진다."
            )
        return (
            f"甲은 {year}. 6. 3. 乙과 다투던 중 재산상 이익 및 신체 침해가 문제되는 행위를 하였고, "
            f"丙은 그 과정에 일부 관여하였다. 甲의 행위가 {topic} 관련 범죄 성립과 처벌에서 문제된다. "
            "행위 전후의 공모관계, 고의, 위법성조각사유, 죄수관계 및 형의 가중ㆍ감경 사유도 함께 검토된다."
        )
    if group == "헌법":
        return (
            f"甲은 {year}. 1. 5. 공권력 행사로 기본권을 침해받았다고 주장하면서 권리구제 절차를 검토하고 있다. "
            f"국가기관 乙의 조치와 국회 丙의 의결이 {topic} 관련 쟁점으로 문제된다. "
            "甲은 헌법소원, 권한쟁의, 위헌법률심판 제청 가능성과 권리보호이익의 존부를 함께 주장한다."
        )
    return (
        f"甲은 {year}. 7. 12. 행정청 乙로부터 불이익한 처분을 받았고, 丙은 같은 처분의 이해관계인이다. "
        f"甲은 {topic} 관련 쟁점에 관하여 행정심판 또는 행정소송을 제기할 수 있는지 검토하고 있다. "
        "처분사유의 추가ㆍ변경, 제소기간, 집행정지, 사정판결 및 간접강제 가능성이 함께 문제된다."
    )


def case_intro_for_item(item: dict, rng: random.Random) -> str:
    if item["kind"] == "statute":
        return case_intro(item["article"], rng)
    subject = item_subject(item)
    title = item["precedent"]["case_name"]
    year = rng.choice([2022, 2023, 2024, 2025])
    if subject == "형사법":
        return (
            f"甲은 {year}. 4. 20. 범죄 혐의로 수사를 받았고, 수사기관의 증거수집 및 법원의 사실인정이 문제되었다. "
            f"검사 S와 변호인 乙은 {title} 관련 판례 법리가 이 사안에 적용되는지를 다투고 있다."
        )
    if subject == "민사법":
        return (
            f"甲과 乙은 {year}. 3. 10. 계약 또는 손해배상 관계를 두고 소송을 진행 중이고, 丙은 관련 권리관계를 주장한다. "
            f"법원은 {title} 판례 법리와 현행 민사법 규정을 함께 검토하고 있다."
        )
    return (
        f"甲은 {year}. 7. 12. 행정청 乙의 처분 또는 공권력 행사로 권리가 침해되었다고 주장한다. "
        f"甲은 {title} 판례 법리와 현행 공법 규정에 따라 권리구제를 검토하고 있다."
    )


def source_reference(item: dict) -> dict[str, str]:
    if item["kind"] == "statute":
        return base.make_reference(item["article"])
    p = item["precedent"]
    return {
        "title": p["case_name"],
        "article": "판결요지",
        "source_url": p["source_url"],
        "source_path": p["source_path"],
        "source_commit": "",
        "effective_date": p["decision_date"],
        "promulgation_date": p["decision_date"],
        "promulgation_number": p["court"],
        "excerpt": p["holding"],
    }


def source_label(item: dict) -> str:
    if item["kind"] == "statute":
        return base.reference_label(item["article"])
    p = item["precedent"]
    court = f"{p['court']} " if p["court"] else ""
    decided = f"{p['decision_date']} " if p["decision_date"] else ""
    return f"{court}{decided}{p['case_name']} 판결요지"


def main_article_for(items: list[dict]) -> base.Article:
    for item in items:
        if item["kind"] == "statute":
            return item["article"]
    raise ValueError("mixed question needs at least one statute item")


def combo_options(rng: random.Random, correct: frozenset[str]) -> tuple[list[str], int]:
    candidates = [s for s in OPTION_SETS if s != correct]
    rng.shuffle(candidates)
    picked = candidates[:4]
    answer_pos = rng.randrange(5)
    picked.insert(answer_pos, correct)
    return [label_combo(s) for s in picked], answer_pos


def assemble_combo_row(
    qid: int,
    qtype: str,
    stem_prefix: str,
    statements: list[dict],
    as_of: date,
) -> dict:
    true_letters = frozenset(LETTERS[i] for i, st in enumerate(statements) if st["truth"])
    rng = random.Random(qid * 7919 + len(stem_prefix))
    choices, answer_pos = combo_options(rng, true_letters)
    block = "\n".join(f"{LETTERS[i]}. {st['text']}" for i, st in enumerate(statements))
    stem = f"{stem_prefix}\n\n{block}"
    refs = []
    ref_keys = set()
    explanation_parts = [f"정답은 {answer_pos + 1}번이다."]
    for i, st in enumerate(statements):
        label = LETTERS[i]
        ref = source_reference(st)
        key = (ref["title"], ref["article"])
        if key not in ref_keys:
            ref_keys.add(key)
            refs.append(ref)
        if st["truth"]:
            explanation_parts.append(f"{label}은 옳다. {source_label(st)}에서 해당 취지를 확인할 수 있다.")
        else:
            explanation_parts.append(
                f"{label}은 옳지 않다. {source_label(st)}의 실제 내용은 \"{st['actual']}\"이다."
            )
    return base.build_row(qid, qtype, main_article_for(statements), stem, choices, answer_pos, " ".join(explanation_parts), refs)


def make_combo_question(
    rng: random.Random,
    qid: int,
    pool: list[dict],
    as_of: date,
    with_case: bool,
) -> dict | None:
    seed = rng.choice(pool)
    related = pick_related(rng, pool, seed, 3)
    selected = [seed, *related]
    rng.shuffle(selected)
    false_candidates = [i for i, item in enumerate(selected) if make_false(item["text"])]
    if len(false_candidates) < 1:
        return None
    false_count = rng.choice([1, 2])
    false_count = min(false_count, len(false_candidates), 3)
    false_indexes = set(rng.sample(false_candidates, false_count))
    statements = []
    for i, item in enumerate(selected):
        text = item["text"]
        if i in false_indexes:
            false_text = make_false(text)
            if not false_text:
                return None
            statements.append({**item, "text": false_text, "truth": False, "actual": text})
        else:
            statements.append({**item, "text": text, "truth": True, "actual": text})

    if with_case:
        intro = case_intro_for_item(seed, rng)
        stem_prefix = (
            f"{intro} 이에 관한 설명 중 옳은 것을 모두 고른 것은? "
            f"(기준일 {as_of.isoformat()} 현재 시행 법령에 의함)"
        )
        qtype = "case_multi_statement"
    else:
        title = item_title(seed)
        stem_prefix = (
            f"{title} 및 관련 법령ㆍ판례의 내용에 관한 설명 중 옳은 것을 모두 고른 것은? "
            "각 지문은 동일한 사안에서 문제될 수 있는 실체법상 요건, 절차법상 효과 및 판례 법리를 교차하여 서술한 것이다. "
            f"(기준일 {as_of.isoformat()} 현재 시행 법령에 의함)"
        )
        qtype = "multi_statement"
    return assemble_combo_row(qid, qtype, stem_prefix, statements, as_of)


def make_negative_question(rng: random.Random, qid: int, pool: list[dict], as_of: date) -> dict | None:
    seed = rng.choice(pool)
    related = pick_related(rng, pool, seed, 4)
    true_items = [seed, *related[:3]]
    false_source = related[3]
    false_text = make_false(false_source["text"])
    if not false_text:
        return None
    choices = [item["text"] for item in true_items] + [false_text]
    refs = [source_reference(item) for item in true_items]
    refs.append(source_reference(false_source))
    answer_pos = rng.randrange(5)
    choices[answer_pos], choices[-1] = choices[-1], choices[answer_pos]
    stem = (
        f"{item_title(seed)} 및 관련 법령ㆍ판례의 내용에 관한 설명으로 옳지 않은 것은? "
        "각 선택지는 서로 다른 법률관계나 절차 단계에서 문제되는 요건을 전제로 한다. "
        f"(기준일 {as_of.isoformat()} 현재 시행 법령에 의함)"
    )
    explanation = (
        f"정답은 {answer_pos + 1}번이다. 선택지 {answer_pos + 1}은 "
        f"{source_label(false_source)}의 실제 내용인 \"{false_source['text']}\"를 달리 바꾼 것이다. "
        "나머지 선택지는 각 참고 조문에서 확인되는 내용이다."
    )
    return base.build_row(qid, "negative_statement", main_article_for([seed, *related]), stem, choices, answer_pos, explanation, refs)


def make_mixed_precedent_question(
    rng: random.Random,
    qid: int,
    statute_pool: list[dict],
    precedent_pool: list[dict],
    subject: str,
    as_of: date,
) -> dict | None:
    statutes = [s for s in statute_pool if item_subject(s) == subject]
    precedents = [p for p in precedent_pool if item_subject(p) == subject]
    if len(statutes) < 2 or len(precedents) < 2:
        return None
    selected = rng.sample(statutes, 2) + rng.sample(precedents, 2)
    rng.shuffle(selected)
    false_candidates = [i for i, item in enumerate(selected) if make_false(item["text"])]
    if len(false_candidates) < 1:
        return None
    false_count = rng.choice([1, 2])
    false_count = min(false_count, len(false_candidates), 3)
    false_indexes = set(rng.sample(false_candidates, false_count))
    statements = []
    for i, item in enumerate(selected):
        actual = item["text"]
        if i in false_indexes:
            false_text = make_false(actual)
            if not false_text:
                return None
            statements.append({**item, "text": false_text, "truth": False, "actual": actual})
        else:
            statements.append({**item, "text": actual, "truth": True, "actual": actual})
    intro = case_intro_for_item(selected[0], rng)
    stem_prefix = (
        f"{intro} 다음 설명은 현행 법령 조문과 관련 판례 법리를 함께 검토한 것이다. "
        f"옳은 것을 모두 고른 것은? (기준일 {as_of.isoformat()} 현재 시행 법령 및 판례 법리에 의함)"
    )
    return assemble_combo_row(qid, "mixed_statute_precedent_case", stem_prefix, statements, as_of)


def count_by(rows: list[dict], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def build_hard_rows(
    articles: list[base.Article],
    target_count: int,
    rng: random.Random,
    as_of: date,
    round15_texts: list[dict[str, object]],
) -> tuple[list[dict], dict]:
    statute_pool = statement_pool(articles)
    precedent_pool = load_precedent_pool()
    pool = statute_pool + precedent_pool
    subject_targets = {"공법": 270, "민사법": 460, "형사법": 270}
    subject_counts = Counter()
    type_targets = {
        "mixed_statute_precedent_case": 360,
        "case_multi_statement": 280,
        "multi_statement": 220,
        "negative_statement": 140,
    }
    type_counts = Counter()
    rows = []
    fps = set()
    rejected_sim = 0
    attempts = 0
    max_sim = 0.0
    while len(rows) < target_count and attempts < target_count * 120:
        attempts += 1
        remaining_subjects = [s for s, n in subject_targets.items() if subject_counts[s] < n]
        subject = max(
            remaining_subjects or list(subject_targets),
            key=lambda s: (subject_targets[s] - subject_counts[s]) / subject_targets[s],
        )
        subject_statutes = [item for item in statute_pool if item_subject(item) == subject]
        subject_precedents = [item for item in precedent_pool if item_subject(item) == subject]
        if len(subject_statutes) < 20:
            continue
        remaining_types = [t for t, n in type_targets.items() if type_counts[t] < n]
        qtype = max(
            remaining_types or list(type_targets),
            key=lambda t: (type_targets[t] - type_counts[t]) / type_targets[t],
        )
        try:
            if qtype == "mixed_statute_precedent_case":
                if len(subject_precedents) < 2:
                    continue
                row = make_mixed_precedent_question(rng, len(rows) + 1, statute_pool, precedent_pool, subject, as_of)
            elif qtype == "case_multi_statement":
                row = make_combo_question(rng, len(rows) + 1, subject_statutes, as_of, with_case=True)
            elif qtype == "multi_statement":
                row = make_combo_question(rng, len(rows) + 1, subject_statutes, as_of, with_case=False)
            else:
                row = make_negative_question(rng, len(rows) + 1, subject_statutes, as_of)
        except Exception:
            continue
        if row is None:
            continue
        sim = base.max_round15_similarity(row["question_text"], round15_texts)
        max_sim = max(max_sim, sim)
        if sim >= base.ROUND15_SIM_THRESHOLD:
            rejected_sim += 1
            continue
        if row["fingerprint"] in fps:
            continue
        fps.add(row["fingerprint"])
        row["round15_max_similarity"] = f"{sim:.4f}"
        rows.append(row)
        subject_counts[row["subject"]] += 1
        type_counts[row["question_type"]] += 1

    if len(rows) != target_count:
        raise RuntimeError(f"Generated {len(rows)} rows after {attempts} attempts")
    for idx, row in enumerate(rows, 1):
        row["id"] = f"ko_current_law_bar_hard_sft_{idx:04d}"
        row["exam_style"] = "bar_exam_multiple_choice_hard_synthetic"
        row["generation_method"] = "hard_statute_case_combo_generation_with_round15_similarity_filter"
    qa = {
        "target_count": target_count,
        "row_count": len(rows),
        "attempts": attempts,
        "subject_counts": dict(subject_counts),
        "question_type_counts": dict(type_counts),
        "law_counts_top20": dict(sorted(count_by(rows, "law_title").items(), key=lambda kv: (-kv[1], kv[0]))[:20]),
        "round15_similarity_threshold": base.ROUND15_SIM_THRESHOLD,
        "round15_max_similarity": round(max_sim, 4),
        "rejected_round15_similarity": rejected_sim,
        "duplicate_fingerprints": len(rows) - len({row["fingerprint"] for row in rows}),
        "difficulty_note": "Harder v2: case-framed and ㄱㄴㄷㄹ multi-statement questions replace direct lookup questions.",
        "precedent_pool_count": len(precedent_pool),
    }
    return rows, qa


def write_readme(path: Path, repo_id: str, as_of: date, qa: dict) -> None:
    text = f"""---
license: other
language:
- ko
pretty_name: Korean Current-Law Bar Exam Hard SFT 1000
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

# Korean Current-Law Bar Exam Hard SFT 1000

대한민국 현행 법령을 기준으로 만든 변호사시험 선택형 **고난도 스타일** SFT 데이터 1,000문항입니다.

초기 직접 조문확인형 생성본은 실제 제14ㆍ15회 변호사시험보다 쉬워서, 이 버전은 다음 기준으로 다시 만들었습니다.

- `ㄱ/ㄴ/ㄷ/ㄹ` 복합정오형 중심
- `甲/乙/丙`, 검사ㆍ사법경찰관ㆍ행정청ㆍ회사ㆍ소송당사자 등이 등장하는 사례형 비중 확대
- 단순 근거 조문 선택형 제거
- 정답뿐 아니라 각 지문별 O/X 이유와 참고 법령 조문 제공
- 제15회 변호사시험 `data/questions.csv`와 높은 유사도 문항 제외

## Files

- `data/questions.csv`: Hugging Face preview용 메인 CSV입니다.
- `sft/train.jsonl`: `messages` 형식 SFT용 JSONL입니다.
- `metadata/qa_report.json`: 생성 수량, 난도 관련 분포, 제15회 유사도 QA 결과입니다.

## Columns

- `question_text`: 문제와 5개 선택지
- `answer`: 정답 번호, `1`부터 `5`
- `answer_text`: 정답 선택지 원문
- `explanation`: 지문별 정오 판단과 근거 조문
- `references_json`: 참고 법령, 조문, 출처 URL, 로컬 경로, git commit, 시행일자
- `prompt`, `response`, `messages_json`: SFT 변환용 텍스트

## Generation Policy

- 기준일: `{as_of.isoformat()}`
- 법령 원천: `legalize-kr/legalize-kr` 로컬 스냅샷 및 git 이력
- 미래 시행본 제외: `시행일자 <= {as_of.isoformat()}`인 버전만 사용
- 기존 기출 복제 금지: `gyung/korean-bar-exam-moj-multiple-choice`의 제15회 문항과 문자열 유사도 `{base.ROUND15_SIM_THRESHOLD}` 이상인 생성 문항 제외
- 문항 유형: 사례형 복합정오, 일반 복합정오, 옳지 않은 설명형

## QA Summary

```json
{json.dumps(qa, ensure_ascii=False, indent=2)}
```

## Sources

- Existing bar-exam reference dataset: https://huggingface.co/datasets/gyung/korean-bar-exam-moj-multiple-choice
- Ministry of Justice, 제15회 변호사시험 기출문제/정답 공지
  - https://www.moj.go.kr/bbs/moj/150/602397/artclView.do
  - https://www.moj.go.kr/bbs/moj/150/602398/artclView.do
  - https://www.moj.go.kr/bbs/moj/150/602399/artclView.do
  - https://www.moj.go.kr/bbs/moj/151/603464/artclView.do
- Korean statutes: https://github.com/legalize-kr/legalize-kr and https://www.law.go.kr

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
    ap.add_argument("--seed", type=int, default=2026061302)
    args = ap.parse_args()

    as_of = date.fromisoformat(args.as_of)
    rng = random.Random(args.seed)
    legal_root = Path(args.legal_root)
    out_dir = Path(args.out_dir)

    docs = base.load_law_docs(legal_root, as_of)
    base_docs = []
    seen = set()
    for doc in docs:
        key = (doc.title, doc.rel_path, doc.source_commit)
        if key not in seen:
            seen.add(key)
            base_docs.append(doc)
    articles = []
    for doc in base_docs:
        articles.extend(base.split_articles(doc))

    weighted_articles = []
    for spec in base.LAW_SPECS:
        for article in articles:
            if article.doc.rel_path == spec.rel_path:
                weighted_articles.extend([article] * spec.weight)
    round15_texts = base.load_round15_texts(Path(args.bar_questions))
    rows, qa = build_hard_rows(weighted_articles, args.count, rng, as_of, round15_texts)
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
    base.write_csv(out_dir / "data" / "questions.csv", rows)
    base.write_sft_jsonl(out_dir / "sft" / "train.jsonl", rows)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata" / "qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(out_dir / "README.md", args.repo_id, as_of, qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
