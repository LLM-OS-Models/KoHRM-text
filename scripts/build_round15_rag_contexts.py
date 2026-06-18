"""Build concise, answer-free RAG contexts for round 15 bar-exam questions.

The older round15_rag_* builders used fuzzy statute directory matching. That can
map short names such as "민법" to "난민법" or "형법" to "군형법". This builder
uses exact statute aliases, short statute snippets, and local precedent matches.
"""
from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CORE_LAWS = [
    "대한민국헌법",
    "헌법재판소법",
    "국회법",
    "국가공무원법",
    "공직선거법",
    "정당법",
    "법원조직법",
    "행정기본법",
    "행정절차법",
    "행정심판법",
    "행정소송법",
    "국가배상법",
    "지방자치법",
    "개인정보보호법",
    "민법",
    "민사소송법",
    "민사집행법",
    "상법",
    "어음법",
    "수표법",
    "부동산등기법",
    "가등기담보등에관한법률",
    "집합건물의소유및관리에관한법률",
    "주택임대차보호법",
    "상가건물임대차보호법",
    "채무자회생및파산에관한법률",
    "가족관계의등록등에관한법률",
    "국제사법",
    "형법",
    "형사소송법",
    "검찰청법",
    "군형법",
    "특정범죄가중처벌등에관한법률",
    "성폭력범죄의처벌등에관한특례법",
    "아동ㆍ청소년의성보호에관한법률",
    "교통사고처리특례법",
    "도로교통법",
    "마약류관리에관한법률",
    "변호사법",
    "부정수표단속법",
]


LAW_ALIASES = {
    "헌법": "대한민국헌법",
    "민소법": "민사소송법",
    "형소법": "형사소송법",
    "특가법": "특정범죄가중처벌등에관한법률",
    "성폭력처벌법": "성폭력범죄의처벌등에관한특례법",
    "아청법": "아동ㆍ청소년의성보호에관한법률",
    "교특법": "교통사고처리특례법",
    "채무자회생법": "채무자회생및파산에관한법률",
}


KEYWORD_LAW_RULES = [
    (r"국회|국회의원|법률안|탄핵|국정감사|국정조사", ["대한민국헌법", "국회법"]),
    (r"대통령|국무총리|국무위원|행정각부", ["대한민국헌법", "정부조직법"]),
    (r"헌법재판|위헌|헌법소원|권한쟁의|탄핵심판", ["대한민국헌법", "헌법재판소법"]),
    (r"선거|정당|비례대표|후보자", ["공직선거법", "정당법", "대한민국헌법"]),
    (r"공무원|징계|직위해제|임용", ["국가공무원법", "대한민국헌법"]),
    (r"처분|행정청|재량|취소소송|무효확인|부작위|행정소송|원고적격|제소기간", ["행정소송법", "행정기본법", "행정절차법"]),
    (r"행정심판|재결", ["행정심판법", "행정소송법"]),
    (r"국가배상|손실보상|공무원의 직무", ["국가배상법", "대한민국헌법"]),
    (r"지방자치|조례|지방의회|지방자치단체", ["지방자치법", "대한민국헌법"]),
    (r"대리|표현대리|무권대리|제126조|제125조|제129조", ["민법"]),
    (r"소유권|점유취득시효|취득시효|물권|유치권|저당권|전세권|공유|부합|명의신탁", ["민법"]),
    (r"채권|채무|계약|해제|해지|손해배상|불법행위|부당이득|상계|소멸시효|보증", ["민법"]),
    (r"상속|유류분|혼인|이혼|친생자|입양|후견", ["민법", "가족관계의등록등에관한법률"]),
    (r"소송|항소|상고|재심|송달|공시송달|변론|자백|자백간주|증거|기판력|청구취지|소송비용", ["민사소송법"]),
    (r"강제집행|압류|추심|전부명령|가압류|가처분|배당", ["민사집행법"]),
    (r"회사|주식|주주|이사|대표이사|감사|상행위|상인|영업양도", ["상법"]),
    (r"환어음|약속어음|어음|배서|인수|지급인|소지인", ["어음법", "상법"]),
    (r"수표", ["수표법", "상법"]),
    (r"등기|가등기|등기권리증", ["부동산등기법", "민법"]),
    (r"임대차|대항력|우선변제|보증금", ["주택임대차보호법", "상가건물임대차보호법", "민법"]),
    (r"회생|파산|면책|부인권", ["채무자회생및파산에관한법률"]),
    (r"죄수|구성요건|위법성|책임|공범|교사|방조|미수|고의|착오|정당방위|긴급피난", ["형법"]),
    (r"사기죄|횡령|배임|절도|강도|상해|폭행|살인|강간|공갈|문서|뇌물|공무집행방해|방화|주거침입", ["형법"]),
    (r"공소|기소|영장|체포|구속|압수|수색|증거능력|전문법칙|자백보강|피의자|피고인|상소|재심|면소", ["형사소송법"]),
    (r"성폭력|카메라|통신매체|추행", ["성폭력범죄의처벌등에관한특례법", "형법"]),
    (r"아동ㆍ청소년|청소년성보호", ["아동ㆍ청소년의성보호에관한법률", "형법"]),
    (r"교통사고|도로교통|음주운전", ["교통사고처리특례법", "도로교통법", "형법"]),
    (r"마약|향정", ["마약류관리에관한법률", "형법"]),
]


CONCEPT_TERMS = [
    "포괄일죄 관계인 범행의 일부",
    "상상적 경합관계",
    "확정판결의 기판력",
    "인신사고로 인한 손해배상",
    "피해자의 기대여명",
    "재판상 자백의 대상",
    "포괄일죄",
    "상상적경합",
    "상상적 경합",
    "기판력",
    "표현대리",
    "무권대리",
    "성명모용",
    "정당한 사유",
    "점유취득시효",
    "재판상 자백",
    "자백간주",
    "공시송달",
    "기대여명",
    "환어음",
    "부단순인수",
    "인수제시",
    "대항력",
    "우선변제권",
    "소멸시효",
    "유치권",
    "명의신탁",
    "부당이득",
    "불법행위",
    "채권자대위",
    "채권자취소",
    "공동불법행위",
    "유류분",
    "이사회",
    "주주총회",
    "대표이사",
    "영업양도",
    "전문법칙",
    "증거능력",
    "압수수색",
    "위법수집증거",
    "공소시효",
    "공소장변경",
    "일사부재리",
    "불이익변경금지",
    "죄형법정주의",
    "정당방위",
    "긴급피난",
    "책임능력",
    "고의",
    "착오",
    "공범",
    "교사",
    "방조",
    "사기죄",
    "횡령죄",
    "배임죄",
    "장물",
    "뇌물",
    "주거침입",
    "강제추행",
    "헌법소원",
    "권한쟁의",
    "위헌법률심판",
    "과잉금지원칙",
    "평등원칙",
    "명확성원칙",
    "신뢰보호",
    "법률유보",
    "원고적격",
    "처분성",
    "재량권",
    "부작위",
    "국가배상",
    "손실보상",
]


STOP_TERMS = {
    "설명",
    "경우",
    "다툼",
    "판례",
    "관한",
    "것은",
    "있는",
    "없는",
    "다음",
    "모두",
    "옳은",
    "옳지",
    "각각",
    "甲",
    "乙",
    "丙",
    "丁",
}


CONCEPT_ARTICLES = {
    "민법": [
        (r"표현대리|무권대리|성명모용", ["제125조", "제126조", "제129조"]),
        (r"점유취득시효|취득시효", ["제245조", "제246조", "제247조"]),
        (r"손해배상|불법행위", ["제750조", "제751조", "제763조"]),
        (r"소멸시효", ["제162조", "제166조", "제168조"]),
        (r"채권자대위", ["제404조"]),
        (r"채권자취소", ["제406조", "제407조"]),
        (r"상계", ["제492조", "제493조", "제496조"]),
        (r"유류분", ["제1112조", "제1113조", "제1115조"]),
    ],
    "민사소송법": [
        (r"자백간주|재판상 자백|자백의 대상|공시송달", ["제150조"]),
        (r"기판력", ["제216조", "제218조"]),
        (r"재심", ["제451조"]),
        (r"상고|항소", ["제390조", "제422조"]),
    ],
    "형법": [
        (r"상상적 ?경합|상상적경합", ["제40조"]),
        (r"교사|방조|공범", ["제30조", "제31조", "제32조"]),
        (r"사기죄", ["제347조"]),
        (r"횡령", ["제355조", "제356조"]),
        (r"장물", ["제362조"]),
    ],
    "형사소송법": [
        (r"기판력|확정판결|면소|포괄일죄|일사부재리", ["제326조"]),
        (r"압수|수색|영장", ["제106조", "제215조", "제216조", "제218조"]),
        (r"전문법칙|증거능력|자백보강", ["제307조", "제308조", "제309조", "제310조의2", "제312조"]),
        (r"공소시효", ["제249조", "제253조"]),
    ],
    "어음법": [
        (r"환어음|인수|인수제시|지급인|소지인|단순한 점유자", ["제21조", "제22조", "제25조", "제26조", "제27조"]),
        (r"배서", ["제11조", "제12조", "제13조", "제14조", "제15조"]),
    ],
    "상법": [
        (r"주주총회", ["제361조", "제363조", "제368조"]),
        (r"이사|대표이사|이사회", ["제382조", "제389조", "제393조"]),
        (r"영업양도", ["제41조", "제42조", "제45조"]),
    ],
}


@dataclass
class LawDoc:
    name: str
    path: Path
    title: str
    source_url: str
    text: str


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def clean_text(text: str, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n..."


def front_value(text: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    if not m:
        return ""
    return m.group(1).strip().strip("'\"")


def build_law_index(root: Path) -> dict[str, LawDoc]:
    out: dict[str, LawDoc] = {}
    for name in CORE_LAWS:
        d = root / name
        if not d.is_dir():
            continue
        candidates = [d / "법률.md", d / "헌법.md", d / f"{name}.md"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            mds = sorted(d.glob("*.md"))
            path = mds[0] if mds else None
        if path is None:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = front_value(text, "제목") or name
        source_url = front_value(text, "출처") or f"https://www.law.go.kr/법령/{name}"
        out[name] = LawDoc(name=name, path=path, title=title, source_url=source_url, text=text)
    return out


def extract_articles(qtext: str) -> list[str]:
    articles = []
    for m in re.finditer(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?", qtext):
        art = m.group(1) + (f"의{m.group(2)}" if m.group(2) else "")
        if art not in articles:
            articles.append(art)
    return articles[:8]


def select_laws(qtext: str, subject: str, law_index: dict[str, LawDoc]) -> list[str]:
    q_norm = compact(qtext)
    explicit: list[str] = []
    found: list[str] = []

    for alias, canonical in LAW_ALIASES.items():
        if compact(alias) in q_norm and canonical in law_index:
            explicit.append(canonical)

    for name in sorted(law_index, key=len, reverse=True):
        if compact(name) in q_norm:
            explicit.append(name)

    subject_allowed = {
        "공법": {
            "대한민국헌법",
            "헌법재판소법",
            "국회법",
            "국가공무원법",
            "공직선거법",
            "정당법",
            "법원조직법",
            "행정기본법",
            "행정절차법",
            "행정심판법",
            "행정소송법",
            "국가배상법",
            "지방자치법",
            "개인정보보호법",
        },
        "민사법": {
            "민법",
            "민사소송법",
            "민사집행법",
            "상법",
            "어음법",
            "수표법",
            "부동산등기법",
            "가등기담보등에관한법률",
            "집합건물의소유및관리에관한법률",
            "주택임대차보호법",
            "상가건물임대차보호법",
            "채무자회생및파산에관한법률",
            "가족관계의등록등에관한법률",
            "국제사법",
        },
        "형사법": {
            "형법",
            "형사소송법",
            "검찰청법",
            "군형법",
            "특정범죄가중처벌등에관한법률",
            "성폭력범죄의처벌등에관한특례법",
            "아동ㆍ청소년의성보호에관한법률",
            "교통사고처리특례법",
            "도로교통법",
            "마약류관리에관한법률",
            "변호사법",
            "부정수표단속법",
        },
    }
    allowed = subject_allowed.get(subject, set())

    found.extend(explicit)

    for pattern, laws in KEYWORD_LAW_RULES:
        if re.search(pattern, qtext):
            found.extend(l for l in laws if l in law_index and l in allowed)

    if subject == "형사법" and re.search(r"기판력|확정판결|약식명령|면소|공소|상소|증거|전문법칙|압수|수색|영장|체포|구속", qtext):
        if "형사소송법" in law_index:
            found.append("형사소송법")
    if subject == "민사법" and re.search(r"소송|항소|상고|송달|공시송달|변론|자백|자백간주|증거|기판력|청구취지|소송비용", qtext):
        if "민사소송법" in law_index:
            found.append("민사소송법")

    if not found:
        fallback = {
            "공법": ["대한민국헌법", "행정소송법"],
            "민사법": ["민법", "민사소송법"],
            "형사법": ["형법", "형사소송법"],
        }
        found = [l for l in fallback.get(subject, []) if l in law_index]

    # Prefer non-fallback laws but keep order; cap context size.
    explicit_set = set(explicit)
    deduped = [l for l in dict.fromkeys(found) if l in allowed or l in explicit_set]
    if "대한민국헌법" in deduped and subject == "공법":
        deduped.insert(0, deduped.pop(deduped.index("대한민국헌법")))
    return deduped[:4]


def extract_terms(qtext: str, law_names: list[str]) -> list[str]:
    q_norm = compact(qtext)
    terms: list[str] = []
    for t in CONCEPT_TERMS:
        if compact(t) in q_norm:
            terms.append(t)
    for m in re.finditer(r"[가-힣]{2,12}(?:죄|권|법|소송|시효|대리|자백|경합|처분|보상|배상|증거|영장|공소|인수|배서|상속|등기)", qtext):
        terms.append(m.group(0))
    for law in law_names:
        terms.append(law)
    cleaned = []
    for t in terms:
        t = re.sub(r"\s+", " ", t).strip()
        if len(compact(t)) < 2 or t in STOP_TERMS:
            continue
        if t not in cleaned:
            cleaned.append(t)
    return cleaned[:12]


def iter_article_blocks(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s*#{1,6}\s*(제\d+조(?:의\d+)?)\b", line)
        if m:
            starts.append((i, m.group(1)))
    blocks: list[tuple[str, str]] = []
    for idx, (start, art) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        blocks.append((art, "\n".join(lines[start:end]).strip()))
    return blocks


def statute_snippets(doc: LawDoc, articles: list[str], terms: list[str]) -> list[tuple[str, str]]:
    blocks = iter_article_blocks(doc.text)
    snippets: list[tuple[str, str]] = []
    wanted = {f"제{a}조" for a in articles}
    joined_terms = " ".join(terms)
    for pattern, mapped_articles in CONCEPT_ARTICLES.get(doc.name, []):
        if re.search(pattern, joined_terms):
            wanted.update(mapped_articles)
    for art, block in blocks:
        if art in wanted:
            snippets.append((art, clean_text(block, 900)))
    if snippets:
        return snippets[:5]

    scored = []
    generic = {doc.name, "대한민국헌법", "민법", "형법", "상법", "민사소송법", "형사소송법"}
    useful_terms = [compact(t) for t in terms if compact(t) not in {compact(x) for x in generic} and len(compact(t)) >= 3]
    for art, block in blocks:
        block_norm = compact(block)
        score = sum(2 if len(t) >= 5 else 1 for t in useful_terms if t and t in block_norm)
        if score:
            scored.append((score, art, block))
    scored.sort(key=lambda x: (-x[0], len(x[2])))
    if scored:
        return [(art, clean_text(block, 900)) for _, art, block in scored[:2]]

    head = "\n".join(doc.text.strip().splitlines()[:24])
    return [("개요", clean_text(head, 700))]


def precedent_roots(subject: str, precedent_root: Path) -> list[Path]:
    if subject == "공법":
        names = ["일반행정", "선거·특별", "세무", "기타"]
    elif subject == "민사법":
        names = ["민사", "가사", "특허"]
    else:
        names = ["형사"]
    return [precedent_root / n for n in names if (precedent_root / n).exists()]


def parse_precedent_meta(text: str, path: Path) -> dict[str, str]:
    return {
        "case_no": front_value(text, "사건번호"),
        "case_name": front_value(text, "사건명") or path.stem,
        "court": front_value(text, "법원명"),
        "date": front_value(text, "선고일자"),
        "source": front_value(text, "출처"),
    }


def best_precedent_snippet(text: str, terms: list[str]) -> str:
    body = text
    for heading in ["## 판결요지", "## 판시사항", "## 판례내용"]:
        idx = text.find(heading)
        if idx >= 0:
            body = text[idx:]
            break
    body_norm = compact(body)
    best_pos = -1
    best_term = ""
    for term in terms:
        t = compact(term)
        if len(t) < 3:
            continue
        pos = body_norm.find(t)
        if pos >= 0 and (best_pos < 0 or pos < best_pos):
            best_pos = pos
            best_term = term
    if best_term:
        plain = re.sub(r"\s+", " ", body)
        pos = plain.find(best_term)
        if pos >= 0:
            start = max(0, pos - 180)
            return clean_text(plain[start : pos + 520], 750)
    lines = [ln for ln in body.splitlines() if ln.strip()]
    return clean_text("\n".join(lines[:12]), 750)


def search_precedents(subject: str, precedent_root: Path, terms: list[str], max_hits: int = 3) -> list[dict[str, str]]:
    roots = precedent_roots(subject, precedent_root)
    if not roots:
        return []
    generic = {
        "민법",
        "형법",
        "상법",
        "대한민국헌법",
        "민사소송법",
        "형사소송법",
        "행정소송법",
        "판례",
        "소송",
        "증거",
    }
    search_terms = [t for t in terms if t not in generic and len(compact(t)) >= 3]
    search_terms = sorted(search_terms, key=lambda x: (-len(compact(x)), x))[:8]
    if not search_terms:
        return []

    candidate_paths: list[Path] = []
    for term in search_terms[:6]:
        root_args = " ".join(shlex.quote(str(r)) for r in roots)
        cmd = f"rg -l -F --glob '*.md' -- {shlex.quote(term)} {root_args} | head -n 80"
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=12)
        except subprocess.TimeoutExpired:
            continue
        for line in proc.stdout.splitlines():
            p = Path(line)
            if p not in candidate_paths:
                candidate_paths.append(p)

    scored = []
    norm_terms = [compact(t) for t in terms if len(compact(t)) >= 3]
    for path in candidate_paths[:120]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        meta = parse_precedent_meta(text, path)
        if "시험불합격처분취소" in meta.get("case_name", ""):
            continue
        if "정답" in text or "답항" in text:
            continue
        text_norm = compact(text)
        score = sum(2 if len(t) >= 5 else 1 for t in norm_terms if t in text_norm)
        if score < 3:
            continue
        if "/대법원/" in str(path):
            score += 3
        scored.append((score, path, meta, best_precedent_snippet(text, terms)))

    scored.sort(key=lambda x: (-x[0], "대법원" not in str(x[1]), str(x[1])))
    hits = []
    seen = set()
    for score, path, meta, snippet in scored:
        key = meta.get("source") or str(path)
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            {
                "score": str(score),
                "path": str(path),
                "case_name": meta.get("case_name", ""),
                "case_no": meta.get("case_no", ""),
                "court": meta.get("court", ""),
                "date": meta.get("date", ""),
                "source": meta.get("source", ""),
                "snippet": snippet,
            }
        )
        if len(hits) >= max_hits:
            break
    return hits


def write_readme(out_dir: Path) -> None:
    readme = """# 제15회 변호사시험 RAG 컨텍스트

이 폴더는 변호사시험 15회 선택형 150개 문항에 붙일 짧은 풀이 보조 컨텍스트입니다.

- 문제/선택지는 포함하되 선택지 번호 판정 정보는 넣지 않았습니다.
- 법령은 `legalize-kr/kr/<정확한 법령명>/...`에서만 가져왔습니다. 부분문자열 매칭은 쓰지 않았습니다.
- 판례는 `../precedent-kr` 실제 판례 문서에서 가져왔고, 행정규칙/훈령 JSONL은 제외했습니다.
- `ordinance-kr`는 자치법규용이라 15회 선택형의 일반 법령·판례 쟁점에는 직접 근거로 넣지 않았습니다.
- 웹 확인은 국가법령정보센터/대법원 공개 URL을 우선했습니다.

토큰을 아끼기 위해 각 문항은 핵심 법령 1-4개, 조문/판례 발췌는 짧게 제한했습니다.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def render_context(row: dict[str, str], law_index: dict[str, LawDoc], precedent_root: Path) -> str:
    subject = row["subject"]
    qno = int(row["question_no"])
    qtext = row["question_text"].strip()
    laws = select_laws(qtext, subject, law_index)
    articles = extract_articles(qtext)
    terms = extract_terms(qtext, laws)
    precedents = search_precedents(subject, precedent_root, terms)

    lines: list[str] = []
    lines.append(f"# 제15회 변호사시험 q{qno:03d} [{subject}] RAG 컨텍스트")
    lines.append("")
    lines.append("## 문제")
    lines.append("```text")
    lines.append(qtext)
    lines.append("```")
    lines.append("")
    lines.append("## 쟁점 키워드")
    lines.append(", ".join(terms[:10]) if terms else "(자동 추출 없음)")
    lines.append("")

    lines.append("## 법령 근거")
    if laws:
        for law in laws:
            doc = law_index[law]
            lines.append(f"### {doc.title}")
            lines.append(f"- 파일: `{doc.path.relative_to(ROOT) if doc.path.is_relative_to(ROOT) else doc.path}`")
            lines.append(f"- 공식 URL: {doc.source_url}")
            for label, snippet in statute_snippets(doc, articles, terms):
                lines.append(f"#### {label}")
                lines.append("```text")
                lines.append(snippet)
                lines.append("```")
            lines.append("")
    else:
        lines.append("직접 연결할 법령을 찾지 못했습니다.")
        lines.append("")

    lines.append("## 판례 근거")
    if precedents:
        for hit in precedents:
            title_bits = [hit["court"], hit["date"], hit["case_no"], hit["case_name"]]
            title = " ".join(x for x in title_bits if x)
            lines.append(f"### {title or Path(hit['path']).stem}")
            lines.append(f"- 파일: `{hit['path']}`")
            if hit["source"]:
                lines.append(f"- 공식 URL: {hit['source']}")
            lines.append("```text")
            lines.append(hit["snippet"])
            lines.append("```")
            lines.append("")
    else:
        lines.append("로컬 판례 코퍼스에서 높은 공통도 문서를 찾지 못했습니다. 공식 검색 링크를 확인하세요.")
        lines.append("")

    lines.append("## 공식 검색 링크")
    for law in laws[:3]:
        lines.append(f"- {law}: https://www.law.go.kr/법령/{law}")
    if terms:
        query = "+".join(terms[:4])
        lines.append(f"- 판례 검색어: https://www.law.go.kr/precSc.do?query={query}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-csv", default=str(ROOT / "data/bar_exam/hf_multiple_choice/data/questions.csv"))
    parser.add_argument("--legalize-root", default=str(ROOT / "legalize-kr/kr"))
    parser.add_argument("--precedent-root", default=str(ROOT.parent / "precedent-kr"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/bar_exam/round15_rag_contexts_20260618"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_readme(out_dir)

    law_index = build_law_index(Path(args.legalize_root))
    rows: list[dict[str, str]] = []
    with Path(args.questions_csv).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["round"] == "15":
                rows.append(row)

    for idx, row in enumerate(rows, 1):
        subject = row["subject"]
        qno = int(row["question_no"])
        text = render_context(row, law_index, Path(args.precedent_root))
        (out_dir / f"q{qno:03d}_{subject}.md").write_text(text, encoding="utf-8")
        if idx % 10 == 0 or idx == len(rows):
            print(f"{idx}/{len(rows)} generated", flush=True)

    print(f"wrote {len(rows)} files to {out_dir}")


if __name__ == "__main__":
    main()
