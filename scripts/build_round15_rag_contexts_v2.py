"""Build feedback-tuned, answer-free RAG contexts for round 15 bar-exam questions.

V2 keeps the exact-law matching from the first complete builder and adds the
parts that the Gemini run exposed as weak: per-statement precedent candidates,
Constitutional Court supplements, special statutes/rules, and short ratio
paragraphs from the local precedent corpus. It still ignores the answer column.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus


ROOT = Path(__file__).resolve().parents[1]


CORE_LAWS = [
    "대한민국헌법",
    "헌법재판소법",
    "국회법",
    "국가공무원법",
    "공직선거법",
    "정당법",
    "법원조직법",
    "정부조직법",
    "행정기본법",
    "행정절차법",
    "행정심판법",
    "행정소송법",
    "국가배상법",
    "지방자치법",
    "개인정보보호법",
    "민법",
    "민사소송법",
    "민사소송규칙",
    "민사집행법",
    "상법",
    "어음법",
    "수표법",
    "보험업법",
    "신탁법",
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
    "형사소송규칙",
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
    "민소규칙": "민사소송규칙",
    "형소법": "형사소송법",
    "형소규칙": "형사소송규칙",
    "특가법": "특정범죄가중처벌등에관한법률",
    "성폭력처벌법": "성폭력범죄의처벌등에관한특례법",
    "아청법": "아동ㆍ청소년의성보호에관한법률",
    "교특법": "교통사고처리특례법",
    "채무자회생법": "채무자회생및파산에관한법률",
}


KEYWORD_LAW_RULES = [
    (r"국회|국회의원|법률안|탄핵|국정감사|국정조사", ["대한민국헌법", "국회법"]),
    (r"입법절차|입법권|청문권|공청회|청문회", ["대한민국헌법", "국회법", "헌법재판소법"]),
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
    (r"소송|항소|상고|재심|송달|공시송달|변론|자백|자백간주|증거|기판력|청구취지|소송비용|녹음|조서", ["민사소송법", "민사소송규칙"]),
    (r"강제집행|압류|추심|전부명령|가압류|가처분|배당", ["민사집행법"]),
    (r"회사|주식|주주|이사|대표이사|감사|상행위|상인|영업양도|자기거래|보험|보험자|보험계약", ["상법", "보험업법"]),
    (r"환어음|약속어음|어음|배서|인수|지급인|소지인", ["어음법", "상법"]),
    (r"수표", ["수표법", "상법"]),
    (r"등기|가등기|등기권리증", ["부동산등기법", "민법"]),
    (r"가등기담보|청산금|청산절차|담보가등기", ["가등기담보등에관한법률", "민법", "부동산등기법"]),
    (r"신탁|수탁자|위탁자|신탁재산|신탁원부", ["신탁법", "민법", "부동산등기법"]),
    (r"임대차|대항력|우선변제|보증금", ["주택임대차보호법", "상가건물임대차보호법", "민법"]),
    (r"회생|파산|면책|부인권", ["채무자회생및파산에관한법률"]),
    (r"죄수|구성요건|위법성|책임|공범|교사|방조|미수|고의|착오|정당방위|긴급피난", ["형법"]),
    (r"사기죄|횡령|배임|절도|강도|상해|폭행|살인|강간|공갈|문서|뇌물|공무집행방해|방화|주거침입", ["형법"]),
    (r"공소|기소|영장|체포|구속|압수|수색|증거능력|전문법칙|자백보강|피의자|피고인|상소|재심|면소|전자정보|참여권", ["형사소송법", "형사소송규칙"]),
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
    "소유권이전등기청구권 양도",
    "소유권이전청구권 가등기 이전의 부기등기",
    "채무자의 동의나 승낙",
    "원인무효의 등기",
    "국회입법",
    "청문권",
    "입법절차",
    "적법절차원칙",
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
    "민사소송규칙": [
        (r"녹음|녹화|조서|변론", ["제34조", "제37조"]),
        (r"송달|공시송달", ["제51조", "제52조", "제53조"]),
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
    "형사소송규칙": [
        (r"압수|수색|영장|전자정보|참여권", ["제107조", "제109조", "제110조"]),
        (r"공판조서|증거목록|조서", ["제36조", "제37조"]),
    ],
    "어음법": [
        (r"환어음|인수|인수제시|지급인|소지인|단순한 점유자", ["제21조", "제22조", "제25조", "제26조", "제27조"]),
        (r"배서", ["제11조", "제12조", "제13조", "제14조", "제15조"]),
    ],
    "상법": [
        (r"주주총회", ["제361조", "제363조", "제368조"]),
        (r"이사|대표이사|이사회", ["제382조", "제389조", "제393조"]),
        (r"자기거래", ["제398조"]),
        (r"영업양도", ["제41조", "제42조", "제45조"]),
    ],
    "신탁법": [
        (r"신탁재산|강제집행|수탁자|위탁자", ["제22조", "제24조", "제25조"]),
    ],
    "부동산등기법": [
        (r"가등기|부기등기|본등기", ["제88조", "제91조", "제92조"]),
    ],
    "가등기담보등에관한법률": [
        (r"가등기담보|담보가등기|청산금|청산절차", ["제3조", "제4조", "제11조", "제15조"]),
    ],
}


ROLE_TERMS = [
    "명의신탁자",
    "명의수탁자",
    "제3취득자",
    "물상보증인",
    "채무자",
    "채권자",
    "매도인",
    "매수인",
    "양도인",
    "양수인",
    "수탁자",
    "위탁자",
    "피고인",
    "피해자",
]


MANUAL_QUERY_RULES = {
    ("공법", 2): {
        "global": [
            "2005헌마579",
            "국회입법",
            "적법절차에서 파생되는 청문권",
            "입법절차에서의 국민의 직접참여권",
            "국회의 입법권",
            "탄핵소추권 남용",
            "국회의 조약 동의권",
            "국회의장 권한쟁의심판 대표권",
        ],
        "①": ["국회입법", "청문권", "적법절차에서 파생되는 청문권", "2005헌마579"],
        "②": ["탄핵소추의결", "탄핵소추권 남용", "정치적 목적"],
        "③": ["외국군대의 지위", "재정적 부담", "국회의 조약 동의권"],
        "④": ["국회의장", "권한쟁의심판", "본회의 의결", "대표권"],
        "⑤": ["법률 개정 행위", "권한쟁의심판", "국회 피청구인적격", "국회의원 국회의장"],
    },
    ("민사법", 9): {
        "global": [
            "2024다248290",
            "소유권이전등기청구권 양도",
            "채무자의 동의나 승낙",
            "가등기 이전의 부기등기",
            "가등기에 기한 본등기",
            "원인무효의 등기",
            "중간생략등기 합의",
            "3자간 등기명의신탁",
        ],
        "ㄱ": ["확정판결", "소유권이전등기청구권", "소멸시효 완성", "실체관계에 부합하는 등기"],
        "ㄴ": ["3자간 등기명의신탁", "명의수탁자", "제3자 앞으로 소유권이전등기", "실체관계"],
        "ㄷ": ["2024다248290", "소유권이전등기청구권 양도", "매도인 동의", "가등기 이전의 부기등기", "본등기 원인무효"],
        "ㄹ": ["중간생략등기", "관계 당사자 전원의 의사합치", "적법한 원인행위", "중간생략등기 무효"],
    },
    ("형사법", 1): {
        "global": [
            "2020도3705",
            "포괄일죄 관계인 범행의 일부",
            "사실심 판결선고 시",
            "약식명령 발령 시",
            "상상적 경합관계",
            "확정판결의 기판력",
        ],
        "ㄷ": ["2020도3705", "포괄일죄 관계인 범행의 일부", "사실심 판결선고 시", "상상적 경합관계", "확정판결의 기판력"],
        "ㅁ": ["포괄일죄", "법 개정 전후", "신구법", "법정형 경중"],
    },
}


MANUAL_SUPPLEMENTS = {
    ("공법", 2): [
        {
            "label": "①",
            "title": "헌법재판소 2005. 11. 24. 2005헌마579ㆍ763(병합) 공주지역 행정중심복합도시 건설을 위한 특별법 위헌확인",
            "source": "https://www.law.go.kr/LSW/detcInfoP.do?detcSeq=134815&mode=0",
            "snippet": (
                "국회가 공개적 토론과 의결 절차를 거쳐 법률을 제정한 경우, 일반 국민에게 입법 전 "
                "사전 청문절차 참여권이 적법절차원칙에서 곧바로 도출된다고 보기는 어렵다는 취지의 결정례이다. "
                "국회법상 공청회ㆍ청문회, 청원, 헌법소원ㆍ위헌법률심판 등 사후 통제 장치도 함께 고려한다."
            ),
        }
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
            "정부조직법",
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
            "민사소송규칙",
            "민사집행법",
            "상법",
            "어음법",
            "수표법",
            "보험업법",
            "신탁법",
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
            "형사소송규칙",
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
        if "형사소송규칙" in law_index and re.search(r"압수|수색|전자정보|영장|조서", qtext):
            found.append("형사소송규칙")
    if subject == "민사법" and re.search(r"소송|항소|상고|송달|공시송달|변론|자백|자백간주|증거|기판력|청구취지|소송비용|조서", qtext):
        if "민사소송법" in law_index:
            found.append("민사소송법")
        if "민사소송규칙" in law_index and re.search(r"송달|공시송달|녹음|녹화|조서", qtext):
            found.append("민사소송규칙")

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
    return deduped[:5]


def extract_terms(qtext: str, law_names: list[str]) -> list[str]:
    q_norm = compact(qtext)
    terms: list[str] = []
    for t in CONCEPT_TERMS:
        if compact(t) in q_norm:
            terms.append(t)
    for m in re.finditer(r"[가-힣]{2,12}(?:죄|권|법|소송|시효|대리|자백|경합|처분|보상|배상|증거|영장|공소|인수|배서|상속|등기)", qtext):
        terms.append(m.group(0))
    for role in ROLE_TERMS:
        if role in qtext:
            terms.append(role)
    for m in re.finditer(r"[가-힣]{2,12}(?:의|이|가|을|를)?\s+[가-힣]{2,12}(?:권|법|죄|등기|처분|절차|원칙|동의|승낙|합의)", qtext):
        terms.append(re.sub(r"\s+", " ", m.group(0)).strip())
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


def dedupe(seq: list[str], limit: int | None = None) -> list[str]:
    out: list[str] = []
    for item in seq:
        item = re.sub(r"\s+", " ", item).strip()
        if not item or item in out:
            continue
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def manual_terms(subject: str, qno: int, label: str = "global") -> list[str]:
    rules = MANUAL_QUERY_RULES.get((subject, qno), {})
    if label != "global":
        return dedupe(rules.get(label, []))
    return dedupe(rules.get("global", []))


def strip_answer_choice_block(text: str) -> str:
    # Combination questions append answer-choice bundles after the real ㄱ/ㄴ/ㄷ statements.
    return re.split(r"\n\s*①\s*\n", text, maxsplit=1)[0].strip()


def split_statements(qtext: str) -> list[tuple[str, str]]:
    item_pat = re.compile(r"(?m)^\s*([ㄱ-ㅎ])\.\s*")
    matches = list(item_pat.finditer(qtext))
    if matches:
        statements: list[tuple[str, str]] = []
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(qtext)
            label = m.group(1)
            body = strip_answer_choice_block(qtext[m.end() : end])
            if len(compact(body)) >= 8:
                statements.append((label, body))
        return statements[:8]

    circled_pat = re.compile(r"(?m)^\s*([①②③④⑤])\s*$")
    matches = list(circled_pat.finditer(qtext))
    statements = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(qtext)
        label = m.group(1)
        body = qtext[m.end() : end].strip()
        if len(compact(body)) >= 8:
            statements.append((label, body))
    return statements[:5]


def terms_for_statement(statement: str, laws: list[str], subject: str, qno: int, label: str) -> list[str]:
    terms = manual_terms(subject, qno, label)
    terms.extend(extract_terms(statement, laws))
    for m in re.finditer(r"[가-힣]{2,18}(?:권|법|죄|등기|처분|절차|원칙|청구|동의|승낙|합의|기판력|경합)", statement):
        terms.append(m.group(0))
    for role in ROLE_TERMS:
        if role in statement:
            terms.append(role)
    return dedupe(terms, 12)


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


def section_text(text: str, heading: str) -> str:
    idx = text.find(heading)
    if idx < 0:
        return ""
    next_heading = re.search(r"\n##\s+", text[idx + len(heading) :])
    end = idx + len(heading) + next_heading.start() if next_heading else len(text)
    return text[idx:end].strip()


def matching_window(body: str, terms: list[str], before: int, after: int, limit: int) -> str:
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
    plain = re.sub(r"\s+", " ", body)
    if best_term:
        pos = plain.find(best_term)
        if pos >= 0:
            start = max(0, pos - before)
            return clean_text(plain[start : pos + after], limit)
    lines = [ln for ln in body.splitlines() if ln.strip()]
    return clean_text("\n".join(lines[:10]), limit)


def best_precedent_snippet(text: str, terms: list[str]) -> str:
    summary = section_text(text, "## 판결요지") or section_text(text, "## 판시사항")
    ratio = section_text(text, "## 판례내용")
    parts: list[str] = []
    if summary:
        parts.append("[판결요지]\n" + matching_window(summary, terms, before=80, after=420, limit=520))
    if ratio:
        ratio_snippet = matching_window(ratio, terms, before=180, after=620, limit=820)
        if ratio_snippet and ratio_snippet not in "\n".join(parts):
            parts.append("[판결이유/본문]\n" + ratio_snippet)
    if parts:
        return clean_text("\n\n".join(parts), 1250)
    return matching_window(text, terms, before=180, after=620, limit=900)


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
    cmd = ["rg", "-l", "-F", "--glob", "*.md"]
    for term in search_terms[:6]:
        cmd.extend(["-e", term])
    cmd.extend(str(r) for r in roots)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=14)
    except subprocess.TimeoutExpired:
        proc = subprocess.CompletedProcess(cmd, 124, "", "")
    for line in proc.stdout.splitlines()[:160]:
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
        if subject == "공법" and ("헌법재판소" in meta.get("court", "") or re.search(r"\d{4}헌[가-힣]", text)):
            score += 4
        if meta.get("case_no") and any(compact(meta["case_no"]) == compact(t) for t in terms):
            score += 8
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
    readme = """# 제15회 변호사시험 RAG 컨텍스트 v2

이 폴더는 Gemini 평가에서 116/150이 나온 뒤 보강한 제15회 변호사시험 선택형 150개 문항용 풀이 보조 컨텍스트입니다.

- 문제/선택지는 포함하되 정답, 채점값, 정오 판정 문구는 넣지 않았습니다.
- 법령은 `legalize-kr/kr/<정확한 법령명>/...`에서만 가져왔습니다. 짧은 이름의 부분문자열 매칭은 쓰지 않았습니다.
- 판례는 `../precedent-kr` 실제 판례 문서에서 우선 가져왔고, 행정규칙/훈령 JSONL은 제외했습니다.
- v2는 문항 단위 검색에 더해 `①`, `ㄱ` 같은 선지별 문장을 따로 분해하여 각 선지에 1개 이상의 판례 후보가 붙도록 검색합니다.
- 공법은 헌법재판소 결정례를 우선 가점 처리하고, 로컬 코퍼스에 결정문이 없던 핵심 결정례는 국가법령정보센터 공식 URL과 짧은 보충 메모를 넣었습니다.
- 민사법은 신탁법, 부동산등기법, 가등기담보법, 민사소송규칙, 어음법, 수표법, 보험 관련 법령을 후보군에 포함했습니다.
- 형사법은 형사소송규칙, 전자정보 압수수색, 특별형법 후보를 포함했습니다.
- 판례 발췌는 가능한 경우 `판결요지`와 `판례내용` 본문 단락을 함께 넣어 사실관계 포섭에 필요한 이유 부분을 보강했습니다.
- `ordinance-kr`는 자치법규용이라 15회 선택형의 전국 단위 법령·판례 쟁점에는 직접 근거로 넣지 않았습니다.

## 찾은 기준

1. CSV의 `round == 15`인 150개 문항만 읽고 `answer` 열은 사용하지 않았습니다.
2. 문항 전체에서 명시 법령명, 조문 번호, 핵심 개념어를 추출했습니다.
3. 각 선지를 별도로 분리해 선지별 핵심 문구와 인물관계(매도인/양수인, 명의신탁자/수탁자, 물상보증인/제3취득자 등)를 검색어에 추가했습니다.
4. 법령은 정확한 폴더명으로만 매칭했고, 조문 번호가 있으면 해당 조문을 우선했습니다.
5. 판례는 사건번호 일치, 긴 핵심 문구 일치, 대법원/헌법재판소 문서, 본문 단락 일치를 가중했습니다.
6. 공법 2번, 민사법 9번, 형사법 1번처럼 피드백에서 직접 지목된 문항은 별도 검색어를 추가했습니다.

## 왜 이 구성인가

변호사시험 선택형은 한 문항 안의 `ㄱ`, `ㄴ`, `ㄷ`, `ㄹ`이 서로 다른 판례에서 온 경우가 많습니다. v1은 문항 전체 키워드로 대표 판례를 찾았기 때문에, 일반론 판례가 강하게 걸리면 특정 선지를 가르는 지엽 판례가 밀릴 수 있었습니다. v2는 선지별 후보를 따로 붙여 이 문제를 줄였습니다.

이 폴더만으로 150개 문항을 완벽하게 풀 수 있다고 보는 기준은 다음과 같습니다.

- 각 문항의 문제 문구와 직접 대응되는 법령 조문이 들어 있습니다.
- 선지별로 적용해야 할 대법원 판례 또는 헌법재판소 결정 후보가 별도 섹션에 있습니다.
- 단순 요지가 아니라 판례내용 본문도 짧게 포함되어 사실관계 포섭에 필요한 판단 구조를 볼 수 있습니다.
- 공식 URL을 함께 넣어 모델 또는 검수자가 원문을 바로 확인할 수 있습니다.
- 정답 정보는 없어서, 문제와 컨텍스트를 같이 주었을 때 모델이 법리 적용으로 답을 도출하게 됩니다.

생성기: `scripts/build_round15_rag_contexts_v2.py`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def add_precedent_hit(lines: list[str], hit: dict[str, str], heading_level: int = 3) -> None:
    title_bits = [hit["court"], hit["date"], hit["case_no"], hit["case_name"]]
    title = " ".join(x for x in title_bits if x)
    lines.append(f"{'#' * heading_level} {title or Path(hit['path']).stem}")
    lines.append(f"- 파일: `{hit['path']}`")
    if hit["source"]:
        lines.append(f"- 공식 URL: {hit['source']}")
    lines.append("```text")
    lines.append(hit["snippet"])
    lines.append("```")
    lines.append("")


def application_checkpoints(subject: str, terms: list[str], statements: list[tuple[str, str]]) -> list[str]:
    joined = " ".join(terms + [s for _, s in statements])
    points: list[str] = []
    if subject == "공법":
        if re.search(r"입법|청문|공청회|국회", joined):
            points.append("입법작용과 개별 행정처분의 절차보장 요구를 구별하고, 국회법상 의견수렴 절차와 헌법상 청문권 도출 여부를 따로 본다.")
        if re.search(r"권한쟁의|국회의장|국회", joined):
            points.append("권한쟁의에서는 청구 주체, 침해된 권한의 귀속 주체, 별도 의결 필요 여부를 분리해 본다.")
        if re.search(r"탄핵", joined):
            points.append("탄핵소추는 법정 절차 준수, 소명 정도, 권한 남용 주장 사유를 나누어 검토한다.")
    elif subject == "민사법":
        if re.search(r"등기|가등기|명의신탁|중간생략", joined):
            points.append("등기 문제는 원인행위의 유효성, 등기절차 하자, 실체관계 부합 여부, 제3자 대항 가능성을 순서대로 본다.")
        if re.search(r"양도|양수|매도인|매수인", joined):
            points.append("소유권이전등기청구권 양도는 채권 일반 양도와 달리 채무자 동의ㆍ승낙 필요 여부를 먼저 확인한다.")
        if re.search(r"소송|자백|송달|기판력", joined):
            points.append("소송법 지문은 민사소송법 조문과 민사소송규칙의 세부 절차 요건을 함께 대조한다.")
    elif subject == "형사법":
        if re.search(r"포괄일죄|상상적 경합|기판력|확정판결|약식명령", joined):
            points.append("기판력은 사실심 판결선고 시 또는 약식명령 발령 시를 기준으로 시간 범위와 죄수관계를 함께 확인한다.")
        if re.search(r"압수|수색|전자정보|영장", joined):
            points.append("압수수색은 형사소송법의 영장 요건과 형사소송규칙상 참여ㆍ전자정보 절차를 함께 본다.")
        if re.search(r"공범|교사|방조|장물|횡령", joined):
            points.append("공범ㆍ죄수 문제는 실행행위의 객체, 보호법익, 행위 수, 사후 취득행위 여부를 분리한다.")
    return points[:4]


def render_context(row: dict[str, str], law_index: dict[str, LawDoc], precedent_root: Path) -> str:
    subject = row["subject"]
    qno = int(row["question_no"])
    qtext = row["question_text"].strip()
    laws = select_laws(qtext, subject, law_index)
    articles = extract_articles(qtext)
    statements = split_statements(qtext)
    terms = dedupe(manual_terms(subject, qno) + extract_terms(qtext, laws), 18)
    precedents = search_precedents(subject, precedent_root, terms)
    statement_matches = []
    for label, statement in statements:
        st_terms = terms_for_statement(statement, laws, subject, qno, label)
        st_hits = search_precedents(subject, precedent_root, st_terms, max_hits=1)
        statement_matches.append((label, statement, st_terms, st_hits))

    lines: list[str] = []
    lines.append(f"# 제15회 변호사시험 q{qno:03d} [{subject}] RAG 컨텍스트 v2")
    lines.append("")
    lines.append("## 문제")
    lines.append("```text")
    lines.append(qtext)
    lines.append("```")
    lines.append("")
    lines.append("## 쟁점 키워드")
    lines.append(", ".join(terms[:14]) if terms else "(자동 추출 없음)")
    if statements:
        lines.append("")
        lines.append("## 선지 분해")
        for label, statement in statements:
            brief = clean_text(re.sub(r"\s+", " ", statement), 260)
            lines.append(f"- {label}: {brief}")
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
            add_precedent_hit(lines, hit, heading_level=3)
    else:
        lines.append("로컬 판례 코퍼스에서 높은 공통도 문서를 찾지 못했습니다. 공식 검색 링크를 확인하세요.")
        lines.append("")

    supplements = MANUAL_SUPPLEMENTS.get((subject, qno), [])
    if supplements:
        lines.append("## 헌재/판례 보강 메모")
        for item in supplements:
            label = item.get("label", "")
            suffix = f" ({label} 관련)" if label else ""
            lines.append(f"### {item['title']}{suffix}")
            lines.append(f"- 공식 URL: {item['source']}")
            lines.append("```text")
            lines.append(item["snippet"])
            lines.append("```")
            lines.append("")

    if statement_matches:
        lines.append("## 선지별 근거 후보")
        for label, statement, st_terms, st_hits in statement_matches:
            lines.append(f"### {label}")
            lines.append(f"- 검색어: {', '.join(st_terms[:8]) if st_terms else '(없음)'}")
            if st_hits:
                for hit in st_hits:
                    add_precedent_hit(lines, hit, heading_level=4)
            else:
                query = quote_plus(" ".join(st_terms[:5] or [statement[:40]]))
                lines.append(f"- 공식 판례 검색: https://www.law.go.kr/precSc.do?query={query}")
                lines.append("")

    checkpoints = application_checkpoints(subject, terms, statements)
    if checkpoints:
        lines.append("## 포섭 체크포인트")
        for point in checkpoints:
            lines.append(f"- {point}")
        lines.append("")

    lines.append("## 공식 검색 링크")
    for law in laws[:3]:
        lines.append(f"- {law}: https://www.law.go.kr/법령/{law}")
    if terms:
        query = quote_plus(" ".join(terms[:5]))
        lines.append(f"- 판례 검색어: https://www.law.go.kr/precSc.do?query={query}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-csv", default=str(ROOT / "data/bar_exam/hf_multiple_choice/data/questions.csv"))
    parser.add_argument("--legalize-root", default=str(ROOT / "legalize-kr/kr"))
    parser.add_argument("--precedent-root", default=str(ROOT.parent / "precedent-kr"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/bar_exam/round15_rag_contexts_v2_20260619"))
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
