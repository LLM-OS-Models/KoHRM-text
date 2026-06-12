#!/usr/bin/env python3
"""Build a Korean bar exam dataset from Ministry of Justice source files.

The script downloads official MOJ attachments, extracts HWP/HWPX text, parses
multiple-choice answers, splits question documents into question-level rows, and
optionally uploads the processed dataset folder to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "bar_exam"
RAW_DIR = OUT_ROOT / "raw" / "downloads"
EXTRACTED_DIR = OUT_ROOT / "extracted"
TEXT_DIR = OUT_ROOT / "text"
PROCESSED_DIR = OUT_ROOT / "processed"
HF_DIR = OUT_ROOT / "hf_dataset"

BASE_URL = "https://www.moj.go.kr"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "Chrome/125.0 Safari/537.36"
)


QUESTION_ARTICLES = [
    (150, 568844),
    (150, 579016),
    (150, 591294),
    (150, 591295),
    (150, 591296),
    (150, 602397),
    (150, 602398),
    (150, 602399),
]

ANSWER_ARTICLES = [
    (151, 579015),
    (151, 591292),
    (151, 602396),
]

FINAL_NOTICE_BY_ROUND = {
    13: "https://www.moj.go.kr/bbs/moj/151/579933/artclView.do",
    14: "https://www.moj.go.kr/bbs/moj/151/592142/artclView.do",
    15: "https://www.moj.go.kr/bbs/moj/151/603464/artclView.do",
}

SOURCE_LIST_URL = (
    "https://www.moj.go.kr/moj/405/subview.do?enc="
    "Zm5jdDF8QEB8JTJGYmJzJTJGbW9qJTJGMTUwJTJGYXJ0Y2xMaXN0LmRvJTNG"
    "dGFibGVfY2F0ZV9zZWxlY3QlM0QxNDElMjZiYnNDbFNlcSUzRDE0MSUyNmlz"
    "Vmlld01pbmUlM0RmYWxzZSUyNmJic09wZW5XcmRTZXElM0QlMjZzcmNoQ29s"
    "dW1uJTNEc2olMjZzcmNoV3JkJTNEJTI2"
)

ANSWER_LIST_URL = (
    "https://www.moj.go.kr/moj/2126/subview.do?enc="
    "Zm5jdDF8QEB8JTJGYmJzJTJGbW9qJTJGMTUxJTJGYXJ0Y2xMaXN0LmRvJTNG"
    "YmJzQ2xTZXElM0QlMjZpc1ZpZXdNaW5lJTNEZmFsc2UlMjZiYnNPcGVuV3Jk"
    "U2VxJTNEJTI2c3JjaENvbHVtbiUzRHNqJTI2c3JjaFdyZCUzRCVFQiVCMyU4"
    "MCVFRCU5OCVCOCVFQyU4MiVBQyVFQyU4QiU5QyVFRCU5NyU5OCslRUMlQTAl"
    "OTUlRUIlOEIlQjUlRUElQjAlODglRUMlOTUlODglMjY%3D"
)

KOG_LICENCE_URL = "https://www.kogl.or.kr/info/licenseType1.do"
SOURCE_LICENSE = "Korea Open Government License Type 1 (KOGL Type 1)"
SOURCE_ATTRIBUTION = "대한민국 법무부 법조인력과, 변호사시험 기출문제 및 선택형 정답"

SUBJECTS = [
    "공법",
    "민사법",
    "형사법",
    "선택과목",
    "국제거래법",
    "국제법",
    "노동법",
    "조세법",
    "지적재산권법",
    "경제법",
    "환경법",
]

QUESTION_TYPES = ["선택형", "사례형", "기록형", "선택과목"]
CHOICE_MARKS = "①②③④⑤⑥⑦⑧⑨➀➁➂➃➄➅➆➇➈"
EXPECTED_MULTIPLE_COUNTS = {"공법": 40, "민사법": 70, "형사법": 40}


@dataclass
class Attachment:
    source_kind: str
    article_bbs: int
    article_id: int
    article_url: str
    article_title: str
    article_date: str
    file_url: str
    file_id: str
    filename: str
    local_path: str


@dataclass
class Document:
    source_kind: str
    article_bbs: int
    article_id: int
    article_url: str
    article_title: str
    article_date: str
    file_url: str
    archive_name: str
    source_filename: str
    local_path: str
    text_path: str
    html_path: str
    sha256: str
    year: int | None
    round: int | None
    subject: str
    question_type: str
    book_type: str
    extension: str
    parse_status: str
    parse_error: str


def ensure_dirs() -> None:
    for path in [RAW_DIR, EXTRACTED_DIR, TEXT_DIR, PROCESSED_DIR, HF_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def article_url(bbs: int, article_id: int) -> str:
    return f"{BASE_URL}/bbs/moj/{bbs}/{article_id}/artclView.do"


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def clean_filename(name: str) -> str:
    name = html.unescape(name).strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip(" .")


def stable_name(prefix: str, filename: str) -> str:
    return f"{prefix}_{clean_filename(filename)}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_article(s: requests.Session, bbs: int, article_id: int) -> tuple[str, BeautifulSoup]:
    url = article_url(bbs, article_id)
    r = s.get(url, timeout=60)
    r.raise_for_status()
    return url, BeautifulSoup(r.text, "html.parser")


def parse_article_attachments(
    s: requests.Session, source_kind: str, bbs: int, article_id: int
) -> list[Attachment]:
    url, soup = fetch_article(s, bbs, article_id)
    title_tag = soup.select_one(".artclViewTitle")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    date = ""
    for dl in soup.select(".infor dl"):
        dt = dl.select_one("dt")
        dd = dl.select_one("dd")
        if dt and dd and "작성일" in dt.get_text(" ", strip=True):
            date = dd.get_text(" ", strip=True)
            break

    attachments: list[Attachment] = []
    for a in soup.select('a[href$="/download.do"]'):
        href = a.get("href", "")
        file_url = urljoin(BASE_URL, href)
        file_id_match = re.search(r"/(\d+)/download\.do", href)
        file_id = file_id_match.group(1) if file_id_match else ""
        filename = clean_filename(a.get_text(" ", strip=True))
        effective_kind = "answer" if "정답" in filename else source_kind
        prefix = f"{effective_kind}_bbs{bbs}_art{article_id}_file{file_id}"
        local_path = RAW_DIR / stable_name(prefix, filename)
        attachments.append(
            Attachment(
                source_kind=effective_kind,
                article_bbs=bbs,
                article_id=article_id,
                article_url=url,
                article_title=title,
                article_date=date,
                file_url=file_url,
                file_id=file_id,
                filename=filename,
                local_path=str(local_path),
            )
        )
    return attachments


def include_attachment(att: Attachment, only_multiple_choice: bool) -> bool:
    if not only_multiple_choice:
        return True
    if att.source_kind == "answer":
        return True
    name = att.filename
    if "선택형" in name:
        return True
    # 제13회 is distributed as one all-in-one ZIP; filter contained files later.
    if att.article_id == 579016 and name.endswith(".zip"):
        return True
    return False


def collect_attachments(only_multiple_choice: bool = False) -> list[Attachment]:
    s = session()
    attachments: list[Attachment] = []
    for bbs, article_id in QUESTION_ARTICLES:
        attachments.extend(
            att
            for att in parse_article_attachments(s, "question", bbs, article_id)
            if include_attachment(att, only_multiple_choice)
        )
    for bbs, article_id in ANSWER_ARTICLES:
        attachments.extend(
            att
            for att in parse_article_attachments(s, "answer", bbs, article_id)
            if include_attachment(att, only_multiple_choice)
        )
    return attachments


def download_attachments(attachments: list[Attachment], force: bool = False) -> None:
    s = session()
    for att in attachments:
        out = Path(att.local_path)
        if out.exists() and out.stat().st_size > 1024 and not force:
            continue
        s.get(att.article_url, timeout=60)
        r = s.get(att.file_url, headers={"Referer": att.article_url}, timeout=180)
        r.raise_for_status()
        if len(r.content) < 1024:
            raise RuntimeError(f"Suspiciously small download: {att.file_url}")
        out.write_bytes(r.content)
        time.sleep(0.2)


def decode_zip_name(raw_name: str) -> str:
    try:
        decoded = raw_name.encode("cp437").decode("cp949")
        if any("\uac00" <= ch <= "\ud7a3" for ch in decoded):
            return decoded
    except Exception:
        pass
    return raw_name


def extract_zip(path: Path, target: Path, force: bool = False) -> list[Path]:
    if force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            decoded = decode_zip_name(info.filename)
            parts = [clean_filename(part) for part in decoded.split("/") if part.strip()]
            if not parts:
                continue
            out = target.joinpath(*parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and out.stat().st_size == info.file_size and not force:
                extracted.append(out)
                continue
            with zf.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out)
    return extracted


def infer_round(text: str) -> int | None:
    patterns = [
        r"제\s*(\d{1,2})\s*회",
        r"(?<!\d)(\d{1,2})\s*회",
        r"20\d{2}\s*년도\s*제\s*(\d{1,2})\s*회",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            value = int(m.group(1))
            if 1 <= value <= 30:
                return value
    return None


def infer_year(text: str, round_no: int | None) -> int | None:
    m = re.search(r"(20\d{2})\s*년도", text)
    if m:
        return int(m.group(1))
    if round_no:
        return 2011 + round_no
    return None


def infer_subject(text: str) -> str:
    hits = [subject for subject in SUBJECTS if subject in text]
    if "선택과목" in hits:
        return "선택과목"
    for subject in ["공법", "민사법", "형사법"]:
        if subject in hits:
            return subject
    return hits[0] if hits else ""


def infer_question_type(text: str, source_kind: str) -> str:
    if source_kind == "answer":
        return "선택형"
    for kind in QUESTION_TYPES:
        if kind in text:
            return kind
    return ""


def infer_book_type(text: str) -> str:
    m = re.search(r"([13])\s*책형", text)
    return f"{m.group(1)}책형" if m else ""


def extract_hwpx_text(path: Path) -> str:
    lines: list[str] = []
    with zipfile.ZipFile(path) as zf:
        section_names = sorted(
            name
            for name in zf.namelist()
            if name.lower().startswith("contents/section") and name.lower().endswith(".xml")
        )
        for name in section_names:
            root = ET.fromstring(zf.read(name))
            for elem in root.iter():
                local = elem.tag.rsplit("}", 1)[-1]
                if local == "p":
                    paragraph = "".join(
                        "".join(t.itertext())
                        for t in elem.iter()
                        if t.tag.rsplit("}", 1)[-1] == "t"
                    )
                    if paragraph.strip():
                        lines.append(paragraph.strip())
    return "\n".join(lines)


def run_hwp5html(path: Path, out_dir: Path) -> tuple[str, str]:
    if out_dir.exists():
        index = out_dir / "index.xhtml"
        if index.exists():
            return index.read_text(encoding="utf-8", errors="ignore"), ""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["hwp5html", "--output", str(out_dir), str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return "", proc.stderr.strip()
    index = out_dir / "index.xhtml"
    if not index.exists():
        return "", "hwp5html did not create index.xhtml"
    return index.read_text(encoding="utf-8", errors="ignore"), proc.stderr.strip()


def run_hwp5txt(path: Path) -> tuple[str, str]:
    cmd = ["hwp5txt", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return "", proc.stderr.strip()
    return proc.stdout, proc.stderr.strip()


def text_from_html(xhtml: str) -> str:
    soup = BeautifulSoup(xhtml, "html.parser")
    for tag in soup(["style", "script"]):
        tag.decompose()
    text = soup.get_text("\n")
    return normalize_text(text)


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_file_id(path: Path) -> str:
    rel = str(path).replace(str(OUT_ROOT), "")
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
    return digest


def convert_document(path: Path) -> tuple[str, str, str, str]:
    ext = path.suffix.lower()
    doc_id = text_file_id(path)
    text_path = TEXT_DIR / f"{doc_id}.txt"
    html_dir = TEXT_DIR / f"{doc_id}_html"
    html_path = html_dir / "index.xhtml"
    if text_path.exists():
        return (
            text_path.read_text(encoding="utf-8", errors="ignore"),
            str(text_path),
            str(html_path if html_path.exists() else ""),
            "ok",
        )
    if ext == ".hwpx":
        text = extract_hwpx_text(path)
        status = "ok" if text else "empty_hwpx"
        text_path.write_text(text, encoding="utf-8")
        return text, str(text_path), "", status
    if ext == ".hwp":
        xhtml, html_err = run_hwp5html(path, html_dir)
        if xhtml:
            text = text_from_html(xhtml)
            text_path.write_text(text, encoding="utf-8")
            return text, str(text_path), str(html_path), "ok"
        text, txt_err = run_hwp5txt(path)
        text = normalize_text(text)
        text_path.write_text(text, encoding="utf-8")
        status = "fallback_hwp5txt" if text else "failed"
        err = html_err or txt_err
        return text, str(text_path), "", f"{status}: {err}" if err else status
    return "", "", "", f"unsupported_extension:{ext}"


def build_documents(
    attachments: list[Attachment],
    force_extract: bool = False,
    only_multiple_choice: bool = False,
) -> list[Document]:
    documents: list[Document] = []
    att_by_path = {Path(att.local_path): att for att in attachments}

    for att in attachments:
        local = Path(att.local_path)
        suffix = local.suffix.lower()
        files: list[tuple[Path, str]] = []
        if suffix == ".zip":
            archive_target = EXTRACTED_DIR / local.stem
            extracted = extract_zip(local, archive_target, force=force_extract)
            files.extend((p, att.filename) for p in extracted)
        else:
            files.append((local, ""))

        for file_path, archive_name in files:
            if file_path.suffix.lower() not in {".hwp", ".hwpx"}:
                continue
            meta_text = " ".join(
                [
                    att.article_title,
                    att.filename,
                    archive_name,
                    str(file_path.relative_to(OUT_ROOT) if OUT_ROOT in file_path.parents else file_path),
                ]
            )
            if only_multiple_choice and att.source_kind == "question" and "선택형" not in meta_text:
                continue
            round_no = infer_round(meta_text)
            year = infer_year(meta_text, round_no)
            subject = infer_subject(meta_text)
            qtype = infer_question_type(meta_text, att.source_kind)
            book_type = infer_book_type(meta_text)
            parse_error = ""
            try:
                text, text_path, html_path, status = convert_document(file_path)
            except Exception as exc:  # keep the run going and report in QA
                text, text_path, html_path, status = "", "", "", "failed"
                parse_error = repr(exc)
            documents.append(
                Document(
                    source_kind=att.source_kind,
                    article_bbs=att.article_bbs,
                    article_id=att.article_id,
                    article_url=att.article_url,
                    article_title=att.article_title,
                    article_date=att.article_date,
                    file_url=att.file_url,
                    archive_name=archive_name,
                    source_filename=file_path.name,
                    local_path=str(file_path),
                    text_path=text_path,
                    html_path=html_path,
                    sha256=sha256_file(file_path),
                    year=year,
                    round=round_no,
                    subject=subject,
                    question_type=qtype,
                    book_type=book_type,
                    extension=file_path.suffix.lower(),
                    parse_status=status,
                    parse_error=parse_error,
                )
            )

    return documents


def useful_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "문" and i + 1 < len(lines) and re.fullmatch(r"\d{1,3}\s*[.．]?", lines[i + 1]):
            line = f"문 {lines[i + 1]}"
            i += 2
        else:
            i += 1
        if not line:
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        if line in {"쪽", "끝"}:
            continue
        if re.fullmatch(r"[헌민형공사기록선택법\s]{2,12}", line):
            continue
        cleaned.append(line)
    return cleaned


def split_question_blocks(text: str, qtype: str) -> list[tuple[str, list[str]]]:
    lines = useful_lines(text)
    if qtype == "선택형":
        start_re = re.compile(r"^문\s*(\d{1,3})\s*[.．]*\s*(.*)$")
    else:
        start_re = re.compile(
            r"^(?:문제\s*)?(\d{1,2})\s*[.．]?$|^문\s*(\d{1,2})\s*[.．]?$|^제\s*(\d{1,2})\s*문"
        )

    starts: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        m = start_re.match(line)
        if m:
            qno = next(group for group in m.groups() if group)
            if qtype == "선택형" and int(qno) > 70 and len(qno) == 3:
                qno = qno[:2]
            rest = ""
            if qtype == "선택형":
                rest = (m.group(2) or "").strip()
            starts.append((i, str(int(qno)), rest))

    # Some essay/record files are not numbered in a way worth splitting. Keep
    # one document-level row instead of dropping content.
    if not starts:
        return [("1", lines)] if lines else []

    blocks: list[tuple[str, list[str]]] = []
    for idx, (start, qno, rest) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        block = ([rest] if rest else []) + lines[start + 1 : end]
        if block:
            blocks.append((qno, block))
    if qtype == "선택형":
        seen: set[str] = set()
        deduped: list[tuple[str, list[str]]] = []
        for qno, block in blocks:
            if qno in seen:
                continue
            seen.add(qno)
            deduped.append((qno, block))
        blocks = deduped
    return blocks


def merge_lines(lines: Iterable[str]) -> str:
    text = "\n".join(line.strip() for line in lines if line.strip())
    text = re.sub(r"\n([·,.;:)\]〉」』])", r"\1", text)
    text = re.sub(r"([｢「『(〈\[])\n", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_choices(block_lines: list[str]) -> tuple[str, list[str], str]:
    stem_lines: list[str] = []
    choices: list[str] = []
    current_choice: list[str] = []
    current_mark = ""

    def flush_choice() -> None:
        nonlocal current_choice, current_mark
        if current_mark:
            body = merge_lines(current_choice)
            choices.append(f"{current_mark} {body}".strip())
        current_choice = []
        current_mark = ""

    for line in block_lines:
        stripped = line.strip()
        if not stripped:
            continue
        segments = [seg for seg in re.split(f"(?=[{CHOICE_MARKS}])", stripped) if seg]
        if len(segments) > 1 and not stripped[0] in CHOICE_MARKS:
            segments = [stripped]
        for segment in segments:
            stripped = segment.strip()
            if not stripped:
                continue
            if stripped and stripped[0] in CHOICE_MARKS:
                flush_choice()
                current_mark = stripped[0]
                rest = stripped[1:].strip()
                if rest:
                    current_choice.append(rest)
                continue
            if re.fullmatch(f"[{CHOICE_MARKS}]", stripped):
                flush_choice()
                current_mark = stripped
                continue
            if current_mark:
                current_choice.append(stripped)
            else:
                stem_lines.append(stripped)
            continue
        continue
        if stripped and stripped[0] in CHOICE_MARKS:
            flush_choice()
            current_mark = stripped[0]
            rest = stripped[1:].strip()
            if rest:
                current_choice.append(rest)
            continue
        if re.fullmatch(f"[{CHOICE_MARKS}]", stripped):
            flush_choice()
            current_mark = stripped
            continue
        if current_mark:
            current_choice.append(stripped)
        else:
            stem_lines.append(stripped)
    flush_choice()

    stem = merge_lines(stem_lines)
    full_text = merge_lines(block_lines)
    return stem, choices, full_text


def parse_answer_tables(doc: Document) -> list[dict]:
    if not doc.html_path:
        return []
    html_path = Path(doc.html_path)
    if not html_path.exists():
        return []
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    rows: list[dict] = []
    for table in soup.find_all("table"):
        cells = [cell.get_text(" ", strip=True) for cell in table.find_all(["td", "th"])]
        joined = " ".join(cells)
        if "문번" not in joined or "정답" not in joined:
            continue
        context_parts: list[str] = []
        for prev in table.find_all_previous(["p", "span"], limit=40):
            txt = prev.get_text(" ", strip=True)
            if txt:
                context_parts.append(txt)
        context = " ".join(reversed(context_parts))
        subject = infer_subject(context) or doc.subject
        values = [re.sub(r"\s+", "", c) for c in cells]
        for i in range(len(values) - 1):
            qno = values[i]
            answer = values[i + 1]
            if not re.fullmatch(r"\d{1,3}", qno):
                continue
            if answer in {"문번", "정답"}:
                continue
            if not re.search(r"[1-9①-⑨복수없음정답]", answer):
                continue
            rows.append(
                {
                    "exam": "변호사시험",
                    "round": doc.round,
                    "year": doc.year,
                    "subject": subject,
                    "question_type": "선택형",
                    "book_type": doc.book_type,
                    "question_no": int(qno),
                    "answer": normalize_answer(answer),
                    "answer_source_status": "final" if (doc.round or 0) <= 12 else "final_notice_same_as_provisional",
                    "source_article_url": FINAL_NOTICE_BY_ROUND.get(doc.round or -1, doc.article_url),
                    "provisional_answer_article_url": doc.article_url,
                    "source_file": doc.source_filename,
                    "source_file_url": doc.file_url,
                    "source_license": SOURCE_LICENSE,
                }
            )
    dedup: dict[tuple, dict] = {}
    for row in rows:
        key = (row["round"], row["subject"], row["question_no"])
        dedup[key] = row
    return list(dedup.values())


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace("①", "1").replace("②", "2").replace("③", "3")
    answer = answer.replace("④", "4").replace("⑤", "5")
    answer = re.sub(r"\s+", "", answer)
    return answer


def parse_answer_rows(documents: list[Document]) -> list[dict]:
    answer_rows: list[dict] = []
    for doc in documents:
        if doc.source_kind != "answer":
            continue
        answer_rows.extend(parse_answer_tables(doc))
    answer_rows.sort(key=lambda r: (r["round"] or 0, r["subject"], r["question_no"]))
    return answer_rows


def answer_lookup(answer_rows: list[dict]) -> dict[tuple[int, str, int], dict]:
    lookup: dict[tuple[int, str, int], dict] = {}
    for row in answer_rows:
        if row["round"] is None:
            continue
        lookup[(int(row["round"]), row["subject"], int(row["question_no"]))] = row
    return lookup


def parse_question_rows(documents: list[Document], answers: list[dict]) -> list[dict]:
    lookup = answer_lookup(answers)
    rows: list[dict] = []
    for doc in documents:
        if doc.source_kind != "question":
            continue
        if not doc.text_path or not Path(doc.text_path).exists():
            continue
        text = Path(doc.text_path).read_text(encoding="utf-8", errors="ignore")
        qtype = doc.question_type or infer_question_type(text[:1000], "question") or "문서"
        blocks = split_question_blocks(text, qtype)
        for qno, block in blocks:
            qno_int = int(qno) if qno.isdigit() else None
            stem, choices, full_text = parse_choices(block) if qtype == "선택형" else ("", [], merge_lines(block))
            if qtype == "선택형" and not choices:
                # Keep the row, but expose lower parser quality for manual review.
                stem = stem or full_text
            ans = lookup.get((doc.round or -1, doc.subject, qno_int or -1), {})
            row_id = make_row_id(doc, qno)
            rows.append(
                {
                    "id": row_id,
                    "exam": "변호사시험",
                    "round": doc.round,
                    "year": doc.year,
                    "subject": doc.subject,
                    "question_type": qtype,
                    "book_type": doc.book_type,
                    "question_no": qno_int,
                    "stem": stem,
                    "choices_json": json.dumps(choices, ensure_ascii=False),
                    "question_text": full_text,
                    "answer": ans.get("answer", ""),
                    "answer_source_status": ans.get("answer_source_status", ""),
                    "source_article_url": doc.article_url,
                    "source_file": doc.source_filename,
                    "source_file_url": doc.file_url,
                    "source_sha256": doc.sha256,
                    "source_license": SOURCE_LICENSE,
                    "attribution": SOURCE_ATTRIBUTION,
                    "parse_quality": quality_for_row(qtype, full_text, choices, ans),
                }
            )
    rows.sort(
        key=lambda r: (
            r["round"] or 0,
            r["question_type"],
            r["subject"],
            r["book_type"],
            r["question_no"] or 0,
            r["source_file"],
        )
    )
    return rows


def make_row_id(doc: Document, qno: str) -> str:
    parts = [
        "moj-bar",
        f"r{doc.round:02d}" if doc.round else "rxx",
        doc.subject or "unknown",
        doc.question_type or "document",
        doc.book_type or "default",
        f"q{qno}",
    ]
    base = "_".join(re.sub(r"[^0-9A-Za-z가-힣]+", "-", p).strip("-") for p in parts)
    return base


def quality_for_row(qtype: str, full_text: str, choices: list[str], ans: dict) -> str:
    if not full_text:
        return "empty"
    if qtype == "선택형":
        if len(choices) >= 4 and ans.get("answer"):
            return "parsed_with_answer"
        if len(choices) >= 4:
            return "parsed_without_answer"
        return "needs_review_choices"
    return "parsed_no_official_answer"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(attachments: list[Attachment], documents: list[Document], answers: list[dict], questions: list[dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(PROCESSED_DIR / "attachments.csv", [asdict(a) for a in attachments])
    write_csv(PROCESSED_DIR / "documents.csv", [asdict(d) for d in documents])
    write_csv(PROCESSED_DIR / "answers.csv", answers)
    write_csv(PROCESSED_DIR / "questions.csv", questions)
    write_jsonl(PROCESSED_DIR / "questions.jsonl", questions)
    write_jsonl(PROCESSED_DIR / "answers.jsonl", answers)
    qa = build_qa_report(documents, answers, questions)
    (PROCESSED_DIR / "qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


def build_qa_report(documents: list[Document], answers: list[dict], questions: list[dict]) -> dict:
    by_doc_status: dict[str, int] = {}
    for doc in documents:
        by_doc_status[doc.parse_status] = by_doc_status.get(doc.parse_status, 0) + 1

    mc_questions: dict[tuple[int, str], int] = {}
    mc_answers: dict[tuple[int, str], int] = {}
    for row in questions:
        if row["question_type"] == "선택형" and row["round"] and row["subject"]:
            key = (int(row["round"]), row["subject"])
            mc_questions[key] = mc_questions.get(key, 0) + 1
    for row in answers:
        if row["round"] and row["subject"]:
            key = (int(row["round"]), row["subject"])
            mc_answers[key] = mc_answers.get(key, 0) + 1

    count_issues = []
    for round_no in range(1, 16):
        for subject, expected in EXPECTED_MULTIPLE_COUNTS.items():
            qc = mc_questions.get((round_no, subject), 0)
            ac = mc_answers.get((round_no, subject), 0)
            if qc not in {0, expected} or ac not in {0, expected}:
                count_issues.append(
                    {
                        "round": round_no,
                        "subject": subject,
                        "expected": expected,
                        "question_rows": qc,
                        "answer_rows": ac,
                    }
                )

    quality_counts: dict[str, int] = {}
    for row in questions:
        quality_counts[row["parse_quality"]] = quality_counts.get(row["parse_quality"], 0) + 1

    return {
        "attachments": len(documents),
        "documents_by_status": by_doc_status,
        "question_rows": len(questions),
        "answer_rows": len(answers),
        "question_quality_counts": quality_counts,
        "multiple_choice_count_issues": count_issues,
    }


def prepare_hf_dataset(
    questions: list[dict],
    answers: list[dict],
    documents: list[Document],
    subset_title: str = "Korean Bar Exam Questions and Answers (MOJ)",
) -> None:
    if HF_DIR.exists():
        shutil.rmtree(HF_DIR)
    (HF_DIR / "data").mkdir(parents=True)
    shutil.copy2(PROCESSED_DIR / "questions.csv", HF_DIR / "data" / "questions.csv")
    shutil.copy2(PROCESSED_DIR / "answers.csv", HF_DIR / "data" / "answers.csv")
    shutil.copy2(PROCESSED_DIR / "documents.csv", HF_DIR / "data" / "documents.csv")
    shutil.copy2(PROCESSED_DIR / "qa_report.json", HF_DIR / "qa_report.json")
    readme = build_readme(questions, answers, documents, subset_title=subset_title)
    (HF_DIR / "README.md").write_text(readme, encoding="utf-8")


def build_readme(
    questions: list[dict],
    answers: list[dict],
    documents: list[Document],
    subset_title: str = "Korean Bar Exam Questions and Answers (MOJ)",
) -> str:
    qa = json.loads((PROCESSED_DIR / "qa_report.json").read_text(encoding="utf-8"))
    rounds = sorted({row["round"] for row in questions if row["round"]})
    return f"""---
license: other
language:
- ko
pretty_name: {subset_title}
task_categories:
- question-answering
- text-classification
tags:
- law
- korean-law
- bar-exam
- public-sector
source_datasets:
- original
---

# {subset_title}

This dataset contains question-level rows extracted from official Korean Ministry of Justice 변호사시험 source files.

## Contents

- `data/questions.csv`: question rows with `round`, `year`, `subject`, `question_type`, `question_no`, `question_text`, `choices_json`, and `answer` where an official multiple-choice answer is available.
- `data/answers.csv`: parsed official multiple-choice answer rows.
- `data/documents.csv`: source document manifest with source URLs and SHA-256 hashes.
- `qa_report.json`: parser and count checks.

Covered rounds: {rounds[0]}-{rounds[-1]}.

Rows:

- Questions: {len(questions)}
- Multiple-choice answer rows: {len(answers)}
- Source documents: {len(documents)}

## Source and Attribution

Source: {SOURCE_ATTRIBUTION}

Primary source pages:

- 기출문제: {SOURCE_LIST_URL}
- 정답가안/확정정답 공지: {ANSWER_LIST_URL}

Each row includes the source article URL, source file URL, file name, and source SHA-256 where applicable.

## License

The source content is provided by the Ministry of Justice under {SOURCE_LICENSE}.
KOGL Type 1 permits online/offline use, derivative works, and commercial use, provided source attribution is shown and users do not imply endorsement by the public institution.

License summary: {KOG_LICENCE_URL}

Dataset packaging and conversion scripts in the originating repository may be Apache-2.0, but the source exam content remains governed by KOGL Type 1 attribution conditions.

## Processing Notes

HWP/HWPX files were converted to text using `pyhwp` (`hwp5html`) for HWP and XML extraction for HWPX. Multiple-choice answer tables were parsed from the converted XHTML table cells. For rounds 13-15, the Ministry of Justice final-answer notices state that the confirmed answers are the same as the posted provisional answer files; those rows are marked `final_notice_same_as_provisional`.

QA summary:

```json
{json.dumps(qa, ensure_ascii=False, indent=2)}
```
"""


def load_hf_token() -> str | None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"(?:export\s+)?(?:HF_TOKEN|HUGGINGFACE_HUB_TOKEN)\s*=\s*['\"]?([^'\"\s]+)", line)
        if m:
            return m.group(1).strip()
    return None


def upload_to_hub(repo_id: str | None, default_repo_name: str = "korean-bar-exam-moj") -> str:
    from huggingface_hub import HfApi

    token = load_hf_token()
    if not token:
        raise RuntimeError("HF token not found in environment or .env")
    api = HfApi(token=token)
    if repo_id is None:
        who = api.whoami(token=token)
        username = who.get("name")
        if not username:
            raise RuntimeError("Could not infer Hugging Face username")
        repo_id = f"{username}/{default_repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(HF_DIR),
        commit_message="Add MOJ Korean bar exam question dataset",
        token=token,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--only-multiple-choice", action="store_true")
    args = parser.parse_args()

    global PROCESSED_DIR, HF_DIR
    if args.only_multiple_choice:
        PROCESSED_DIR = OUT_ROOT / "processed_multiple_choice"
        HF_DIR = OUT_ROOT / "hf_multiple_choice"

    ensure_dirs()
    print("Collecting attachment metadata...")
    attachments = collect_attachments(only_multiple_choice=args.only_multiple_choice)
    print(f"Found {len(attachments)} attachments.")
    print("Downloading attachments...")
    download_attachments(attachments, force=args.force_download)
    print("Extracting and converting documents...")
    documents = build_documents(
        attachments,
        force_extract=args.force_extract,
        only_multiple_choice=args.only_multiple_choice,
    )
    print("Parsing answers...")
    answers = parse_answer_rows(documents)
    print("Parsing questions...")
    questions = parse_question_rows(documents, answers)
    print("Writing processed outputs...")
    write_outputs(attachments, documents, answers, questions)
    subset_title = (
        "Korean Bar Exam Multiple-Choice Questions and Answers (MOJ)"
        if args.only_multiple_choice
        else "Korean Bar Exam Questions and Answers (MOJ)"
    )
    prepare_hf_dataset(questions, answers, documents, subset_title=subset_title)
    qa = json.loads((PROCESSED_DIR / "qa_report.json").read_text(encoding="utf-8"))
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if args.upload:
        print("Uploading to Hugging Face Hub...")
        default_repo_name = (
            "korean-bar-exam-moj-multiple-choice"
            if args.only_multiple_choice
            else "korean-bar-exam-moj"
        )
        url = upload_to_hub(args.repo_id, default_repo_name=default_repo_name)
        print(url)


if __name__ == "__main__":
    main()
