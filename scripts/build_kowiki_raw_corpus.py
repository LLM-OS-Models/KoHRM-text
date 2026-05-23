"""Build Korean Wikipedia raw-text JSONL for HRM-Text pretraining."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator


COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
REF_RE = re.compile(r"<ref\b[^>/]*?>.*?</ref>|<ref\b[^>]*/>", flags=re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
TABLE_RE = re.compile(r"\{\|.*?\|\}", flags=re.DOTALL)
FILE_LINK_RE = re.compile(r"\[\[(?:파일|File|그림|Image):[^\]]+\]\]", flags=re.IGNORECASE)
CATEGORY_RE = re.compile(r"\[\[(?:분류|Category):([^\]|]+)(?:\|[^\]]*)?\]\]", flags=re.IGNORECASE)
WIKI_LINK_WITH_LABEL_RE = re.compile(r"\[\[[^\]|]+\|([^\]]+)\]\]")
WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)\]\]")
HEADING_RE = re.compile(r"(?m)^={2,}\s*(.+?)\s*={2,}$")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def iter_pages(path: Path) -> Iterator[tuple[str, str, str | None]]:
    title = ""
    namespace = ""
    text = ""
    redirect: str | None = None
    in_revision = False

    for event, elem in ET.iterparse(path, events=("start", "end")):
        name = local_name(elem.tag)
        if event == "start":
            if name == "page":
                title = ""
                namespace = ""
                text = ""
                redirect = None
                in_revision = False
            elif name == "revision":
                in_revision = True
            elif name == "redirect":
                redirect = elem.attrib.get("title") or ""
            continue

        if name == "title":
            title = elem.text or ""
        elif name == "ns":
            namespace = elem.text or ""
        elif name == "text" and in_revision:
            text = elem.text or ""
        elif name == "revision":
            in_revision = False
        elif name == "page":
            if namespace == "0" and title and text:
                yield title, text, redirect
            elem.clear()


def strip_templates(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = TEMPLATE_RE.sub("", text)
    return text


def clean_wikitext(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = COMMENT_RE.sub("", text)
    text = TABLE_RE.sub("\n", text)
    text = REF_RE.sub("", text)
    text = FILE_LINK_RE.sub("", text)
    text = CATEGORY_RE.sub(r"\1", text)
    text = strip_templates(text)
    text = HEADING_RE.sub(lambda m: "\n# " + m.group(1).strip() + "\n", text)
    text = WIKI_LINK_WITH_LABEL_RE.sub(r"\1", text)
    text = WIKI_LINK_RE.sub(r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", "", text)
    text = TAG_RE.sub("", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> Iterator[str]:
    if len(text) <= max_chars:
        yield text
        return

    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        end = hard_end
        if hard_end < len(text):
            window = text[start:hard_end]
            para = window.rfind("\n\n", int(max_chars * 0.55))
            if para > 0:
                end = start + para
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)


def is_redirect_text(text: str) -> bool:
    head = text.lstrip()[:40].lower()
    return head.startswith("#redirect") or head.startswith("#넘겨주기")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-chars", type=int, default=5600)
    ap.add_argument("--overlap-chars", type=int, default=160)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--progress-interval", type=int, default=10000)
    args = ap.parse_args()

    input_path = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    pages = 0
    kept_pages = 0
    bytes_written = 0
    with out.open("w", encoding="utf-8") as f:
        for title, raw_text, redirect in iter_pages(input_path):
            pages += 1
            if redirect is not None or is_redirect_text(raw_text):
                continue
            text = clean_wikitext(raw_text)
            if len(text) < args.min_chars:
                continue
            chunks = list(chunk_text(text, args.max_chars, args.overlap_chars))
            if not chunks:
                continue
            kept_pages += 1
            for i, chunk in enumerate(chunks):
                row = {
                    "instruction": (
                        "다음은 한국어 위키백과 문서 원문 일부입니다. "
                        "백과사전식 한국어, 고유명사, 날짜, 기술/사회/문화 지식을 그대로 학습하십시오.\n\n"
                        f"[문서명]\n{title}\n\n[부분]\n{i + 1}/{len(chunks)}"
                    ),
                    "response": chunk,
                    "condition": "direct",
                    "source": "kowiki_raw_20260501",
                    "title": title,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                }
                line = json.dumps(row, ensure_ascii=False) + "\n"
                f.write(line)
                rows += 1
                bytes_written += len(line.encode("utf-8"))
            if args.progress_interval and pages % args.progress_interval == 0:
                print(
                    f"pages={pages:,} kept_pages={kept_pages:,} rows={rows:,} bytes={bytes_written:,}",
                    flush=True,
                )

    stats = {
        "input": str(input_path),
        "rows": rows,
        "pages": pages,
        "kept_pages": kept_pages,
        "bytes": bytes_written,
        "max_chars": args.max_chars,
        "overlap_chars": args.overlap_chars,
        "min_chars": args.min_chars,
    }
    stats_path = out.with_suffix(out.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
