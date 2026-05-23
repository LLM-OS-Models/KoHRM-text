"""Build Korean legal/regulation/precedent SFT JSONL from local markdown corpora."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path


ARTICLE_RE = re.compile(r"(?m)^#####\s+(제\d+조[^\n]*)\n")
SECTION_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")


def walk_md(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            if name.endswith(".md") and name != "README.md":
                yield Path(dirpath) / name


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    raw = text[4:end]
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip("'\"")
    return meta, text[end + 4 :].strip()


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def title_from(meta: dict[str, str], body: str, path: Path) -> str:
    for key in ("제목", "사건명"):
        if meta.get(key):
            return meta[key]
    m = re.search(r"^#\s+(.+)$", body, flags=re.M)
    return m.group(1).strip() if m else path.stem


def first_sentences(text: str, max_chars: int = 420) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    cut = compact[:max_chars]
    pos = max(cut.rfind("."), cut.rfind("다."), cut.rfind("한다."))
    return (cut[: pos + 1] if pos > 120 else cut).strip()


def article_chunks(body: str):
    matches = list(ARTICLE_RE.finditer(body))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        heading = m.group(1).strip()
        content = clean(body[start:end])
        if len(content) >= 80:
            yield heading, content[:7000]


def precedent_sections(body: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(body))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[name] = clean(body[start:end])
    return sections


def make_rows(path: Path, root_name: str, text: str):
    meta, body = split_frontmatter(text)
    body = clean(body)
    title = title_from(meta, body, path)
    rel = str(path)

    if root_name == "precedent":
        sections = precedent_sections(body)
        holding = sections.get("판결요지") or sections.get("판시사항")
        content = sections.get("판례내용", "")
        if holding and len(holding) >= 20:
            instruction = (
                "다음 한국 판례 문서에서 사건명, 법원, 선고일자, 판결요지를 JSON으로 정리하라.\n\n"
                f"[문서]\n{body[:6000]}"
            )
            response = json.dumps(
                {
                    "사건명": title,
                    "법원": meta.get("법원명", ""),
                    "선고일자": meta.get("선고일자", ""),
                    "판결요지": first_sentences(holding, 700),
                    "출처": meta.get("출처", ""),
                },
                ensure_ascii=False,
            )
            yield instruction, response, "korean_legal_precedent", rel
        if content and len(content) >= 200:
            instruction = (
                "다음 판례 본문을 읽고 주문과 핵심 법리를 한국어로 간결하게 요약하라.\n\n"
                f"[판례]\n{content[:5000]}"
            )
            response = f"사건명: {title}\n핵심 법리: {first_sentences(holding or content, 800)}"
            yield instruction, response, "korean_legal_precedent", rel
        return

    made = 0
    for heading, content in article_chunks(body):
        instruction = (
            "다음 한국 법령/행정규칙 발췌문에서 조문명, 적용 대상, 핵심 의무를 JSON으로 추출하라.\n\n"
            f"[문서명]\n{title}\n\n[조문]\n{heading}\n{content}"
        )
        response = json.dumps(
            {
                "문서명": title,
                "조문": heading,
                "핵심요지": first_sentences(content, 520),
                "출처": meta.get("출처", ""),
            },
            ensure_ascii=False,
        )
        yield instruction, response, f"korean_{root_name}", rel
        made += 1
        if made >= 8:
            break

    if made == 0 and len(body) >= 300:
        excerpt = body[:5000]
        instruction = f"다음 한국 법령/규정 문서를 핵심 항목 5개 이하로 요약하라.\n\n[문서]\n{excerpt}"
        response = f"{title}: {first_sentences(excerpt, 900)}"
        yield instruction, response, f"korean_{root_name}", rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-output-mib", type=int, default=512, help="0 means no output cap.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--root",
        action="append",
        default=[],
        help="name=path, e.g. law=HRM-Text/legalize-kr",
    )
    args = ap.parse_args()

    roots = []
    for item in args.root:
        name, raw_path = item.split("=", 1)
        roots.append((name, Path(raw_path)))
    if not roots:
        roots = [
            ("law", Path("HRM-Text/legalize-kr")),
            ("ordinance", Path("HRM-Text/ordinance-kr")),
            ("admrule", Path("admrule-kr")),
            ("precedent", Path("precedent-kr")),
        ]

    paths = []
    for name, root in roots:
        for p in walk_md(root):
            paths.append((name, p))
    rng = random.Random(args.seed)
    rng.shuffle(paths)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = None if args.max_output_mib <= 0 else args.max_output_mib * 1024 * 1024
    rows = 0
    bytes_written = 0
    by_source: dict[str, int] = {}
    with open(out, "w", encoding="utf-8") as f:
        for root_name, path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for instruction, response, source, rel in make_rows(path, root_name, text):
                row = {
                    "instruction": instruction,
                    "response": response,
                    "condition": "direct",
                    "source": source,
                    "path": rel,
                }
                line = json.dumps(row, ensure_ascii=False) + "\n"
                encoded = line.encode("utf-8")
                if max_bytes is not None and bytes_written + len(encoded) > max_bytes:
                    print(f"wrote {rows:,} rows, {bytes_written:,} bytes to {out}")
                    print(json.dumps(by_source, ensure_ascii=False, indent=2))
                    return
                f.write(line)
                rows += 1
                bytes_written += len(encoded)
                by_source[source] = by_source.get(source, 0) + 1

    print(f"wrote {rows:,} rows, {bytes_written:,} bytes to {out}")
    print(json.dumps(by_source, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
