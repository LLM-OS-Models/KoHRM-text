"""Build raw Korean legal Markdown corpus JSONL for HRM-Text pretraining.

Unlike build_korean_legal_sft_data.py, this keeps the original legal text as
the response target. Long documents are split into overlapping chunks so the
full corpus can be used without context-window truncation losing most content.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterator


FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", flags=re.DOTALL)
HEADING_RE = re.compile(r"(?m)^(#{1,6}\s+.+|제\d+조(?:의\d+)?\s*\([^)]+\).*)$")


def strip_frontmatter(text: str) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end >= 0:
            raw = text[4:end]
            for line in raw.splitlines():
                if ":" in line and not line.startswith((" ", "-")):
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip().strip("'\"")
            text = text[end + 4 :]
    return meta, text


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = FRONTMATTER_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def git_md_files(root: Path) -> list[Path]:
    git_dir = root / ".git"
    if git_dir.exists():
        proc = subprocess.run(
            ["git", "-C", str(root), "-c", "core.quotepath=false", "ls-files", "-z", "--", "*.md"],
            check=True,
            stdout=subprocess.PIPE,
        )
        paths: list[Path] = []
        for raw_line in proc.stdout.split(b"\0"):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="surrogateescape")
            if Path(line).name == "README.md":
                continue
            paths.append(root / line)
        return paths

    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            if name.endswith(".md") and name != "README.md":
                paths.append(Path(dirpath) / name)
    return sorted(paths)


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
            candidates = [m.start() for m in HEADING_RE.finditer(window) if m.start() > max_chars * 0.55]
            if candidates:
                end = start + candidates[-1]
            else:
                para = window.rfind("\n\n", int(max_chars * 0.55))
                if para > 0:
                    end = start + para
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)


def title_from(meta: dict[str, str], body: str, path: Path) -> str:
    for key in ("제목", "자치법규명", "사건명"):
        value = meta.get(key)
        if value:
            return value
    match = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
    return match.group(1).strip() if match else path.stem


def build_rows(root_name: str, root: Path, max_chars: int, overlap_chars: int) -> Iterator[dict]:
    files = git_md_files(root)
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        meta, body = strip_frontmatter(raw)
        body = clean_text(body)
        if len(body) < 80:
            continue
        title = title_from(meta, body, path)
        rel = str(path.relative_to(root))
        chunks = list(chunk_text(body, max_chars=max_chars, overlap_chars=overlap_chars))
        for i, chunk in enumerate(chunks):
            instruction = (
                "다음은 대한민국 법령/자치법규 원문 일부입니다. "
                "법률 한국어, 조문 구조, 번호 체계, 기관명, 시행일자 표현을 그대로 학습하십시오.\n\n"
                f"[자료종류]\n{root_name}\n\n[문서명]\n{title}\n\n[경로]\n{rel}\n\n[부분]\n{i + 1}/{len(chunks)}"
            )
            yield {
                "instruction": instruction,
                "response": chunk,
                "condition": "direct",
                "source": f"korean_legal_raw_{root_name}",
                "path": rel,
                "chunk_index": i,
                "chunk_count": len(chunks),
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--root",
        action="append",
        default=[],
        help="name=path. Example: law=/repo/legalize-kr",
    )
    ap.add_argument("--max-chars", type=int, default=5600)
    ap.add_argument("--overlap-chars", type=int, default=160)
    ap.add_argument("--progress-interval", type=int, default=10000)
    args = ap.parse_args()

    roots: list[tuple[str, Path]] = []
    for item in args.root:
        name, path = item.split("=", 1)
        roots.append((name, Path(path)))
    if not roots:
        raise SystemExit("--root is required")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    bytes_written = 0
    by_source: dict[str, int] = {}
    with out.open("w", encoding="utf-8") as f:
        for root_name, root in roots:
            for row in build_rows(root_name, root, args.max_chars, args.overlap_chars):
                line = json.dumps(row, ensure_ascii=False) + "\n"
                f.write(line)
                rows += 1
                bytes_written += len(line.encode("utf-8"))
                by_source[row["source"]] = by_source.get(row["source"], 0) + 1
                if args.progress_interval and rows % args.progress_interval == 0:
                    print(
                        f"rows={rows:,} bytes={bytes_written:,} source={row['source']}",
                        flush=True,
                    )

    stats = {
        "rows": rows,
        "bytes": bytes_written,
        "roots": {name: str(path) for name, path in roots},
        "by_source": by_source,
        "max_chars": args.max_chars,
        "overlap_chars": args.overlap_chars,
    }
    stats_path = out.with_suffix(out.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
