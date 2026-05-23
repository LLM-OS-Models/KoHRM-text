"""Convert local terminal trajectory parquet files to HRM-Text SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq


ROLE_LABELS = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "terminal",
    "terminal": "terminal",
    "function": "tool",
}


def normalize_role(role: object) -> str:
    value = str(role or "user").strip().lower()
    return ROLE_LABELS.get(value, value or "user")


def trim_left(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return "[이전 대화 일부 생략]\n" + text[-max_chars:]


def chunk_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        cut = end
        if end < len(text):
            newline = text.rfind("\n", start + int(max_chars * 0.6), end)
            if newline > start:
                cut = newline
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = max(cut, start + 1)
    return chunks


def format_turn(role: str, content: str) -> str:
    return f"[{role}]\n{content.strip()}"


def conversation_turns(raw: object) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    if not isinstance(raw, list):
        return turns
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if content is None:
            continue
        text = str(content).strip()
        if not text:
            continue
        turns.append((normalize_role(item.get("role")), text))
    return turns


def iter_parquet_rows(paths: Iterable[Path], batch_size: int) -> Iterator[tuple[Path, dict]]:
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield path, row


def build_examples(
    path: Path,
    row: dict,
    max_instruction_chars: int,
    max_response_chars: int,
    min_response_chars: int,
) -> Iterator[dict]:
    turns = conversation_turns(row.get("conversations"))
    if len(turns) < 2:
        return

    prior: list[str] = []
    example_index = 0
    base_source = path.stem
    condition = "cot" if row.get("enable_thinking") else "direct"
    metadata = []
    for key in ("task", "episode", "agent", "model", "source"):
        value = row.get(key)
        if value:
            metadata.append(f"{key}: {value}")
    metadata_text = "[metadata]\n" + "\n".join(metadata) if metadata else ""

    for turn_index, (role, content) in enumerate(turns):
        if role == "assistant" and prior and len(content.strip()) >= min_response_chars:
            context = "\n\n".join(([metadata_text] if metadata_text else []) + prior)
            context = trim_left(context, max_instruction_chars)
            chunks = chunk_text(content, max_response_chars)
            for chunk_index, chunk in enumerate(chunks):
                if len(chunk.strip()) < min_response_chars:
                    continue
                chunk_note = ""
                if len(chunks) > 1:
                    chunk_note = f"\n\n[응답 부분]\n{chunk_index + 1}/{len(chunks)}"
                yield {
                    "instruction": (
                        "다음 터미널/코딩 작업 대화 맥락에서 assistant가 이어서 수행할 분석, 계획, "
                        "명령 JSON 또는 최종 응답을 작성하십시오.\n\n"
                        f"{context}{chunk_note}"
                    ),
                    "response": chunk,
                    "condition": condition,
                    "source": f"local_terminal_{base_source}",
                    "task": row.get("task"),
                    "episode": row.get("episode"),
                    "run_id": row.get("run_id"),
                    "turn_index": turn_index,
                    "example_index": example_index,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                }
                example_index += 1
        prior.append(format_turn(role, content))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-instruction-chars", type=int, default=24000)
    ap.add_argument("--max-response-chars", type=int, default=12000)
    ap.add_argument("--min-response-chars", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--progress-interval", type=int, default=5000)
    args = ap.parse_args()

    paths = [Path(p) for p in args.input]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    examples = 0
    bytes_written = 0
    by_source: dict[str, int] = {}
    with out.open("w", encoding="utf-8") as f:
        for path, row in iter_parquet_rows(paths, args.batch_size):
            rows += 1
            if args.max_rows and rows > args.max_rows:
                break
            for ex in build_examples(
                path,
                row,
                args.max_instruction_chars,
                args.max_response_chars,
                args.min_response_chars,
            ):
                line = json.dumps(ex, ensure_ascii=False) + "\n"
                f.write(line)
                examples += 1
                bytes_written += len(line.encode("utf-8"))
                source = ex["source"]
                by_source[source] = by_source.get(source, 0) + 1
            if args.progress_interval and rows % args.progress_interval == 0:
                print(
                    f"rows={rows:,} examples={examples:,} bytes={bytes_written:,} file={path.name}",
                    flush=True,
                )

    stats = {
        "rows": rows,
        "examples": examples,
        "bytes": bytes_written,
        "inputs": [str(p) for p in paths],
        "by_source": by_source,
        "max_instruction_chars": args.max_instruction_chars,
        "max_response_chars": args.max_response_chars,
    }
    stats_path = out.with_suffix(out.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
