"""Audit local HRM-Text training data sizes and token estimates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer


TEXT_EXTS = {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".csv"}


def walk_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "__MACOSX"}]
        for name in filenames:
            if name.startswith("._") or name == ".DS_Store":
                continue
            yield Path(dirpath) / name


def folder_bytes(root: Path) -> tuple[int, int]:
    files, total = 0, 0
    for path in walk_files(root):
        try:
            total += path.stat().st_size
            files += 1
        except OSError:
            pass
    return files, total


def sample_text(root: Path, max_bytes: int) -> tuple[str, int]:
    chunks: list[str] = []
    read_bytes = 0
    for path in walk_files(root):
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            with open(path, "rb") as f:
                data = f.read(max(0, max_bytes - read_bytes))
        except OSError:
            continue
        if not data:
            continue
        read_bytes += len(data)
        chunks.append(data.decode("utf-8", errors="ignore"))
        if read_bytes >= max_bytes:
            break
    return "\n".join(chunks), read_bytes


def exact_jsonl_rows(path: Path) -> int | None:
    if not path.exists() or path.suffix != ".jsonl":
        return None
    rows = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                rows += 1
    return rows


def prepared_stats(path: Path) -> dict | None:
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    out = {
        "prepared_tokens_exact": meta.get("total_length"),
        "context_size": meta.get("max_seq_len"),
    }
    for name in ("preprocess_stats.json", "merge_stats.json"):
        p = path / name
        if p.exists():
            out[name.removesuffix(".json")] = json.loads(p.read_text(encoding="utf-8"))
    return out


def audit_source(name: str, path: Path, tok: Tokenizer, sample_mib: int) -> dict:
    files, total_bytes = folder_bytes(path)
    source = {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "files": files,
        "bytes": total_bytes,
        "gib": total_bytes / (1024 ** 3),
    }
    rows = exact_jsonl_rows(path) if path.is_file() else None
    if rows is not None:
        source["rows"] = rows

    prep = prepared_stats(path)
    if prep:
        source |= prep
        return source

    text, sampled_bytes = sample_text(path, sample_mib * 1024 * 1024)
    if sampled_bytes and total_bytes:
        token_count = len(tok.encode(text, add_special_tokens=False).ids)
        source["sampled_bytes"] = sampled_bytes
        source["sampled_tokens"] = token_count
        source["estimated_tokens"] = int(token_count * (total_bytes / sampled_bytes))
        source["estimate_note"] = "sample extrapolation"
    return source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--sample-mib", type=int, default=32)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    sources = [
        ("SWE-ZERO raw sample", Path("/home/work/.data/huggingface/hrm_text_extra/sft/swe_zero_terminal_sft_sample.jsonl")),
        ("GLM reasoning raw sample", Path("/home/work/.data/huggingface/hrm_text_extra/sft/glm_5_1_reasoning_sft_sample.jsonl")),
        ("structured wikipedia tokenizer sample", Path("/home/work/.data/huggingface/hrm_text_extra/tokenizer_corpus/structured_wikipedia_en_sample.jsonl")),
        ("Claude Opus reasoning raw snapshot", Path("/home/work/.data/huggingface/hrm_text_extra/raw/angrygiraffe__claude-opus-4.6-4.7-reasoning-8.7k")),
        ("DeepSeek-v4-Pro-Agent partial raw", Path("/home/work/.data/huggingface/hrm_text_extra/raw/TeichAI__DeepSeek-v4-Pro-Agent")),
        ("HRM ToolBench train", Path("HRM-Text/data_toolbench/data/toolllama_G123_dfs_train.json")),
        ("HRM ToolBench eval", Path("HRM-Text/data_toolbench/data/toolllama_G123_dfs_eval.json")),
        ("HRM data.zip", Path("HRM-Text/data.zip")),
        ("Korean laws legalize-kr", Path("HRM-Text/legalize-kr")),
        ("Korean ordinances", Path("HRM-Text/ordinance-kr")),
        ("Korean administrative rules", Path("admrule-kr")),
        ("Korean precedents", Path("precedent-kr")),
        ("prepared SWE", Path("/home/work/.data/hrm_text_prepared/sft_swe_zero_v1")),
        ("prepared GLM", Path("/home/work/.data/hrm_text_prepared/sft_glm_reasoning_v1")),
        ("prepared SWE+GLM mix", Path("/home/work/.data/hrm_text_prepared/sft_swe_glm_mix_v1")),
    ]
    result = [audit_source(name, path, tok, args.sample_mib) for name, path in sources]

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# HRM-Text 데이터 상태 감사",
        "",
        "생성일: 2026-05-23",
        "",
        "| 데이터 | 상태 | 파일/rows | 크기 | 토큰 | 비고 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in result:
        status = "있음" if item["exists"] else "없음"
        count = item.get("rows", item.get("files", 0))
        gib = item["gib"]
        tokens = item.get("prepared_tokens_exact") or item.get("estimated_tokens")
        token_text = f"{tokens:,}" if tokens is not None else "-"
        note = "exact prepared" if item.get("prepared_tokens_exact") else item.get("estimate_note", "")
        lines.append(f"| {item['name']} | {status} | {count:,} | {gib:.2f} GiB | {token_text} | {note} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
