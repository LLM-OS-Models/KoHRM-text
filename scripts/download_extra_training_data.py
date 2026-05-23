"""Download and sample extra datasets for the Korean terminal HRM-Text run.

The goal is deliberately conservative on disk usage:
- small, high-value datasets are downloaded as snapshots;
- very large datasets are streamed into capped JSONL shards.

The capped JSONL files use HRM-Text SFT-compatible fields:
    {"instruction": "...", "response": "...", "condition": "...", "source": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import snapshot_download


SMALL_SNAPSHOTS = [
    "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k",
    "TuringEnterprises/Open-MM-RL",
]

# These repos contain thousands of tiny files. Unauthenticated HF downloads hit
# rate limits quickly, so keep them opt-in and prefer streaming/partial samples
# unless a token is configured.
MANY_SMALL_FILE_SNAPSHOTS = [
    "TeichAI/DeepSeek-v4-Pro-Agent",
    "actava/chi-bench",
]


def safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        text = obj.strip()
        if text:
            yield text
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_strings(value)


class CappedJsonlWriter:
    def __init__(self, path: Path, cap_mib: int):
        self.path = path
        self.cap_bytes = cap_mib * 1024 * 1024
        self.bytes_written = 0
        self.rows_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            for row in rows:
                line = json.dumps(row, ensure_ascii=False) + "\n"
                b = len(line.encode("utf-8"))
                if self.bytes_written and self.bytes_written + b > self.cap_bytes:
                    break
                f.write(line)
                self.bytes_written += b
                self.rows_written += 1


def format_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "unknown")
        content = (m.get("content") or "").strip()
        if content:
            parts.append(f"<|{role}|>\n{content}")
    return "\n\n".join(parts)


def stream_swe_zero(out_dir: Path, cap_mib: int) -> dict[str, Any]:
    out_path = out_dir / "sft" / "swe_zero_terminal_sft_sample.jsonl"
    writer = CappedJsonlWriter(out_path, cap_mib)

    def rows() -> Iterable[dict[str, Any]]:
        ds = load_dataset("AlienKevin/SWE-ZERO-12M-trajectories", split="train", streaming=True)
        for item in ds:
            prior: list[dict[str, str]] = []
            for message in item.get("messages") or []:
                role = message.get("role")
                content = (message.get("content") or "").strip()
                if role == "assistant" and content and prior:
                    yield {
                        "instruction": format_messages(prior),
                        "response": content,
                        "condition": "direct",
                        "source": "AlienKevin/SWE-ZERO-12M-trajectories",
                        "instance_id": item.get("instance_id"),
                        "repo": item.get("repo"),
                        "exit_status": item.get("exit_status"),
                    }
                if role and content:
                    prior.append({"role": role, "content": content})

    writer.write_rows(rows())
    return {"path": str(out_path), "rows": writer.rows_written, "bytes": writer.bytes_written}


def stream_glm_reasoning(out_dir: Path, cap_mib: int) -> dict[str, Any]:
    out_path = out_dir / "sft" / "glm_5_1_reasoning_sft_sample.jsonl"
    writer = CappedJsonlWriter(out_path, cap_mib)

    def rows() -> Iterable[dict[str, Any]]:
        ds = load_dataset("Jackrong/GLM-5.1-Reasoning-1M-Cleaned", split="train", streaming=True)
        for item in ds:
            instruction = (item.get("input") or "").strip()
            response = (item.get("output") or "").strip()
            if instruction and response:
                yield {
                    "instruction": instruction,
                    "response": response,
                    "condition": "cot" if "<think>" in response else "direct",
                    "source": "Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
                    "domain": item.get("domain"),
                    "teacher_model": (item.get("meta") or {}).get("teacher_model"),
                }

    writer.write_rows(rows())
    return {"path": str(out_path), "rows": writer.rows_written, "bytes": writer.bytes_written}


def stream_structured_wikipedia(out_dir: Path, cap_mib: int) -> dict[str, Any]:
    out_path = out_dir / "tokenizer_corpus" / "structured_wikipedia_en_sample.jsonl"
    writer = CappedJsonlWriter(out_path, cap_mib)

    def rows() -> Iterable[dict[str, Any]]:
        ds = load_dataset(
            "wikimedia/structured-wikipedia",
            "enwiki_namespace_0",
            split="train",
            streaming=True,
        )
        for item in ds:
            texts = list(iter_strings(item))
            text = "\n".join(texts)
            if len(text) >= 200:
                yield {
                    "text": text,
                    "source": "wikimedia/structured-wikipedia/enwiki_namespace_0",
                }

    writer.write_rows(rows())
    return {"path": str(out_path), "rows": writer.rows_written, "bytes": writer.bytes_written}


def download_small_snapshots(out_dir: Path, include_many_small_files: bool = False) -> list[dict[str, Any]]:
    results = []
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    repo_ids = list(SMALL_SNAPSHOTS)
    if include_many_small_files:
        repo_ids.extend(MANY_SMALL_FILE_SNAPSHOTS)
    for repo_id in repo_ids:
        local_dir = raw_dir / safe_name(repo_id)
        path = snapshot_download(
            repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            max_workers=16,
        )
        results.append({"repo_id": repo_id, "path": path})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/work/.data/huggingface/hrm_text_extra")
    ap.add_argument("--swe-mib", type=int, default=1024)
    ap.add_argument("--glm-mib", type=int, default=1024)
    ap.add_argument("--wiki-mib", type=int, default=256)
    ap.add_argument("--skip-snapshots", action="store_true")
    ap.add_argument("--skip-streaming", action="store_true")
    ap.add_argument(
        "--include-many-small-file-snapshots",
        action="store_true",
        help="Also snapshot datasets like TeichAI/DeepSeek and chi-bench. This may hit HF rate limits.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"out_dir": str(out_dir), "snapshots": [], "samples": []}

    if not args.skip_snapshots:
        manifest["snapshots"] = download_small_snapshots(out_dir, args.include_many_small_file_snapshots)

    if not args.skip_streaming:
        manifest["samples"].append(stream_swe_zero(out_dir, args.swe_mib))
        manifest["samples"].append(stream_glm_reasoning(out_dir, args.glm_mib))
        manifest["samples"].append(stream_structured_wikipedia(out_dir, args.wiki_mib))

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
