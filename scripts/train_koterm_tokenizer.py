"""Train a Korean/terminal/tool-call BPE tokenizer for HRM-Text.

This script trains from text-like local files and JSONL SFT samples without
requiring a monolithic corpus file. It is intended for tokenizer iteration;
the final tokenizer should be trained on a larger, audited corpus manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


SPECIAL_TOKENS = [
    "<|PAD|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_response|>",
    "<|function|>",
    "<|/function|>",
    "<|execute|>",
    "<|result|>",
    "<|terminal|>",
    "<|/terminal|>",
    "<|command|>",
    "<|output|>",
    "<|error|>",
    "<|exit_code|>",
    "<|json_start|>",
    "<|json_end|>",
    "<|xml_start|>",
    "<|xml_end|>",
    "<|code_start|>",
    "<|code_end|>",
    "<think>",
    "</think>",
    # Existing HRM-Text/control tokens. Keep these because prepare_sft_data.py
    # maps direct/cot/noisy/synth to the object/quad tokens by default.
    "<|direct|>",
    "<|cot|>",
    "<|noisy|>",
    "<|synth|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]

TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".py",
    ".sh",
    ".xml",
    ".csv",
}


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


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


def clean_xmlish(line: str) -> str:
    line = TAG_RE.sub(" ", line)
    return SPACE_RE.sub(" ", line).strip()


def iter_json_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            yield from iter_strings(obj)


def iter_plain_text(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = clean_xmlish(line) if path.suffix == ".xml" else line.strip()
            if len(line) >= 20:
                yield line


def iter_files(paths: list[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path
        elif path.is_dir():
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in {".git", ".cache", "__pycache__"}]
                for name in files:
                    file_path = Path(root) / name
                    if file_path.suffix.lower() in TEXT_SUFFIXES:
                        yield file_path


def iter_training_text(paths: list[Path], max_bytes: int, max_bytes_per_input: int | None) -> Iterable[str]:
    total_seen_bytes = 0
    for input_path in paths:
        input_seen_bytes = 0
        for path in iter_files([input_path]):
            iterator = iter_json_lines(path) if path.suffix.lower() in {".json", ".jsonl"} else iter_plain_text(path)
            for text in iterator:
                if len(text) < 20:
                    continue
                b = len(text.encode("utf-8", errors="ignore"))
                if max_bytes_per_input is not None and input_seen_bytes + b > max_bytes_per_input:
                    break
                if total_seen_bytes + b > max_bytes:
                    return
                input_seen_bytes += b
                total_seen_bytes += b
                yield text
            if max_bytes_per_input is not None and input_seen_bytes >= max_bytes_per_input:
                break


def iter_training_text_unbalanced(paths: list[Path], max_bytes: int) -> Iterable[str]:
    seen_bytes = 0
    for path in iter_files(paths):
        iterator = iter_json_lines(path) if path.suffix.lower() in {".json", ".jsonl"} else iter_plain_text(path)
        for text in iterator:
            if len(text) < 20:
                continue
            seen_bytes += len(text.encode("utf-8", errors="ignore"))
            if seen_bytes > max_bytes:
                return
            yield text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="Files or directories to read.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=131072)
    ap.add_argument("--max-gib", type=float, default=2.0)
    ap.add_argument(
        "--max-mib-per-input",
        type=float,
        default=None,
        help="Optional cap per top-level input path. Use this to prevent early large corpora from dominating.",
    )
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    input_paths = [Path(p) for p in args.input]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>", byte_fallback=True))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    max_bytes = int(args.max_gib * 1024**3)
    max_bytes_per_input = None
    if args.max_mib_per_input is not None:
        max_bytes_per_input = int(args.max_mib_per_input * 1024**2)
    tokenizer.train_from_iterator(iter_training_text(input_paths, max_bytes, max_bytes_per_input), trainer=trainer)
    tokenizer.save(str(output_dir / "tokenizer.json"))

    meta = {
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "requested_vocab_size": args.vocab_size,
        "max_gib": args.max_gib,
        "max_mib_per_input": args.max_mib_per_input,
        "min_frequency": args.min_frequency,
        "special_tokens": SPECIAL_TOKENS,
        "inputs": [str(p) for p in input_paths],
    }
    (output_dir / "tokenizer_training_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
