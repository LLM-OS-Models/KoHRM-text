"""Prepare SFT data for HRM-Text fine-tuning.

Converts a generic JSONL into the V1Dataset binary layout that
`dataset_new.py` consumes directly. Prompt construction (few-shot demos,
schema injection, task framing, etc.) is the caller's responsibility —
the `instruction` field is tokenized verbatim.

Input JSONL (one object per line):
    {"instruction": "<full prompt>", "response": "<expected output>",
     "condition": "direct"}      # condition optional; defaults to "direct"

Output directory:
    <out>/metadata.json
    <out>/tokens.npy
    <out>/tokenizer_info.json
    <out>/tokenizer.json                   # copy for self-containment
    <out>/epoch_0/{inst_start,inst_len,resp_start,resp_len}.npy
    <out>/epoch_1/...
    ...

Usage:
    python scripts/prepare_sft_data.py \
        --train my_sft.jsonl \
        --tokenizer /path/to/tokenizer.json \
        --output /tmp/sft_data \
        --epochs 10
"""
import argparse
from array import array
import json
import re
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
from tokenizers import Tokenizer


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
SUPPORTED_INPUT_SUFFIXES = {".jsonl", ".parquet"}


def strip_think_blocks(text: str) -> str:
    """Remove teacher private reasoning blocks for response-only pretraining."""
    return THINK_BLOCK_RE.sub("", text).strip()


def truncate_instruction_middle(inst_ids: list[int], budget: int, head_tokens: int) -> list[int]:
    if len(inst_ids) <= budget:
        return inst_ids
    if budget <= 0:
        return []

    keep_head = min(head_tokens, max(0, budget // 2))
    keep_tail = budget - keep_head
    if keep_tail <= 0:
        return inst_ids[:budget]
    return inst_ids[:keep_head] + inst_ids[-keep_tail:]


def normalize_condition(condition: str) -> str:
    labels = {part.strip() for part in str(condition).split(",") if part.strip()}
    # Preserve explicit CoT first, then synthetic/noisy provenance.
    for key in ("cot", "synth", "noisy", "direct"):
        if key in labels:
            return key
    return str(condition)


def collect_input_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(
                sorted(
                    f for f in p.rglob("*")
                    if f.is_file() and f.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
                )
            )
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
                raise ValueError(f"unsupported input suffix for {p}; expected .jsonl or .parquet")
            files.append(p)
        else:
            matches = sorted(Path().glob(item))
            if not matches:
                raise FileNotFoundError(item)
            files.extend(
                f for f in matches
                if f.is_file() and f.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
            )

    if not files:
        raise ValueError("no supported input files found")
    return files


def iter_jsonl_rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_parquet_rows(path: Path, batch_size: int) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required for parquet input") from exc

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        rows = batch.to_pydict()
        keys = list(rows)
        for i in range(batch.num_rows):
            yield {k: rows[k][i] for k in keys}


def iter_input_rows(files: list[Path], parquet_batch_size: int, max_rows_per_file: int) -> Iterator[tuple[Path, dict]]:
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            row_iter = iter_jsonl_rows(path)
        elif suffix == ".parquet":
            row_iter = iter_parquet_rows(path, parquet_batch_size)
        else:
            continue

        file_rows = 0
        for row in row_iter:
            file_rows += 1
            yield path, row
            if max_rows_per_file and file_rows >= max_rows_per_file:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="Input JSONL/parquet files, globs, or directories.")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    ap.add_argument("--output", required=True, help="Output directory.")
    ap.add_argument("--epochs", type=int, required=True,
                    help="Pre-compute N epoch shuffles (must match training epochs).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context-size", type=int, default=4097,
                    help="Must be at least max sample length + 1 (AR shift).")
    ap.add_argument(
        "--overflow-policy",
        choices=["error", "drop", "truncate-instruction-middle"],
        default="error",
        help="How to handle samples longer than context-size - 1.",
    )
    ap.add_argument(
        "--truncate-head-tokens",
        type=int,
        default=512,
        help="For truncate-instruction-middle, keep up to this many instruction tokens from the front.",
    )
    ap.add_argument(
        "--strip-think-blocks",
        action="store_true",
        help="Remove <think>...</think> blocks from responses before tokenization.",
    )
    ap.add_argument(
        "--condition-override",
        default=None,
        help="Force every sample to this condition label after optional response cleanup.",
    )
    ap.add_argument("--min-response-tokens", type=int, default=1)
    ap.add_argument("--progress-interval", type=int, default=5000)
    ap.add_argument("--parquet-batch-size", type=int, default=2048)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Stop after this many input rows across all files; 0 means no cap.")
    ap.add_argument("--max-rows-per-file", type=int, default=0,
                    help="Stop after this many rows per input file; useful for huge clustered parquet sets.")
    ap.add_argument("--target-tokens", type=int, default=0,
                    help="Stop after approximately this many written tokens; 0 means no cap.")
    ap.add_argument("--normalize-composite-condition", action=argparse.BooleanOptionalAction, default=True,
                    help="Map composite labels such as synth,direct or noisy,cot to a known condition bucket.")
    ap.add_argument("--unknown-condition-policy", choices=["error", "direct", "drop"], default="direct")
    ap.add_argument("--boq", default="<|im_start|>")
    ap.add_argument("--eoq", default="<|im_end|>")
    ap.add_argument("--eoa", default="<|box_end|>")
    ap.add_argument(
        "--conditions",
        default="direct=<|object_ref_start|>,cot=<|object_ref_end|>,noisy=<|quad_start|>,synth=<|quad_end|>",
        help="Comma-separated key=token pairs mapping condition labels to vocab tokens.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_files = collect_input_files(args.train)
    print(f"Preparing {len(input_files):,} input files", flush=True)

    tok = Tokenizer.from_file(args.tokenizer)

    def _id(name: str) -> int:
        tid = tok.token_to_id(name)
        if tid is None:
            raise ValueError(f"special token {name!r} not in tokenizer vocab")
        return tid

    boq_id, eoq_id, eoa_id = _id(args.boq), _id(args.eoq), _id(args.eoa)

    cond_map: dict[str, int] = {}
    cond_mapping_tokens: dict[str, str] = {}
    for pair in args.conditions.split(","):
        k, v = pair.split("=")
        cond_map[k] = _id(v)
        cond_mapping_tokens[k] = v

    max_sample_len = args.context_size - 1

    all_tokens = array("i")
    inst_start: list[int] = []
    inst_len: list[int] = []
    resp_start: list[int] = []
    resp_len: list[int] = []

    stats = {
        "total_rows": 0,
        "kept_rows": 0,
        "truncated_rows": 0,
        "dropped_long_rows": 0,
        "dropped_response_too_long_rows": 0,
        "dropped_empty_response_rows": 0,
        "dropped_missing_field_rows": 0,
        "dropped_unknown_condition_rows": 0,
        "max_original_sample_len": 0,
        "max_written_sample_len": 0,
        "input_files": len(input_files),
    }

    for source_path, r in iter_input_rows(input_files, args.parquet_batch_size, args.max_rows_per_file):
            stats["total_rows"] += 1
            if args.max_rows and stats["total_rows"] > args.max_rows:
                break

            if "instruction" not in r or "response" not in r:
                stats["dropped_missing_field_rows"] += 1
                continue

            condition = args.condition_override or r.get("condition", "direct")
            if args.normalize_composite_condition:
                condition = normalize_condition(condition)
            if condition not in cond_map:
                if args.unknown_condition_policy == "direct":
                    condition = "direct"
                elif args.unknown_condition_policy == "drop":
                    stats["dropped_unknown_condition_rows"] += 1
                    continue
                else:
                    raise ValueError(
                        f"sample condition {condition!r} from {source_path} not in --conditions map "
                        f"(known: {sorted(cond_map)})"
                    )

            instruction_text = r["instruction"]
            response_text = r["response"]
            if instruction_text is None or response_text is None:
                stats["dropped_missing_field_rows"] += 1
                continue
            instruction_text = str(instruction_text)
            response_text = str(response_text)

            if args.strip_think_blocks:
                response_text = strip_think_blocks(response_text)

            if not response_text.strip():
                stats["dropped_empty_response_rows"] += 1
                continue

            inst_ids = tok.encode(instruction_text, add_special_tokens=False).ids
            resp_ids = tok.encode(response_text, add_special_tokens=False).ids

            if len(resp_ids) < args.min_response_tokens:
                stats["dropped_empty_response_rows"] += 1
                continue

            sample_len = 3 + len(inst_ids) + len(resp_ids) + 1
            stats["max_original_sample_len"] = max(stats["max_original_sample_len"], sample_len)

            if sample_len > max_sample_len:
                if args.overflow_policy == "error":
                    raise ValueError(
                        f"sample {stats['total_rows']} is {sample_len} tokens but "
                        f"--context-size permits {max_sample_len}; use an overflow policy."
                    )

                if args.overflow_policy == "drop":
                    stats["dropped_long_rows"] += 1
                    continue

                resp_with_eoa_len = len(resp_ids) + 1
                inst_budget = max_sample_len - resp_with_eoa_len
                inst_ids_budget = inst_budget - 3  # BOQ + condition + EOQ
                if inst_ids_budget < 0:
                    stats["dropped_response_too_long_rows"] += 1
                    continue

                inst_ids = truncate_instruction_middle(inst_ids, inst_ids_budget, args.truncate_head_tokens)
                sample_len = 3 + len(inst_ids) + len(resp_ids) + 1
                stats["truncated_rows"] += 1

            stats["max_written_sample_len"] = max(stats["max_written_sample_len"], sample_len)

            if condition not in cond_map:
                raise ValueError(
                    f"sample condition {condition!r} not in --conditions map "
                    f"(known: {sorted(cond_map)})"
                )

            i_start = len(all_tokens)
            all_tokens.append(boq_id)
            all_tokens.append(cond_map[condition])
            all_tokens.extend(inst_ids)
            all_tokens.append(eoq_id)
            inst_start.append(i_start)
            inst_len.append(len(all_tokens) - i_start)

            r_start = len(all_tokens)
            all_tokens.extend(resp_ids)
            all_tokens.append(eoa_id)
            resp_start.append(r_start)
            resp_len.append(len(all_tokens) - r_start)

            stats["kept_rows"] += 1
            if args.target_tokens and len(all_tokens) >= args.target_tokens:
                break
            if args.progress_interval and (stats["total_rows"] % args.progress_interval == 0):
                print(
                    f"Processed {stats['total_rows']:,} rows; kept={stats['kept_rows']:,} "
                    f"truncated={stats['truncated_rows']:,} dropped={stats['dropped_long_rows'] + stats['dropped_response_too_long_rows'] + stats['dropped_empty_response_rows']:,}",
                    flush=True,
                )

    print(
        f"Loaded {stats['total_rows']:,} rows from {len(input_files):,} files; "
        f"kept={stats['kept_rows']:,} truncated={stats['truncated_rows']:,}",
        flush=True,
    )

    if not inst_start:
        raise ValueError("No samples were kept after filtering.")

    tokens_np = np.array(all_tokens, dtype=np.int32)
    inst_start_np = np.array(inst_start, dtype=np.int64)
    inst_len_np = np.array(inst_len, dtype=np.int64)
    resp_start_np = np.array(resp_start, dtype=np.int64)
    resp_len_np = np.array(resp_len, dtype=np.int64)

    sample_lens = inst_len_np + resp_len_np
    max_len = int(sample_lens.max())
    if max_len >= args.context_size:
        raise ValueError(
            f"longest sample is {max_len} tokens but --context-size is "
            f"{args.context_size}; bump --context-size."
        )
    print(f"Tokens: {len(all_tokens):,}  avg sample = {sample_lens.mean():.1f}  max = {max_len}")

    np.save(out_dir / "tokens.npy", tokens_np)

    # Self-contained tokenizer copy so downstream tools (inference / convert_to_hf)
    # can find it without needing the original path.
    shutil.copyfile(args.tokenizer, out_dir / "tokenizer.json")

    tokenizer_info = {
        "tokenizer_path": str(out_dir),
        "boq": args.boq,
        "eoq": args.eoq,
        "eoa": args.eoa,
        "condition_mapping": cond_mapping_tokens,
        "vocab_size": tok.get_vocab_size(with_added_tokens=True),
    }
    with open(out_dir / "tokenizer_info.json", "w") as f:
        json.dump(tokenizer_info, f)

    # vocab_size MUST be None at the meta level; the training loop derives the
    # padded vocab from the model arch (see dataset_new.py).
    meta = {
        "tokenizer_info": tokenizer_info,
        "vocab_size": None,
        "max_seq_len": args.context_size,
        "total_length": int(sample_lens.sum()),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f)

    stats |= {
        "context_size": args.context_size,
        "max_sample_len": max_sample_len,
        "overflow_policy": args.overflow_policy,
        "strip_think_blocks": args.strip_think_blocks,
        "condition_override": args.condition_override,
        "normalize_composite_condition": args.normalize_composite_condition,
        "unknown_condition_policy": args.unknown_condition_policy,
        "max_rows": args.max_rows,
        "max_rows_per_file": args.max_rows_per_file,
        "target_tokens": args.target_tokens,
        "total_tokens": int(len(all_tokens)),
        "avg_written_sample_len": float(sample_lens.mean()),
    }
    with open(out_dir / "preprocess_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    rng = np.random.Generator(np.random.Philox(seed=args.seed))
    for epoch in range(args.epochs):
        perm = rng.permutation(len(inst_start_np))
        ep_dir = out_dir / f"epoch_{epoch}"
        ep_dir.mkdir(exist_ok=True)
        np.save(ep_dir / "inst_start.npy", inst_start_np[perm])
        np.save(ep_dir / "inst_len.npy", inst_len_np[perm])
        np.save(ep_dir / "resp_start.npy", resp_start_np[perm])
        np.save(ep_dir / "resp_len.npy", resp_len_np[perm])

    print(f"Wrote {args.epochs} epoch shuffles to {out_dir}")
    print(f"Done. Point cfg.data.path to: {out_dir}")


if __name__ == "__main__":
    main()
