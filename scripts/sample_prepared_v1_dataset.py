"""Sample a compact subset from an HRM-Text V1Dataset.

The input dataset is already tokenized. This script selects examples from
epoch_0, copies only the selected instruction/response token spans into a new
compact tokens.npy, and writes fresh epoch shuffles. It is intended for small
SFT/LoRA candidate sets derived from larger pretraining-prepared datasets.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input prepared V1Dataset directory.")
    ap.add_argument("--output", required=True, help="Output prepared V1Dataset directory.")
    ap.add_argument("--target-tokens", type=int, default=0, help="Approximate selected token cap. 0 means no cap.")
    ap.add_argument("--max-samples", type=int, default=0, help="Maximum selected samples. 0 means no cap.")
    ap.add_argument("--epochs", type=int, required=True, help="Number of epoch shuffles to write.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-response-tokens", type=int, default=1)
    ap.add_argument("--max-sample-len", type=int, default=0, help="Drop samples longer than this token length. 0 disables.")
    ap.add_argument("--copy-tokenizer", action="store_true")
    ap.add_argument("--source-name", default=None)
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_json(in_dir / "metadata.json")
    ep0 = in_dir / "epoch_0"
    tokens = np.load(in_dir / "tokens.npy", mmap_mode="r")
    inst_start = np.load(ep0 / "inst_start.npy", mmap_mode="r")
    inst_len = np.load(ep0 / "inst_len.npy", mmap_mode="r")
    resp_start = np.load(ep0 / "resp_start.npy", mmap_mode="r")
    resp_len = np.load(ep0 / "resp_len.npy", mmap_mode="r")

    sample_lens = inst_len.astype(np.int64) + resp_len.astype(np.int64)
    rng = np.random.Generator(np.random.Philox(seed=args.seed))
    order = rng.permutation(inst_len.shape[0])

    selected: list[int] = []
    selected_tokens = 0
    dropped_short_response = 0
    dropped_long_sample = 0
    for idx in order:
        idx_int = int(idx)
        if resp_len[idx_int] < args.min_response_tokens:
            dropped_short_response += 1
            continue
        length = int(sample_lens[idx_int])
        if args.max_sample_len and length > args.max_sample_len:
            dropped_long_sample += 1
            continue
        selected.append(idx_int)
        selected_tokens += length
        if args.max_samples and len(selected) >= args.max_samples:
            break
        if args.target_tokens and selected_tokens >= args.target_tokens:
            break

    if not selected:
        raise ValueError("no samples selected")

    out_tokens = np.lib.format.open_memmap(
        out_dir / "tokens.npy",
        mode="w+",
        dtype=np.int32,
        shape=(selected_tokens,),
    )
    out_inst_start = np.empty(len(selected), dtype=np.int64)
    out_inst_len = np.empty(len(selected), dtype=np.int64)
    out_resp_start = np.empty(len(selected), dtype=np.int64)
    out_resp_len = np.empty(len(selected), dtype=np.int64)

    cursor = 0
    for out_i, src_i in enumerate(selected):
        i_start = int(inst_start[src_i])
        i_len = int(inst_len[src_i])
        r_start = int(resp_start[src_i])
        r_len = int(resp_len[src_i])

        out_inst_start[out_i] = cursor
        out_inst_len[out_i] = i_len
        out_tokens[cursor: cursor + i_len] = tokens[i_start: i_start + i_len]
        cursor += i_len

        out_resp_start[out_i] = cursor
        out_resp_len[out_i] = r_len
        out_tokens[cursor: cursor + r_len] = tokens[r_start: r_start + r_len]
        cursor += r_len
    out_tokens.flush()

    tokenizer_info = dict(meta["tokenizer_info"])
    tokenizer_info["tokenizer_path"] = str(out_dir)
    with (out_dir / "tokenizer_info.json").open("w", encoding="utf-8") as f:
        json.dump(tokenizer_info, f)

    out_meta = {
        "tokenizer_info": tokenizer_info,
        "vocab_size": None,
        "max_seq_len": meta["max_seq_len"],
        "total_length": int(selected_tokens),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(out_meta, f)

    if args.copy_tokenizer and (in_dir / "tokenizer.json").exists():
        shutil.copyfile(in_dir / "tokenizer.json", out_dir / "tokenizer.json")

    shuffle_rng = np.random.Generator(np.random.Philox(seed=args.seed + 1009))
    for epoch in range(args.epochs):
        perm = shuffle_rng.permutation(len(selected))
        ep_dir = out_dir / f"epoch_{epoch}"
        ep_dir.mkdir(exist_ok=True)
        np.save(ep_dir / "inst_start.npy", out_inst_start[perm])
        np.save(ep_dir / "inst_len.npy", out_inst_len[perm])
        np.save(ep_dir / "resp_start.npy", out_resp_start[perm])
        np.save(ep_dir / "resp_len.npy", out_resp_len[perm])

    stats = {
        "input": str(in_dir),
        "source_name": args.source_name or in_dir.name,
        "source_samples": int(inst_len.shape[0]),
        "source_tokens": int(meta["total_length"]),
        "selected_samples": int(len(selected)),
        "selected_tokens": int(selected_tokens),
        "avg_sample_len": float((out_inst_len + out_resp_len).mean()),
        "max_sample_len": int((out_inst_len + out_resp_len).max()),
        "target_tokens": int(args.target_tokens),
        "max_samples": int(args.max_samples),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "dropped_short_response": int(dropped_short_response),
        "dropped_long_sample": int(dropped_long_sample),
    }
    with (out_dir / "sample_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"Sampled {stats['selected_samples']:,} examples, "
        f"{stats['selected_tokens']:,} tokens from {in_dir} -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
