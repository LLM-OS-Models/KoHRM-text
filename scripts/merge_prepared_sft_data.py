"""Merge HRM-Text V1Dataset directories.

This combines already-tokenized prepared datasets without re-tokenizing raw
JSONL. It concatenates tokens.npy, offsets sample starts, and writes fresh epoch
shuffles for the merged dataset.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Prepared V1Dataset directories.")
    ap.add_argument("--output", required=True, help="Merged output directory.")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-tokenizer", action="store_true")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = [load_json(p / "metadata.json") for p in inputs]
    tokenizer_infos = [m["tokenizer_info"] for m in metas]
    ref_info = dict(tokenizer_infos[0])
    ref_info_no_path = dict(ref_info)
    ref_info_no_path.pop("tokenizer_path", None)

    for p, meta, info in zip(inputs, metas, tokenizer_infos):
        if meta["max_seq_len"] != metas[0]["max_seq_len"]:
            raise ValueError(f"{p} max_seq_len differs: {meta['max_seq_len']} != {metas[0]['max_seq_len']}")
        info_no_path = dict(info)
        info_no_path.pop("tokenizer_path", None)
        if info_no_path != ref_info_no_path:
            raise ValueError(f"{p} tokenizer_info differs from first input")

    token_arrays = [np.load(p / "tokens.npy", mmap_mode="r") for p in inputs]
    token_lengths = [int(a.shape[0]) for a in token_arrays]
    total_tokens = sum(token_lengths)

    merged_tokens = np.lib.format.open_memmap(out_dir / "tokens.npy", mode="w+", dtype=np.int32, shape=(total_tokens,))
    token_offsets = []
    cursor = 0
    for arr in token_arrays:
        token_offsets.append(cursor)
        merged_tokens[cursor: cursor + arr.shape[0]] = arr
        cursor += arr.shape[0]
    merged_tokens.flush()

    inst_start_parts = []
    inst_len_parts = []
    resp_start_parts = []
    resp_len_parts = []
    input_stats = []
    for p, offset, token_len in zip(inputs, token_offsets, token_lengths):
        ep0 = p / "epoch_0"
        inst_start = np.load(ep0 / "inst_start.npy", mmap_mode="r").astype(np.int64) + offset
        resp_start = np.load(ep0 / "resp_start.npy", mmap_mode="r").astype(np.int64) + offset
        inst_len = np.load(ep0 / "inst_len.npy", mmap_mode="r").astype(np.int64)
        resp_len = np.load(ep0 / "resp_len.npy", mmap_mode="r").astype(np.int64)

        inst_start_parts.append(inst_start)
        inst_len_parts.append(inst_len)
        resp_start_parts.append(resp_start)
        resp_len_parts.append(resp_len)
        input_stats.append({
            "path": str(p),
            "samples": int(inst_len.shape[0]),
            "tokens": int(token_len),
        })

    inst_start_np = np.concatenate(inst_start_parts)
    inst_len_np = np.concatenate(inst_len_parts)
    resp_start_np = np.concatenate(resp_start_parts)
    resp_len_np = np.concatenate(resp_len_parts)

    sample_lens = inst_len_np + resp_len_np
    rng = np.random.Generator(np.random.Philox(seed=args.seed))
    for epoch in range(args.epochs):
        perm = rng.permutation(inst_start_np.shape[0])
        ep_dir = out_dir / f"epoch_{epoch}"
        ep_dir.mkdir(exist_ok=True)
        np.save(ep_dir / "inst_start.npy", inst_start_np[perm])
        np.save(ep_dir / "inst_len.npy", inst_len_np[perm])
        np.save(ep_dir / "resp_start.npy", resp_start_np[perm])
        np.save(ep_dir / "resp_len.npy", resp_len_np[perm])

    tokenizer_info = dict(tokenizer_infos[0])
    tokenizer_info["tokenizer_path"] = str(out_dir)
    with open(out_dir / "tokenizer_info.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_info, f)

    meta = {
        "tokenizer_info": tokenizer_info,
        "vocab_size": None,
        "max_seq_len": metas[0]["max_seq_len"],
        "total_length": int(total_tokens),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    if args.copy_tokenizer:
        tokenizer_src = inputs[0] / "tokenizer.json"
        if tokenizer_src.exists():
            shutil.copyfile(tokenizer_src, out_dir / "tokenizer.json")

    stats = {
        "inputs": input_stats,
        "samples": int(inst_start_np.shape[0]),
        "tokens": int(total_tokens),
        "avg_sample_len": float(sample_lens.mean()),
        "max_sample_len": int(sample_lens.max()),
        "epochs": args.epochs,
    }
    with open(out_dir / "merge_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(
        f"Merged {len(inputs)} datasets: samples={stats['samples']:,} "
        f"tokens={stats['tokens']:,} avg={stats['avg_sample_len']:.1f} "
        f"max={stats['max_sample_len']}"
    )
    print(f"Done. Point cfg.data.path to: {out_dir}")


if __name__ == "__main__":
    main()
