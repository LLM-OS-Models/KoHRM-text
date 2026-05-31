"""Export small decoded samples from KoHRM prepared V1Dataset folders.

The real training datasets are tokenized V1Dataset directories. This helper
keeps the repo-readable sample files small while preserving the fields that
matter for understanding the training input/output split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


DEFAULT_DATASETS = {
    "stage1_fastcap": "/home/work/.data/hrm_text_prepared/koterm_hrm_cleaned_fastcap_stage1_v1",
    "stage2_full_nocap": "/home/work/.data/hrm_text_prepared/koterm_hrm_cleaned_full_nocap_v1",
    "stage3_terminal": "/home/work/.data/hrm_text_prepared/local_terminal_conversations_ctx9k_resp6k_v1",
    "stage4_korean_tool_finance": "/home/work/.data/hrm_text_prepared/koterm_korean_tool_finance_mix_v1",
    "sft_korean_legal": "/home/work/.data/hrm_text_prepared/sft_korean_legal_v1",
    "sft_bcai_finance": "/home/work/.data/hrm_text_prepared/sft_bcai_finance_kor_v1",
    "sft_toolbench": "/home/work/.data/hrm_text_prepared/sft_toolbench_v1",
    "sft_swe_glm_mix": "/home/work/.data/hrm_text_prepared/sft_swe_glm_mix_v1",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_tokenizer_path(meta: dict, dataset_dir: Path) -> Path:
    raw = Path(meta["tokenizer_info"]["tokenizer_path"])
    candidates = [
        raw,
        raw / "tokenizer.json",
        dataset_dir / "tokenizer.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"tokenizer.json not found for {dataset_dir}; tried {candidates}")


def decode_excerpt(tokenizer: Tokenizer, ids: np.ndarray, max_tokens: int) -> tuple[str, bool]:
    ids_list = ids[:max_tokens].astype(int).tolist()
    text = tokenizer.decode(ids_list, skip_special_tokens=False)
    return text, ids.shape[0] > max_tokens


def choose_indices(inst_len: np.ndarray, resp_len: np.ndarray, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(inst_len.shape[0])
    selected: list[int] = []
    for idx in order:
        i = int(idx)
        if int(inst_len[i]) <= 0 or int(resp_len[i]) <= 0:
            continue
        selected.append(i)
        if len(selected) >= count:
            break
    return sorted(selected)


def export_dataset(
    name: str,
    dataset_dir: Path,
    out_dir: Path,
    count: int,
    seed: int,
    max_instruction_tokens: int,
    max_response_tokens: int,
) -> dict:
    meta = load_json(dataset_dir / "metadata.json")
    tokenizer = Tokenizer.from_file(str(resolve_tokenizer_path(meta, dataset_dir)))
    tokens = np.load(dataset_dir / "tokens.npy", mmap_mode="r")
    ep0 = dataset_dir / "epoch_0"
    inst_start = np.load(ep0 / "inst_start.npy", mmap_mode="r")
    inst_len = np.load(ep0 / "inst_len.npy", mmap_mode="r")
    resp_start = np.load(ep0 / "resp_start.npy", mmap_mode="r")
    resp_len = np.load(ep0 / "resp_len.npy", mmap_mode="r")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.jsonl"
    indices = choose_indices(inst_len, resp_len, count=count, seed=seed)
    with out_path.open("w", encoding="utf-8") as f:
        for row_id, idx in enumerate(indices):
            i_start = int(inst_start[idx])
            i_len = int(inst_len[idx])
            r_start = int(resp_start[idx])
            r_len = int(resp_len[idx])
            inst_text, inst_truncated = decode_excerpt(tokenizer, tokens[i_start : i_start + i_len], max_instruction_tokens)
            resp_text, resp_truncated = decode_excerpt(tokenizer, tokens[r_start : r_start + r_len], max_response_tokens)
            row = {
                "dataset": name,
                "source_dataset_dir": str(dataset_dir),
                "sample_no": row_id,
                "epoch_0_index": idx,
                "instruction_len_tokens": i_len,
                "response_len_tokens": r_len,
                "instruction_truncated": inst_truncated,
                "response_truncated": resp_truncated,
                "instruction_text": inst_text,
                "response_text": resp_text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "name": name,
        "path": str(out_path),
        "source_dataset_dir": str(dataset_dir),
        "samples": len(indices),
        "source_samples": int(inst_len.shape[0]),
        "source_total_tokens": int(meta["total_length"]),
        "max_seq_len_metadata": int(meta["max_seq_len"]),
        "vocab_size": int(meta["tokenizer_info"]["vocab_size"]),
    }


def parse_dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be name=/path/to/prepared_dataset")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="samples/prepared_training_data")
    ap.add_argument("--samples-per-dataset", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--max-instruction-tokens", type=int, default=384)
    ap.add_argument("--max-response-tokens", type=int, default=256)
    ap.add_argument(
        "--dataset",
        type=parse_dataset_arg,
        action="append",
        help="Override defaults with name=/path/to/prepared_dataset. Can be repeated.",
    )
    args = ap.parse_args()

    datasets = dict(args.dataset) if args.dataset else {k: Path(v) for k, v in DEFAULT_DATASETS.items()}
    out_dir = Path(args.output)
    index = []
    for offset, (name, dataset_dir) in enumerate(datasets.items()):
        if not dataset_dir.exists():
            raise FileNotFoundError(dataset_dir)
        index.append(
            export_dataset(
                name=name,
                dataset_dir=dataset_dir,
                out_dir=out_dir,
                count=args.samples_per_dataset,
                seed=args.seed + offset,
                max_instruction_tokens=args.max_instruction_tokens,
                max_response_tokens=args.max_response_tokens,
            )
        )

    with (out_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(index)} sample files under {out_dir}")


if __name__ == "__main__":
    main()
