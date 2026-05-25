"""Build additional HRM sampled epochs without copying the huge token array.

Upstream data_io sampling is deterministic across epochs. This helper recreates
the same sampling stream, discards the already trained logical epochs, and writes
the following epochs into a new V1Dataset directory that hardlinks the existing
`tokens.npy`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pydantic
import yaml
from tqdm import tqdm


LongContextMode = Literal["drop", "truncate"]


class PrefixConfig(pydantic.BaseModel):
    max_per_file: Optional[int] = None
    long_context: LongContextMode = "truncate"
    repeat: int = 1


@dataclass
class TaskIndices:
    inst_start: np.ndarray
    inst_len: np.ndarray
    resp_start: np.ndarray
    resp_len: np.ndarray


@dataclass
class Task:
    name: str
    indices: TaskIndices
    prefix_config: PrefixConfig
    mmap_base_offset: int = 0
    perm: Optional[np.ndarray] = None
    perm_cursor: int = 0


def load_prefix_configs(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def prefix_config_for(name: str, prefix_configs: list[dict]) -> PrefixConfig:
    for item in prefix_configs:
        if name.startswith(item["prefix"]):
            return PrefixConfig(**item)
    return PrefixConfig()


def truncate_and_filter(task: Task, context_size: int, min_resp_length: int) -> None:
    keep_mask = task.indices.resp_len >= min_resp_length
    allowed_resp = context_size - np.minimum(task.indices.inst_len, context_size)
    if task.prefix_config.long_context == "truncate":
        keep_mask &= allowed_resp >= 1
        task.indices.resp_len = np.minimum(task.indices.resp_len, allowed_resp)
    else:
        keep_mask &= task.indices.resp_len <= allowed_resp

    for field in fields(TaskIndices):
        setattr(task.indices, field.name, getattr(task.indices, field.name)[keep_mask])


def load_tasks(tokenized_path: Path, prefix_config_path: Path, context_size: int, min_resp_length: int) -> list[Task]:
    prefix_configs = load_prefix_configs(prefix_config_path)
    tasks: list[Task] = []
    offset = 0
    for dataset_dir in tqdm(sorted(tokenized_path.iterdir()), desc="Reading task indices"):
        if not dataset_dir.is_dir():
            continue
        task = Task(
            name=dataset_dir.name,
            indices=TaskIndices(
                inst_start=np.load(dataset_dir / "inst_start.npy", mmap_mode="r"),
                inst_len=np.load(dataset_dir / "inst_len.npy", mmap_mode="r"),
                resp_start=np.load(dataset_dir / "resp_start.npy", mmap_mode="r"),
                resp_len=np.load(dataset_dir / "resp_len.npy", mmap_mode="r"),
            ),
            prefix_config=prefix_config_for(dataset_dir.name, prefix_configs),
            mmap_base_offset=offset,
        )
        offset += int(np.load(dataset_dir / "tokens.npy", mmap_mode="r").shape[0])
        truncate_and_filter(task, context_size, min_resp_length)
        tasks.append(task)
    return tasks


def sample_one_epoch(tasks: list[Task], rng: np.random.Generator) -> TaskIndices:
    total_rows = 0
    for task in tasks:
        rows = min(task.prefix_config.max_per_file, len(task.indices.inst_start)) if task.prefix_config.max_per_file is not None else len(task.indices.inst_start)
        total_rows += rows * task.prefix_config.repeat

    out = TaskIndices(**{field.name: np.empty((total_rows,), dtype=np.int64) for field in fields(TaskIndices)})
    cursor = 0
    for task in tasks:
        rows_to_sample = min(task.prefix_config.max_per_file, len(task.indices.inst_start)) if task.prefix_config.max_per_file is not None else len(task.indices.inst_start)
        rows_to_sample *= task.prefix_config.repeat
        rows_fetched = 0
        while rows_fetched < rows_to_sample:
            if task.perm is None or task.perm_cursor >= len(task.perm):
                task.perm = rng.permutation(len(task.indices.inst_len))
                task.perm_cursor = 0
            take = min(len(task.perm) - task.perm_cursor, rows_to_sample - rows_fetched)
            indices = task.perm[task.perm_cursor : task.perm_cursor + take]
            task.perm_cursor += take
            rows_fetched += take

            out.inst_start[cursor : cursor + take] = task.indices.inst_start[indices] + task.mmap_base_offset
            out.inst_len[cursor : cursor + take] = task.indices.inst_len[indices]
            out.resp_start[cursor : cursor + take] = task.indices.resp_start[indices] + task.mmap_base_offset
            out.resp_len[cursor : cursor + take] = task.indices.resp_len[indices]
            cursor += take

    perm = rng.permutation(cursor)
    return TaskIndices(**{field.name: getattr(out, field.name)[perm] for field in fields(TaskIndices)})


def write_epoch(epoch_dir: Path, indices: TaskIndices) -> int:
    epoch_dir.mkdir(parents=True, exist_ok=True)
    total_tokens = int(np.sum(indices.inst_len) + np.sum(indices.resp_len))
    for field in fields(TaskIndices):
        np.save(epoch_dir / f"{field.name}.npy", getattr(indices, field.name))
    return total_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenized-path", required=True)
    ap.add_argument("--base-prepared", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prefix-config-path", required=True)
    ap.add_argument("--discard-epochs", type=int, default=1)
    ap.add_argument("--write-epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context-size", type=int, default=4097)
    ap.add_argument("--min-resp-length", type=int, default=2)
    args = ap.parse_args()

    tokenized_path = Path(args.tokenized_path)
    base_prepared = Path(args.base_prepared)
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    os.link(base_prepared / "tokens.npy", output / "tokens.npy")
    if (base_prepared / "tokenizer_info.json").exists():
        os.link(base_prepared / "tokenizer_info.json", output / "tokenizer_info.json")

    tasks = load_tasks(tokenized_path, Path(args.prefix_config_path), args.context_size, args.min_resp_length)
    rng = np.random.Generator(np.random.Philox(seed=args.seed))

    for _ in tqdm(range(args.discard_epochs), desc="Discarding logical epochs"):
        sample_one_epoch(tasks, rng)

    totals: list[int] = []
    for out_epoch in tqdm(range(args.write_epochs), desc="Writing extra epochs"):
        indices = sample_one_epoch(tasks, rng)
        totals.append(write_epoch(output / f"epoch_{out_epoch}", indices))

    metadata = json.loads((base_prepared / "metadata.json").read_text(encoding="utf-8"))
    metadata["total_length"] = int(round(sum(totals) / max(1, len(totals))))
    metadata["kohrm_extra_epochs"] = {
        "source_base_prepared": str(base_prepared),
        "source_tokenized_path": str(tokenized_path),
        "discarded_logical_epochs": args.discard_epochs,
        "written_logical_epochs": list(range(args.discard_epochs, args.discard_epochs + args.write_epochs)),
        "per_epoch_total_lengths": totals,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata["kohrm_extra_epochs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
