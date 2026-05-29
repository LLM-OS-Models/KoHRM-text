#!/usr/bin/env python3
"""Upload KoHRM SFT/LoRA prepared V1Dataset folders to Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DEFAULT_REPO_ID = "LLM-OS-Models/KoHRM-Text-1.4B-sft-lora-data"
DEFAULT_DATA_ROOT = Path("/home/work/.data/hrm_text_prepared")
DEFAULT_README = Path("docs/HF_DATASET_CARD_KoHRM-Text-SFT-LoRA-Data.md")

DATASETS = [
    "kohrm_sft_behavior_mini_v1",
    "kohrm_sft_terminal_tool_core_v1",
    "kohrm_sft_korean_domain_core_v1",
    "kohrm_sft_behavior_core_v1",
    "kohrm_sft_comp_terminal_80m_v1",
    "kohrm_sft_comp_toolbench_30m_v1",
    "kohrm_sft_comp_swe_zero_30m_v1",
    "kohrm_sft_comp_glm_reasoning_20m_v1",
    "kohrm_sft_comp_agent_reasoning_25m_v1",
    "kohrm_sft_comp_korean_legal_50m_v1",
    "kohrm_sft_comp_finance_50m_v1",
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--readme", type=Path, default=DEFAULT_README)
    ap.add_argument("--env-file", type=Path, default=Path("/home/work/.projects/LLM-OS-Models/Terminal/.env"))
    ap.add_argument("--only", nargs="*", default=None, help="Upload only selected dataset folder names.")
    ap.add_argument("--skip-existing", action="store_true", help="Reserved for future manifest checks.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Put it in .env or export it before running.")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    selected = args.only or DATASETS
    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="dataset", token=token, exist_ok=True)

    if args.readme.exists():
        print(f"[upload] README -> {args.repo_id}/README.md", flush=True)
        api.upload_file(
            path_or_fileobj=str(args.readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Update SFT LoRA dataset card",
        )

    for name in selected:
        local_dir = args.data_root / name
        if not local_dir.exists():
            raise FileNotFoundError(local_dir)
        size_gib = folder_size(local_dir) / 2**30
        print(f"[upload] {name} ({size_gib:.2f} GiB) -> {args.repo_id}/{name}/", flush=True)
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Upload {name}",
            ignore_patterns=["*.lock", "__pycache__/*"],
        )

    info = api.dataset_info(args.repo_id)
    print(f"[upload] done: https://huggingface.co/datasets/{args.repo_id}", flush=True)
    print(f"[upload] sha: {info.sha}", flush=True)


if __name__ == "__main__":
    main()

