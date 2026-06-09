#!/usr/bin/env python3
"""Upload the KoHRM CPU runtime pack to Hugging Face.

This uploads code and documentation only. The runtime downloads the original
weights from ``LLM-OS-Models/KoHRM-Text-1.4B`` at execution time.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DEFAULT_REPO = "LLM-OS-Models/KoHRM-Text-1.4B-CPU-Runtime"


def read_hf_token() -> str | None:
    candidates = [
        PROJECT_ROOT / ".env",
        ROOT / ".env",
        Path.cwd() / ".env",
        Path.home() / ".cache" / "huggingface" / "token",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.name == "token":
            token = path.read_text(encoding="utf-8").strip()
            return token or None
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key.split(None, 1)[1]
            if key in {"HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                token = value.strip().strip('"').strip("'")
                return token or None
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def build_readme() -> str:
    body = (ROOT / "docs" / "KOHRM_CPU_RUNTIME_PACK_2026-06-09.md").read_text(encoding="utf-8")
    header = """---
license: apache-2.0
base_model: LLM-OS-Models/KoHRM-Text-1.4B
base_model_relation: quantized
library_name: pytorch
tags:
- kohrm
- hrm-text
- cpu
- int8
- int4
- korean
- terminal
---

# KoHRM-Text-1.4B CPU Runtime

This repository contains a CPU-oriented inference runtime for
`LLM-OS-Models/KoHRM-Text-1.4B`.

It does not duplicate the original model weights. The runtime downloads the
base model from Hugging Face and applies CPU quantization at load time.

"""
    return header + body


def stage_files(stage: Path) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "inference").mkdir(parents=True)
    (stage / "notebooks").mkdir(parents=True)
    shutil.copy2(ROOT / "inference" / "kohrm_cpu_runtime.py", stage / "inference" / "kohrm_cpu_runtime.py")
    shutil.copy2(ROOT / "inference" / "requirements-cpu.txt", stage / "inference" / "requirements-cpu.txt")
    shutil.copy2(ROOT / "notebooks" / "kohrm_colab_generate.py", stage / "notebooks" / "kohrm_colab_generate.py")
    (stage / "README.md").write_text(build_readme(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload KoHRM CPU runtime pack to Hugging Face.")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--stage", default="/home/work/.data/hf_upload_stage/kohrm_cpu_runtime_pack")
    args = ap.parse_args()

    token = read_hf_token()
    if not token:
        raise SystemExit("HF token not found in .env or HF cache")

    stage = Path(args.stage)
    stage_files(stage)

    create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True, token=token)
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(stage),
        commit_message="Add KoHRM CPU quantized runtime pack",
    )
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
