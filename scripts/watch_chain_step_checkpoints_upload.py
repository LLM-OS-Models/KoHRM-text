"""Upload step checkpoints from the KoHRM continuation chain.

This watcher is intentionally independent from the training chain. It monitors
known checkpoint directories and uploads each complete `fsdp2_step_*` checkpoint
at most once. Raw FSDP2 checkpoints go to the raw checkpoint repo; converted
EMA safetensors overwrite the main model repo as the latest public artifact.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


HRM_ROOT = Path("/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text")
CKPT_ROOT = Path("/home/work/.data/hrm_text_checkpoints")
LOG_ROOT = Path("/home/work/.data/hrm_text_logs")
STAGE_ROOT = Path("/home/work/.data/hrm_text_hf_upload_stage")
TOKENIZER_PATH = Path("/home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1")
MODEL_CARD = HRM_ROOT / "docs" / "MODEL_CARD_KoHRM-Text-1.4B.md"
RAW_REPO = "LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints"
MODEL_REPO = "LLM-OS-Models/KoHRM-Text-1.4B"


STAGES = [
    ("stage3-local-terminal", CKPT_ROOT / "KoHRM-Text-1.4B-stage3-local-terminal-gbs180"),
    ("stage4-korean-tool-finance", CKPT_ROOT / "KoHRM-Text-1.4B-stage4-korean-tool-finance-gbs180"),
    ("stage1b-hrm-fastcap-repeat", CKPT_ROOT / "KoHRM-Text-1.4B-stage1b-hrm-fastcap-repeat-gbs180"),
    ("stage2b-hrm-full-nocap-extra-epoch1", CKPT_ROOT / "KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1-gbs180"),
    ("stage3b-local-terminal-repeat", CKPT_ROOT / "KoHRM-Text-1.4B-stage3b-local-terminal-repeat-gbs180"),
    ("stage4b-korean-tool-finance-repeat", CKPT_ROOT / "KoHRM-Text-1.4B-stage4b-korean-tool-finance-repeat-gbs180"),
    ("stage1c-hrm-fastcap-repeat2", CKPT_ROOT / "KoHRM-Text-1.4B-stage1c-hrm-fastcap-repeat2-gbs180"),
    ("stage2c-hrm-full-nocap-repeat2", CKPT_ROOT / "KoHRM-Text-1.4B-stage2c-hrm-full-nocap-repeat2-gbs180"),
    ("stage3c-local-terminal-repeat2", CKPT_ROOT / "KoHRM-Text-1.4B-stage3c-local-terminal-repeat2-gbs180"),
    ("stage4c-korean-tool-finance-repeat2", CKPT_ROOT / "KoHRM-Text-1.4B-stage4c-korean-tool-finance-repeat2-gbs180"),
]


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def load_env() -> dict[str, str]:
    env = os.environ.copy()
    env_path = HRM_ROOT / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"').strip("'")
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
            "NUMEXPR_MAX_THREADS": "256",
            "NUMEXPR_NUM_THREADS": "64",
            "OMP_NUM_THREADS": "32",
        }
    )
    return env


def marker_dir() -> Path:
    path = STAGE_ROOT / ".step_upload_markers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def marker(stage_name: str, step: int) -> Path:
    safe = stage_name.replace("/", "_")
    return marker_dir() / f"{safe}_step_{step}.done"


def is_complete(ckpt_path: Path, step: int) -> bool:
    tag = f"step_{step}"
    fsdp = ckpt_path / f"fsdp2_{tag}"
    if not fsdp.exists():
        return False
    if not (fsdp / ".metadata").exists():
        return False
    if len(list(fsdp.glob("__*_0.distcp"))) < 8:
        return False
    if len(list(ckpt_path.glob(f"carry_{tag}.*.pt"))) < 8:
        return False
    if not (ckpt_path / f"{tag}_info.json").exists():
        return False
    return True


def discover_steps(ckpt_path: Path) -> list[int]:
    steps = []
    if not ckpt_path.exists():
        return steps
    for item in ckpt_path.glob("fsdp2_step_*"):
        try:
            steps.append(int(item.name.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return sorted(set(steps))


def stage_raw_folder(stage_name: str, ckpt_path: Path, step: int) -> Path:
    raw_root = STAGE_ROOT / f"LLM-OS-Models__KoHRM-Text-1.4B-raw-checkpoints-{stage_name}-step{step}-{int(time.time())}"
    dest = raw_root / f"{stage_name}-step{step}"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    dest.mkdir(parents=True, exist_ok=True)
    tag = f"step_{step}"
    for name in ["all_config.yaml", "train_metadata.yaml", "latest_checkpoint.txt", f"{tag}_info.json"]:
        src = ckpt_path / name
        if src.exists():
            os.link(src, dest / name)
    shutil.copytree(ckpt_path / f"fsdp2_{tag}", dest / f"fsdp2_{tag}", copy_function=os.link)
    for carry in ckpt_path.glob(f"carry_{tag}.*.pt"):
        os.link(carry, dest / carry.name)
    (dest / "upload_manifest.json").write_text(
        f"{stage_name} {tag} raw resume checkpoint\n",
        encoding="utf-8",
    )
    return raw_root


def run_upload(stage_name: str, ckpt_path: Path, step: int) -> None:
    env = load_env()
    raw_root = stage_raw_folder(stage_name, ckpt_path, step)
    raw_log = LOG_ROOT / f"upload_{stage_name}_step{step}_raw_auto.log"
    model_log = LOG_ROOT / f"upload_{stage_name}_step{step}_converted_model_auto.log"
    out = STAGE_ROOT / f"KoHRM-Text-1.4B-converted-{stage_name}-step{step}"

    raw_cmd = [
        sys.executable,
        "scripts/upload_folder_to_hf.py",
        "--folder",
        str(raw_root),
        "--repo-id",
        RAW_REPO,
        "--repo-type",
        "model",
        "--large",
        "--num-workers",
        "4",
    ]
    log(f"uploading raw checkpoint: stage={stage_name} step={step}")
    with raw_log.open("ab") as f:
        subprocess.run(raw_cmd, cwd=HRM_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)

    convert_cmd = (
        f"rm -rf {out} && "
        f"{sys.executable} conversion/convert_to_hf.py "
        f"--ckpt_path {ckpt_path} --ckpt_step {step} --ckpt_use_ema true "
        f"--out_dir {out} --tokenizer_path {TOKENIZER_PATH} --device cpu && "
        f"cp {MODEL_CARD} {out}/README.md && "
        f"{sys.executable} scripts/upload_folder_to_hf.py "
        f"--folder {out} --repo-id {MODEL_REPO} --repo-type model --large --num-workers 4"
    )
    log(f"uploading converted model: stage={stage_name} step={step}")
    with model_log.open("ab") as f:
        subprocess.run(["bash", "-lc", convert_cmd], cwd=HRM_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)

    marker(stage_name, step).write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()), encoding="utf-8")
    log(f"uploaded step checkpoint: stage={stage_name} step={step}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-step", type=int, default=190000)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    marker_dir()
    log(f"step checkpoint upload watcher started min_step={args.min_step}")
    while True:
        for stage_name, ckpt_path in STAGES:
            for step in discover_steps(ckpt_path):
                if step < args.min_step:
                    continue
                if marker(stage_name, step).exists():
                    continue
                if not is_complete(ckpt_path, step):
                    continue
                try:
                    run_upload(stage_name, ckpt_path, step)
                except Exception as exc:
                    log(f"upload failed: stage={stage_name} step={step} error={exc!r}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
