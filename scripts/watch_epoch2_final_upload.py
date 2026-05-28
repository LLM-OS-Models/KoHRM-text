"""Upload a clearly named KoHRM epoch-2 final artifact after stage4b finishes.

Epoch 2 in the current KoHRM schedule ends after:

    stage4b-korean-tool-finance-repeat

The normal upload watcher keeps the rolling latest model repo updated, but that
repo will later be overwritten by epoch-3 checkpoints. This watcher creates an
explicit fixed Hugging Face artifact named `KoHRM-Text-1.4B-Epoch2` and also
uploads a raw resume checkpoint folder with `epoch2-final` in its path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi


HRM_ROOT = Path("/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text")
CKPT = Path("/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage4b-korean-tool-finance-repeat-gbs180")
LOG_ROOT = Path("/home/work/.data/hrm_text_logs")
STAGE_ROOT = Path("/home/work/.data/hrm_text_hf_upload_stage")
TOKENIZER_PATH = Path("/home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1")
MODEL_CARD = HRM_ROOT / "docs" / "MODEL_CARD_KoHRM-Text-1.4B.md"

RUN_NAME = "KoHRM-Text-1.4B-stage4b-korean-tool-finance-repeat"
RAW_REPO = "LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints"
EPOCH2_MODEL_REPO = "LLM-OS-Models/KoHRM-Text-1.4B-Epoch2"
MAIN_MODEL_REPO = "LLM-OS-Models/KoHRM-Text-1.4B"
MAIN_MODEL_BRANCH = "epoch-2-final"


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env() -> dict[str, str]:
    out = os.environ.copy()
    for path in [HRM_ROOT / ".env", HRM_ROOT.parent / ".env", Path.home() / ".env"]:
        out.update(parse_env_file(path))
    out.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
            "NUMEXPR_MAX_THREADS": "256",
            "NUMEXPR_NUM_THREADS": "64",
            "OMP_NUM_THREADS": "32",
        }
    )
    return out


def token() -> str:
    e = env()
    value = e.get("HF_TOKEN") or e.get("HUGGINGFACE_TOKEN")
    if not value:
        raise RuntimeError("HF_TOKEN is not available")
    return value


def marker_path() -> Path:
    path = STAGE_ROOT / ".epoch_final_markers"
    path.mkdir(parents=True, exist_ok=True)
    return path / "epoch2-final-stage4b-uploaded.done"


def checkpoint_ready() -> bool:
    return (
        (CKPT / "fsdp2_epoch_1").exists()
        and (CKPT / "fsdp2_epoch_1" / ".metadata").exists()
        and len(list((CKPT / "fsdp2_epoch_1").glob("__*_0.distcp"))) >= 8
        and len(list(CKPT.glob("carry_epoch_1.*.pt"))) >= 8
        and (CKPT / "epoch_1_info.json").exists()
    )


def epoch_global_step() -> int:
    info = json.loads((CKPT / "epoch_1_info.json").read_text(encoding="utf-8"))
    return int(info["global_step"])


def live_training_pids() -> list[int]:
    out = subprocess.run(
        ["ps", "-eo", "pid=,stat=,cmd="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    ).stdout
    pids: list[int] = []
    marker = f"+run_name={RUN_NAME}"
    for line in out.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_s, stat, cmd = parts
        if stat.startswith("Z") or marker not in cmd:
            continue
        if "torchrun" in cmd or "pretrain.py" in cmd:
            pids.append(int(pid_s))
    return pids


def wait_for_epoch2_final() -> int:
    log(f"waiting for epoch2 final checkpoint under {CKPT}")
    while True:
        if checkpoint_ready():
            step = epoch_global_step()
            pids = live_training_pids()
            if not pids:
                log(f"epoch2 final checkpoint ready: global_step={step}")
                return step
            log(f"epoch2 checkpoint exists; waiting for stage4b process exit: pids={pids}")
        time.sleep(60)


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_raw(step: int) -> Path:
    root = STAGE_ROOT / f"KoHRM-Text-1.4B-Epoch2-raw-final-{step}-{int(time.time())}"
    dest = root / f"epoch2-final-stage4b-korean-tool-finance-repeat-globalstep-{step}"
    if root.exists():
        shutil.rmtree(root)
    dest.mkdir(parents=True, exist_ok=True)

    for name in ["all_config.yaml", "train_metadata.yaml", "latest_checkpoint.txt", "epoch_1_info.json"]:
        src = CKPT / name
        if src.exists():
            link_or_copy(src, dest / name)
    shutil.copytree(CKPT / "fsdp2_epoch_1", dest / "fsdp2_epoch_1", copy_function=os.link)
    for carry in CKPT.glob("carry_epoch_1.*.pt"):
        link_or_copy(carry, dest / carry.name)

    (dest / "README.md").write_text(
        "\n".join(
            [
                "# KoHRM-Text-1.4B Epoch 2 Final Raw Checkpoint",
                "",
                f"- Epoch label: `epoch2-final`",
                f"- Final stage: `stage4b-korean-tool-finance-repeat`",
                f"- Global step: `{step}`",
                f"- Checkpoint tag: `epoch_1` inside the stage4b checkpoint directory",
                "- Purpose: fixed raw FSDP2 resume checkpoint for the end of KoHRM epoch 2.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (dest / "upload_manifest.json").write_text(
        json.dumps(
            {
                "artifact": "KoHRM-Text-1.4B Epoch 2 Final Raw Checkpoint",
                "epoch_label": "epoch2-final",
                "stage": "stage4b-korean-tool-finance-repeat",
                "global_step": step,
                "checkpoint_tag": "epoch_1",
                "source_checkpoint": str(CKPT),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def upload_large(folder: Path, repo_id: str, repo_type: str = "model", revision: str | None = None) -> None:
    api = HfApi(token=token())
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    if revision is not None:
        api.create_branch(repo_id=repo_id, repo_type=repo_type, branch=revision, revision="main", exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        folder_path=folder,
        num_workers=4,
        print_report=True,
        print_report_every=120,
    )


def convert_and_upload(step: int) -> Path:
    out = STAGE_ROOT / f"KoHRM-Text-1.4B-Epoch2-converted-final-globalstep-{step}"
    if out.exists():
        shutil.rmtree(out)
    cmd = [
        sys.executable,
        "conversion/convert_to_hf.py",
        "--ckpt_path",
        str(CKPT),
        "--ckpt_epoch",
        "1",
        "--ckpt_use_ema",
        "true",
        "--out_dir",
        str(out),
        "--tokenizer_path",
        str(TOKENIZER_PATH),
        "--device",
        "cpu",
    ]
    log("converting epoch2 final checkpoint to HF safetensors")
    subprocess.run(cmd, cwd=HRM_ROOT, env=env(), check=True)

    base_card = MODEL_CARD.read_text(encoding="utf-8")
    banner = "\n".join(
        [
            "---",
            "license: apache-2.0",
            "language:",
            "- ko",
            "- en",
            "tags:",
            "- kohrm",
            "- epoch2-final",
            "- hrm-text",
            "- korean",
            "- terminal",
            "- tool-use",
            "library_name: transformers",
            "---",
            "",
            "# KoHRM-Text-1.4B Epoch 2 Final",
            "",
            "This repository/revision is the fixed **Epoch 2 Final** export of KoHRM-Text-1.4B.",
            "",
            f"- Epoch label: `epoch2-final`",
            f"- Final stage: `stage4b-korean-tool-finance-repeat`",
            f"- Global step: `{step}`",
            "- Export: EMA weights converted to `model.safetensors`",
            f"- Rolling latest repo: `https://huggingface.co/{MAIN_MODEL_REPO}`",
            f"- Raw resume checkpoints: `https://huggingface.co/{RAW_REPO}`",
            "",
            "한국어: 이 저장소/리비전은 KoHRM-Text-1.4B의 **에폭 2 완료본**을 고정 저장한 것입니다. 이후 학습이 계속 진행되어 rolling latest가 바뀌어도 이 artifact는 epoch 2 완료 지점을 가리킵니다.",
            "",
            "---",
            "",
        ]
    )
    (out / "README.md").write_text(banner + base_card, encoding="utf-8")
    (out / "EPOCH_2_FINAL.json").write_text(
        json.dumps(
            {
                "artifact": "KoHRM-Text-1.4B Epoch 2 Final",
                "epoch_label": "epoch2-final",
                "stage": "stage4b-korean-tool-finance-repeat",
                "global_step": step,
                "checkpoint_tag": "epoch_1",
                "source_checkpoint": str(CKPT),
                "main_model_repo_revision": MAIN_MODEL_BRANCH,
                "pinned_model_repo": EPOCH2_MODEL_REPO,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log(f"uploading epoch2 final converted model to {EPOCH2_MODEL_REPO}")
    upload_large(out, EPOCH2_MODEL_REPO)

    log(f"uploading epoch2 final converted model to {MAIN_MODEL_REPO}@{MAIN_MODEL_BRANCH}")
    upload_large(out, MAIN_MODEL_REPO, revision=MAIN_MODEL_BRANCH)

    api = HfApi(token=token())
    tag = f"epoch-2-final-step-{step}"
    api.create_tag(
        repo_id=MAIN_MODEL_REPO,
        repo_type="model",
        tag=tag,
        tag_message=f"KoHRM-Text-1.4B epoch 2 final at global step {step}",
        revision=MAIN_MODEL_BRANCH,
        exist_ok=True,
    )
    log(f"created/kept tag {MAIN_MODEL_REPO}@{tag}")
    return out


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    marker = marker_path()
    if marker.exists():
        log(f"epoch2 final upload already marked done: {marker}")
        return

    step = wait_for_epoch2_final()
    raw_root = stage_raw(step)
    log(f"uploading epoch2 final raw checkpoint to {RAW_REPO}")
    upload_large(raw_root, RAW_REPO)
    out = convert_and_upload(step)

    marker.write_text(
        json.dumps(
            {
                "epoch_label": "epoch2-final",
                "global_step": step,
                "raw_stage": str(raw_root),
                "converted_stage": str(out),
                "pinned_model_repo": EPOCH2_MODEL_REPO,
                "main_model_branch": MAIN_MODEL_BRANCH,
                "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"epoch2 final upload complete; marker={marker}")


if __name__ == "__main__":
    main()
