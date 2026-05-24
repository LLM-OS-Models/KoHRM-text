"""Wait for the current KoHRM stage-1 run, then start the next continuations.

This is intentionally conservative:
- it does not delete or move canonical prepared datasets;
- it waits for the stage-1 final epoch checkpoint to become stable;
- it trains large prepared datasets as separate stages instead of merging the
  very large HRM full/no-cap token backing file;
- it merges only the smaller Korean/tool/finance datasets for the final
  continuation stage.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


HRM_ROOT = Path("/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text")
DATA_ROOT = Path("/home/work/.data/hrm_text_prepared")
CKPT_ROOT = Path("/home/work/.data/hrm_text_checkpoints")
LOG_ROOT = Path("/home/work/.data/hrm_text_logs")
UPLOAD_STAGE_ROOT = Path("/home/work/.data/hrm_text_hf_upload_stage")
RAW_CHECKPOINT_REPO = "LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints"
MODEL_REPO = "LLM-OS-Models/KoHRM-Text-1.4B"
TOKENIZER_PATH = Path("/home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1")

GLOBAL_BATCH = 180_224
TOTAL_STEPS_OVERRIDE = 238_117

CURRENT_STAGE = CKPT_ROOT / "KoHRM-Text-1.4B-stage1-hrm-fastcap-gbs180"

SMALL_MIX = DATA_ROOT / "koterm_korean_tool_finance_mix_v1"
SMALL_INPUTS = [
    DATA_ROOT / "sft_bcai_finance_kor_v1",
    DATA_ROOT / "korean_legal_tasks_full_v1",
    DATA_ROOT / "kowiki_raw_full_v1",
    DATA_ROOT / "korean_legal_raw_full_v1",
    DATA_ROOT / "korean_admrule_precedent_raw_full_v1",
    DATA_ROOT / "sft_toolbench_v1",
    DATA_ROOT / "sft_swe_zero_v1",
    DATA_ROOT / "sft_glm_reasoning_v1",
    DATA_ROOT / "hf_extra_reasoning_agent_mm_v1",
]

STAGES = [
    {
        "name": "stage2-hrm-full-nocap",
        "data": DATA_ROOT / "koterm_hrm_cleaned_full_nocap_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage2-hrm-full-nocap-gbs180",
    },
    {
        "name": "stage3-local-terminal",
        "data": DATA_ROOT / "local_terminal_conversations_ctx9k_resp6k_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage3-local-terminal-gbs180",
    },
    {
        "name": "stage4-korean-tool-finance",
        "data": SMALL_MIX,
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage4-korean-tool-finance-gbs180",
    },
]


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def file_tree_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def wait_stable(path: Path, seconds: int = 180) -> None:
    last = -1
    stable_since = time.monotonic()
    while True:
        current = file_tree_size(path)
        if current == last:
            if time.monotonic() - stable_since >= seconds:
                return
        else:
            last = current
            stable_since = time.monotonic()
        time.sleep(30)


def wait_final_checkpoint(root: Path) -> None:
    log(f"waiting for final stage-1 checkpoint under {root}")
    while True:
        epoch = root / "fsdp2_epoch_1"
        carries = sorted(root.glob("carry_epoch_1.*.pt"))
        if epoch.exists() and len(carries) >= 8:
            log("stage-1 epoch checkpoint detected; waiting for stable files")
            wait_stable(root)
            log("stage-1 checkpoint is stable")
            return
        time.sleep(120)


def metadata_tokens(path: Path) -> int:
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return int(meta["total_length"])


def validate_dataset(path: Path) -> None:
    missing = []
    for rel in [
        "metadata.json",
        "tokens.npy",
        "epoch_0/inst_start.npy",
        "epoch_0/inst_len.npy",
        "epoch_0/resp_start.npy",
        "epoch_0/resp_len.npy",
    ]:
        if not (path / rel).exists():
            missing.append(rel)
    if missing:
        raise FileNotFoundError(f"{path} missing {missing}")


def run_logged(cmd: list[str], log_path: Path, cwd: Path = HRM_ROOT) -> None:
    log(f"running: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as f:
        f.write((f"\n\n===== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} =====\n").encode())
        proc = subprocess.run(cmd, cwd=cwd, env=training_env(), stdout=f, stderr=subprocess.STDOUT)
        f.write((f"\n===== exit {proc.returncode} =====\n").encode())
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}")


def ensure_small_mix() -> None:
    if (SMALL_MIX / "metadata.json").exists() and (SMALL_MIX / "tokens.npy").exists():
        log(f"small mix already exists: {SMALL_MIX}")
        return
    for path in SMALL_INPUTS:
        validate_dataset(path)
    if SMALL_MIX.exists():
        shutil.rmtree(SMALL_MIX)
    cmd = [
        "python",
        "scripts/merge_prepared_sft_data.py",
        "--inputs",
        *[str(p) for p in SMALL_INPUTS],
        "--output",
        str(SMALL_MIX),
        "--epochs",
        "1",
        "--seed",
        "20260524",
        "--copy-tokenizer",
    ]
    run_logged(cmd, LOG_ROOT / "koterm_korean_tool_finance_mix_v1_merge.log")


def training_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "WANDB_MODE": "offline",
            "WANDB_DIR": "/home/work/.data/wandb",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NCCL_DEBUG": "WARN",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        }
    )
    return env


def train_stage(stage: dict[str, Path | str], resume_from: Path, resume_step_offset: int) -> tuple[Path, int]:
    data_path = Path(stage["data"])
    checkpoint_path = Path(stage["checkpoint"])
    name = str(stage["name"])
    validate_dataset(data_path)
    tokens = metadata_tokens(data_path)
    steps = tokens // GLOBAL_BATCH
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        "pretrain.py",
        "arch/size@arch=XL",
        f"data.path={data_path}",
        f"resume_from={resume_from}",
        f"+checkpoint_path={checkpoint_path}",
        "+project_name=KoHRM-Text",
        f"+run_name=KoHRM-Text-1.4B-{name}",
        "epochs=1",
        f"global_batch_size={GLOBAL_BATCH}",
        "lr_warmup_steps=2000",
        f"resume_step_offset={resume_step_offset}",
        f"total_steps_override={TOTAL_STEPS_OVERRIDE}",
        "+log_interval=5",
        "checkpoint_step_interval=10000",
        "checkpoint_keep_last=2",
        "checkpoint_interval=1",
    ]
    run_logged(cmd, LOG_ROOT / f"KoHRM-Text-1.4B-{name}.log")
    return checkpoint_path, steps


def checkpoint_order(checkpoint_path: Path, tag: str) -> tuple[int, int]:
    info_path = checkpoint_path / f"{tag}_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            return int(info.get("global_step", 0)), 1 if tag.startswith("epoch_") else 0
        except Exception:
            pass
    try:
        return int(tag.rsplit("_", 1)[-1]), 1 if tag.startswith("epoch_") else 0
    except ValueError:
        return 0, 0


def stage_latest_checkpoints(checkpoint_path: Path, stage_name: str, keep_last: int = 2) -> Path:
    tags = []
    for path in checkpoint_path.glob("fsdp2_*"):
        if path.is_dir():
            tags.append(path.name[len("fsdp2_"):])
    tags = sorted(set(tags), key=lambda tag: checkpoint_order(checkpoint_path, tag))[-keep_last:]
    if not tags:
        raise RuntimeError(f"no checkpoints found in {checkpoint_path}")

    stage_root = UPLOAD_STAGE_ROOT / f"KoHRM-Text-1.4B-raw-checkpoints-{stage_name}-latest"
    dest = stage_root / stage_name
    if stage_root.exists():
        shutil.rmtree(stage_root)
    dest.mkdir(parents=True, exist_ok=True)

    for name in ["all_config.yaml", "train_metadata.yaml", "latest_checkpoint.txt"]:
        src = checkpoint_path / name
        if src.exists():
            os.link(src, dest / name)

    for tag in tags:
        shutil.copytree(checkpoint_path / f"fsdp2_{tag}", dest / f"fsdp2_{tag}", copy_function=os.link)
        for carry in checkpoint_path.glob(f"carry_{tag}.*.pt"):
            os.link(carry, dest / carry.name)
        info = checkpoint_path / f"{tag}_info.json"
        if info.exists():
            os.link(info, dest / info.name)

    (dest / "upload_manifest.json").write_text(
        json.dumps(
            {
                "source": str(checkpoint_path),
                "stage": stage_name,
                "keep_last": keep_last,
                "tags": tags,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return stage_root


def start_latest_checkpoint_upload(checkpoint_path: Path, stage_name: str) -> None:
    stage_root = stage_latest_checkpoints(checkpoint_path, stage_name, keep_last=2)
    log_path = LOG_ROOT / f"upload_{stage_name}_latest_checkpoints.log"
    cmd = (
        "python scripts/upload_folder_to_hf.py "
        f"--folder {stage_root} "
        f"--repo-id {RAW_CHECKPOINT_REPO} "
        "--repo-type model --large --num-workers 4 "
        f"&& rm -rf {stage_root}"
    )
    log(f"starting latest checkpoint upload: {cmd}")
    f = log_path.open("ab")
    subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=HRM_ROOT,
        env=training_env(),
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def start_converted_model_upload(checkpoint_path: Path, stage_name: str) -> None:
    """Convert the latest epoch checkpoint to HF safetensors and upload main repo.

    This runs on CPU so it does not steal VRAM from the next training stage.
    Raw FSDP2 checkpoints still go to RAW_CHECKPOINT_REPO; this upload targets
    the user-facing MODEL_REPO.
    """
    out_dir = UPLOAD_STAGE_ROOT / f"KoHRM-Text-1.4B-converted-{stage_name}"
    log_path = LOG_ROOT / f"upload_{stage_name}_converted_model.log"
    model_card = HRM_ROOT / "MODEL_CARD_KoHRM-Text-1.4B.md"
    cmd = (
        f"rm -rf {out_dir} && "
        "python conversion/convert_to_hf.py "
        f"--ckpt_path {checkpoint_path} "
        "--ckpt_epoch 1 "
        "--ckpt_use_ema true "
        f"--out_dir {out_dir} "
        f"--tokenizer_path {TOKENIZER_PATH} "
        "--device cpu && "
        f"cp {model_card} {out_dir}/README.md && "
        "python scripts/upload_folder_to_hf.py "
        f"--folder {out_dir} "
        f"--repo-id {MODEL_REPO} "
        "--repo-type model --large --num-workers 4"
    )
    log(f"starting converted model upload to {MODEL_REPO}: {cmd}")
    f = log_path.open("ab")
    subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=HRM_ROOT,
        env=training_env(),
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log("next continuation watcher started")
    wait_final_checkpoint(CURRENT_STAGE)
    start_converted_model_upload(CURRENT_STAGE, "stage1-hrm-fastcap")
    ensure_small_mix()

    resume_from = CURRENT_STAGE
    offset = 88_522
    for stage in STAGES:
        checkpoint, steps = train_stage(stage, resume_from, offset)
        start_latest_checkpoint_upload(checkpoint, str(stage["name"]))
        start_converted_model_upload(checkpoint, str(stage["name"]))
        offset += steps
        resume_from = checkpoint
        log(f"completed {stage['name']}: steps={steps}, next_offset={offset}")
    log("all scheduled continuation stages completed")


if __name__ == "__main__":
    main()
