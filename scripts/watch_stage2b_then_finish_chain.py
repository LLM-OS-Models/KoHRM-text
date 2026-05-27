"""Continue KoHRM training after a manually restarted stage2b run.

The active stage2b process was started manually. This watcher must not launch a
second stage2b job. It waits for that checkpoint, uploads it, and then keeps the
GPUs occupied with:

    stage3b -> stage4b -> stage1c -> stage2c -> stage3c -> stage4c
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from watch_stage1_then_train_next import (
    CKPT_ROOT,
    DATA_ROOT,
    GLOBAL_BATCH,
    LOG_ROOT,
    run_logged,
    start_converted_model_upload,
    start_latest_checkpoint_upload,
    validate_dataset,
)


TOTAL_STEPS_OVERRIDE = 700_000

STAGE2B = CKPT_ROOT / "KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1-gbs180"
STAGE2B_RUN_NAME = "KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1"

REMAINING_STAGES = [
    {
        "name": "stage3b-local-terminal-repeat",
        "data": DATA_ROOT / "local_terminal_conversations_ctx9k_resp6k_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage3b-local-terminal-repeat-gbs180",
    },
    {
        "name": "stage4b-korean-tool-finance-repeat",
        "data": DATA_ROOT / "koterm_korean_tool_finance_mix_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage4b-korean-tool-finance-repeat-gbs180",
    },
    {
        "name": "stage1c-hrm-fastcap-repeat2",
        "data": DATA_ROOT / "koterm_hrm_cleaned_fastcap_stage1_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage1c-hrm-fastcap-repeat2-gbs180",
    },
    {
        "name": "stage2c-hrm-full-nocap-repeat2",
        "data": DATA_ROOT / "koterm_hrm_cleaned_full_nocap_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage2c-hrm-full-nocap-repeat2-gbs180",
    },
    {
        "name": "stage3c-local-terminal-repeat2",
        "data": DATA_ROOT / "local_terminal_conversations_ctx9k_resp6k_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage3c-local-terminal-repeat2-gbs180",
    },
    {
        "name": "stage4c-korean-tool-finance-repeat2",
        "data": DATA_ROOT / "koterm_korean_tool_finance_mix_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage4c-korean-tool-finance-repeat2-gbs180",
    },
]


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def checkpoint_global_step(root: Path) -> int:
    info_path = root / "epoch_1_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info["global_step"])


def wait_epoch_checkpoint(root: Path, label: str) -> int:
    log(f"waiting for {label} epoch checkpoint under {root}")
    while True:
        epoch = root / "fsdp2_epoch_1"
        carries = sorted(root.glob("carry_epoch_1.*.pt"))
        info = root / "epoch_1_info.json"
        if epoch.exists() and len(carries) >= 8 and info.exists():
            step = checkpoint_global_step(root)
            log(f"{label} epoch checkpoint detected: global_step={step}")
            return step
        time.sleep(60)


def live_training_processes(run_name: str) -> list[int]:
    out = subprocess.run(
        ["ps", "-eo", "pid=,stat=,cmd="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    marker = f"+run_name={run_name}"
    pids: list[int] = []
    for line in out.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_s, stat, cmd = parts
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if stat.startswith("Z"):
            continue
        if marker in cmd and ("torchrun" in cmd or "pretrain.py" in cmd):
            pids.append(pid)
    return pids


def wait_training_process_exit(run_name: str) -> None:
    last_log = 0.0
    while True:
        pids = live_training_processes(run_name)
        if not pids:
            log(f"training process for {run_name} has exited")
            return
        now = time.monotonic()
        if now - last_log >= 60:
            log(f"waiting for {run_name} processes to exit before next stage: pids={pids}")
            last_log = now
        time.sleep(10)


def read_finished_step(checkpoint_path: Path, fallback: int) -> int:
    info_path = checkpoint_path / "epoch_1_info.json"
    if info_path.exists():
        return checkpoint_global_step(checkpoint_path)
    return fallback


def train_stage(stage: dict[str, Path | str], resume_from: Path, resume_step_offset: int) -> tuple[Path, int]:
    data_path = Path(stage["data"])
    checkpoint_path = Path(stage["checkpoint"])
    name = str(stage["name"])
    validate_dataset(data_path)
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
    return checkpoint_path, read_finished_step(checkpoint_path, TOTAL_STEPS_OVERRIDE)


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log("stage2b continuation watcher started")

    offset = wait_epoch_checkpoint(STAGE2B, "stage2b-hrm-full-nocap-extra-epoch1")
    wait_training_process_exit(STAGE2B_RUN_NAME)
    start_latest_checkpoint_upload(STAGE2B, "stage2b-hrm-full-nocap-extra-epoch1")
    start_converted_model_upload(STAGE2B, "stage2b-hrm-full-nocap-extra-epoch1")

    resume_from = STAGE2B
    for stage in REMAINING_STAGES:
        if offset >= TOTAL_STEPS_OVERRIDE:
            log(f"total_steps_override reached at {offset}; skipping remaining stages")
            break
        checkpoint, offset = train_stage(stage, resume_from, offset)
        start_latest_checkpoint_upload(checkpoint, str(stage["name"]))
        start_converted_model_upload(checkpoint, str(stage["name"]))
        resume_from = checkpoint
        log(f"completed {stage['name']}: next_offset={offset}")

    log("stage2b continuation chain completed")


if __name__ == "__main__":
    main()
