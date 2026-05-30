"""Continue KoHRM training from pass 3 into pass 4.

The active chain already owns pass 3:

    stage1c -> stage2c -> stage3c -> stage4c

This watcher must not interfere with that process.  It waits for the stage4c
final checkpoint and for the stage4c training process to exit, then runs the
fourth full data pass:

    stage1d -> stage2d -> stage3d -> stage4d

The d-stage names avoid overwriting earlier checkpoint directories while making
the pass number clear in checkpoint and upload paths.
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
    metadata_tokens,
    run_logged,
    start_converted_model_upload,
    start_latest_checkpoint_upload,
    validate_dataset,
)


STAGE4C = CKPT_ROOT / "KoHRM-Text-1.4B-stage4c-korean-tool-finance-repeat2-gbs180"
STAGE4C_RUN_NAME = "KoHRM-Text-1.4B-stage4c-korean-tool-finance-repeat2"

EPOCH4_STAGES = [
    {
        "name": "stage1d-hrm-fastcap-repeat3",
        "data": DATA_ROOT / "koterm_hrm_cleaned_fastcap_stage1_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage1d-hrm-fastcap-repeat3-gbs180",
    },
    {
        "name": "stage2d-hrm-full-nocap-repeat3",
        "data": DATA_ROOT / "koterm_hrm_cleaned_full_nocap_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage2d-hrm-full-nocap-repeat3-gbs180",
    },
    {
        "name": "stage3d-local-terminal-repeat3",
        "data": DATA_ROOT / "local_terminal_conversations_ctx9k_resp6k_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage3d-local-terminal-repeat3-gbs180",
    },
    {
        "name": "stage4d-korean-tool-finance-repeat3",
        "data": DATA_ROOT / "koterm_korean_tool_finance_mix_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage4d-korean-tool-finance-repeat3-gbs180",
    },
]


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def checkpoint_global_step(root: Path) -> int:
    info_path = root / "epoch_1_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info["global_step"])


def stage_is_complete(checkpoint_path: Path) -> bool:
    return (
        (checkpoint_path / "fsdp2_epoch_1").exists()
        and (checkpoint_path / "epoch_1_info.json").exists()
        and len(list(checkpoint_path.glob("carry_epoch_1.*.pt"))) >= 8
    )


def wait_epoch_checkpoint(root: Path, label: str) -> int:
    log(f"waiting for {label} epoch checkpoint under {root}")
    while True:
        if stage_is_complete(root):
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
            log(f"waiting for {run_name} processes to exit before epoch 4: pids={pids}")
            last_log = now
        time.sleep(10)


def stage_steps(stage: dict[str, Path | str]) -> int:
    data_path = Path(stage["data"])
    validate_dataset(data_path)
    return metadata_tokens(data_path) // GLOBAL_BATCH


def planned_total_steps(start_offset: int) -> int:
    # Keep a small margin above the estimated full fourth pass so each stage can
    # finish by epoch length rather than by the global cap.
    return start_offset + sum(stage_steps(stage) for stage in EPOCH4_STAGES) + 1_000


def read_finished_step(checkpoint_path: Path, fallback: int) -> int:
    info_path = checkpoint_path / "epoch_1_info.json"
    if info_path.exists():
        return checkpoint_global_step(checkpoint_path)
    return fallback


def train_stage(
    stage: dict[str, Path | str],
    resume_from: Path,
    resume_step_offset: int,
    total_steps_override: int,
) -> tuple[Path, int]:
    data_path = Path(stage["data"])
    checkpoint_path = Path(stage["checkpoint"])
    name = str(stage["name"])
    validate_dataset(data_path)
    expected_steps = stage_steps(stage)
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
        f"total_steps_override={total_steps_override}",
        "+log_interval=5",
        "checkpoint_step_interval=10000",
        "checkpoint_keep_last=2",
        "checkpoint_interval=1",
    ]
    run_logged(cmd, LOG_ROOT / f"KoHRM-Text-1.4B-{name}.log")
    return checkpoint_path, read_finished_step(checkpoint_path, resume_step_offset + expected_steps)


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log("epoch4 continuation watcher started")

    offset = wait_epoch_checkpoint(STAGE4C, "stage4c-korean-tool-finance-repeat2")
    wait_training_process_exit(STAGE4C_RUN_NAME)
    total_steps_override = planned_total_steps(offset)
    log(f"epoch4 planned_total_steps={total_steps_override} start_offset={offset}")

    resume_from = STAGE4C
    for stage in EPOCH4_STAGES:
        stage_name = str(stage["name"])
        checkpoint = Path(stage["checkpoint"])
        expected_steps = stage_steps(stage)
        if stage_is_complete(checkpoint):
            offset = read_finished_step(checkpoint, offset + expected_steps)
            resume_from = checkpoint
            log(f"{stage_name} already complete; next_offset={offset}")
            continue

        checkpoint, offset = train_stage(stage, resume_from, offset, total_steps_override)
        start_latest_checkpoint_upload(checkpoint, stage_name)
        start_converted_model_upload(checkpoint, stage_name)
        resume_from = checkpoint
        log(f"completed {stage_name}: next_offset={offset}")

    log("epoch4 continuation chain completed")


if __name__ == "__main__":
    main()
