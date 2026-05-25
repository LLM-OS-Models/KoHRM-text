"""Continue KoHRM training after the manually launched stage-2 run.

The stage-2 run was started manually from stage-1 step_85000. This watcher does
not launch another stage-2 job. It waits for the stage-2 epoch checkpoint, uploads
the latest artifacts, then runs the remaining prepared datasets.
"""

from __future__ import annotations

import time
from pathlib import Path

from watch_stage1_then_train_next import (
    CKPT_ROOT,
    DATA_ROOT,
    GLOBAL_BATCH,
    LOG_ROOT,
    TOTAL_STEPS_OVERRIDE,
    ensure_small_mix,
    log,
    run_logged,
    start_converted_model_upload,
    start_latest_checkpoint_upload,
    training_env,
    validate_dataset,
    wait_stable,
    metadata_tokens,
)


STAGE2 = CKPT_ROOT / "KoHRM-Text-1.4B-stage2-hrm-full-nocap-gbs180"
STAGE2_END_OFFSET = 165_753

REMAINING_STAGES = [
    {
        "name": "stage3-local-terminal",
        "data": DATA_ROOT / "local_terminal_conversations_ctx9k_resp6k_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage3-local-terminal-gbs180",
    },
    {
        "name": "stage4-korean-tool-finance",
        "data": DATA_ROOT / "koterm_korean_tool_finance_mix_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage4-korean-tool-finance-gbs180",
    },
]


def wait_stage2_epoch() -> None:
    log(f"waiting for manual stage-2 checkpoint under {STAGE2}")
    while True:
        epoch = STAGE2 / "fsdp2_epoch_1"
        carries = sorted(STAGE2.glob("carry_epoch_1.*.pt"))
        if epoch.exists() and len(carries) >= 8:
            log("stage-2 epoch checkpoint detected; waiting for stable files")
            wait_stable(STAGE2)
            log("stage-2 checkpoint is stable")
            return
        time.sleep(120)


def train_stage(stage: dict[str, Path | str], resume_from: Path, resume_step_offset: int) -> tuple[Path, int]:
    data_path = Path(stage["data"])
    checkpoint_path = Path(stage["checkpoint"])
    name = str(stage["name"])
    validate_dataset(data_path)
    steps = metadata_tokens(data_path) // GLOBAL_BATCH
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


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log("manual stage-2 continuation watcher started")
    wait_stage2_epoch()
    start_latest_checkpoint_upload(STAGE2, "stage2-hrm-full-nocap")
    start_converted_model_upload(STAGE2, "stage2-hrm-full-nocap")

    ensure_small_mix()
    resume_from = STAGE2
    offset = STAGE2_END_OFFSET
    for stage in REMAINING_STAGES:
        checkpoint, steps = train_stage(stage, resume_from, offset)
        start_latest_checkpoint_upload(checkpoint, str(stage["name"]))
        start_converted_model_upload(checkpoint, str(stage["name"]))
        offset += steps
        resume_from = checkpoint
        log(f"completed {stage['name']}: steps={steps}, next_offset={offset}")
    log("remaining scheduled continuation stages completed")


if __name__ == "__main__":
    main()
