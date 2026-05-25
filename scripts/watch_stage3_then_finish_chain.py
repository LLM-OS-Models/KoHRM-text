"""Continue KoHRM training after an already-started stage-3 run.

This is a recovery orchestrator for the case where stage-3 was launched but
the parent stage-chain process was interrupted. It does not start stage-3
again. It waits for the stage-3 epoch checkpoint, uploads it, then runs:

    stage4 -> stage1b -> stage2b -> stage3b -> stage4b
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
    ensure_small_mix,
    log,
    metadata_tokens,
    run_logged,
    start_converted_model_upload,
    start_latest_checkpoint_upload,
    validate_dataset,
)


TOTAL_STEPS_OVERRIDE = 465_000

STAGE3 = CKPT_ROOT / "KoHRM-Text-1.4B-stage3-local-terminal-gbs180"
SMALL_MIX = DATA_ROOT / "koterm_korean_tool_finance_mix_v1"
HRM_EXTRA_EPOCHS = DATA_ROOT / "koterm_hrm_cleaned_full_nocap_extra_epochs_1_3_v1"

CHAIN_REST = [
    {
        "name": "stage4-korean-tool-finance",
        "data": SMALL_MIX,
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage4-korean-tool-finance-gbs180",
    },
    {
        "name": "stage1b-hrm-fastcap-repeat",
        "data": DATA_ROOT / "koterm_hrm_cleaned_fastcap_stage1_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage1b-hrm-fastcap-repeat-gbs180",
    },
    {
        "name": "stage2b-hrm-full-nocap-extra-epoch1",
        "data": HRM_EXTRA_EPOCHS,
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1-gbs180",
    },
    {
        "name": "stage3b-local-terminal-repeat",
        "data": DATA_ROOT / "local_terminal_conversations_ctx9k_resp6k_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage3b-local-terminal-repeat-gbs180",
    },
    {
        "name": "stage4b-korean-tool-finance-repeat",
        "data": SMALL_MIX,
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage4b-korean-tool-finance-repeat-gbs180",
    },
]


def checkpoint_global_step(root: Path) -> int:
    info_path = root / "epoch_1_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return int(info["global_step"])
    raise FileNotFoundError(info_path)


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


def merge_process_running() -> bool:
    out = subprocess.run(
        ["pgrep", "-af", "merge_prepared_sft_data.py.*koterm_korean_tool_finance_mix_v1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return out.returncode == 0


def ensure_or_wait_small_mix() -> None:
    while True:
        try:
            validate_dataset(SMALL_MIX)
            log(f"small mix ready: {SMALL_MIX}")
            return
        except FileNotFoundError:
            if merge_process_running():
                log("small mix merge is already running; waiting")
                time.sleep(120)
                continue
            log("small mix missing and no merge process found; building now")
            ensure_small_mix()


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
    log("stage3 recovery continuation watcher started")

    offset = wait_epoch_checkpoint(STAGE3, "stage3-local-terminal")
    start_latest_checkpoint_upload(STAGE3, "stage3-local-terminal")
    start_converted_model_upload(STAGE3, "stage3-local-terminal")

    resume_from = STAGE3
    for stage in CHAIN_REST:
        data_path = Path(stage["data"])
        if data_path == SMALL_MIX:
            ensure_or_wait_small_mix()
        else:
            validate_dataset(data_path)
        checkpoint, steps = train_stage(stage, resume_from, offset)
        start_latest_checkpoint_upload(checkpoint, str(stage["name"]))
        start_converted_model_upload(checkpoint, str(stage["name"]))
        offset += steps
        resume_from = checkpoint
        log(f"completed {stage['name']}: steps={steps}, next_offset={offset}")

    log("stage3 recovery continuation chain completed")


if __name__ == "__main__":
    main()
