"""Run the KoHRM two-pass continuation chain after the active stage-2 job.

This watcher is intentionally a single orchestration entry point. It waits for
the manually launched stage-2 checkpoint, then runs:

    stage3 -> stage4 -> stage1b -> stage2b -> stage3b -> stage4b

Each training run still uses its own prepared dataset path to avoid building a
single huge merged token file. Checkpoint retention is handled by pretrain.py
with checkpoint_keep_last=2, and uploads are spawned after each stage.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from watch_stage1_then_train_next import (
    CKPT_ROOT,
    DATA_ROOT,
    GLOBAL_BATCH,
    LOG_ROOT,
    RAW_CHECKPOINT_REPO,
    MODEL_REPO,
    TOKENIZER_PATH,
    ensure_small_mix,
    log,
    metadata_tokens,
    run_logged,
    start_converted_model_upload,
    start_latest_checkpoint_upload,
    validate_dataset,
    wait_stable,
)


TOTAL_STEPS_OVERRIDE = 465_000

STAGE2_ACTIVE = CKPT_ROOT / "KoHRM-Text-1.4B-stage2-hrm-full-nocap-gbs180"
STAGE2_ACTIVE_FALLBACK_END = 165_753

HRM_EXTRA_EPOCHS = DATA_ROOT / "koterm_hrm_cleaned_full_nocap_extra_epochs_1_3_v1"

CHAIN = [
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
        "data": DATA_ROOT / "koterm_korean_tool_finance_mix_v1",
        "checkpoint": CKPT_ROOT / "KoHRM-Text-1.4B-stage4b-korean-tool-finance-repeat-gbs180",
    },
]


def wait_dataset(path: Path) -> None:
    while True:
        try:
            validate_dataset(path)
            return
        except FileNotFoundError as exc:
            log(f"waiting for prepared dataset {path}: {exc}")
            time.sleep(300)


def wait_stage2_epoch() -> int:
    log(f"waiting for active stage-2 checkpoint under {STAGE2_ACTIVE}")
    while True:
        epoch = STAGE2_ACTIVE / "fsdp2_epoch_1"
        carries = sorted(STAGE2_ACTIVE.glob("carry_epoch_1.*.pt"))
        if epoch.exists() and len(carries) >= 8:
            log("active stage-2 epoch checkpoint detected; starting continuation immediately")
            info_path = STAGE2_ACTIVE / "epoch_1_info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    return int(info.get("global_step", STAGE2_ACTIVE_FALLBACK_END))
                except Exception:
                    pass
            return STAGE2_ACTIVE_FALLBACK_END
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
    log(
        "two-pass continuation watcher started "
        f"(raw_repo={RAW_CHECKPOINT_REPO}, model_repo={MODEL_REPO}, tokenizer={TOKENIZER_PATH})"
    )
    offset = wait_stage2_epoch()
    start_latest_checkpoint_upload(STAGE2_ACTIVE, "stage2-hrm-full-nocap")
    start_converted_model_upload(STAGE2_ACTIVE, "stage2-hrm-full-nocap")

    small_mix_thread = threading.Thread(target=ensure_small_mix, name="ensure-small-mix", daemon=False)
    small_mix_thread.start()

    resume_from = STAGE2_ACTIVE
    for stage in CHAIN:
        if Path(stage["data"]) == DATA_ROOT / "koterm_korean_tool_finance_mix_v1":
            log("waiting for Korean/tool/finance mix before this stage")
            small_mix_thread.join()
        if Path(stage["data"]) == HRM_EXTRA_EPOCHS:
            wait_dataset(HRM_EXTRA_EPOCHS)
        checkpoint, steps = train_stage(stage, resume_from, offset)
        start_latest_checkpoint_upload(checkpoint, str(stage["name"]))
        start_converted_model_upload(checkpoint, str(stage["name"]))
        offset += steps
        resume_from = checkpoint
        log(f"completed {stage['name']}: steps={steps}, next_offset={offset}")

    log("two-pass continuation chain completed")


if __name__ == "__main__":
    main()
