"""Resume KoHRM training from completed stage4b into the final pass.

stage4b saved a valid epoch_1 checkpoint, but the earlier continuation watcher
did not hand off to stage1c after a SIGSEGV during process teardown.  This
watcher starts from that saved stage4b checkpoint and keeps the remaining GPUs
busy with:

    stage1c -> stage2c -> stage3c -> stage4c

It intentionally reuses the same train/upload helpers as the existing
stage2b continuation watcher so batch size, checkpoint retention, upload
behavior, and LR schedule handling remain identical.
"""

from __future__ import annotations

import os
from pathlib import Path

from watch_stage2b_then_finish_chain import (
    CKPT_ROOT,
    REMAINING_STAGES,
    TOTAL_STEPS_OVERRIDE,
    checkpoint_global_step,
    log,
    read_finished_step,
    start_converted_model_upload,
    start_latest_checkpoint_upload,
    train_stage,
)


STAGE4B = CKPT_ROOT / "KoHRM-Text-1.4B-stage4b-korean-tool-finance-repeat-gbs180"
START_INDEX = 2  # REMAINING_STAGES[2:] == stage1c, stage2c, stage3c, stage4c


def require_epoch_checkpoint(root: Path, label: str) -> int:
    epoch = root / "fsdp2_epoch_1"
    info = root / "epoch_1_info.json"
    carries = sorted(root.glob("carry_epoch_1.*.pt"))
    if not epoch.exists() or not info.exists() or len(carries) < 8:
        raise RuntimeError(
            f"{label} is not ready: epoch={epoch.exists()} "
            f"info={info.exists()} carries={len(carries)} root={root}"
        )
    return checkpoint_global_step(root)


def stage_is_complete(checkpoint_path: Path) -> bool:
    return (
        (checkpoint_path / "fsdp2_epoch_1").exists()
        and (checkpoint_path / "epoch_1_info.json").exists()
        and len(list(checkpoint_path.glob("carry_epoch_1.*.pt"))) >= 8
    )


def main() -> None:
    log("stage1c continuation watcher started")
    offset = require_epoch_checkpoint(STAGE4B, "stage4b-korean-tool-finance-repeat")
    log(f"resuming from stage4b epoch_1: global_step={offset}")

    # Stage4b is the user-visible epoch-2 completion point.  Uploading is async
    # and does not consume VRAM, so start it before training the next stage.
    if os.environ.get("KOHRM_SKIP_INITIAL_UPLOADS") == "1":
        log("skipping initial stage4b uploads because KOHRM_SKIP_INITIAL_UPLOADS=1")
    else:
        start_latest_checkpoint_upload(STAGE4B, "stage4b-korean-tool-finance-repeat")
        start_converted_model_upload(STAGE4B, "stage4b-korean-tool-finance-repeat")

    resume_from = STAGE4B
    for stage in REMAINING_STAGES[START_INDEX:]:
        stage_name = str(stage["name"])
        checkpoint = Path(stage["checkpoint"])
        if stage_is_complete(checkpoint):
            offset = read_finished_step(checkpoint, offset)
            resume_from = checkpoint
            log(f"{stage_name} already complete; next_offset={offset}")
            continue

        if offset >= TOTAL_STEPS_OVERRIDE:
            log(f"total_steps_override reached at {offset}; skipping {stage_name} and later stages")
            break

        checkpoint, offset = train_stage(stage, resume_from, offset)
        start_latest_checkpoint_upload(checkpoint, stage_name)
        start_converted_model_upload(checkpoint, stage_name)
        resume_from = checkpoint
        log(f"completed {stage_name}: next_offset={offset}")

    log("stage1c continuation chain completed")


if __name__ == "__main__":
    main()
