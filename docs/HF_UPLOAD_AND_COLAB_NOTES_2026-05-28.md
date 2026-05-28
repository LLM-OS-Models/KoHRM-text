# HF Upload And Colab Notes - 2026-05-28

## Current Hub Targets

Main public model repo:

- `LLM-OS-Models/KoHRM-Text-1.4B`
- Contains the latest converted EMA export as `model.safetensors`.
- Also contains tokenizer files, `config.json`, and the model card copied as `README.md`.

Raw resume checkpoint repo:

- `LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints`
- Contains raw FSDP2 resume checkpoints under stage/step-specific folders.
- These are for continuation, recovery, and reproducibility. They are not meant for one-line inference.

Prepared data repo:

- `LLM-OS-Models/KoHRM-Text-1.4B-prepared-data`
- Contains prepared/tokenized datasets where upload has been completed.

## Upload Policy

The long pretraining run saves step checkpoints every `10,000` steps and keeps the newest local step checkpoints through `checkpoint_keep_last=2`.

For each complete step checkpoint, `scripts/watch_chain_step_checkpoints_upload.py` does two uploads:

1. Raw checkpoint upload to `LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints`.
2. CPU conversion to a single `model.safetensors`, then upload to `LLM-OS-Models/KoHRM-Text-1.4B`.

The public model repo is a rolling latest export. Older converted public exports are not preserved there unless uploaded to a separate repo or revision/tag later.

## 2026-05-28 Upload Fix

After moving project documents under `docs/`, the upload code still tried to copy:

```text
MODEL_CARD_KoHRM-Text-1.4B.md
```

from the repository root.

The actual path is now:

```text
docs/MODEL_CARD_KoHRM-Text-1.4B.md
```

This broke converted model uploads after conversion succeeded. Raw checkpoint uploads still completed, but converted model uploads failed at the model-card copy step.

Fixed files:

- `scripts/watch_chain_step_checkpoints_upload.py`
- `scripts/watch_stage1_then_train_next.py`
- `scripts/watch_and_upload_hrm_checkpoints.py`

Operational action:

- The broken `watch_chain_step_checkpoints_upload.py` process was stopped.
- A new watcher was started with the fixed code.
- `stage3b-local-terminal-repeat` `step_420000` was manually converted and uploaded to the main public model repo after the path fix.
- The fixed watcher was then restarted with `--min-step 430000` so future checkpoints are uploaded without repeatedly reprocessing older already-uploaded raw checkpoints.

## Current Known Limitation

The public converted model repo does not yet include the custom Hugging Face remote-code implementation for:

```text
HrmTextForCausalLM
```

Therefore:

- `AutoTokenizer.from_pretrained("LLM-OS-Models/KoHRM-Text-1.4B")` works.
- `config.json` and `model.safetensors` inspection works.
- Plain `AutoModelForCausalLM.from_pretrained(...)` is expected to fail today.
- Actual generation still requires project-side HRM-Text code and compatible raw checkpoints until a remote-code wrapper is released.

## Colab T4 Notebook

Notebook:

- `notebooks/KoHRM_Text_1_4B_Colab_T4_Smoke_Test.ipynb`

Purpose:

- Check the latest Hugging Face revision from Colab.
- Download tokenizer/config and optionally `model.safetensors`.
- Validate Korean/terminal/tool-call tokenizer behavior.
- Inspect safetensors tensor shapes without loading the full model into GPU memory.
- Confirm that plain Transformers generation is not yet supported and fails for the expected custom-architecture reason.

T4 design choice:

- The notebook avoids full model load/generation.
- Safetensors inspection uses file-header slices to avoid unnecessarily loading all 1.38B parameters into RAM/GPU memory.
- Weight download can be disabled with `DOWNLOAD_WEIGHTS = False` for a very quick tokenizer/config smoke test.

## Manual Verification Commands

Check the active upload watcher:

```bash
ps -eo pid,etimes,args | rg 'watch_chain_step_checkpoints_upload.py'
```

Check recent upload logs:

```bash
ls -lt /home/work/.data/hrm_text_logs | rg 'upload_.*stage3b|watch_chain' | head -30
tail -80 /home/work/.data/hrm_text_logs/watch_chain_step_checkpoints_upload_20260528_restart.log
```

Check local complete checkpoints:

```bash
find /home/work/.data/hrm_text_checkpoints -maxdepth 2 -type d \
  \( -name 'fsdp2_step_*' -o -name 'fsdp2_epoch_*' \) | sort | tail -40
```

Check upload markers:

```bash
find /home/work/.data/hrm_text_hf_upload_stage/.step_upload_markers -type f \
  -printf '%f %TY-%Tm-%Td %TH:%TM:%TS\n' | sort | tail -40
```
