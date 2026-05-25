---
license: other
language:
- ko
- en
tags:
- hrm-text
- korean
- terminal
- tool-use
- code
- pretraining
pipeline_tag: text-generation
---

# KoHRM-Text-1.4B

`KoHRM-Text-1.4B` is a scratch-pretrained Korean/English/code/terminal/tool-use model based on the `sapientinc/HRM-Text` PrefixLM training stack.

This is not a continued finetune of `sapientinc/HRM-Text-1B`. It uses a new Korean/terminal-oriented 131K byte-level BPE tokenizer and a new scratch training run.

## Links

| Item | Link |
|---|---|
| HF model | https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B |
| Project code | https://github.com/LLM-OS-Models/KoHRM-text |
| Prepared training data | https://huggingface.co/datasets/LLM-OS-Models/KoHRM-Text-1.4B-prepared-data |
| Upstream HRM-Text code | https://github.com/sapientinc/HRM-Text |
| HRM-Text paper | https://arxiv.org/html/2605.20613 |
| Tokenizer | https://huggingface.co/LLM-OS-Models/HRM-Text-Ko-Terminal-Tokenizer-131K |
| Raw resume checkpoints | https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints |

## Release Policy

The main model repository is intended to expose the latest model-only artifact:

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `README.md`

It is not intended to keep every training checkpoint as visible model files. Intermediate FSDP2 `.distcp` checkpoints are large resume artifacts and are kept separately in `LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints` when needed. The main repo may still have normal Hugging Face git history, but the current file tree should be treated as the latest public model export.

Current public artifact: `stage3` local-terminal continuation checkpoint at `step_180000`, converted with EMA weights to `safetensors`. Training is still in progress; this is an intermediate checkpoint from the ongoing `stage3-local-terminal` run.

## Model Details

| Field | Value |
|---|---|
| Model id | `LLM-OS-Models/KoHRM-Text-1.4B` |
| Standard name | `KoHRM-Text-1.4B` |
| Training origin | scratch |
| Architecture family | HRM-Text PrefixLM |
| Architecture size | `XL` |
| Parameters | 1,384,120,320 |
| Context length | 4,096 tokens |
| Training dtype | bfloat16 |
| Tokenizer | byte-level BPE, NFC normalization |
| Vocabulary size | 131,072 |
| Objective | PrefixLM response-only loss |
| Optimizer | Adam-atan2 from upstream HRM-Text |
| EMA | 0.9999 |

The model config uses `model_type: hrm_text` and `architectures: ["HrmTextForCausalLM"]`. At the time of this checkpoint, `HrmTextForCausalLM` is a project-side custom architecture, not a built-in Transformers architecture.

## Tokenizer

The tokenizer was trained for Korean, English, code, shell/terminal text, and JSON/tool-call formats. It intentionally keeps common chat/tool special tokens as stable single tokens where possible.

| Sample bucket | chars/token |
|---|---:|
| Korean general text | 2.60 |
| Korean legal text | 2.36 |
| Korean terminal instruction | 2.18 |
| shell command | 2.68 |
| tool-call JSON | 3.32 |
| Python code | 3.37 |
| English | 4.40 |

Important formatting tokens include:

- `<|im_start|>`
- `<|im_end|>`
- `<|box_end|>`
- `<|object_ref_start|>` for direct condition
- `<|object_ref_end|>` for cot condition
- `<|quad_start|>` for noisy condition
- `<|quad_end|>` for synth condition

## Usage

### Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "LLM-OS-Models/KoHRM-Text-1.4B",
    use_fast=True,
)

prompt = "<|im_start|><|object_ref_start|>한국어로 현재 디렉터리의 큰 파일을 찾는 명령을 알려주세요.<|im_end|>"
ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
print(len(ids), ids[:20])
```

### Model Weights

The repo currently contains a model-only `safetensors` export. Because the architecture is custom (`hrm_text`), direct `AutoModelForCausalLM.from_pretrained(...)` generation requires an HRM-Text-compatible modeling wrapper or remote-code integration. Until that wrapper is added to the model repo, use the project code and raw FSDP2 checkpoint path for internal inference/resume workflows.

Raw checkpoint inference pattern:

```python
from simple_inference_engine import inference_load_checkpoint, inference_generate

ckpt = inference_load_checkpoint(
    ckpt_path="/path/to/KoHRM-Text-1.4B-stage1-hrm-fastcap-gbs180",
    ckpt_epoch=None,
    ckpt_step=85000,
    ckpt_use_ema=True,
    device="cuda",
)

prompts = iter([
    (0, ("direct", "한국어로 `du`와 `df`의 차이를 설명해주세요.")),
])

for _, text in inference_generate(
    ckpt,
    prompts,
    max_tokens=4096,
    max_generation=512,
    batch_size=1,
    temp=0.0,
):
    print(text)
```

For code and training scripts, see https://github.com/LLM-OS-Models/KoHRM-text.

## Training Data

Prepared data artifacts are uploaded to the Hugging Face dataset repository, not to the model repository:

https://huggingface.co/datasets/LLM-OS-Models/KoHRM-Text-1.4B-prepared-data

All datasets are converted into HRM-Text V1Dataset style records with `instruction`, `response`, and `condition` fields where possible. The training objective is PrefixLM response-only loss, so the model is trained to predict the response span after seeing the instruction/prompt span.

Completed and prepared datasets:

| Dataset | Tokens | Disk | Use |
|---|---:|---:|---|
| `koterm_pretrain_mix_v1` | 711.3M | 2.8G | stage-0/stage0b |
| HRM cleaned base sample | 250.0M | 994M | included in stage-0 mix |
| SWE-ZERO + GLM pilot mix | 251.2M | 990M | included in stage-0 mix |
| Korean legal SFT/task data | 83.1M | 336M | included in stage-0 mix |
| ToolBench train tool-call data | 127.0M | 500M | included in stage-0 mix |
| HRM cleaned fast-cap stage-1 | 14.55B | 148G | completed through latest saved `step_85000` |
| Korean statutes/local ordinances raw full | 308.9M | 1.2G | prepared for later stages |
| Korean administrative rules + precedents raw full | 271.7M | 1.1G | prepared for later stages |
| Korean legal/admin full task data | 629.0M | 2.5G | uploaded to prepared dataset repo |
| Korean Wikipedia raw full | 462.5M | 1.8G | prepared for later stages |
| HF extra reasoning/agent/mm subset | 112.6M | 444M | prepared, limited weight |
| Local terminal conversations | 9.39B | 36G | prepared for terminal-heavy later stages |
| BCAI Finance Korean | 857.7M | 3.3G | prepared and uploaded for later Korean finance/domain stages |
| SWE-ZERO prepared | 182.7M | 720M | pretraining and later SFT |
| GLM reasoning prepared | 68.5M | 282M | pretraining and later SFT |

Major source groups and provenance:

| Source group | Origin | Prepared dataset usage |
|---|---|---|
| HRM-Text cleaned pretraining data | https://huggingface.co/datasets/sapientinc/HRM-Text-data-io-cleaned-20260515 | `hrm_cleaned_base_sample_v1`, `koterm_hrm_cleaned_fastcap_stage1_v1`; full no-cap retokenization is still running |
| Korean Wikipedia | https://dumps.wikimedia.org/kowiki/20260501/ | `kowiki_raw_full_v1` |
| Korean statutes | https://github.com/legalize-kr/legalize-kr | `korean_legal_raw_full_v1`, `sft_korean_legal_v1`, `korean_legal_tasks_full_v1` |
| Korean local ordinances | https://github.com/legalize-kr/ordinance-kr | `korean_legal_raw_full_v1`, `sft_korean_legal_v1`, `korean_legal_tasks_full_v1` |
| Korean administrative rules | local Markdown snapshot at `admrule-kr/` | `korean_admrule_precedent_raw_full_v1`, `korean_legal_tasks_full_v1` |
| Korean precedents | local Markdown snapshot at `precedent-kr/` | `korean_admrule_precedent_raw_full_v1`, `korean_legal_tasks_full_v1` |
| ToolBench train data | local extraction under `data_toolbench/data/`; eval split excluded | `sft_toolbench_v1` |
| SWE-ZERO trajectories | https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories | `sft_swe_zero_v1`, `sft_swe_glm_mix_v1` |
| GLM reasoning | https://huggingface.co/datasets/Jackrong/GLM-5.1-Reasoning-1M-Cleaned | `sft_glm_reasoning_v1`, `sft_swe_glm_mix_v1` |
| Claude reasoning sample | https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k | small reviewed reasoning subset inside `hf_extra_reasoning_agent_mm_v1` |
| Open-MM-RL text subset | https://huggingface.co/datasets/TuringEnterprises/Open-MM-RL | text-only reviewed subset inside `hf_extra_reasoning_agent_mm_v1` |
| DeepSeek agent traces | https://huggingface.co/datasets/TeichAI/DeepSeek-v4-Pro-Agent | limited agent/tool-use subset; license-sensitive |
| structured Wikipedia | https://huggingface.co/datasets/wikimedia/structured-wikipedia | tokenizer/general text support |
| Local terminal/code/math conversations | local `swe`, `code`, and `math` parquet conversations | `local_terminal_conversations_ctx9k_resp6k_v1` |
| BCAI Finance Kor | https://huggingface.co/datasets/BCCard/BCAI-Finance-Kor-1862K | `sft_bcai_finance_kor_v1` |

The full Korean legal/admin task upload is present in the dataset repository at:

- `korean_legal_tasks_full_v1/`
- `raw_jsonl/korean_legal_tasks_full_20260524.jsonl`
- `LEGAL_FULL_TASKS_README.md`
- `sft_bcai_finance_kor_v1/`
- `raw_jsonl/bcai_finance_kor_hrm_20260524.jsonl`
- `FINANCE_BCAI_README.md`

Evaluation-like data is excluded from training where identified, including ToolBench eval, Terminal Bench 2 style data, and benchmark-oriented `chi-bench` data.

Licenses and terms remain those of the original sources. The prepared dataset upload does not relicense upstream content.

## Training Run

The current public checkpoint was produced through staged pretraining:

1. Train `stage-0` on `koterm_pretrain_mix_v1` with 711.3M tokens.
2. Continue once more on the same available mix as `stage0b`.
3. Continue to `stage-1` on HRM cleaned fast-cap data with 14.55B tokens.
4. Convert intermediate EMA weights to `safetensors` and upload to the main model repo for public inspection.
5. Continue from `stage1 step_85000` into `stage2` on full/no-cap HRM cleaned data.

Current long-running stage-3 settings:

| Field | Value |
|---|---|
| Hardware | 8 x NVIDIA H200 |
| Data | `local_terminal_conversations_ctx9k_resp6k_v1` |
| Tokens in current stage dataset | 9.39B |
| Global batch | 180,224 tokens |
| Local token slots/GPU | 22,528 |
| Context | 4,096 |
| LR | 2.2e-4 |
| LR warmup | 2,000 steps |
| Checkpoint interval | 10,000 steps |
| Current public export | `stage3 step_180000`, EMA, safetensors |

The run uses staged continuation. The checkpoint carries model, optimizer, EMA, and recurrent carry state forward. `resume_step_offset` and `total_steps_override` are used so the learning-rate schedule follows the intended longer pretraining run rather than resetting at every data stage.

The stage-2 full/no-cap HRM continuation has completed and produced a final epoch checkpoint. The public artifact is now being updated through the stage-3 local-terminal continuation while the remaining `stage4 -> stage1b -> stage2b -> stage3b -> stage4b` chain continues in the background.

## Intended Use

This checkpoint is intended for:

- continued pretraining experiments
- Korean tokenizer and HRM-Text architecture experiments
- terminal/tool-call/code pretraining research
- checkpoint conversion and evaluation work

It is not yet intended as a finished assistant model.

## Limitations

- This is an intermediate checkpoint, not a final aligned instruct model.
- It has not completed the full planned 40B+ token continuation.
- It has not completed final SFT or safety tuning.
- Public benchmark scores for this new checkpoint are not final.
- Direct Transformers generation requires adding the custom `hrm_text` modeling wrapper or remote-code files.
- Tool-call JSON validity and terminal action safety must be evaluated before production use.

## Citation

This work builds on the HRM-Text architecture and training stack:

- Paper: https://arxiv.org/html/2605.20613
- Upstream code: https://github.com/sapientinc/HRM-Text
