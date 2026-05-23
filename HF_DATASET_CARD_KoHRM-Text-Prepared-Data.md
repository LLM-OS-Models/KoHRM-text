---
license: other
language:
- ko
- en
task_categories:
- text-generation
tags:
- hrm-text
- korean
- terminal
- tool-use
- code
- pretraining
- tokenized
pretty_name: KoHRM-Text 1.4B Prepared Data
---

# KoHRM-Text-1.4B Prepared Data

This dataset repository contains prepared HRM-Text V1Dataset artifacts for `KoHRM-Text-1.4B`.

The data is intended for continued pretraining and staged training with the project code at:

- https://github.com/LLM-OS-Models/KoHRM-text
- https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B
- https://huggingface.co/LLM-OS-Models/HRM-Text-Ko-Terminal-Tokenizer-131K

The upstream architecture and training method are based on:

- Paper: https://arxiv.org/html/2605.20613
- Upstream code: https://github.com/sapientinc/HRM-Text

## Format

Each subdirectory is an HRM-Text V1Dataset-style prepared dataset. The common layout is:

```text
dataset_name/
  metadata.json
  tokens.npy
  epoch_0/
    indices.npy or equivalent epoch index files
    inst_start.npy
    inst_len.npy
    resp_start.npy
    resp_len.npy
```

The datasets are not plain raw-text corpora. They are already tokenized or packed for HRM-Text PrefixLM training with response-only loss.

## Tokenizer

All real prepared datasets in this upload use the KoHRM Korean/terminal tokenizer:

| Field | Value |
|---|---|
| HF repo | https://huggingface.co/LLM-OS-Models/HRM-Text-Ko-Terminal-Tokenizer-131K |
| Local training name | `hrm-ko-terminal-131k-v1` |
| Type | byte-level BPE |
| Unicode normalization | NFC |
| Vocabulary size | 131,072 |

The tokenizer was trained with an intentional mix of Korean, English, code, terminal, JSON/tool-call, and reasoning text.

Tokenizer corpus design:

| Bucket | Target share | Purpose |
|---|---:|---|
| Korean general/legal/admin | 35-40% | Korean morphology, legal/admin terminology, long-form Korean |
| English instruction/general | 20-25% | Preserve upstream English instruction behavior |
| Code/terminal/SWE | 20-25% | CLI commands, stack traces, patches, test output |
| Tool-call/JSON/API | 10-15% | Stable JSON arguments, schemas, API names |
| Math/STEM/reasoning | 5-10% | Equations, reasoning text, symbolic patterns |

Measured chars/token:

| Sample bucket | chars/token |
|---|---:|
| Korean general | 2.60 |
| Korean legal | 2.36 |
| Korean terminal instruction | 2.18 |
| shell command | 2.68 |
| tool-call JSON | 3.32 |
| Python code | 3.37 |
| English | 4.40 |

Important special tokens:

- `<|im_start|>`
- `<|im_end|>`
- `<|box_end|>`
- `<|object_ref_start|>` for direct condition
- `<|object_ref_end|>` for cot condition
- `<|quad_start|>` for noisy condition
- `<|quad_end|>` for synth condition

## Included Prepared Datasets

| Folder | Tokens | Approx size | Description |
|---|---:|---:|---|
| `koterm_pretrain_mix_v1` | 711.3M | 2.8G | Initial stage-0 mixture: HRM sample, SWE/GLM, Korean legal task data, ToolBench train |
| `hrm_cleaned_base_sample_v1` | 250.0M | 994M | Retokenized sample from upstream HRM cleaned data |
| `sft_swe_glm_mix_v1` | 251.2M | 990M | SWE-ZERO plus GLM reasoning pilot mix |
| `sft_swe_zero_v1` | 182.7M | 720M | SWE-ZERO terminal/code trajectories |
| `sft_glm_reasoning_v1` | 68.5M | 282M | GLM reasoning samples |
| `sft_korean_legal_v1` | 83.1M | 336M | Korean legal task-style data |
| `sft_toolbench_v1` | 127.0M | 500M | ToolBench train tool-call data |
| `koterm_hrm_cleaned_fastcap_stage1_v1` | 14.55B | 148G | HRM cleaned fast-cap stage-1 dataset |
| `korean_legal_raw_full_v1` | 308.9M | 1.2G | Korean statutes and local ordinances, chunked raw/task style |
| `korean_admrule_precedent_raw_full_v1` | 271.7M | 1.1G | Korean administrative rules and precedents |
| `kowiki_raw_full_v1` | 462.5M | 1.8G | Korean Wikipedia articles converted into training chunks |
| `hf_extra_reasoning_agent_mm_v1` | 112.6M | 444M | Small prepared subset from extra reasoning/agent/multimodal text sources |
| `local_terminal_conversations_ctx9k_resp6k_v1` | 9.39B | 36G | Local terminal/code/math conversations converted into HRM-Text training records |
| `korean_legal_tasks_full_v1` | 629.0M | 2.5G | Uncapped task-style Korean legal/admin data generated from statutes, local ordinances, administrative rules, and precedents |
| `sft_bcai_finance_kor_v1` | 857.7M | 3.3G | Korean finance instruction-response data from BCAI Finance Kor |

`smoke_hrm_parquet_v1` is a local smoke-test dataset and is intentionally not part of the main upload unless explicitly needed.

## Scheduled Follow-Up Uploads

The first public prepared-data upload contains the completed datasets listed above. The uncapped Korean legal/admin task dataset was uploaded as a follow-up on 2026-05-23 UTC. One large follow-up addition is still scheduled from the KoHRM training machine:

| Folder | Status | Description |
|---|---|---|
| `koterm_hrm_cleaned_full_nocap_v1` | waiting for tokenizer finish | Full/no-cap retokenized upstream HRM 328G cleaned corpus packed as HRM-Text V1Dataset |

The follow-up uploads use the same KoHRM 131K tokenizer and the same HRM-Text PrefixLM response-only training layout.

## Korean Legal Full Task Upload

The full Korean legal/admin task upload is available in the repository files:

| Path | Description |
|---|---|
| `korean_legal_tasks_full_v1/` | Prepared V1Dataset, about 629M tokens and 2.5G on disk |
| `raw_jsonl/korean_legal_tasks_full_20260524.jsonl` | Raw task JSONL, 1,383,749 rows and about 4.12GB |
| `LEGAL_FULL_TASKS_README.md` | Source note for the legal/admin full task upload |
| `sft_bcai_finance_kor_v1/` | Prepared V1Dataset, 857,699,372 tokens and about 3.3G on disk |
| `raw_jsonl/bcai_finance_kor_hrm_20260524.jsonl` | Raw HRM-converted finance JSONL, 1,862,508 rows and about 5.3G |
| `FINANCE_BCAI_README.md` | Source note for the BCAI finance upload |

## Source Attribution

Major sources used while constructing these prepared datasets:

| Source | Link / origin | Usage |
|---|---|---|
| HRM-Text cleaned data | https://huggingface.co/datasets/sapientinc/HRM-Text-data-io-cleaned-20260515 | Retokenized sample and fast-cap stage-1 data |
| HRM-Text code/paper | https://github.com/sapientinc/HRM-Text, https://arxiv.org/html/2605.20613 | Training format, PrefixLM objective, V1Dataset style |
| Korean Wikipedia | https://dumps.wikimedia.org/kowiki/20260501/ | Korean general/wiki text |
| Korean statutes | https://github.com/legalize-kr/legalize-kr | Korean legal raw text and task data |
| Korean local ordinances | https://github.com/legalize-kr/ordinance-kr | Korean local law raw text |
| Korean administrative rules | local Markdown snapshot at `/home/work/.projects/LLM-OS-Models/Terminal/admrule-kr` | Korean administrative-rule text and task rows |
| Korean precedents | local Markdown snapshot at `/home/work/.projects/LLM-OS-Models/Terminal/precedent-kr` | Korean precedent text and task rows |
| ToolBench | local `data_toolbench` extraction from ToolBench train data | Tool-call/API/JSON trajectories; eval split excluded |
| SWE-ZERO | https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories | Terminal/code trajectory subset |
| GLM reasoning | https://huggingface.co/datasets/Jackrong/GLM-5.1-Reasoning-1M-Cleaned | Reasoning/instruction subset |
| Claude reasoning sample | https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k | Small reviewed reasoning subset |
| Open-MM-RL | https://huggingface.co/datasets/TuringEnterprises/Open-MM-RL | Text-only subset review |
| DeepSeek agent traces | https://huggingface.co/datasets/TeichAI/DeepSeek-v4-Pro-Agent | Limited agent/tool-use subset; license-sensitive |
| structured Wikipedia | https://huggingface.co/datasets/wikimedia/structured-wikipedia | Tokenizer/general text support |
| BCAI Finance Kor | https://huggingface.co/datasets/BCCard/BCAI-Finance-Kor-1862K | Korean finance instruction-response data in `sft_bcai_finance_kor_v1` |

Licenses and terms remain those of the original data sources. This upload does not relicense upstream content. Users should verify source licenses before downstream redistribution or commercial use.

## Training Plan

These prepared datasets are used in staged pretraining:

1. `koterm_pretrain_mix_v1` for stage-0 and stage0b.
2. `koterm_hrm_cleaned_fastcap_stage1_v1` for the current stage-1 run.
3. Korean raw full, Wikipedia, terminal, SWE, ToolBench, and extra reasoning datasets for later mixed stages.
4. Full no-cap retokenization of the upstream HRM 328G cleaned corpus is still in progress and will be added as a later dataset when complete.
5. The uncapped Korean legal/admin task dataset is generated as a follow-up so the raw legal corpus and task-style legal corpus are both available.
6. BCAI Finance Kor is prepared and uploaded as a Korean finance/domain instruction dataset for the next staged continuation.

SFT-style datasets are also used during pretraining first. A cleaner, more strongly weighted SFT pass is planned after the pretraining continuation.

## Contamination Policy

The following are excluded from training where identified:

- ToolBench eval split
- Terminal Bench 2 style evaluation data
- `actava/chi-bench` benchmark data
- benchmark-only or evaluation-only splits

## Reproduction

Use the code repository for preprocessing scripts and staged training commands:

```bash
git clone https://github.com/LLM-OS-Models/KoHRM-text
cd KoHRM-text
pip install -r requirements.txt
```

The important scripts are in `scripts/`, especially the SFT/V1Dataset preparation scripts, Korean corpus builders, terminal conversation builders, merge scripts, and HRM retokenization runbooks documented in the repository.
