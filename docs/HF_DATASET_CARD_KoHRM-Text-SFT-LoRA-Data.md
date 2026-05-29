---
license: other
language:
  - ko
  - en
task_categories:
  - text-generation
  - question-answering
pretty_name: KoHRM-Text-1.4B SFT and LoRA Prepared Data
tags:
  - kohrm
  - hrm-text
  - prefixlm
  - sft
  - lora
  - korean
  - terminal
  - tool-use
---

# KoHRM-Text-1.4B SFT and LoRA Prepared Data

This dataset repo stores curated KoHRM SFT/LoRA subsets in the same tokenized
HRM-Text V1Dataset format used by training. It is intended for quick behavior
alignment experiments after KoHRM pretraining.

Model repo:

```text
https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B
```

Code repo:

```text
https://github.com/LLM-OS-Models/KoHRM-text
```

## Format

Each folder is a prepared V1Dataset:

```text
<dataset-name>/
  metadata.json
  tokenizer_info.json
  tokenizer.json
  tokens.npy
  epoch_0/
    inst_start.npy
    inst_len.npy
    resp_start.npy
    resp_len.npy
  ...
```

The token layout follows the KoHRM/HRM-Text PrefixLM convention:

```text
<|im_start|><|object_ref_start|>instruction<|im_end|>response<|box_end|>
```

Instruction/prefix tokens are read bidirectionally and excluded from the loss.
Response tokens are trained with causal response-only cross entropy.

## Tokenizer

```text
type:       byte-level BPE
vocab:      131,072
context:    4,096 model tokens
max_seq_len in prepared metadata: 4,097 including shift/packing convention
boq:        <|im_start|>
eoq:        <|im_end|>
eoa:        <|box_end|>
direct:     <|object_ref_start|>
cot:        <|object_ref_end|>
noisy:      <|quad_start|>
synth:      <|quad_end|>
```

## Recommended Use

Start with a small LoRA run, not full SFT:

```text
1. kohrm_sft_behavior_mini_v1
2. kohrm_sft_korean_domain_core_v1 if Korean legal/finance answers are weak
3. kohrm_sft_terminal_tool_core_v1 if terminal/tool behavior is weak
4. kohrm_sft_behavior_core_v1 for a broader final behavior pass
```

The current public KoHRM checkpoints are pretraining checkpoints. If they
produce repeated phrases, English agent traces, or malformed JSON, that is a
behavior-alignment issue rather than evidence that the tokenizer or prepared
data format is unusable. These SFT/LoRA subsets exist to correct those output
habits after pretraining.

## Datasets

### Experiment Mixes

```text
kohrm_sft_behavior_mini_v1
  tokens: 60,000,387
  samples: 61,810
  size: about 251M
  purpose: quick LoRA/SFT smoke test for Korean answer style, JSON/tool-call
           form, terminal command behavior, and repetition risk.

kohrm_sft_terminal_tool_core_v1
  tokens: 165,007,375
  samples: 55,934
  size: about 652M
  purpose: terminal trajectories, tool-call JSON, SWE/code workflow, and
           agent next-action behavior.

kohrm_sft_korean_domain_core_v1
  tokens: 100,000,654
  samples: 219,072
  size: about 428M
  purpose: Korean legal/admin-rule extraction, Korean finance QA, concise
           Korean explanation style.

kohrm_sft_behavior_core_v1
  tokens: 285,008,218
  samples: 291,382
  size: about 1.2G
  purpose: broad behavior alignment mix across terminal/tool/code/reasoning
           and Korean legal/finance data.
```

### Component Subsets

```text
kohrm_sft_comp_terminal_80m_v1
  tokens: 80,001,183
  samples: 23,374
  source: local terminal/code conversations
  purpose: terminal and coding trajectory behavior.

kohrm_sft_comp_toolbench_30m_v1
  tokens: 30,002,879
  samples: 15,210
  source: ToolBench/ToolLLaMA train data
  purpose: tool selection and JSON argument generation.

kohrm_sft_comp_swe_zero_30m_v1
  tokens: 30,001,177
  samples: 8,826
  source: AlienKevin/SWE-ZERO-12M-trajectories subset
  purpose: SWE/code trajectory behavior.

kohrm_sft_comp_glm_reasoning_20m_v1
  tokens: 20,000,189
  samples: 16,376
  source: Jackrong/GLM-5.1-Reasoning-1M-Cleaned subset
  purpose: final-answer reasoning behavior. Long private reasoning is not the
           primary target.

kohrm_sft_comp_agent_reasoning_25m_v1
  tokens: 25,002,136
  samples: 8,524
  source: small extra reasoning/agent/multimodal-text subset from reviewed HF
          candidates such as Claude-style reasoning, DeepSeek agent traces,
          and Open-MM-RL text portions
  purpose: agent/reasoning dialogue behavior.

kohrm_sft_comp_korean_legal_50m_v1
  tokens: 50,000,209
  samples: 110,578
  source: Korean legal/admin-rule prepared tasks
  purpose: Korean legal extraction and grounded Korean explanation.

kohrm_sft_comp_finance_50m_v1
  tokens: 50,000,445
  samples: 108,494
  source: BCCard/BCAI-Finance-Kor-1862K prepared subset
  purpose: Korean finance QA and domain explanation.
```

## Sources

The prepared data is derived from multiple local and public sources:

```text
local terminal/code conversations:
  /home/work/.data/hrm_text_prepared/local_terminal_conversations_ctx9k_resp6k_v1

ToolBench / ToolLLaMA train data:
  local HRM-Text data_toolbench extraction
  eval split excluded

SWE-ZERO:
  https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories

GLM reasoning:
  https://huggingface.co/datasets/Jackrong/GLM-5.1-Reasoning-1M-Cleaned

BCAI Finance Kor:
  https://huggingface.co/datasets/BCCard/BCAI-Finance-Kor-1862K

extra reasoning/agent text subsets:
  https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k
  https://huggingface.co/datasets/TeichAI/DeepSeek-v4-Pro-Agent
  https://huggingface.co/datasets/TuringEnterprises/Open-MM-RL
```

Benchmark/evaluation-like data is excluded where identified, including
ToolBench eval, Terminal Bench style evaluation data, and benchmark-oriented
`chi-bench` data.

## License and Use Notes

This dataset repo is a prepared training artifact with mixed upstream sources.
Do not assume a single permissive license for all examples. Check the source
dataset licenses and terms before redistributing derivatives or training a
public commercial model on a particular subset.

The KoHRM model repo may use Apache-2.0 for code/model release metadata, but
that does not automatically relicense the upstream data sources.

## Loading in KoHRM Training

Example LoRA command:

```bash
export RESUME_FROM=/path/to/KoHRM/full/checkpoint
bash scripts/run_kohrm_lora_experiments.sh behavior-mini
```

Manual command:

```bash
torchrun --standalone --nproc_per_node=8 train_lora.py \
  --config-name cfg_lora \
  arch/size@arch=XL \
  data.path=/home/work/.data/hrm_text_prepared/kohrm_sft_behavior_mini_v1 \
  resume_from=/path/to/KoHRM/full/checkpoint \
  checkpoint_path=/home/work/.data/hrm_text_lora/KoHRM-Text-1.4B-lora-behavior-mini-v1 \
  run_name=KoHRM-Text-1.4B-lora-behavior-mini-v1 \
  global_batch_size=32768 \
  epochs=1 \
  lr=8.0e-5 \
  checkpoint_step_interval=1000 \
  checkpoint_keep_last=2 \
  lora.rank=16 \
  lora.alpha=32.0
```
