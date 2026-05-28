---
license: apache-2.0
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
- prefix-lm
library_name: pytorch
pipeline_tag: text-generation
---

# KoHRM-Text-1.4B

**Language / 언어:** [English](#english) | [한국어](#korean)

<a id="english"></a>

## English

`KoHRM-Text-1.4B` is a scratch-pretrained Korean/English/code/terminal/tool-use model built from the `sapientinc/HRM-Text` PrefixLM training stack.

This is **not** a continued finetune of `sapientinc/HRM-Text-1B`. It uses a new Korean/terminal-oriented 131K byte-level BPE tokenizer and a new scratch training run.

### Current Status

This repository is a rolling **latest public model export**. Training is still in progress.

- Main repo: `LLM-OS-Models/KoHRM-Text-1.4B`
- Current public files: `model.safetensors`, `config.json`, tokenizer files, and this `README.md`
- Raw FSDP2 resume checkpoints: `LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints`
- Prepared data: `LLM-OS-Models/KoHRM-Text-1.4B-prepared-data`
- Project code: https://github.com/LLM-OS-Models/KoHRM-text
- Upstream HRM-Text code: https://github.com/sapientinc/HRM-Text
- HRM-Text paper: https://arxiv.org/html/2605.20613
- Tokenizer repo: `LLM-OS-Models/HRM-Text-Ko-Terminal-Tokenizer-131K`

The main branch is overwritten with the newest converted EMA `safetensors` export as training checkpoints are uploaded. To test the latest public weight, download `revision="main"`.

### Training Method At A Glance

KoHRM-Text is best understood as **instruction pretraining from scratch**.

It is not ordinary raw-text causal LM pretraining, and it is not only a small SFT pass on top of an existing base model.

```text
raw data -> tokenizer -> V1Dataset -> PrefixLM batches
         -> HRM H/L recurrence -> LM head -> response-only loss
```

The input context is handled as a PrefixLM prefix:

```text
instruction / prefix: bidirectional attention, no loss
response:             causal attention, response-only CE loss
```

The architecture keeps the upstream HRM-Text recurrent design:

```text
H module: slower strategic state
L module: faster execution state
schedule: H2L3 recurrent computation
```

For a readable full explanation of the training method, architecture, PT/SFT distinction, staged continuation, and checkpoint naming, see the project document:

[MODEL_TRAINING_ARCHITECTURE_GUIDE_2026-05-28.md](MODEL_TRAINING_ARCHITECTURE_GUIDE_2026-05-28.md) in https://github.com/LLM-OS-Models/KoHRM-text

### Important Compatibility Note

The public repo currently contains the converted model weights and tokenizer, but it does **not yet** include a Hugging Face `trust_remote_code` modeling implementation for `HrmTextForCausalLM`.

What works today:

- Download the latest public weights.
- Load the tokenizer directly with `tokenizers.Tokenizer.from_file("tokenizer.json")`.
- Inspect `config.json`.
- Verify `model.safetensors` on CPU or Colab T4.

What is not supported yet in plain Transformers:

- `AutoModelForCausalLM.from_pretrained("LLM-OS-Models/KoHRM-Text-1.4B")`
- One-line hosted text generation from this repo

Expected reason: `model_type: "hrm_text"` is a custom HRM-Text architecture. Public generation will require adding the compatible `HrmTextForCausalLM` remote-code files to this model repo or releasing a standard wrapper.

### Model Details

| Field | Value |
|---|---:|
| Model id | `LLM-OS-Models/KoHRM-Text-1.4B` |
| Standard name | `KoHRM-Text-1.4B` |
| Training origin | scratch |
| Architecture family | HRM-Text PrefixLM |
| Architecture size | `XL` |
| Parameters | 1,384,120,320 |
| Context length | 4,096 tokens |
| Training dtype | bfloat16 |
| Public export dtype | bfloat16 EMA `safetensors` |
| Tokenizer | byte-level BPE, NFC normalization |
| Vocabulary size | 131,072 |
| Objective | PrefixLM response-only loss |
| Optimizer | Adam-atan2 from upstream HRM-Text |
| EMA | 0.9999 |

Converted config highlights:

```json
{
  "model_type": "hrm_text",
  "architectures": ["HrmTextForCausalLM"],
  "vocab_size": 131072,
  "hidden_size": 1536,
  "num_hidden_layers": 32,
  "num_attention_heads": 12,
  "max_position_embeddings": 4096,
  "prefix_lm": true
}
```

### Compared With The HRM-Text Paper

This run can take longer than the paper recipe even on 8 x H200 because the setup is not identical:

- The paper reference used 16 x H100; this run uses 8 x H200.
- KoHRM uses a larger 131K tokenizer vocabulary, compared with the upstream 65K tokenizer.
- The public KoHRM size is about 1.38B parameters.
- The stable long-run batch is `180,224` tokens/step after OOM probing; larger batches were possible briefly but not chosen for reliability.
- The continuation includes extra Korean, terminal, tool-call, legal, finance, wiki, and repeated HRM-cleaned stages.

This does not automatically guarantee better benchmark scores. The expected upside is domain-specific: Korean tokenization efficiency, Korean legal/finance/wiki coverage, terminal trajectories, tool-call formatting, and code-oriented behavior should have a better chance than the upstream English/general checkpoint. Final claims require evaluation after the planned continuation and SFT finish.

### Tokenizer

The tokenizer was trained for Korean, English, code, shell/terminal text, and JSON/tool-call formats. It keeps common chat/tool special tokens as stable single tokens where possible.

| Sample bucket | chars/token |
|---|---:|
| Korean general text | 2.60 |
| Korean legal text | 2.36 |
| Korean terminal instruction | 2.18 |
| shell command | 2.68 |
| tool-call JSON | 3.32 |
| Python code | 3.37 |
| English | 4.40 |

Formatting tokens:

```text
<|im_start|>         instruction start
<|im_end|>           instruction end
<|box_end|>          response/end marker
<|object_ref_start|> direct condition
<|object_ref_end|>   chain-of-thought style condition
<|quad_start|>       noisy condition
<|quad_end|>         synthetic condition
```

Prompt format used by the project-side inference code:

```text
<|im_start|><|object_ref_start|>YOUR_PROMPT_HERE<|im_end|>
```

### Colab T4 Quick Generation

A ready-to-run Colab notebook is available in the project repo:

https://github.com/LLM-OS-Models/KoHRM-text/blob/main/notebooks/KoHRM_Text_1_4B_Colab_T4_Smoke_Test.ipynb

The notebook downloads the latest public files and runs a short generation test on a Colab T4.

It intentionally avoids `transformers`, `AutoTokenizer`, and `AutoModelForCausalLM`. Instead, it uses:

- `tokenizers.Tokenizer.from_file("tokenizer.json")`
- `safetensors.torch.load_file("model.safetensors")`
- `kohrm_colab_generate.py`, a small PyTorch SDPA runtime for the HRM-Text architecture

```python
!pip -q install -U huggingface_hub hf_transfer tokenizers safetensors
```

```python
from pathlib import Path
import json
import importlib.util
import sys
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

repo_id = "LLM-OS-Models/KoHRM-Text-1.4B"

repo_dir = Path(snapshot_download(
    repo_id,
    revision="main",
    allow_patterns=[
        "README.md",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "kohrm_colab_generate.py",
    ],
))

print("Downloaded to:", repo_dir)
config = json.loads((repo_dir / "config.json").read_text())
print("model_type:", config["model_type"])
print("hidden_size:", config["hidden_size"])
print("vocab_size:", config["vocab_size"])
print("context:", config["max_position_embeddings"])

tokenizer = Tokenizer.from_file(str(repo_dir / "tokenizer.json"))
wrapped = "<|im_start|><|object_ref_start|>한국어로 현재 디렉터리에서 가장 큰 파일 10개를 찾는 명령을 알려주세요.<|im_end|>"
ids = tokenizer.encode(wrapped).ids
print("prompt tokens:", len(ids))
print("first token ids:", ids[:20])

spec = importlib.util.spec_from_file_location(
    "kohrm_colab_generate",
    repo_dir / "kohrm_colab_generate.py",
)
kohrm = importlib.util.module_from_spec(spec)
sys.modules["kohrm_colab_generate"] = kohrm
spec.loader.exec_module(kohrm)

output = kohrm.generate_text(
    repo_dir,
    "한국어로 현재 디렉터리에서 가장 큰 파일 10개를 찾는 bash 명령을 알려주세요. 명령만 간단히 답하세요.",
    max_new_tokens=64,
    max_seq_len=512,
    temperature=0.0,
)
print(output)
```

Expected result:

- `model_type` should be `hrm_text`.
- `vocab_size` should be `131072`.
- The helper should load the 1.38B public `model.safetensors` export.
- On Colab T4, generation runs in fp16 through PyTorch scaled-dot-product attention.
- First generation can take a few minutes because it downloads and loads the full weight file.

Prompt format used by the helper:

```text
<|im_start|><|object_ref_start|>PROMPT<|im_end|>
```

Plain `AutoModelForCausalLM.generate()` is still not the supported path. This model is a custom `hrm_text` architecture, so ordinary Transformers generation requires a future `trust_remote_code` wrapper. Use the notebook/helper above for public `model.safetensors` generation today.

### Internal Raw-Checkpoint Generation

For training-machine debugging and exact raw FSDP2 checkpoint recovery, the project still includes the upstream-style inference path:

- `simple_inference_engine.py`
- raw checkpoints from `LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints`
- CUDA/FlashAttention-oriented execution

That path is mainly for internal continuation/evaluation, not the easiest Colab test.


### Training Data

Prepared data artifacts are uploaded to:

https://huggingface.co/datasets/LLM-OS-Models/KoHRM-Text-1.4B-prepared-data

The training objective is PrefixLM response-only loss. Instruction/prompt tokens are visible as context, while loss is applied to the response span.

Major prepared data groups:

| Dataset group | Tokens | Use |
|---|---:|---|
| `koterm_pretrain_mix_v1` | 711.3M | stage-0/stage0b |
| HRM cleaned fast-cap stage1/stage1b | 14.55B | HRM-style instruction pretraining |
| HRM cleaned full/no-cap stage2 | 14.55B | completed continuation |
| HRM cleaned full/no-cap extra stage2b | 14.55B | active continuation |
| Local terminal conversations | 9.39B | terminal/code/tool-heavy continuation |
| Korean tool/legal/wiki/finance mix | 3.02B | Korean domain and tool continuation |
| BCAI Finance Korean | 857.7M | Korean finance/domain data |
| Korean legal/admin task data | 629.0M | Korean legal/admin data |
| Korean Wikipedia | 462.5M | Korean general text |
| ToolBench train tool-call data | 127.0M | tool-call pretraining |
| SWE-ZERO + GLM reasoning subsets | 251.2M | code/reasoning data |

Evaluation-like datasets are excluded where identified, including ToolBench eval, Terminal Bench style evaluation data, and benchmark-oriented `chi-bench` data.

### Training Run

The current run uses staged continuation:

```text
stage0
-> stage0b
-> stage1
-> stage2
-> stage3
-> stage4
-> stage1b
-> stage2b
-> stage3b
-> stage4b
-> stage1c
-> stage2c
-> stage3c
-> stage4c
```

The checkpoint carries model weights, optimizer state, EMA weights, and recurrent carry state. `resume_step_offset` and `total_steps_override` are used so the learning-rate schedule follows the intended longer run instead of resetting at each stage.

As of 2026-05-27, `stage2b` is active. The continuation watcher is scheduled to launch `stage3b -> stage4b -> stage1c -> stage2c -> stage3c -> stage4c` after each completed checkpoint. The handoff reads the actual `epoch_1_info.json` `global_step` from each completed checkpoint before starting the next stage.

### Intended Use

This checkpoint is intended for:

- continued pretraining experiments
- Korean tokenizer and HRM-Text architecture experiments
- terminal/tool-call/code pretraining research
- checkpoint conversion and evaluation work

It is not yet intended as a finished assistant model.

### Limitations

- This is an intermediate checkpoint, not a final aligned instruct model.
- The full planned continuation has not finished.
- Final SFT and safety tuning have not been completed.
- Public benchmark scores for this new checkpoint are not final.
- Plain Transformers generation requires adding the custom `hrm_text` modeling wrapper or remote-code files.
- Tool-call JSON validity and terminal action safety must be evaluated before production use.

### Citation

This work builds on HRM-Text:

- Paper: https://arxiv.org/html/2605.20613
- Upstream code: https://github.com/sapientinc/HRM-Text

<a id="korean"></a>

## 한국어

`KoHRM-Text-1.4B`는 `sapientinc/HRM-Text`의 PrefixLM 학습 스택을 기반으로 처음부터 학습 중인 한국어/영어/코드/터미널/툴콜 모델입니다.

이 모델은 `sapientinc/HRM-Text-1B`를 이어서 파인튜닝한 모델이 아닙니다. 한국어와 터미널/툴콜 형식에 맞춰 새로 만든 131K byte-level BPE tokenizer를 사용하며, 가중치도 scratch pretraining으로 학습합니다.

### 현재 상태

이 저장소는 최신 공개 변환본을 계속 덮어쓰는 rolling latest model repo입니다. 학습은 아직 진행 중입니다.

- 메인 모델 repo: `LLM-OS-Models/KoHRM-Text-1.4B`
- 현재 공개 파일: `model.safetensors`, `config.json`, tokenizer 파일, `README.md`
- raw FSDP2 resume checkpoint: `LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints`
- prepared data: `LLM-OS-Models/KoHRM-Text-1.4B-prepared-data`
- 프로젝트 코드: https://github.com/LLM-OS-Models/KoHRM-text
- 원본 HRM-Text 코드: https://github.com/sapientinc/HRM-Text
- HRM-Text 논문: https://arxiv.org/html/2605.20613
- tokenizer repo: `LLM-OS-Models/HRM-Text-Ko-Terminal-Tokenizer-131K`

최신 공개 weight를 테스트하려면 `revision="main"`으로 다운로드하면 됩니다. 학습 중 10,000 step 단위로 새 checkpoint가 변환되어 올라오면 같은 파일명이 최신 EMA `safetensors`로 갱신됩니다.

### 학습 방식 한눈에 보기

KoHRM-Text는 **scratch instruction pretraining**으로 보는 것이 가장 정확합니다.

일반적인 raw-text causal LM 사전학습도 아니고, 이미 완성된 base model 위에 짧게 얹는 SFT만도 아닙니다.

```text
raw data -> tokenizer -> V1Dataset -> PrefixLM batches
         -> HRM H/L recurrence -> LM head -> response-only loss
```

입력 컨텍스트는 PrefixLM prefix로 처리합니다.

```text
instruction / prefix: 양방향 attention, loss 없음
response:             causal attention, response-only CE loss
```

아키텍처는 원본 HRM-Text recurrent design을 유지합니다.

```text
H module: 느리게 변하는 전략 state
L module: 빠르게 변하는 실행 state
schedule: H2L3 recurrent computation
```

학습 방식, 아키텍처, PT/SFT 차이, staged continuation, checkpoint 이름을 쉽게 풀어 쓴 전체 설명은 프로젝트 문서를 기준으로 보면 됩니다.

[MODEL_TRAINING_ARCHITECTURE_GUIDE_2026-05-28.md](MODEL_TRAINING_ARCHITECTURE_GUIDE_2026-05-28.md) in https://github.com/LLM-OS-Models/KoHRM-text

### 중요한 호환성 안내

현재 공개 repo에는 변환된 model weight와 tokenizer가 있지만, 아직 Hugging Face `trust_remote_code`용 `HrmTextForCausalLM` 구현 파일은 포함되어 있지 않습니다.

현재 바로 가능한 것:

- 최신 공개 weight 다운로드
- `tokenizers.Tokenizer.from_file("tokenizer.json")`로 tokenizer 로드
- `config.json` 확인
- CPU 또는 Colab T4에서 `model.safetensors` 무결성 확인

아직 일반 Transformers에서 바로 안 되는 것:

- `AutoModelForCausalLM.from_pretrained("LLM-OS-Models/KoHRM-Text-1.4B")`
- 이 repo만으로 one-line text generation 실행

이유는 `model_type: "hrm_text"`가 custom HRM-Text architecture이기 때문입니다. 공개 generation을 하려면 이 model repo에 `HrmTextForCausalLM` remote-code wrapper가 추가되어야 합니다.

### 모델 상세

| 항목 | 값 |
|---|---:|
| 모델 ID | `LLM-OS-Models/KoHRM-Text-1.4B` |
| 표준 이름 | `KoHRM-Text-1.4B` |
| 학습 출발점 | scratch |
| 아키텍처 계열 | HRM-Text PrefixLM |
| 아키텍처 크기 | `XL` |
| 파라미터 | 1,384,120,320 |
| 컨텍스트 길이 | 4,096 tokens |
| 학습 dtype | bfloat16 |
| 공개 변환본 dtype | bfloat16 EMA `safetensors` |
| tokenizer | byte-level BPE, NFC normalization |
| vocabulary size | 131,072 |
| objective | PrefixLM response-only loss |
| optimizer | HRM-Text의 Adam-atan2 |
| EMA | 0.9999 |

변환된 config 주요 값:

```json
{
  "model_type": "hrm_text",
  "architectures": ["HrmTextForCausalLM"],
  "vocab_size": 131072,
  "hidden_size": 1536,
  "num_hidden_layers": 32,
  "num_attention_heads": 12,
  "max_position_embeddings": 4096,
  "prefix_lm": true
}
```

### HRM-Text 논문 대비

현재 run은 논문 recipe보다 더 오래 걸릴 수 있습니다. 설정이 완전히 같지 않기 때문입니다.

- 논문 기준은 16 x H100이고, 현재 run은 8 x H200입니다.
- KoHRM은 원본 65K tokenizer보다 큰 131K tokenizer vocab을 씁니다.
- 공개 KoHRM 크기는 약 1.38B parameters입니다.
- 안정 장기 run batch는 OOM probe 이후 `180,224` tokens/step으로 잡았습니다. 더 큰 batch는 초반에 가능해 보여도 장기 안정성이 떨어졌습니다.
- 한국어, 터미널, 툴콜, 법률, 금융, 위키, HRM-cleaned 반복 stage가 추가됐습니다.

이것이 자동으로 모든 benchmark 점수 상승을 보장하지는 않습니다. 다만 한국어 토크나이저 효율, 한국어 법률/금융/위키 coverage, 터미널 trajectory, tool-call formatting, code-oriented behavior 쪽은 원본 영어/general checkpoint보다 좋아질 가능성이 있습니다. 최종 주장은 continuation과 SFT가 끝난 뒤 평가로 확인해야 합니다.

### 토크나이저

토크나이저는 한국어, 영어, 코드, shell/terminal 텍스트, JSON/tool-call 형식을 고려해서 만들었습니다. 자주 쓰는 chat/tool special token은 가능한 한 안정적인 단일 token으로 유지합니다.

| 샘플 종류 | chars/token |
|---|---:|
| 한국어 일반 | 2.60 |
| 한국어 법률 | 2.36 |
| 한국어 터미널 지시 | 2.18 |
| shell command | 2.68 |
| tool-call JSON | 3.32 |
| Python code | 3.37 |
| 영어 | 4.40 |

포맷 token:

```text
<|im_start|>         instruction 시작
<|im_end|>           instruction 종료
<|box_end|>          response/end marker
<|object_ref_start|> direct condition
<|object_ref_end|>   chain-of-thought style condition
<|quad_start|>       noisy condition
<|quad_end|>         synthetic condition
```

프로젝트 내부 inference code가 쓰는 prompt 형식:

```text
<|im_start|><|object_ref_start|>여기에_프롬프트를_넣습니다<|im_end|>
```

### Colab T4 빠른 생성

바로 실행할 수 있는 Colab 노트북은 project repo에 있습니다.

https://github.com/LLM-OS-Models/KoHRM-text/blob/main/notebooks/KoHRM_Text_1_4B_Colab_T4_Smoke_Test.ipynb

이 노트북은 Colab T4에서 최신 공개 파일을 다운로드하고 짧은 생성을 직접 실행합니다.

일부 Colab 환경에서 `transformers`가 `torchvision::nms` import 오류를 내거나 custom architecture를 못 찾는 문제가 생길 수 있으므로, 이 노트북은 `AutoTokenizer`와 `AutoModelForCausalLM`을 쓰지 않습니다. 대신 아래 경로를 사용합니다.

- `tokenizers.Tokenizer.from_file("tokenizer.json")`
- `safetensors.torch.load_file("model.safetensors")`
- HRM-Text 구조를 직접 구현한 `kohrm_colab_generate.py`

```python
!pip -q install -U huggingface_hub hf_transfer tokenizers safetensors
```

```python
from pathlib import Path
import json
import importlib.util
import sys
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

repo_id = "LLM-OS-Models/KoHRM-Text-1.4B"

repo_dir = Path(snapshot_download(
    repo_id,
    revision="main",
    allow_patterns=[
        "README.md",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "kohrm_colab_generate.py",
    ],
))

print("Downloaded to:", repo_dir)
config = json.loads((repo_dir / "config.json").read_text())
print("model_type:", config["model_type"])
print("hidden_size:", config["hidden_size"])
print("vocab_size:", config["vocab_size"])
print("context:", config["max_position_embeddings"])

tokenizer = Tokenizer.from_file(str(repo_dir / "tokenizer.json"))
wrapped = "<|im_start|><|object_ref_start|>한국어로 현재 디렉터리에서 가장 큰 파일 10개를 찾는 명령을 알려주세요.<|im_end|>"
ids = tokenizer.encode(wrapped).ids
print("prompt tokens:", len(ids))
print("first token ids:", ids[:20])

spec = importlib.util.spec_from_file_location(
    "kohrm_colab_generate",
    repo_dir / "kohrm_colab_generate.py",
)
kohrm = importlib.util.module_from_spec(spec)
sys.modules["kohrm_colab_generate"] = kohrm
spec.loader.exec_module(kohrm)

output = kohrm.generate_text(
    repo_dir,
    "한국어로 현재 디렉터리에서 가장 큰 파일 10개를 찾는 bash 명령을 알려주세요. 명령만 간단히 답하세요.",
    max_new_tokens=64,
    max_seq_len=512,
    temperature=0.0,
)
print(output)
```

정상 결과:

- `model_type`은 `hrm_text`입니다.
- `vocab_size`는 `131072`입니다.
- helper가 1.38B 공개 `model.safetensors` 변환본을 로드합니다.
- Colab T4에서는 fp16 PyTorch scaled-dot-product attention으로 생성합니다.
- 첫 실행은 2.8 GiB급 weight 다운로드와 로드 때문에 몇 분 걸릴 수 있습니다.

helper가 쓰는 prompt 형식:

```text
<|im_start|><|object_ref_start|>PROMPT<|im_end|>
```

일반 `AutoModelForCausalLM.generate()`는 아직 지원 경로가 아닙니다. 이 모델은 custom `hrm_text` architecture이므로, 일반 Transformers generation은 추후 `trust_remote_code` wrapper가 추가된 뒤 지원하는 것이 맞습니다. 지금 공개 `model.safetensors`로 바로 생성하려면 위 노트북/helper를 쓰면 됩니다.

### 내부 raw-checkpoint 생성

학습 머신에서 디버깅하거나 raw FSDP2 checkpoint를 정확히 복구해서 평가할 때는 upstream 스타일 inference 경로도 유지합니다.

- `simple_inference_engine.py`
- `LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints`의 raw checkpoints
- CUDA/FlashAttention 중심 실행

이 경로는 내부 continuation/evaluation용에 가깝고, Colab에서 가장 쉽게 확인하려면 위 공개 `model.safetensors` helper를 쓰는 것이 낫습니다.

### 학습 데이터

prepared data는 아래 dataset repo에 업로드합니다.

https://huggingface.co/datasets/LLM-OS-Models/KoHRM-Text-1.4B-prepared-data

학습 objective는 PrefixLM response-only loss입니다. instruction/prompt token은 context로 보고, loss는 response span에만 적용합니다.

주요 prepared data group:

| 데이터 그룹 | Tokens | 용도 |
|---|---:|---|
| `koterm_pretrain_mix_v1` | 711.3M | stage-0/stage0b |
| HRM cleaned fast-cap stage1/stage1b | 14.55B | HRM-style instruction pretraining |
| HRM cleaned full/no-cap stage2 | 14.55B | 완료된 continuation |
| HRM cleaned full/no-cap extra stage2b | 14.55B | 진행 중인 continuation |
| local terminal conversations | 9.39B | terminal/code/tool-heavy continuation |
| Korean tool/legal/wiki/finance mix | 3.02B | 한국어 domain/tool continuation |
| BCAI Finance Korean | 857.7M | 한국어 금융/domain data |
| Korean legal/admin task data | 629.0M | 한국어 법률/행정 data |
| Korean Wikipedia | 462.5M | 한국어 일반 텍스트 |
| ToolBench train tool-call data | 127.0M | tool-call pretraining |
| SWE-ZERO + GLM reasoning subsets | 251.2M | code/reasoning data |

평가 성격 데이터는 확인되는 범위에서 train에서 제외합니다. 예시는 ToolBench eval, Terminal Bench 계열 평가 데이터, benchmark 성격의 `chi-bench`입니다.

### 학습 진행

현재 run은 staged continuation 방식입니다.

```text
stage0
-> stage0b
-> stage1
-> stage2
-> stage3
-> stage4
-> stage1b
-> stage2b
-> stage3b
-> stage4b
-> stage1c
-> stage2c
-> stage3c
-> stage4c
```

checkpoint는 model weights, optimizer state, EMA weights, recurrent carry state를 이어갑니다. `resume_step_offset`과 `total_steps_override`를 써서 stage마다 learning-rate schedule이 리셋되지 않고 긴 pretraining run처럼 이어지게 합니다.

2026-05-27 기준 `stage2b`가 진행 중입니다. continuation watcher가 이후 `stage3b -> stage4b -> stage1c -> stage2c -> stage3c -> stage4c`를 이어서 실행하도록 예약되어 있습니다. handoff는 각 stage의 실제 `epoch_1_info.json` `global_step`을 읽고 다음 stage를 시작합니다.

### 사용 목적

이 checkpoint는 다음 목적에 적합합니다.

- continued pretraining 실험
- 한국어 tokenizer 및 HRM-Text architecture 실험
- terminal/tool-call/code pretraining 연구
- checkpoint conversion 및 evaluation 작업

아직 완성된 assistant model은 아닙니다.

### 제한 사항

- 중간 checkpoint이며 최종 aligned instruct model이 아닙니다.
- 전체 planned continuation이 아직 끝나지 않았습니다.
- 최종 SFT와 safety tuning이 아직 끝나지 않았습니다.
- 새 checkpoint의 public benchmark score는 아직 final이 아닙니다.
- 일반 Transformers generation은 custom `hrm_text` modeling wrapper 또는 remote-code file이 추가되어야 가능합니다.
- tool-call JSON 유효성과 terminal action safety는 실제 사용 전에 별도 평가가 필요합니다.

### 인용

이 작업은 HRM-Text architecture와 training stack을 기반으로 합니다.

- 논문: https://arxiv.org/html/2605.20613
- 원본 코드: https://github.com/sapientinc/HRM-Text
