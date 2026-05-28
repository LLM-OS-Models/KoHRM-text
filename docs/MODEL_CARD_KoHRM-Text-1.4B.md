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
library_name: transformers
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

### CPU / Colab T4 Quick Test

Use this to test the **latest public weight files** on CPU or a Colab T4 runtime. This verifies that the tokenizer, config, and `model.safetensors` are downloadable and readable.

It does not run text generation yet, because the public repo does not yet ship the custom HRM-Text modeling wrapper.

A ready-to-run Colab notebook is available in the project repo:

https://github.com/LLM-OS-Models/KoHRM-text/blob/main/notebooks/KoHRM_Text_1_4B_Colab_T4_Smoke_Test.ipynb

The notebook is optimized for a Colab T4 smoke test: it can download the latest public files, run tokenizer experiments without importing `transformers`, inspect `model.safetensors` shapes without fully loading all weights into GPU memory, and confirm the current expected Transformers compatibility limitation.

```python
!pip -q install -U huggingface_hub hf_transfer tokenizers safetensors
```

```python
from pathlib import Path
import json
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from safetensors import safe_open

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
    ],
))

print("Downloaded to:", repo_dir)
config = json.loads((repo_dir / "config.json").read_text())
print("model_type:", config["model_type"])
print("hidden_size:", config["hidden_size"])
print("vocab_size:", config["vocab_size"])
print("context:", config["max_position_embeddings"])

tokenizer = Tokenizer.from_file(str(repo_dir / "tokenizer.json"))
prompt = "<|im_start|><|object_ref_start|>한국어로 현재 디렉터리에서 가장 큰 파일 10개를 찾는 명령을 알려주세요.<|im_end|>"
ids = tokenizer.encode(prompt).ids
print("prompt tokens:", len(ids))
print("first token ids:", ids[:20])

# Fast CPU weight metadata check. This reads tensor shapes without loading every tensor.
with safe_open(repo_dir / "model.safetensors", framework="pt", device="cpu") as f:
    keys = list(f.keys())
    num_params = sum(__import__("math").prod(f.get_slice(k).get_shape()) for k in keys)
    first_key = keys[0]
    first_shape = f.get_slice(first_key).get_shape()

print("num_tensors:", len(keys))
print("num_params:", f"{num_params:,}")
print("first tensor:", first_key, tuple(first_shape))
```

Expected result:

- `model_type` should be `hrm_text`.
- `vocab_size` should be `131072`.
- `num_params` should be around `1.38B`.
- Tokenizer loading through `tokenizers.Tokenizer.from_file` should work on CPU and Colab T4.
- `AutoModelForCausalLM` generation is expected to be unavailable until remote-code support is added.

Do not use `AutoTokenizer` or `AutoModelForCausalLM` in this Colab smoke test. The public export is a custom `hrm_text` architecture and does not yet ship the remote-code model wrapper needed for plain Transformers generation.

### Internal / Project-Side Generation

For actual generation today, use the project code and raw FSDP2 checkpoints. This is the currently supported copy-paste path for CUDA machines. A BF16-capable GPU with enough VRAM is recommended; Colab T4 is useful for the smoke test above, not for this raw-checkpoint generation path.

```bash
git clone https://github.com/LLM-OS-Models/KoHRM-text
cd KoHRM-text
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
export TOKENIZERS_PARALLELISM=false
export NUMEXPR_MAX_THREADS=128
```

Download the latest uploaded raw checkpoint example. This example uses `stage1b-hrm-fastcap-repeat-step310000`, which is available in the raw checkpoint repo. When a newer raw checkpoint is uploaded, change both the include path and `ckpt_step`.

```bash
mkdir -p checkpoints/kohm-raw
huggingface-cli download LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints \
  --include "stage1b-hrm-fastcap-repeat-step310000/**" \
  --local-dir checkpoints/kohm-raw
```

Create and run a minimal generation script:

```bash
cat > run_kohrm_raw_generate.py <<'PY'
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "128")

from simple_inference_engine import inference_load_checkpoint, inference_generate

ckpt_dir = "checkpoints/kohm-raw/stage1b-hrm-fastcap-repeat-step310000"

prompts = [
    (
        0,
        (
            "direct",
            "한국어 존댓말로 현재 디렉터리에서 용량이 가장 큰 파일 10개를 찾는 bash 명령을 제안해 주세요.",
        ),
    ),
    (
        1,
        (
            "direct",
            "Write a Python function that validates a JSON tool-call object with name and arguments.",
        ),
    ),
]

ckpt = inference_load_checkpoint(
    ckpt_path=ckpt_dir,
    ckpt_epoch=None,
    ckpt_step=310000,
    ckpt_use_ema=True,
    device="cuda",
)

for pid, text in inference_generate(
    ckpt,
    iter(prompts),
    max_tokens=1024,
    max_generation=256,
    batch_size=1,
    temp=0.0,
):
    print(f"\n### sample {pid}\n{text}")
PY

python run_kohrm_raw_generate.py
```

Prompt format is handled by `InferenceCheckpoint.tokenize_prompt`. The first tuple item is the condition string, usually `"direct"`, and the second item is the user prompt. Internally this becomes:

```text
<|im_start|><|object_ref_start|>PROMPT<|im_end|>
```

If you want to test a newer raw checkpoint:

1. Check the raw checkpoint repo for the newest uploaded stage/step.
2. Change the `huggingface-cli download --include` pattern.
3. Change `ckpt_dir`.
4. Change `ckpt_step`.

Plain `AutoModelForCausalLM` generation from `model.safetensors` will be added later when the public `trust_remote_code` wrapper is available.

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

### CPU / Colab T4 빠른 테스트

아래 코드는 CPU 환경이나 Colab T4 런타임에서 최신 공개 weight 파일을 확인하는 용도입니다. tokenizer, config, `model.safetensors`가 정상적으로 받아지고 읽히는지 검증합니다.

아직 public repo에 custom HRM-Text modeling wrapper가 없기 때문에 이 코드는 text generation을 실행하지 않습니다.

바로 실행할 수 있는 Colab 노트북은 project repo에 있습니다.

https://github.com/LLM-OS-Models/KoHRM-text/blob/main/notebooks/KoHRM_Text_1_4B_Colab_T4_Smoke_Test.ipynb

이 노트북은 T4 Colab에서 최신 공개 파일 다운로드, `transformers`를 import하지 않는 tokenizer 실험, `model.safetensors` shape 검사, 현재 Transformers 호환성 제한 확인을 빠르게 수행하도록 작성되어 있습니다. 전체 weight를 GPU에 올려 생성하는 노트북이 아니라 공개 artifact 스모크 테스트용입니다.

```python
!pip -q install -U huggingface_hub hf_transfer tokenizers safetensors
```

```python
from pathlib import Path
import json
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from safetensors import safe_open

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
    ],
))

print("Downloaded to:", repo_dir)
config = json.loads((repo_dir / "config.json").read_text())
print("model_type:", config["model_type"])
print("hidden_size:", config["hidden_size"])
print("vocab_size:", config["vocab_size"])
print("context:", config["max_position_embeddings"])

tokenizer = Tokenizer.from_file(str(repo_dir / "tokenizer.json"))
prompt = "<|im_start|><|object_ref_start|>한국어로 현재 디렉터리에서 가장 큰 파일 10개를 찾는 명령을 알려주세요.<|im_end|>"
ids = tokenizer.encode(prompt).ids
print("prompt tokens:", len(ids))
print("first token ids:", ids[:20])

# 빠른 CPU weight metadata check. 모든 tensor를 RAM에 올리지 않고 shape만 읽습니다.
with safe_open(repo_dir / "model.safetensors", framework="pt", device="cpu") as f:
    keys = list(f.keys())
    num_params = sum(__import__("math").prod(f.get_slice(k).get_shape()) for k in keys)
    first_key = keys[0]
    first_shape = f.get_slice(first_key).get_shape()

print("num_tensors:", len(keys))
print("num_params:", f"{num_params:,}")
print("first tensor:", first_key, tuple(first_shape))
```

정상 결과:

- `model_type`은 `hrm_text`입니다.
- `vocab_size`는 `131072`입니다.
- `num_params`는 약 `1.38B`입니다.
- tokenizer는 `tokenizers.Tokenizer.from_file` 경로로 CPU와 Colab T4에서 정상 로드됩니다.
- `AutoModelForCausalLM` generation은 remote-code wrapper가 추가되기 전까지는 안 되는 것이 정상입니다.

이 Colab smoke test에서는 `AutoTokenizer`나 `AutoModelForCausalLM`를 쓰지 마십시오. 현재 public export는 custom `hrm_text` architecture이고, plain Transformers generation에 필요한 remote-code model wrapper는 아직 포함되어 있지 않습니다.

### 내부 / 프로젝트 코드 기반 생성

현재 실제 generation을 하려면 프로젝트 코드와 raw FSDP2 checkpoint를 사용합니다. 이것이 지금 바로 쓸 수 있는 CUDA 환경용 경로입니다. BF16이 되는 충분한 VRAM의 GPU를 권장합니다. Colab T4는 위 smoke test에는 쓸 수 있지만, raw checkpoint generation 권장 경로는 아닙니다.

```bash
git clone https://github.com/LLM-OS-Models/KoHRM-text
cd KoHRM-text
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
export TOKENIZERS_PARALLELISM=false
export NUMEXPR_MAX_THREADS=128
```

현재 바로 받을 수 있는 raw checkpoint 예시입니다. 아래 예시는 raw checkpoint repo에 올라온 `stage1b-hrm-fastcap-repeat-step310000`을 사용합니다. 더 최신 raw checkpoint가 올라오면 include path와 `ckpt_step`을 같이 바꾸면 됩니다.

```bash
mkdir -p checkpoints/kohm-raw
huggingface-cli download LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints \
  --include "stage1b-hrm-fastcap-repeat-step310000/**" \
  --local-dir checkpoints/kohm-raw
```

최소 generation script:

```bash
cat > run_kohrm_raw_generate.py <<'PY'
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "128")

from simple_inference_engine import inference_load_checkpoint, inference_generate

ckpt_dir = "checkpoints/kohm-raw/stage1b-hrm-fastcap-repeat-step310000"

prompts = [
    (
        0,
        (
            "direct",
            "한국어 존댓말로 현재 디렉터리에서 용량이 가장 큰 파일 10개를 찾는 bash 명령을 제안해 주세요.",
        ),
    ),
    (
        1,
        (
            "direct",
            "Write a Python function that validates a JSON tool-call object with name and arguments.",
        ),
    ),
]

ckpt = inference_load_checkpoint(
    ckpt_path=ckpt_dir,
    ckpt_epoch=None,
    ckpt_step=310000,
    ckpt_use_ema=True,
    device="cuda",
)

for pid, text in inference_generate(
    ckpt,
    iter(prompts),
    max_tokens=1024,
    max_generation=256,
    batch_size=1,
    temp=0.0,
):
    print(f"\n### sample {pid}\n{text}")
PY

python run_kohrm_raw_generate.py
```

Prompt formatting은 `InferenceCheckpoint.tokenize_prompt`가 처리합니다. tuple의 첫 번째 값은 condition string이고 보통 `"direct"`를 씁니다. 두 번째 값은 사용자 prompt입니다. 내부적으로는 다음 형식이 됩니다.

```text
<|im_start|><|object_ref_start|>PROMPT<|im_end|>
```

더 최신 raw checkpoint를 테스트하려면:

1. raw checkpoint repo에서 가장 최신 stage/step을 확인합니다.
2. `huggingface-cli download --include` pattern을 바꿉니다.
3. `ckpt_dir`를 바꿉니다.
4. `ckpt_step`을 바꿉니다.

공개 `model.safetensors`에서 바로 `AutoModelForCausalLM` generation을 하는 경로는 public `trust_remote_code` wrapper를 추가한 뒤 지원할 예정입니다.

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
