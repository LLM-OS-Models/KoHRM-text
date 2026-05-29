# KoHRM LoRA Training Guide - 2026-05-28

이 문서는 KoHRM-Text에서 SFT 후보별 LoRA 모델을 따로 만드는 방법을 정리합니다.

## 결론

LoRA는 full SFT 전에 빠르게 여러 행동 보정 버전을 비교하기 위한 경로입니다.

현재 만든 네 가지 실험 후보:

```text
behavior-mini     -> 가장 먼저 돌릴 smoke / quick behavior check
korean-domain     -> 한국어 법률/금융 응답 정렬
terminal-tool     -> 터미널, tool-call, 코딩 행동 강화
behavior-core     -> 전체 행동 보정 core
```

2026-05-29 Colab smoke 기준으로 공개 pretraining checkpoint는 다음 문제가 보였습니다.

```text
terminal command prompt:
  명령만 요구했는데 영어 agent reasoning으로 새는 경우가 있음.

finance QA:
  "환율 변동" 같은 짧은 구문 반복이 발생.

legal JSON:
  JSON 형태는 어느 정도 맞추지만 필드명/요지 환각이 남음.
```

따라서 첫 LoRA는 지식 추가가 아니라 output behavior correction입니다. 추천 순서는
`behavior-mini -> korean-domain -> terminal-tool -> behavior-core`입니다. 한국어 자체가 흔들리면
`korean-domain`을 먼저 보고, terminal/tool 명령 형식이 더 큰 문제면 `terminal-tool`을 먼저 봅니다.

## 추가된 코드

```text
models/lora.py
  LoRALinear, LoRA injection, adapter 저장

train_lora.py
  HRM checkpoint를 EMA weight로 로드하고 LoRA adapter만 학습하는 entrypoint

config/cfg_lora.yaml
  LoRA 학습 기본 Hydra config

scripts/run_kohrm_lora_experiments.sh
  후보별 LoRA 실행 wrapper
```

## Upstream 지원 상태

원본 `sapientinc/HRM-Text` 기준:

```text
full-parameter SFT:
  있음.
  scripts/prepare_sft_data.py + cfg_sft + pretrain.py --config-name cfg_sft 경로.

LoRA SFT:
  없음.
  이 fork에서 models/lora.py와 train_lora.py로 새로 추가.

native HRM vLLM:
  아직 없음.
  upstream README도 native vLLM support는 in progress라고 적고 있음.

vLLM baseline eval:
  있음.
  evaluation/config/vllm_benchmarking.yaml은 HF/vLLM 호환 baseline 모델 평가용.
  HRM raw checkpoint를 vLLM으로 바로 serving하는 경로는 아님.

CPU generation:
  실사용 경로 없음.
  CPU에서 convert_to_hf.py --device cpu, safetensors 무결성 확인은 가능.
  실제 HRM generation은 simple_inference_engine.py의 CUDA/FlashAttention 경로가 현실적.

Transformers one-line generation:
  아직 없음.
  model_type=hrm_text remote-code wrapper가 public repo에 들어가야 가능.
```

논문은 학습 방법론과 결과를 설명하지만, 실무적인 SFT 실행법, vLLM serving, CPU serving recipe를 상세히 제공하는 문서는 아닙니다. 실행법은 주로 GitHub repo에 있습니다.

## 기본 실행 전제

반드시 최종 또는 중간 pretraining checkpoint 경로를 지정해야 합니다.

예:

```bash
export RESUME_FROM=/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1-gbs180
```

LoRA 출력 기본 위치:

```text
/home/work/.data/hrm_text_lora
```

공통 기본값:

```text
GPUs:                8
global_batch_size:   32,768 token slots
LoRA rank:           16
LoRA alpha:          32.0
LR:                  8.0e-5
epochs:              1
base weight:         EMA weight
step save interval:  1,000
keep local steps:    latest 2 step adapters
```

## 후보별 실행

### 1. behavior-mini

가장 먼저 돌릴 smoke 후보입니다.

```bash
export RESUME_FROM=/path/to/base/checkpoint
bash scripts/run_kohrm_lora_experiments.sh behavior-mini
```

입력:

```text
/home/work/.data/hrm_text_prepared/kohrm_sft_behavior_mini_v1
```

출력:

```text
/home/work/.data/hrm_text_lora/KoHRM-Text-1.4B-lora-behavior-mini-v1
```

목적:

```text
한국어 응답 형식
tool-call JSON 안정성
터미널 명령 제안/해석
SFT가 기본 pretraining 능력을 훼손하는지 확인
```

### 2. terminal-tool

터미널/툴콜/코딩 행동이 부족할 때 씁니다.

```bash
export RESUME_FROM=/path/to/base/checkpoint
bash scripts/run_kohrm_lora_experiments.sh terminal-tool
```

입력:

```text
/home/work/.data/hrm_text_prepared/kohrm_sft_terminal_tool_core_v1
```

출력:

```text
/home/work/.data/hrm_text_lora/KoHRM-Text-1.4B-lora-terminal-tool-core-v1
```

### 3. korean-domain

한국어 법률/금융 응답 스타일이 부족할 때 씁니다.

```bash
export RESUME_FROM=/path/to/base/checkpoint
bash scripts/run_kohrm_lora_experiments.sh korean-domain
```

입력:

```text
/home/work/.data/hrm_text_prepared/kohrm_sft_korean_domain_core_v1
```

출력:

```text
/home/work/.data/hrm_text_lora/KoHRM-Text-1.4B-lora-korean-domain-core-v1
```

주의:

```text
이 후보만 길게 돌리면 도메인 QA 말투로 기울 수 있다.
1 epoch 이하로 먼저 확인하는 것이 맞다.
```

### 4. behavior-core

전체 행동 보정 후보입니다.

```bash
export RESUME_FROM=/path/to/base/checkpoint
bash scripts/run_kohrm_lora_experiments.sh behavior-core
```

입력:

```text
/home/work/.data/hrm_text_prepared/kohrm_sft_behavior_core_v1
```

출력:

```text
/home/work/.data/hrm_text_lora/KoHRM-Text-1.4B-lora-behavior-core-v1
```

## 모든 후보 순차 실행

```bash
export RESUME_FROM=/path/to/base/checkpoint
bash scripts/run_kohrm_lora_experiments.sh all
```

이 명령은 다음 순서로 실행합니다.

```text
behavior-mini
korean-domain
terminal-tool
behavior-core
```

GPU를 오래 잡으므로, 실제로는 `behavior-mini`부터 확인하는 것을 권장합니다.

첫 진단용 세 후보만 돌리려면:

```bash
export RESUME_FROM=/path/to/base/checkpoint
bash scripts/run_kohrm_lora_experiments.sh phase1
```

이 명령은 다음 순서입니다.

```text
behavior-mini
korean-domain
terminal-tool
```

## 직접 torchrun 실행

wrapper를 쓰지 않고 직접 실행할 수도 있습니다.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
WANDB_DIR=/home/work/.data/wandb \
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 train_lora.py \
  --config-name cfg_lora \
  arch/size@arch=XL \
  data.path=/home/work/.data/hrm_text_prepared/kohrm_sft_behavior_mini_v1 \
  resume_from=/path/to/base/checkpoint \
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

## Adapter 저장 형식

각 출력 폴더에는 다음이 저장됩니다.

```text
lora_epoch_1.pt
lora_epoch_1_info.json
latest_lora.txt
lora_train_config.json
```

`lora_epoch_1.pt`는 LoRA A/B tensor만 담습니다. base model 전체 weight를 다시 저장하지 않습니다.

step checkpoint는 `checkpoint_keep_last` 값에 맞춰 최신 N개만 남기도록 pruning합니다.
epoch checkpoint는 실험 결과 비교를 위해 남깁니다.

## 현재 제한

```text
LoRA adapter merge-to-full-weight 스크립트는 아직 없음.
LoRA adapter를 적용한 inference helper는 아직 없음.
현 단계 목적은 SFT/RL 후보의 학습 산출물을 빠르게 만드는 것.
```

## SFT/LoRA prepared data on Hugging Face

별도 데이터셋 repo:

```text
https://huggingface.co/datasets/LLM-OS-Models/KoHRM-Text-1.4B-sft-lora-data
```

업로드 스크립트:

```bash
python scripts/upload_sft_lora_data_to_hf.py
```

부분 업로드:

```bash
python scripts/upload_sft_lora_data_to_hf.py --only kohrm_sft_behavior_mini_v1
```

이후 필요하면 다음을 추가합니다.

```text
1. LoRA adapter를 base checkpoint에 merge하는 스크립트
2. simple_inference_engine.py에서 LoRA adapter를 load하는 경로
3. HF safetensors export에 LoRA adapter를 반영하는 경로
```
