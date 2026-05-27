# KoHRM-Text

`KoHRM-Text-1.4B`는 [`sapientinc/HRM-Text`](https://github.com/sapientinc/HRM-Text)를 fork해서 만든 작업 저장소에서 학습 중인 scratch model입니다. 원본 HRM-Text의 PrefixLM/HRM 학습 구조를 유지하면서, 한국어/영어/코드/터미널/툴콜을 잘 처리하도록 tokenizer, 데이터 mix, 운영 문서를 KoHRM 목적에 맞게 확장했습니다.

원본 HRM-Text README는 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)에 보존했습니다.

## 빠른 이동

- [현재 모델](#현재-모델)
- [포크 출처와 차이점](#포크-출처와-차이점)
- [핵심 개념](#핵심-개념)
- [학습 상태](#학습-상태)
- [문서 지도](#문서-지도)
- [데이터](#데이터)
- [토크나이저](#토크나이저)
- [운영 메모](#운영-메모)

## 현재 모델

```text
name:        KoHRM-Text-1.4B
origin:      scratch training
base code:   sapientinc/HRM-Text
arch:        HRM XL
params:      1,384,120,320
context:     4,096 tokens
tokenizer:   131,072 vocab byte-level BPE
HF model:    LLM-OS-Models/KoHRM-Text-1.4B
HF data:     LLM-OS-Models/KoHRM-Text-1.4B-prepared-data
GitHub:      https://github.com/LLM-OS-Models/KoHRM-text.git
```

이 모델은 `sapientinc/HRM-Text-1B` 가중치를 이어 학습한 모델이 아닙니다. 한국어/터미널용 131K 토크나이저와 새 데이터 mix로 처음부터 학습합니다.

## 포크 출처와 차이점

이 저장소는 원본 [`sapientinc/HRM-Text`](https://github.com/sapientinc/HRM-Text)를 기반으로 합니다. 따라서 핵심 학습 코드의 방향은 HRM-Text 논문과 upstream 구현을 따릅니다.

그대로 유지한 부분:

```text
HRM recurrent architecture
PrefixLM attention
instruction -> response V1Dataset format
response-only loss
Adam-atan2 optimizer
EMA checkpointing
bf16/FSDP2 training path
```

KoHRM에서 바꾼 부분:

```text
model target:     Korean / English / code / terminal / tool-call
tokenizer:        upstream 65K BPE -> KoHRM 131K byte-level BPE
training origin:  upstream weight continuation 아님, scratch training
model size:       1.4B급, 큰 vocab 때문에 upstream 1B보다 큼
data mix:         HRM cleaned + Korean legal/wiki/finance + terminal/tool/code data
operation:        staged continuation, checkpoint upload watcher, HF prepared dataset 공개
documentation:    KoHRM 운영/데이터/아키텍처 문서를 docs/에 별도 정리
```

따라서 이 repo는 단순 README 수정본이 아니라, 원본 HRM-Text 학습 스택을 KoHRM 데이터와 토크나이저로 재학습하기 위한 fork입니다.

## 핵심 개념

KoHRM-Text는 일반적인 raw-text causal LM 사전학습도 아니고, 완성된 base model 위에 얹는 전통적인 SFT만도 아닙니다.

```text
raw LM PT                    KoHRM instruction PT                    conventional SFT
---------                    ---------------------                    ----------------
raw text                     instruction-response                     instruction-response
all-token loss               response-only loss                       response-only loss
from scratch                 from scratch                             from pretrained model
broad corpus                 broad corpus + task data                 curated small data
```

핵심은 HRM-Text 논문식 single-stage instruction pretraining입니다.

```text
instruction / prefix
  - 입력 컨텍스트
  - 양방향 attention
  - loss 없음

response
  - 모델이 맞혀야 하는 출력
  - causal attention
  - response-only CE loss
```

전체 구조:

```text
raw data -> tokenizer -> V1Dataset -> PrefixLM batches -> HRM H/L recurrence -> LM head -> response loss
```

처음 읽는 사람은 [docs/MODEL_TRAINING_ARCHITECTURE_GUIDE_2026-05-28.md](docs/MODEL_TRAINING_ARCHITECTURE_GUIDE_2026-05-28.md)를 먼저 보면 됩니다. 학습 방식, PrefixLM, HRM 구조, PT/SFT 차이, stage pass 개념을 한 문서에 정리했습니다.

## 학습 상태

기준: 2026-05-28 KST

현재 실행은 `stage2b-hrm-full-nocap-extra-epoch1`입니다. `stage1b-hrm-fastcap-repeat` final checkpoint에서 이어받아 8 x H200으로 학습 중입니다.

짧은 상태:

```text
active stage:     stage2b-hrm-full-nocap-extra-epoch1
pass view:        pass 2, data 2
resume step:      317,814
global batch:     180,224 token slots/step
per GPU batch:    22,528 token slots/step
context length:   4,096 tokens
speed:            about 1.02 step/s
checkpoint:       every 10,000 steps, keep latest 2 locally
upload:           watcher uploads selected raw + converted checkpoints
```

데이터 pass 기준:

```text
pass 1: data 1/2/3/4 완료
pass 2: data 1 완료, data 2 진행 중
pass 3: data 1/2/3/4 예약
```

세부 checkpoint map은 [docs/EPOCH_PASS_CHECKPOINT_MAP_2026-05-28.md](docs/EPOCH_PASS_CHECKPOINT_MAP_2026-05-28.md)를 기준으로 봅니다.

## 문서 지도

문서는 [docs/](docs/)에 모았습니다. 전체 인덱스는 [docs/README.md](docs/README.md)를 봅니다.

### 처음 읽기

- [docs/MODEL_TRAINING_ARCHITECTURE_GUIDE_2026-05-28.md](docs/MODEL_TRAINING_ARCHITECTURE_GUIDE_2026-05-28.md)

  모델 구조, PrefixLM, response-only loss, PT/SFT 관계, staged continuation 설명입니다.

- [docs/METHODOLOGY_ARCHITECTURE_NOTES_2026-05-24.md](docs/METHODOLOGY_ARCHITECTURE_NOTES_2026-05-24.md)

  HRM-Text 논문 방식과 KoHRM 적용 차이를 정리했습니다.

- [docs/BATCH_AND_CONTEXT_LENGTH_NOTES_2026-05-27.md](docs/BATCH_AND_CONTEXT_LENGTH_NOTES_2026-05-27.md)

  pretraining/SFT batch size, token-based batch, context length 4096의 의미입니다.

### 데이터와 학습 계획

- [docs/PRETRAINING_SFT_DATA_MIX_2026-05-23.md](docs/PRETRAINING_SFT_DATA_MIX_2026-05-23.md)

  사전학습/SFT 데이터 구성, 비중, 제외 기준입니다.

- [docs/TRAINING_PLAN_2026-05-23.md](docs/TRAINING_PLAN_2026-05-23.md)

  전체 학습 전략, 토크나이저, 실행 정책입니다.

- [docs/STAGED_TRAINING_RUNBOOK_2026-05-23.md](docs/STAGED_TRAINING_RUNBOOK_2026-05-23.md)

  완료된 전처리 데이터부터 학습하고, 새 데이터가 생기면 이어 학습하는 절차입니다.

- [docs/EPOCH_PASS_CHECKPOINT_MAP_2026-05-28.md](docs/EPOCH_PASS_CHECKPOINT_MAP_2026-05-28.md)

  데이터 1/2/3/4 pass 기준으로 stage와 checkpoint를 찾는 문서입니다.

### 운영과 품질 확인

- [docs/TRAINING_OPERATIONS_LOG_2026-05-26.md](docs/TRAINING_OPERATIONS_LOG_2026-05-26.md)

  장기 학습 운영 로그, stage chain, 업로드 watcher, 속도 분석입니다.

- [docs/CHAIN_HANDOFF_STATUS_2026-05-26.md](docs/CHAIN_HANDOFF_STATUS_2026-05-26.md)

  stage handoff 상태, stage 이름, 용량, ETA, watcher 보정 기록입니다.

- [docs/TRAINING_LOSS_ANALYSIS_2026-05-26.md](docs/TRAINING_LOSS_ANALYSIS_2026-05-26.md)

  train loss와 token accuracy 해석, 계속 진행 여부 판단입니다.

- [docs/VRAM_OOM_NOTES_2026-05-24.md](docs/VRAM_OOM_NOTES_2026-05-24.md)

  VRAM 증가/OOM 원인과 batch 정책입니다.

### 공개용 카드와 인벤토리

- [docs/MODEL_CARD_KoHRM-Text-1.4B.md](docs/MODEL_CARD_KoHRM-Text-1.4B.md)

  Hugging Face model card 초안입니다.

- [docs/HF_DATASET_CARD_KoHRM-Text-Prepared-Data.md](docs/HF_DATASET_CARD_KoHRM-Text-Prepared-Data.md)

  Hugging Face prepared dataset card 초안입니다.

- [docs/AVAILABLE_DATA.md](docs/AVAILABLE_DATA.md)

  로컬 데이터 인벤토리와 용량입니다.

- [docs/PROGRESS_2026-05-23.md](docs/PROGRESS_2026-05-23.md)

  실제 진행 로그입니다.

## 데이터

현재 학습은 HRM cleaned 데이터와 한국어/터미널/툴콜/법률/금융/wiki 데이터를 함께 사용합니다. SFT 후보 데이터도 pretraining에 먼저 넣고, 이후 고품질 subset으로 SFT를 한 번 더 하는 정책입니다.

주요 prepared dataset:

```text
koterm_hrm_cleaned_fastcap_stage1_v1              14.55B tokens
koterm_hrm_cleaned_full_nocap_v1                  14.55B tokens
koterm_hrm_cleaned_full_nocap_extra_epochs_1_3_v1 14.55B tokens per written logical epoch view
local_terminal_conversations_ctx9k_resp6k_v1       9.39B tokens
koterm_korean_tool_finance_mix_v1                  3.02B tokens
sft_bcai_finance_kor_v1                            857.7M tokens
```

전체 데이터 설명과 비중은 [docs/PRETRAINING_SFT_DATA_MIX_2026-05-23.md](docs/PRETRAINING_SFT_DATA_MIX_2026-05-23.md)를 봅니다.

평가 오염 위험이 있는 데이터는 train에서 제외합니다. 예: `tb2_lite`, Terminal Bench 2, ToolBench eval, chi-bench 평가 split.

## 토크나이저

```text
local path:  /home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1
HF repo:     LLM-OS-Models/HRM-Text-Ko-Terminal-Tokenizer-131K
method:      byte-level BPE
vocab:       131,072
normalize:   NFC
```

검증된 chars/token:

```text
Korean general:       2.60
Korean legal:         2.36
Korean terminal:      2.18
shell command:        2.68
tool JSON:            3.32
Python code:          3.37
English:              4.40
```

131K vocab 덕분에 한국어/터미널/tool-call 효율은 좋아졌지만, embedding과 LM head가 커져 모델 크기와 VRAM 사용량은 늘었습니다.

## 운영 메모

로컬 git repo에는 원문 데이터와 대형 checkpoint를 커밋하지 않습니다. 재현 가능한 코드와 문서만 남깁니다.

큰 산출물 위치:

```text
/home/work/.data/hrm_text_prepared
/home/work/.data/hrm_text_checkpoints
/home/work/.data/hrm_text_logs
```

Hugging Face:

```text
model latest export:
  https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B

raw checkpoints:
  https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints

prepared data:
  https://huggingface.co/datasets/LLM-OS-Models/KoHRM-Text-1.4B-prepared-data
```

현재 운영 원칙:

```text
1. GPU를 놀리지 않는다.
2. OOM 위험이 큰 무리한 batch보다 안정적인 장기 run을 우선한다.
3. stage final은 checkpoint metadata의 actual global_step으로 이어받는다.
4. 로컬 checkpoint는 최신 2개 중심으로 유지한다.
5. 업로드는 watcher로 분리해 학습 프로세스와 충돌을 줄인다.
```

## 코드 진입점

```text
pretrain.py
  +-- dataset_new.py
  +-- multipack_sampler.py
  +-- models/lm_head.py
  +-- models/flash_attention_prefixlm_v2.py
  +-- models/baselines/hrm_nocarry_bp_warmup.py

conversion/convert_to_hf.py
  +-- FSDP2 checkpoint -> safetensors export

scripts/watch_stage2b_then_finish_chain.py
  +-- stage2b 이후 stage3b/4b/1c/2c/3c/4c 자동 continuation

scripts/watch_chain_step_checkpoints_upload.py
  +-- step checkpoint raw + converted upload watcher
```

## 다음 작업

```text
1. stage2b를 끊기지 않게 완료한다.
2. watcher가 stage3b -> stage4b -> stage1c -> stage2c -> stage3c -> stage4c를 이어가게 둔다.
3. checkpoint 업로드와 model card 갱신을 계속한다.
4. planned continuation 이후 evaluation과 SFT subset 구성을 진행한다.
```
