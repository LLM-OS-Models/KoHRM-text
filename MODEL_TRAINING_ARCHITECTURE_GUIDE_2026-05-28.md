# KoHRM-Text Training And Architecture Guide - 2026-05-28

이 문서는 `KoHRM-Text-1.4B`를 처음 읽는 사람이 학습 방법, 모델 구조, PrefixLM attention, PT/SFT 관계, staged continuation 운영을 한 번에 이해하도록 정리한 설명서입니다.

핵심 결론은 다음입니다.

```text
KoHRM-Text는 일반적인 raw-text causal LM 사전학습 모델이 아닙니다.
또한 완성된 base model 위에 얹는 전통적인 SFT만도 아닙니다.

HRM-Text 논문 방식처럼,
처음부터 instruction -> response 데이터로 scratch 학습하는
single-stage instruction pretraining 모델입니다.
```

## 한 줄 요약

`KoHRM-Text-1.4B`는 한국어/영어/코드/터미널/툴콜 데이터를 새 131K 토크나이저로 다시 패킹하고, HRM-Text의 recurrent architecture와 PrefixLM response-only loss로 처음부터 학습하는 모델입니다.

## 전체 그림

```text
raw sources
  |
  |  Korean / English / code / terminal / tool-call / legal / finance / wiki
  v
new 131K tokenizer
  |
  v
V1Dataset
  |
  |  tokens.npy
  |  epoch_N/inst_start.npy, inst_len.npy, resp_start.npy, resp_len.npy
  v
packed token batches
  |
  |  instruction tokens = context only
  |  response tokens    = supervised target
  v
LMHead
  |
  v
HRM recurrent backbone
  |
  |  H module: slower strategic state
  |  L module: faster execution state
  v
vocab logits
  |
  v
response-only cross entropy loss
```

## 왜 PT도 SFT도 아닌 중간인가

일반적인 LLM 학습을 단순화하면 보통 다음처럼 나눕니다.

```text
raw LM pretraining
  - 웹/책/코드 텍스트를 그대로 다음 토큰 예측
  - prompt와 answer 구분 없음
  - 거의 모든 토큰에 loss

SFT
  - 이미 학습된 base model을 instruction-answer 데이터로 보정
  - 보통 answer token에 loss
  - 데이터는 더 작고 품질 기준은 더 엄격함
```

KoHRM-Text는 그 중간에 있습니다.

```text
raw LM PT                    KoHRM instruction PT                    conventional SFT
---------                    ---------------------                    ----------------
raw text                     instruction-response                     instruction-response
all-token loss               response-only loss                       response-only loss
very broad corpus            broad corpus + task data                 curated smaller data
from scratch                 from scratch                             from pretrained model
base ability formation       base ability formation + instruction     behavior alignment
```

따라서 이름은 pretraining이지만, 목적함수는 SFT와 매우 가깝습니다. 차이는 데이터 규모, scratch 여부, optimizer trajectory, 학습률, 반복 방식입니다.

## 학습 샘플 구조

각 학습 샘플은 크게 두 부분입니다.

```text
instruction / prefix / context
response / target
```

프로젝트에서 쓰는 대표 prompt 형태는 다음입니다.

```text
<|im_start|><condition_token>instruction text<|im_end|>response text<|box_end|>
```

condition token은 응답 스타일을 알려주는 태그입니다.

```text
<|object_ref_start|> direct
<|object_ref_end|>   cot-style condition
<|quad_start|>       noisy
<|quad_end|>         synth
```

주의할 점은 instruction token도 입력에는 들어가지만, loss는 response token에만 걸린다는 점입니다.

```text
tokens:
  [ instruction tokens                         ][ response tokens          ]

labels:
  [ IGNORE_LABEL_ID / loss 없음                 ][ next response token loss ]

attention:
  [ bidirectional prefix context                ][ causal autoregressive    ]
```

## PrefixLM이 의미하는 것

일반 causal LM은 앞 token만 볼 수 있습니다.

```text
causal LM:

t0 -> t1 -> t2 -> t3 -> t4
```

PrefixLM은 instruction 구간을 더 자유롭게 봅니다. instruction/prefix 내부에서는 양방향 attention을 허용하고, response는 autoregressive로 생성합니다.

```text
PrefixLM:

instruction / prefix
┌───────────────────────────────┐
│ tokens can attend bidirectionally
└───────────────────────────────┘
                |
                v
response
r0 -> r1 -> r2 -> r3 -> ...

response token은 prefix 전체와 이전 response token을 봅니다.
미래 response token은 보지 못합니다.
```

이 구조 덕분에 모델은 문제 설명, 파일 내용, 터미널 로그, tool schema 같은 입력 컨텍스트를 양방향으로 읽고, 응답은 순차적으로 생성하는 방식으로 학습됩니다.

## 수식으로 본 loss

현재 PT와 SFT의 핵심 loss는 거의 같습니다.

```text
L(theta) = - sum_t m_t log p_theta(y_t | prefix, y_<t)
```

여기서:

- `prefix`: instruction, context, terminal log, tool schema 등 입력 구간
- `y_t`: response token
- `m_t = 0`: instruction token, loss 없음
- `m_t = 1`: response token, loss 있음

즉 모델은 “입력 컨텍스트를 이해하고 다음 응답 token을 맞히는 것”을 학습합니다.

## 코드상 구현 위치

핵심 구현은 다음 파일에 있습니다.

- `dataset_new.py`: instruction/response span을 읽고, response-only label을 만듭니다.
- `models/flash_attention_prefixlm_v2.py`: prefix bidirectional + response causal attention을 실행합니다.
- `models/lm_head.py`: `IGNORE_LABEL_ID`를 제외하고 cross entropy와 accuracy를 계산합니다.
- `models/baselines/hrm_nocarry_bp_warmup.py`: HRM H/L recurrent backbone입니다.
- `pretrain.py`: FSDP2, optimizer, EMA, checkpoint, staged resume를 관리합니다.

데이터 로딩에서 가장 중요한 부분은 다음입니다.

```text
dataset_new.py

inputs:
  instruction tokens
  response tokens shifted right

labels:
  instruction span -> IGNORE_LABEL_ID
  response span    -> actual target token ids

prefix_lens:
  instruction length

causal_lens:
  response length
```

## HRM 아키텍처

현재 표준 모델은 `arch/size@arch=XL`입니다.

짧은 요약:

- 약 1.384B parameters
- hidden size 1536
- 32 configured layers
- half-layers split으로 H module 16 layers, L module 16 layers
- 12 attention heads
- head dim 128
- context length 4096
- vocab 131072
- bf16 training

HRM 구조는 H module과 L module을 반복해서 씁니다.

```text
input embeddings
  |
  v
z_H initial state
z_L initial state
  |
  v
┌───────────────────────────────────────────┐
│ H cycle 1                                 │
│                                           │
│   L module update 1                       │
│   L module update 2                       │
│   L module update 3                       │
│                                           │
│   H module update 1                       │
└───────────────────────────────────────────┘
  |
  v
┌───────────────────────────────────────────┐
│ H cycle 2                                 │
│                                           │
│   L module update 1                       │
│   L module update 2                       │
│   L module update 3                       │
│                                           │
│   H module update 1                       │
└───────────────────────────────────────────┘
  |
  v
LM head -> vocab logits
```

직관적으로 보면:

- H module은 더 느리게 변하는 전략/상위 추론 state입니다.
- L module은 더 빠르게 변하는 실행/세부 처리 state입니다.
- L이 여러 번 세부 처리를 하고, H가 그 결과를 받아 상위 state를 갱신합니다.

이 때문에 단순 Transformer stack과 달리, 같은 depth를 한 번 통과하는 구조가 아니라 recurrent computation을 통해 내부 계산을 반복합니다.

## 모듈별 역할

```text
Tokenizer
  Korean, English, code, shell, JSON/tool-call을 131K BPE로 tokenization

V1Dataset
  instruction span과 response span을 따로 저장
  multipack sampler로 token budget에 맞게 batch 구성

PrefixLM Attention
  instruction은 양방향 context
  response는 causal generation

HRM Backbone
  H/L recurrent modules
  H2L3 schedule
  bp warmup으로 recurrent backprop 안정화

LM Head
  token embedding
  vocab projection
  response-only CE loss

Optimizer / EMA
  Adam-atan2
  bf16
  EMA 0.9999
  staged continuation에서 optimizer/EMA도 이어감
```

## 현재 실행 설정

현재 장기 pretraining은 안정성을 우선해서 다음 설정으로 운영합니다.

```text
GPUs:                 8 x H200
architecture:          HRM XL
global batch:          180,224 token slots/step
per-GPU token slots:   22,528
context length:        4,096
dtype:                 bfloat16
optimizer:             Adam-atan2
EMA:                   0.9999
checkpoint policy:     step checkpoint, keep latest 2 locally
upload policy:          watcher process uploads selected checkpoints
```

`global_batch_size`는 sample 개수가 아닙니다. token slots입니다.

```text
180,224 / 8 GPUs = 22,528 token slots per GPU
```

샘플 길이가 짧으면 한 step에 많은 샘플이 들어가고, 터미널 trajectory처럼 길면 샘플 수는 줄어듭니다. 하지만 token budget은 같습니다.

## context length 4096의 의미

prepared metadata에는 `max_seq_len=4097`처럼 보일 수 있습니다. 이것은 autoregressive shift 때문입니다.

실제 모델이 보는 최대 context는 4096 tokens입니다.

```text
metadata max_seq_len: 4097
dataset_new.py load:  metadata.max_seq_len -= 1
model context:        4096
```

## PT와 SFT 차이

현재 KoHRM recipe에서는 PT와 SFT의 수식은 거의 같습니다. 둘 다 instruction-response PrefixLM response-only loss입니다.

차이는 운영 regime입니다.

PT:

- 처음부터 모델 능력을 형성합니다.
- 데이터가 큽니다.
- 한국어/영어/코드/터미널/툴콜/법률/금융/wiki 등을 넓게 넣습니다.
- learning rate가 높습니다.
- optimizer와 EMA를 이어가며 긴 trajectory를 유지합니다.

SFT:

- 이미 학습된 checkpoint를 행동적으로 다듬습니다.
- 데이터는 작고 품질 기준이 더 엄격합니다.
- tool-call JSON validity, 터미널 action, 한국어 존댓말, formatting을 더 강하게 봅니다.
- learning rate가 낮습니다.
- EMA weight만 가져오고 optimizer를 reset하는 선택지도 자연스럽습니다.

따라서 “SFT 데이터를 PT에 넣으면 안 되나?”에 대한 현재 정책은 다음입니다.

```text
넣습니다.

다만 PT에서는 넓게 넣고,
SFT에서는 같은 계열 데이터 중 품질 높은 subset을 다시 선별해서 한 번 더 씁니다.
```

## staged continuation이란

논문식 개념은 single-stage instruction pretraining입니다. 하지만 실제 운영은 checkpoint를 끊어 이어갑니다.

이유:

- 전처리 데이터가 순차적으로 완성됐습니다.
- GPU를 놀리지 않기 위해 준비된 데이터부터 학습했습니다.
- OOM 안정성을 실측해야 했습니다.
- HF 업로드와 checkpoint 보존이 필요합니다.
- 긴 학습에서 중단 없이 이어가는 것이 더 중요합니다.

운영 구조:

```text
stage1  -> stage2  -> stage3  -> stage4
  |          |          |          |
  v          v          v          v
pass 1 data1 pass1 data2 pass1 data3 pass1 data4

stage1b -> stage2b -> stage3b -> stage4b
  |          |          |          |
  v          v          v          v
pass 2 data1 pass2 data2 pass2 data3 pass2 data4

stage1c -> stage2c -> stage3c -> stage4c
  |          |          |          |
  v          v          v          v
pass 3 data1 pass3 data2 pass3 data3 pass3 data4
```

각 stage는 코드상 `epochs=1`로 실행됩니다. 그래서 final checkpoint 이름은 모두 `fsdp2_epoch_1`입니다. “몇 번째 pass인가”는 parent stage 이름으로 구분해야 합니다.

예:

```text
stage2/fsdp2_epoch_1   = pass 1 data 2 완료
stage2b/fsdp2_epoch_1  = pass 2 data 2 완료
stage2c/fsdp2_epoch_1  = pass 3 data 2 완료 예정
```

## 데이터 pass 기준 현재 상태

운영상 데이터 묶음은 다음입니다.

1. HRM fast-cap
2. HRM full/no-cap
3. local terminal/code/tool trajectory
4. Korean/tool/legal/wiki/finance mix

현재 기준:

```text
pass 1: data 1/2/3/4 완료
pass 2: data 1 완료, data 2 진행 중
pass 3: data 1/2/3/4 예약
```

세부 체크포인트 지도는 `EPOCH_PASS_CHECKPOINT_MAP_2026-05-28.md`를 기준으로 봅니다.

## 왜 새 토크나이저인가

기존 HRM-Text는 65K BPE입니다. KoHRM은 한국어, 터미널, tool-call JSON, 코드의 효율을 높이기 위해 131K byte-level BPE를 새로 만들었습니다.

효과:

- 한국어 조각화 감소
- 법률/행정 문서의 긴 한자어/고유 표현 처리 개선
- shell command, path, stack trace, JSON key/value 처리 개선
- tool-call format tokenization 안정화

단점:

- vocab이 커져 embedding/LM head가 커집니다.
- 같은 hidden size에서도 전체 parameter 수가 증가합니다.
- VRAM과 학습 시간이 늘 수 있습니다.

현재 1.4B 규모가 된 가장 큰 이유 중 하나가 131K vocab입니다.

## 관리자/운영자 관점

학습 운영을 볼 때는 다음 네 개만 먼저 보면 됩니다.

```text
1. GPU가 도는가
   nvidia-smi 기준 8장 99% 근처면 정상

2. step이 증가하는가
   log의 N/465000 같은 progress가 계속 증가해야 함

3. checkpoint가 저장되는가
   checkpoint_step_interval마다 step checkpoint 생성

4. HF 업로드가 되는가
   raw checkpoint repo와 main safetensors repo에 watcher가 업로드
```

현재 목표는 무리한 batch 증대보다 끊기지 않는 장기 run입니다. H200 8장 환경에서 VRAM을 크게 쓰고 있으므로, OOM 한 번으로 장시간 손실이 나는 것보다 안정 batch로 계속 가는 편이 낫습니다.

## 실무자/개발자 관점

코드를 볼 때의 진입점은 다음 순서가 좋습니다.

```text
pretrain.py
  |
  +-- config/cfg_pretrain.yaml
  +-- config/cfg_sft.yaml
  |
  +-- dataset_new.py
  |     +-- V1Dataset
  |     +-- MultipackDistributedBatchSampler
  |
  +-- models/lm_head.py
  |     +-- token embedding
  |     +-- response-only CE loss
  |
  +-- models/baselines/hrm_nocarry_bp_warmup.py
  |     +-- H/L recurrent model
  |
  +-- models/flash_attention_prefixlm_v2.py
        +-- prefix bidirectional attention
        +-- response causal attention
```

## 읽는 순서

처음 보는 사람은 다음 순서로 읽으면 됩니다.

1. 이 문서
2. `README.md`
3. `PRETRAINING_SFT_DATA_MIX_2026-05-23.md`
4. `BATCH_AND_CONTEXT_LENGTH_NOTES_2026-05-27.md`
5. `EPOCH_PASS_CHECKPOINT_MAP_2026-05-28.md`
6. `TRAINING_LOSS_ANALYSIS_2026-05-26.md`
7. `METHODOLOGY_ARCHITECTURE_NOTES_2026-05-24.md`

## 결론

KoHRM-Text는 “일반 PT 후 SFT”가 아니라, 처음부터 instruction-response 데이터로 학습하는 HRM-Text식 instruction pretraining입니다.

학습 objective는 SFT와 비슷하지만, 규모와 역할은 pretraining입니다. PrefixLM 구조 덕분에 입력 컨텍스트는 양방향으로 읽고, 응답은 causal하게 생성합니다. HRM H/L recurrent architecture는 내부 계산을 반복해 문제 해결 능력을 만들도록 설계되어 있습니다.

현재 운영은 이 방법론을 유지하면서 checkpoint resume과 stage chain으로 안전하게 길게 돌리는 방식입니다.
