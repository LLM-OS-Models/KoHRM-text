# KoHRM-Text Methodology and Architecture Notes

작성일: 2026-05-24

이 문서는 `KoHRM-Text-1.4B`가 HRM-Text 논문 방식과 어떤 점에서 같고, 어떤 점에서 운영상 다른지 정리합니다.

참고 문서:

- HRM-Text paper: https://arxiv.org/html/2605.20613
- Upstream code: https://github.com/sapientinc/HRM-Text
- KoHRM-Text code: https://github.com/LLM-OS-Models/KoHRM-text

## 결론

우리의 현재 학습은 방법론 관점에서는 HRM-Text 논문식 single-stage instruction pretraining에 맞춰져 있습니다.

다만 실행 운영 관점에서는 논문과 완전히 같지 않습니다. 논문은 40B unique tokens를 단일 연속 run으로 학습했고, 중간 checkpointing/crash recovery를 쓰지 않았다고 설명합니다. 우리는 데이터 준비와 HF 업로드, OOM 회피, 체크포인트 보존 때문에 stage-0, stage0b, stage-1처럼 나누어 실행하고 있습니다.

핵심 차이는 다음입니다.

| 항목 | HRM-Text 논문 | KoHRM-Text 현재 방식 |
|---|---|---|
| 학습 목적 | instruction-response task-completion objective | 동일 |
| loss | response-only NLL | 동일 |
| attention | PrefixLM, instruction bidirectional + response causal | 동일 코드 경로 사용 |
| raw LM pretraining 후 SFT | 하지 않음 | 하지 않음 |
| SFT 후보 데이터 | instruction pretraining에 포함 | 포함 |
| 실행 형태 | 단일 연속 run | staged resume run |
| checkpoint | 논문은 중간 checkpointing 없음 | 운영상 5,000 step마다 저장 |
| tokenizer | 65,536 BPE | 131,072 Korean/terminal BPE |
| hardware | 16 x H100 | 8 x H200 |

따라서 “논문처럼 single-stage 지시문 사전학습인가?”에 대한 답은 다음처럼 정리하는 것이 정확합니다.

> 학습 objective와 데이터 포맷은 single-stage instruction pretraining입니다.  
> 그러나 실행은 한 프로세스의 단일 연속 run이 아니라, 같은 objective를 유지하면서 checkpoint resume으로 이어가는 staged pretraining입니다.

## 논문 방법론 요약

HRM-Text 논문은 기존 LLM의 대규모 raw-text causal LM 사전학습 후 mid-training/SFT로 가는 다단계 recipe를 비효율적이라고 보고, 처음부터 instruction-response pair만으로 학습합니다.

논문 핵심:

1. raw text 전체 토큰을 예측하지 않습니다.
2. instruction tokens에는 loss를 주지 않습니다.
3. response tokens에만 negative log-likelihood loss를 줍니다.
4. instruction segment는 PrefixLM mask로 bidirectional attention을 허용합니다.
5. response segment는 autoregressive causal attention을 유지합니다.
6. 데이터는 direct, cot, synth, noisy 같은 condition tag를 instruction 앞에 붙여 응답 스타일을 제어합니다.
7. `<think>...</think>` 같은 explicit long-CoT trace는 제거해 내부 recurrent computation이 문제 해결을 맡게 합니다.

논문은 1B HRM-Text를 scratch로 학습했고, 약 40B unique tokens 및 16 x H100에서 약 46시간을 사용했다고 설명합니다. 공개 평가에는 EMA checkpoint를 사용합니다.

## 현재 KoHRM-Text 학습 방식

현재 `KoHRM-Text-1.4B`도 raw causal LM이 아니라 HRM-Text V1Dataset 포맷으로 학습합니다.

각 샘플은 기본적으로 다음 구조를 갖습니다.

```text
instruction span -> response span
```

토큰 레벨에서는 다음처럼 구성됩니다.

```text
<|im_start|> condition_tokens instruction <|im_end|> response <|box_end|>
```

condition token은 다음 mapping을 사용합니다.

| condition | token |
|---|---|
| direct | `<|object_ref_start|>` |
| cot | `<|object_ref_end|>` |
| noisy | `<|quad_start|>` |
| synth | `<|quad_end|>` |

`dataset_new.py` 기준으로 instruction span은 `inputs`에는 들어가지만 `target_only=True`일 때 label은 `IGNORE_LABEL_ID`가 됩니다. response span만 실제 cross entropy loss를 받습니다.

즉 현재 학습은 “문서를 처음부터 끝까지 다 맞히는 raw LM”이 아니라 “주어진 지시/문맥을 보고 응답을 완성하는 task-completion pretraining”입니다.

## PrefixLM 구현 확인

현재 코드에서 PrefixLM 경로는 다음 파일들로 확인됩니다.

| 파일 | 역할 |
|---|---|
| `dataset_new.py` | instruction/response span 분리, response-only label 구성 |
| `models/flash_attention_prefixlm_v2.py` | prefix bidirectional attention + response causal attention 구현 |
| `models/layers.py` | attention type `prefixlm` 사용 |
| `models/lm_head.py` | `IGNORE_LABEL_ID`를 제외하고 response label에만 CE loss 계산 |

`dataset_new.py`에서는 각 샘플에 대해 instruction 길이를 `prefix_lens`, response 길이를 `causal_lens`로 넘깁니다.

`flash_attention_prefixlm_v2.py`는 attention을 두 부분으로 처리합니다.

1. prefix 구간: instruction tokens끼리 bidirectional attention
2. causal 구간: response tokens가 prefix 전체와 이전 response tokens를 보는 causal attention

이 구조가 논문의 PrefixLM과 맞는 핵심입니다.

## 모델 아키텍처

현재 표준 모델명은 `KoHRM-Text-1.4B`이고, `arch/size@arch=XL`에 해당하는 설정을 사용합니다.

| 항목 | 값 |
|---|---:|
| params | 1,384,120,320 |
| hidden size | 1,536 |
| total configured layers | 32 |
| half layers | true |
| H module layers | 16 |
| L module layers | 16 |
| heads | 12 |
| head dim | 128 |
| expansion | 4 |
| intermediate size | 4,096 |
| context | 4,096 |
| RoPE theta | 10,000 |
| norm | RMSNorm-style parameterless norm |
| init | LeCun normal |
| dtype | bfloat16 |
| tokenizer vocab | 131,072 |

HRM recurrent schedule:

| 항목 | 값 |
|---|---:|
| H cycles | 2 |
| L cycles per H cycle | 3 |
| effective H/L recurrence | H2L3 |
| bp min steps | 2 |
| bp max steps | 5 |
| bp warmup ratio | 0.2 |

코드상 흐름은 다음과 같습니다.

1. token embedding에서 시작한 hidden state를 `z_H`로 둡니다.
2. learned/fixed low-level initial state `z_L_init`에서 `z_L`을 시작합니다.
3. 각 H cycle마다 L module을 3번 반복 업데이트합니다.
4. 그 뒤 H module을 1번 업데이트합니다.
5. 총 2번의 H cycle을 수행합니다.
6. 최종 `z_H`에 LM head를 붙여 vocabulary logits를 냅니다.

논문 표현으로는 slow-evolving strategic layer인 H module과 fast-evolving execution layer인 L module의 dual-timescale recurrent design입니다.

## MagicNorm / 안정화

논문은 recurrent depth가 깊어지면 activation variance와 gradient instability가 커지므로 MagicNorm과 warmup deep credit assignment를 사용합니다.

현재 코드에서는 `norm_type: pre`를 쓰되, Transformer module 끝에 final RMSNorm을 적용합니다. 즉 내부 block은 PreNorm 스타일이고 module output은 norm으로 capped됩니다. 이것이 논문에서 말하는 MagicNorm 계열 안정화와 대응됩니다.

Backward는 처음부터 모든 recurrent step을 다 통과하지 않고, `bp_steps`를 warmup합니다.

현재 설정:

```yaml
bp_warmup_ratio: 0.2
bp_min_steps: 2
bp_max_steps: 5
```

초기에는 마지막 2 recurrent steps 위주로 gradient를 흘리고, 학습이 진행되며 최대 5 steps까지 늘립니다. 이 점도 논문 방식과 맞습니다.

## Optimizer / 스케줄

현재 pretraining config:

| 항목 | 값 |
|---|---:|
| optimizer | Adam-atan2 |
| beta1 | 0.9 |
| beta2 | 0.95 |
| weight decay | 0.1 |
| lr | 2.2e-4 |
| lr warmup | 2,000 steps |
| lr min ratio | 1.0 |
| EMA | 0.9999 |
| gradient clipping | 없음 |

논문도 Adam-atan2, warmup 2,000 steps, weight decay 0.1, EMA 0.9999, bf16을 사용합니다. 공개/평가는 EMA checkpoint를 기준으로 합니다.

## 현재 staged run이 논문과 다른 이유

논문은 “single continuous run”이라고 설명합니다. 우리는 다음 현실적 이유 때문에 staged run으로 운영합니다.

1. 한국어/터미널/tokenizer 전처리가 순차적으로 끝나고 있습니다.
2. GPU를 놀리지 않기 위해 준비된 데이터부터 학습을 시작했습니다.
3. H200 8장 환경에서 131K vocab 때문에 batch OOM 안정 마진을 실측해야 했습니다.
4. HF 업로드와 raw checkpoint 보존이 필요합니다.
5. full HRM 328G no-cap retokenization이 아직 진행 중입니다.

하지만 각 stage가 다른 목적함수로 바뀌는 것은 아닙니다.

| Stage | Objective | 성격 |
|---|---|---|
| stage-0 | PrefixLM response-only | 준비 완료된 711.3M mix |
| stage0b | PrefixLM response-only | 같은 mix 추가 pass |
| stage-1 | PrefixLM response-only | HRM fast-cap 14.55B |
| later stage | PrefixLM response-only | full HRM 328G retokenized + 한국어/터미널/툴콜 mix |
| final SFT | PrefixLM response-only 또는 SFT용 response-only | 품질 높은 subset으로 후처리 |

중요한 점은 stage-0에서 stage-1로 넘어갈 때 model/optimizer/EMA/carry를 이어받고, `resume_step_offset`과 `total_steps_override`로 global step/LR schedule을 이어가도록 수정했다는 것입니다.

즉 “학습 방법론”은 single-stage instruction pretraining이고, “운영 방식”은 staged continuation입니다.

## 현재 데이터가 single-stage 원칙에 맞는지

현재 prepared dataset들은 전부 가능한 한 instruction-response 형태로 변환했습니다.

| 데이터 | single-stage 원칙 적합성 |
|---|---|
| HRM cleaned data | 원래 HRM instruction/response/condition 구조라 적합 |
| ToolBench | tool instruction -> tool-call/answer response 구조라 적합 |
| SWE-ZERO/local terminal | terminal context -> next action/answer 구조라 적합 |
| GLM/Claude reasoning | final answer 중심으로 정리하면 적합 |
| 한국어 법률/위키 원문 | chunked instruction/response task로 바꿔 투입하므로 적합 |

주의할 점:

- 한국어 위키/법률 raw chunk는 “그냥 다음 텍스트 예측”처럼 넣으면 논문식 task-completion에서 멀어집니다.
- 따라서 title/context를 instruction으로 두고 chunk/summary/extraction을 response로 두는 식이 더 맞습니다.
- local terminal dataset은 objective에 잘 맞지만 전체 비중이 너무 커지면 일반 지식/한국어 균형이 무너질 수 있습니다.

## 현재 방식 평가

현재 방식은 논문 핵심과 잘 맞습니다.

맞는 부분:

- scratch training
- HRM H2L3 recurrent architecture
- PrefixLM attention
- response-only loss
- condition token 사용
- Adam-atan2
- bf16
- EMA 0.9999
- 4,096 context
- LeCun normal init

다른 부분:

- vocab 65,536이 아니라 131,072입니다.
- 16 x H100이 아니라 8 x H200입니다.
- 논문은 단일 연속 run, 우리는 staged resume run입니다.
- 논문은 40B unique tokens를 보고했고, 현재 public checkpoint는 stage-1 fast-cap 중간 산출물입니다.
- 한국어/터미널/툴콜 비중이 논문보다 훨씬 큽니다.

위 차이는 의도된 변경입니다. 특히 한국어/터미널/툴콜을 목표로 하므로 tokenizer와 데이터 분포를 바꾼 것은 맞습니다. 다만 논문의 효율성을 재현하려면 최종적으로 full HRM cleaned data와 balanced Korean/terminal/tool mix를 합쳐 40B+ token 수준으로 이어 학습해야 합니다.

## 운영 결론

현재는 다음 기준으로 계속 가는 것이 맞습니다.

1. 현재 stage-1은 계속 유지합니다.
2. 전처리 완료 데이터는 HF dataset repo에 올려 다른 머신에서도 재현 가능하게 둡니다.
3. full HRM 328G no-cap retokenization이 끝나면 next stage로 이어 학습합니다.
4. SFT 후보 데이터도 pretraining에 먼저 포함합니다.
5. 별도 final SFT는 마지막에 품질 높은 subset으로 다시 수행합니다.
6. model repo는 최신 safetensors 중심, raw checkpoint repo는 resume용으로 분리합니다.

