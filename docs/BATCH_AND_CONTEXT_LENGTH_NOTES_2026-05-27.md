# Batch Size And Context Length Notes - 2026-05-27

기준 시각: 2026-05-27 KST

이 문서는 KoHRM-Text 학습에서 말하는 pretraining batch size, SFT batch size, context length가 코드상 어떤 의미인지 정리합니다.

## Short Answer

SFT batch size와 pretraining batch size는 완전히 다른 개념은 아닙니다.

둘 다 `pretrain.py`의 같은 `global_batch_size` 필드를 쓰며, 단위도 모두 sample count가 아니라 token slots입니다. 다만 pretraining과 SFT는 데이터 분포, 학습률, optimizer resume 정책, 안정성 목표가 다르므로 실무적으로는 별도의 하이퍼파라미터처럼 잡습니다.

수식적으로는 현재 KoHRM-Text recipe의 PT와 SFT는 거의 같습니다. 둘 다 instruction-response PrefixLM response-only cross entropy입니다. 차이는 objective의 수식보다 데이터 규모와 품질, LR/batch/epoch/optimizer 정책, 평가 기준에서 납니다.

현재 장기 pretraining 실행값은 다음과 같습니다.

| 항목 | 값 |
|---|---:|
| GPUs | 8 x H200 |
| current pretraining global batch | `180,224` token slots/step |
| local batch per GPU | `22,528` token slots/step |
| prepared metadata context size | `4,097` tokens |
| model/train context length | `4,096` tokens |
| tokenizer vocab | `131,072` |
| current architecture | HRM `XL`, hidden 1536, 32 layers, 12 heads |

`4,097`과 `4,096`이 같이 보이는 이유는 autoregressive shift 때문입니다. 전처리 파일의 `metadata.json`에는 `max_seq_len=4097`이 저장됩니다. `dataset_new.py`가 로딩할 때 `metadata.max_seq_len -= 1`을 적용하므로 실제 모델 config와 RoPE/attention에서 쓰는 길이는 `4096`입니다.

## Where The Values Come From

Pretraining config:

```yaml
# config/cfg_pretrain.yaml
global_batch_size: 196608
epochs: 4
lr: 2.2e-4
lr_warmup_steps: 2000
ema: 0.9999
```

현재 장기 실행에서는 OOM 안정성을 위해 CLI override로 다음 값을 씁니다.

```text
global_batch_size=180224
checkpoint_step_interval=10000
checkpoint_keep_last=2
```

SFT config:

```yaml
# config/cfg_sft.yaml
global_batch_size: 32768
epochs: 5
lr: 3.0e-5
lr_warmup_steps: 0
ema: 0.999
```

즉 기본 설정 기준으로 SFT global batch는 pretraining보다 훨씬 작습니다.

| Mode | Default global batch | Current/typical override | Unit |
|---|---:|---:|---|
| pretraining | `196,608` | `180,224` | token slots/step |
| SFT | `32,768` | final SFT 때 재조정 | token slots/step |

## PT And SFT: Same Math, Different Regime

현재 KoHRM-Text에서는 PT와 SFT가 수식적으로 크게 다르지 않습니다.

일반적인 LLM 문맥에서 pretraining은 raw text causal LM으로 모든 token에 next-token loss를 거는 경우가 많습니다. 하지만 HRM-Text 논문 방식과 현재 KoHRM-Text recipe는 처음부터 instruction-response pair를 PrefixLM으로 학습합니다. 그래서 우리의 PT도 이미 SFT와 같은 형태의 supervised instruction pretraining입니다.

기본 loss는 다음처럼 볼 수 있습니다.

```text
L(theta) = - sum_t m_t log p_theta(y_t | prefix, y_<t)
```

여기서:

| 기호 | 의미 |
|---|---|
| `prefix` | instruction/prompt/context span |
| `y_t` | response token |
| `m_t` | loss mask |
| `m_t = 0` | prompt/instruction token, context로만 사용 |
| `m_t = 1` | response token, loss 적용 |

현재 코드 기준으로는 `data.target_only=True`가 기본이라 instruction token은 `IGNORE_LABEL_ID`로 마스킹되고 response token에만 CE loss가 걸립니다.

즉 수식 관점의 핵심은 다음입니다.

| 항목 | Pretraining | SFT | 현재 KoHRM에서 같은가 |
|---|---|---|---|
| model forward | HRM PrefixLM | HRM PrefixLM | 같음 |
| token prediction | next response token | next response token | 같음 |
| prompt token loss | 없음 | 없음 | 같음 |
| response token loss | 있음 | 있음 | 같음 |
| attention 방식 | prefix + causal response | prefix + causal response | 같음 |
| objective | response-only CE | response-only CE | 같음 |
| code path | `pretrain.py` | `pretrain.py` + `cfg_sft.yaml` | 거의 같음 |

따라서 “PT냐 SFT냐”를 나누는 실제 기준은 loss 수식이 아니라 학습 regime입니다.

| 구분 | Pretraining/PT | SFT |
|---|---|---|
| 목적 | 기본 언어/지식/코드/터미널/툴콜 능력 형성 | 응답 양식, 지시 이행, tool-call 정확도, 터미널 행동 보정 |
| 데이터 규모 | 큼, 수십 B token 이상 | 작음, 고품질 subset 중심 |
| 데이터 품질 기준 | 폭넓은 coverage와 중복/오염 관리 | 포맷 정확도와 행동 품질을 더 엄격히 봄 |
| 데이터 반복 | 큰 corpus를 staged 반복 | 작은 corpus를 여러 epoch 반복 가능 |
| batch | 큼, 처리량/안정성 우선 | 보통 작게 시작, update 수와 세밀한 보정 우선 |
| learning rate | 큼, 현재 pretrain config `2.2e-4` | 작음, 현재 SFT config `3.0e-5` |
| warmup | 있음, 현재 `2000` steps | 보통 없음 또는 짧음 |
| optimizer | 연속 pretraining trajectory 유지 | EMA weight에서 optimizer reset 가능 |
| EMA | 긴 학습용 `0.9999` | 짧은 SFT용 `0.999` |
| 평가 | loss/accuracy + downstream broad eval | tool-call exactness, terminal trajectory, Korean style, formatting |

## What Is Actually Different In Code

PT와 SFT가 같은 `pretrain.py`를 쓰더라도 config가 다릅니다.

| 설정 | PT config | SFT config | 의미 |
|---|---:|---:|---|
| `global_batch_size` | `196608` 기본, 현재 `180224` | `32768` 기본 | token slots/step |
| `epochs` | `4` 기본, staged run은 보통 stage별 `1` | `5` 기본 | 데이터 반복 횟수 |
| `lr` | `2.2e-4` | `3.0e-5` | SFT가 훨씬 낮음 |
| `lr_warmup_steps` | `2000` | `0` | SFT는 이미 학습된 weight에서 시작 |
| `ema` | `0.9999` | `0.999` | 짧은 SFT trajectory를 더 빨리 따라가게 함 |
| `weights_only_resume_from_ema` | 보통 `false` | 상황에 따라 `true` 권장 | SFT 때 optimizer reset 여부 |
| `arch.bp_warmup_ratio` | HRM 기본 `0.2` | `0.0` | SFT에서는 BP warmup 없이 바로 fine-tune |

중요한 점은 SFT config도 모델 구조를 바꾸지 않는다는 것입니다. arch는 PT checkpoint와 맞아야 하고, tokenizer/context/vocab도 동일해야 합니다.

## Dataset Difference

현재 KoHRM-Text recipe에서는 SFT 후보 데이터도 PT에 먼저 넣습니다. 그래서 “SFT 데이터는 PT에서 제외”가 아닙니다.

정책은 다음입니다.

| 데이터 종류 | PT에서 사용 | SFT에서 재사용 | 이유 |
|---|---|---|---|
| HRM cleaned instruction data | 사용 | 일부 가능 | 기본 instruction ability |
| 한국어 법률/행정/위키 원문 task | 사용 | 고품질 subset 가능 | 한국어 지식과 문체 |
| terminal trajectory | 사용 | 강하게 재사용 | terminal action 품질 핵심 |
| tool-call data | 사용 | 강하게 재사용 | 함수 호출 형식 안정화 |
| finance/legal QA | 사용 | 고품질 subset 재사용 | 도메인 지시 이행 |
| reasoning/coding SFT 후보 | 사용 | 선별 재사용 | reasoning/coding 행동 |

PT에서는 coverage를 넓게 가져갑니다. SFT에서는 같은 계열이라도 더 엄격하게 고릅니다.

SFT에서 더 엄격히 보는 제거 기준:

- 깨진 JSON/tool-call
- 불완전한 trajectory
- benchmark contamination 위험
- 과도한 private reasoning trace
- 한국어 응답 품질이 낮은 샘플
- instruction과 response가 불일치하는 샘플
- response가 너무 짧거나 포맷만 있고 내용이 없는 샘플

## Batch Size Consequence

PT와 SFT가 같은 token-based batch를 쓰므로, 기술적으로 SFT도 큰 batch를 쓸 수 있습니다. 다만 batch를 키우면 update 수가 줄어듭니다.

```text
steps_per_epoch = total_tokens / global_batch_size
```

예를 들어 SFT 데이터가 1B tokens라면:

| global batch | 1 epoch steps |
|---:|---:|
| 32,768 | 약 30,518 |
| 65,536 | 약 15,259 |
| 131,072 | 약 7,629 |
| 180,224 | 약 5,548 |

따라서 나중에 기술적으로는 대형 batch SFT도 가능합니다. 다만 그 경우에는 다음을 같이 조정해야 합니다.

- LR을 유지할지, batch scaling에 맞춰 바꿀지
- SFT epoch 수를 늘릴지
- response token 비율이 충분한지
- formatting/tool-call exact match가 실제로 좋아지는지
- 큰 batch로 broad SFT 후 작은 batch로 final polish를 할지

실무적으로는 다음 순서가 합리적입니다.

| 단계 | batch 방향 | 목적 |
|---|---|---|
| broad SFT | `65k~131k`까지 실험 가능 | 넓은 고품질 instruction 행동 주입 |
| final tool/terminal/Korean polish | `16k~32k` 우선 | 형식, 말투, exactness, 오류 복구 보정 |
| PT급 large-batch SFT | 가능하나 실험 필요 | 빠르고 부드러운 보정, update 수 부족 위험 |

즉 나중에 큰 batch SFT는 기술적으로 열려 있습니다. 다만 SFT가 “섬세한 행동 보정”이라는 점 때문에 무조건 큰 batch가 정답은 아닙니다.

## Optimizer And Resume Difference

PT continuation에서는 보통 optimizer state까지 이어갑니다. 이때 AdamATan2 state와 EMA가 같이 유지되어 긴 pretraining trajectory가 이어집니다.

SFT에서는 두 선택지가 있습니다.

| 방식 | 설정 | 의미 |
|---|---|---|
| optimizer까지 이어받기 | `weights_only_resume_from_ema=false` | PT momentum/optimizer trajectory 유지 |
| EMA weight만 가져오고 optimizer reset | `weights_only_resume_from_ema=true` | 깨끗한 fine-tune 시작 |

최종 SFT는 보통 두 번째가 더 자연스럽습니다. SFT는 pretraining의 다음 token distribution을 그대로 더 밀기보다, selected weight에서 작은 LR로 행동을 정렬하는 성격이 강하기 때문입니다.

## Evaluation Difference

PT와 SFT는 loss가 같아도 평가 기준이 다릅니다.

PT에서 보는 것:

- train loss 하락 안정성
- token accuracy
- OOM 없이 긴 run이 유지되는지
- broad validation/eval 성능
- checkpoint resume 안정성

SFT에서 추가로 봐야 하는 것:

- 한국어 존댓말/응답 스타일
- tool-call JSON validity
- terminal command/action exactness
- multi-turn 지시 이행
- 코드 수정 task 성공률
- 불필요한 reasoning 노출 여부
- benchmark/eval contamination 회피

그래서 SFT는 loss만 낮다고 성공이 아닙니다. response formatting과 실제 task success를 반드시 같이 봐야 합니다.

## Not Sample Count

이 코드에서 `global_batch_size`는 examples/samples 개수가 아닙니다. `pretrain.py`는 다음처럼 처리합니다.

```python
local_batch_size = config.global_batch_size // world_size
V1DatasetConfig(batch_max_length=local_batch_size)
```

8 GPU에서 `global_batch_size=180224`이면 GPU당 `22528` token slots를 받습니다.

```text
180,224 / 8 = 22,528 token slots per GPU
```

`dataset_new.py`는 `MultipackDistributedBatchSampler`를 사용해 여러 instruction-response sample을 이 token budget 안에 packed sequence로 채웁니다. 따라서 한 step의 실제 sample 개수는 sample 길이에 따라 매번 달라집니다.

예를 들어 평균 sample 길이가 짧으면 한 GPU batch 안에 많은 sample이 들어가고, terminal trajectory처럼 긴 sample이 많으면 sample 수는 줄어듭니다. 하지만 token slots budget은 동일합니다.

## Why SFT Batch Is Smaller

SFT와 pretraining이 같은 단위의 `global_batch_size`를 쓰더라도 SFT에서 더 작은 값을 쓰는 이유는 다음과 같습니다.

1. SFT는 데이터가 더 작고 반복 epoch 수가 적어서, 너무 큰 batch를 쓰면 update 수가 지나치게 줄어듭니다.
2. SFT는 형식, 말투, tool-call, terminal action 같은 좁은 행동을 맞추는 단계라 큰 batch보다 충분한 update 횟수와 낮은 learning rate가 중요합니다.
3. SFT는 보통 pretrained checkpoint에서 시작하므로 `lr=3e-5`처럼 낮은 learning rate를 쓰고, warmup도 없거나 짧게 둡니다.
4. SFT는 긴 prompt와 짧은 answer가 섞인 데이터가 많아 batch별 packed sample 수 변동이 더 큽니다.
5. response-only loss에서는 prompt token은 context로만 쓰이고 loss token이 아닙니다. 같은 token slots라도 실제 supervised response token 비율이 데이터마다 달라집니다.

따라서 같은 `32,768 tokens/step`이라도 SFT에서는 “더 많은 optimizer update를 확보하는 작은 batch”에 가깝고, pretraining의 `180,224 tokens/step`은 “처리량과 안정성을 우선한 큰 batch”에 가깝습니다.

## Current Context Length

현재 KoHRM-Text 준비 데이터는 대부분 다음 방식으로 만들어졌습니다.

```text
--context-size 4097
```

전처리에서 sample은 다음 구조로 저장됩니다.

```text
<|begin_of_question|> + condition + instruction + <|end_of_question|> + response + <|end_of_answer|>
```

전처리 단계에서는 `sample_len < context_size`를 강제합니다. 즉 `context_size=4097`이면 저장되는 최대 sample length는 `4096` 이하입니다.

학습 로더는 PrefixLM 입력/label을 만들 때 response를 한 칸 shift합니다.

```text
input  = instruction tokens + response[:-1]
label  = ignore(prompt span) + response
```

이 shift 때문에 `dataset_new.py`는 metadata의 `max_seq_len`에서 1을 빼고, 모델에는 `4096`을 넘깁니다.

정리하면:

| 표기 위치 | 값 | 의미 |
|---|---:|---|
| preprocess CLI | `4097` | AR shift 포함 전처리 cap |
| prepared `metadata.json` | `4097` | 저장된 context-size metadata |
| `dataset_new.py` 로딩 후 | `4096` | 모델 학습용 max sequence length |
| model/RoPE/attention | `4096` | 현재 실제 context length |

따라서 사용자가 묻는 “지금 컨텍스트 랭스”는 모델 기준으로 `4096 tokens`라고 답하는 것이 맞습니다.

## PrefixLM And Loss Span

현재 학습은 raw causal LM이 아니라 PrefixLM response-only objective입니다.

instruction/prompt token은 모델이 양방향 prefix attention으로 읽을 수 있는 context 역할을 합니다. response 구간은 causal 방식으로 다음 token을 맞춥니다. 기본 설정에서 `data.target_only=True`이므로 instruction token label은 `IGNORE_LABEL_ID`로 마스킹되고, loss는 response token에만 걸립니다.

이 점 때문에 같은 global token batch라도 실제 gradient에 기여하는 supervised token 수는 데이터마다 다릅니다.

| 데이터 성격 | Prompt 비중 | Response 비중 | 같은 token batch에서 supervised token 효율 |
|---|---:|---:|---|
| 긴 터미널 history + 짧은 action | 높음 | 낮음 | 낮아질 수 있음 |
| 법률 원문 chunk 요약/재구성 | 중간 | 중간~높음 | 보통 |
| tool-call next action | 중간 | 짧음 | 형식 학습에는 유효하나 loss token은 적음 |
| long reasoning answer | 중간 | 높음 | 높음 |

그래서 SFT에서는 단순 token batch뿐 아니라 response token 비율, formatting accuracy, tool-call exactness를 같이 봐야 합니다.

## Current Running Stage

2026-05-27 현재 장기 학습은 `stage2b-hrm-full-nocap-extra-epoch1`입니다.

| 항목 | 값 |
|---|---:|
| run name | `KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1` |
| data | `koterm_hrm_cleaned_full_nocap_extra_epochs_1_3_v1` |
| resume checkpoint | `KoHRM-Text-1.4B-stage1b-hrm-fastcap-repeat-gbs180` |
| resume step offset | `317,814` |
| global batch | `180,224` |
| local token slots/GPU | `22,528` |
| checkpoint interval | `10,000` steps |
| local retention | latest `2` checkpoints |

The active continuation watcher now waits for stage2b final checkpoint and then launches:

```text
stage3b -> stage4b -> stage1c -> stage2c -> stage3c -> stage4c
```

The watcher was updated to avoid reusing already completed checkpoint directory names for the second repeat. The `c` stages use new names:

| Stage | Data | Purpose |
|---|---|---|
| `stage1c-hrm-fastcap-repeat2` | `koterm_hrm_cleaned_fastcap_stage1_v1` | HRM fast-cap repeat |
| `stage2c-hrm-full-nocap-repeat2` | `koterm_hrm_cleaned_full_nocap_v1` | HRM full/no-cap repeat |
| `stage3c-local-terminal-repeat2` | `local_terminal_conversations_ctx9k_resp6k_v1` | terminal/code repeat |
| `stage4c-korean-tool-finance-repeat2` | `koterm_korean_tool_finance_mix_v1` | Korean/tool/finance repeat |

## Practical Guidance

For the current long pretraining run:

- Keep `global_batch_size=180224`.
- Do not increase batch size mid-chain unless the current full chain finishes cleanly.
- The previous `229376` and `262144` tests looked possible early but later OOM risk was real.
- The stable run is more valuable than a small theoretical throughput gain.

For final SFT:

- Start from `global_batch_size=32768`.
- Use lower learning rate, likely `3e-5` or below.
- Prefer `weights_only_resume_from_ema=true` when starting final SFT from the selected pretraining checkpoint if the goal is clean fine-tuning rather than continuing the exact optimizer trajectory.
- Measure both response loss and downstream behavior. SFT success is not only lower loss; tool-call validity, Korean instruction following, terminal action quality, and formatting stability matter.

## One-Line Answer

SFT batch와 pretraining batch는 코드상 같은 token-based `global_batch_size`지만, 학습 목적이 달라서 값은 별도로 잡습니다. 현재 pretraining은 `180,224` tokens/step, 8 GPU 기준 GPU당 `22,528` token slots이고, 실제 모델 context length는 `4,096` tokens입니다.
