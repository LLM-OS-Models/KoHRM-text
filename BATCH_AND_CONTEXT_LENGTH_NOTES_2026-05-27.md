# Batch Size And Context Length Notes - 2026-05-27

기준 시각: 2026-05-27 KST

이 문서는 KoHRM-Text 학습에서 말하는 pretraining batch size, SFT batch size, context length가 코드상 어떤 의미인지 정리합니다.

## Short Answer

SFT batch size와 pretraining batch size는 완전히 다른 개념은 아닙니다.

둘 다 `pretrain.py`의 같은 `global_batch_size` 필드를 쓰며, 단위도 모두 sample count가 아니라 token slots입니다. 다만 pretraining과 SFT는 데이터 분포, 학습률, optimizer resume 정책, 안정성 목표가 다르므로 실무적으로는 별도의 하이퍼파라미터처럼 잡습니다.

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
