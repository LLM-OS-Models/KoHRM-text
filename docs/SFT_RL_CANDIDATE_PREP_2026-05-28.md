# SFT/RL Candidate Preparation - 2026-05-28

이 문서는 KoHRM-Text pretraining 이후 사용할 SFT, LoRA SFT, RL 후보 데이터를 정리합니다.

핵심 결론:

```text
SFT는 지식을 더 많이 넣는 단계가 아니라,
한국어 응답 습관, 터미널 행동, tool-call 형식, 코딩 작업 흐름을 정렬하는 단계다.

따라서 전체 pretraining 데이터를 다시 SFT하지 않는다.
이미 pretraining에 들어간 데이터 중 행동 품질을 맞출 subset만 작게 뽑아 쓴다.
```

## 현재 의견

현재 만든 후보들을 SFT하는 것은 **적절합니다**. 다만 “전부를 길게 full SFT”하는 것은 적절하지 않습니다.

추천 판단은 다음입니다.

```text
1. SFT 후보 자체는 적절하다.
2. 첫 실험은 full SFT가 아니라 LoRA/짧은 SFT가 맞다.
3. RL만으로 바로 성능을 올리겠다는 접근은 아직 위험하다.
4. pretraining checkpoint의 기본 형식이 충분히 좋으면 SFT를 최소화하고 RL 비중을 높인다.
5. tool-call/terminal/한국어 말투가 흔들리면 RL 전에 SFT가 필요하다.
```

이유:

```text
SFT는 "정답 행동 분포를 보여주는" 데 강하다.
RL은 "이미 낼 수 있는 행동 중 더 좋은 것을 선택/강화하는" 데 강하다.

모델이 아직 올바른 tool-call JSON, terminal action, 한국어 응답 스타일을 안정적으로 못 내면
RL reward를 줘도 탐색 공간이 나쁘고 보상이 불안정해진다.

반대로 모델이 이미 형식은 잘 지키고 성공률/선호도만 부족하면
큰 SFT보다 RL이 더 효율적일 수 있다.
```

따라서 현재 최적 전략은 다음입니다.

```text
pretraining final eval
-> behavior_mini_v1 LoRA 또는 짧은 SFT 1 epoch
-> terminal/tool/Korean quick eval
-> 한국어 반복/도메인 응답 문제가 크면 korean_domain_core_v1 LoRA 추가
-> terminal/tool 형식 문제가 크면 terminal_tool_core_v1 LoRA 추가
-> 형식 문제가 줄면 RL 중심
-> 최종적으로 reward 기반 RL
```

2026-05-29 공개 checkpoint Colab smoke에서 확인한 현상:

```text
legal_json:
  JSON 구조는 일부 맞추지만 필드명과 요지에서 환각이 있음.

finance_qa:
  동일 표현 반복이 나타남.

terminal_command:
  명령만 요구해도 영어 agent reasoning으로 새는 경우가 있음.
```

이 결과만으로 사전학습이 실패했다고 보기는 어렵습니다. 현재 checkpoint는 아직 pretraining 중간본이고,
assistant-style final answer alignment가 덜 된 상태입니다. 다만 SFT/LoRA가 필요하다는 신호는 분명합니다.
우선순위는 한국어 짧은 응답 안정화, repetition 억제, terminal command-only 형식, JSON fidelity입니다.

## 후보가 적절한 이유

현재 후보는 KoHRM의 실제 목표와 직접 맞습니다.

### 적절한 점

```text
terminal component:
  모델이 "명령을 실제로 제안하고 출력에서 다음 행동을 판단하는" 행동을 배운다.

toolbench component:
  tool-call JSON, 함수명, argument 구조 같은 형식 안정화에 직접적이다.

SWE/code component:
  repo 탐색, 파일 수정, 테스트 루프에 필요하다.

Korean legal/finance component:
  한국어 도메인 질의응답, 존댓말 설명, 근거 기반 요약에 필요하다.

reasoning/agent component:
  multi-turn context에서 다음 assistant 행동을 만드는 데 도움이 된다.
```

### 조심할 점

```text
terminal 데이터만 과하게 돌리면 일반 대화/한국어 설명이 거칠어질 수 있다.
법률/금융만 과하게 돌리면 답변이 도메인 QA 말투로 굳을 수 있다.
reasoning 데이터는 긴 사족이나 불필요한 추론 노출을 늘릴 수 있다.
ToolBench는 형식은 좋지만 실제 로컬 터미널 작업과 분포가 다르다.
```

그래서 작은 mix로 시작하고, 평가 결과에 따라 component를 추가하는 방식이 맞습니다.

## RL만으로 충분한가

지금 단계에서 “RL만으로 충분하다”고 보기는 어렵습니다.

RL이 잘 먹히는 조건:

```text
모델이 이미 유효한 action 후보를 자주 낸다.
JSON/tool-call 형식 오류가 낮다.
터미널 명령이 실제로 실행 가능한 수준이다.
한국어 응답 스타일이 기본적으로 안정적이다.
reward가 명확하다.
```

현재 예상되는 KoHRM 위험:

```text
pretraining은 넓은 행동을 배웠지만, 최종 answer format이 아직 흔들릴 수 있다.
tool-call JSON validity는 작은 오류도 실패로 이어진다.
터미널 명령은 한 글자 오류도 task 실패가 될 수 있다.
한국어 존댓말과 concise response는 reward만으로 안정화하기 어렵다.
```

따라서 RL 전에 최소한의 SFT/LoRA로 형식 기반을 잡는 것이 더 안전합니다.

## 언제 SFT를 생략할 수 있나

다음 평가를 통과하면 full SFT는 생략하거나 아주 작게 줄일 수 있습니다.

```text
tool-call JSON validity >= 95%
간단한 terminal command task 성공률이 충분히 높음
한국어 존댓말/간결성 샘플 평가가 안정적
코딩 patch/test 루프에서 명령 형식이 안정적
불필요한 reasoning 노출이 적음
```

이 경우 추천:

```text
behavior_mini_v1 LoRA smoke만 하고
바로 RL 또는 preference optimization으로 넘어간다.
```

## 언제 SFT가 꼭 필요한가

다음 문제가 보이면 SFT가 필요합니다.

```text
JSON이 자주 깨진다.
tool name/argument key를 자주 틀린다.
터미널 명령 대신 설명문만 낸다.
명령 실행 결과를 보고 다음 행동을 못 고른다.
한국어 답변이 반말/번역투/장문 사족으로 흔들린다.
코딩 작업에서 파일 수정과 테스트 순서가 불안정하다.
```

이 경우 추천:

```text
LoRA on behavior_mini_v1
-> 그래도 부족하면 LoRA/full short SFT on terminal_tool_core_v1
-> 한국어 문제만 남으면 korean_domain_core_v1을 1 epoch 이하로 보조 사용
```

## 현재 만든 prepared 후보

모든 경로는 `/home/work/.data/hrm_text_prepared` 아래입니다.

### Component Subsets

| 이름 | 토큰 | 샘플 | 용도 |
|---|---:|---:|---|
| `kohrm_sft_comp_terminal_80m_v1` | 80.0M | 23,374 | 터미널/코딩 trajectory |
| `kohrm_sft_comp_toolbench_30m_v1` | 30.0M | 15,210 | tool-call 형식 |
| `kohrm_sft_comp_swe_zero_30m_v1` | 30.0M | 8,826 | SWE/코딩 행동 |
| `kohrm_sft_comp_glm_reasoning_20m_v1` | 20.0M | 16,376 | reasoning 응답 |
| `kohrm_sft_comp_agent_reasoning_25m_v1` | 25.0M | 8,524 | agent/reasoning 대화 |
| `kohrm_sft_comp_korean_legal_50m_v1` | 50.0M | 110,578 | 한국어 법률/행정 QA |
| `kohrm_sft_comp_finance_50m_v1` | 50.0M | 108,494 | 한국어 금융 QA |

### Experiment Mixes

| 이름 | 토큰 | 샘플 | 용도 |
|---|---:|---:|---|
| `kohrm_sft_behavior_mini_v1` | 60.0M | 61,810 | 빠른 LoRA/SFT smoke, format/행동 확인 |
| `kohrm_sft_terminal_tool_core_v1` | 165.0M | 55,934 | 터미널 + tool-call + coding 강화 |
| `kohrm_sft_korean_domain_core_v1` | 100.0M | 219,072 | 한국어 법률/금융 도메인 응답 정렬 |
| `kohrm_sft_behavior_core_v1` | 285.0M | 291,382 | 전체 행동 보정 core SFT 후보 |

모든 후보는 다음 조건을 확인했습니다.

```text
metadata.total_length == tokens.npy length
epoch_0 index arrays load OK
max sample length < metadata.max_seq_len
max_seq_len = 4097
actual model context = 4096
```

## 전처리 방법

새 스크립트:

```text
scripts/sample_prepared_v1_dataset.py
```

역할:

```text
이미 tokenized된 V1Dataset에서 subset을 뽑는다.
선택한 instruction/response token span만 compact tokens.npy로 다시 쓴다.
fresh epoch shuffle을 만든다.
```

이 방식을 쓴 이유:

```text
1. raw JSONL을 다시 토크나이즈하는 것보다 빠르다.
2. pretraining 때 이미 검증한 tokenizer/V1Dataset 포맷을 그대로 쓴다.
3. 큰 데이터에서 20M~80M token 단위 후보를 안전하게 만들 수 있다.
4. LoRA와 full SFT 실험 크기를 빠르게 바꿀 수 있다.
```

merge는 기존 스크립트를 사용했습니다.

```text
scripts/merge_prepared_sft_data.py
```

## 후보별 권장 사용

### 1. 빠른 LoRA/짧은 SFT

먼저 쓸 후보:

```text
/home/work/.data/hrm_text_prepared/kohrm_sft_behavior_mini_v1
```

목적:

```text
한국어 말투가 망가지지 않는지 확인
tool-call JSON 형식이 좋아지는지 확인
터미널 명령 제안/해석이 좋아지는지 확인
SFT가 pretraining 능력을 훼손하지 않는지 확인
```

권장:

```text
LoRA 또는 very short full SFT smoke
1 epoch 먼저 확인
eval 좋으면 2~3 epoch
```

### 2. 터미널/툴콜 특화

후보:

```text
/home/work/.data/hrm_text_prepared/kohrm_sft_terminal_tool_core_v1
```

목적:

```text
터미널 작업 trajectory
tool-call JSON validity
코딩/파일 수정 흐름
명령 출력 해석
```

이 후보는 KoHRM 목표와 가장 직접적으로 맞습니다. terminal/tool benchmark에서 약하면 이 세트를 먼저 씁니다.

### 3. 한국어 도메인 정렬

후보:

```text
/home/work/.data/hrm_text_prepared/kohrm_sft_korean_domain_core_v1
```

목적:

```text
한국어 법률/금융 QA
존댓말 설명
근거 기반 요약
짧고 명확한 한국어 답변
```

주의:

```text
이 세트만 길게 돌리면 모델이 도메인 QA 말투로 기울 수 있다.
terminal/tool 성능 확인 후 보조로 쓰는 편이 낫다.
```

### 4. 전체 행동 core SFT

후보:

```text
/home/work/.data/hrm_text_prepared/kohrm_sft_behavior_core_v1
```

목적:

```text
terminal/tool/coding/reasoning/Korean-domain을 한 번에 정렬
```

권장:

```text
LoRA 결과가 좋거나, pretraining final checkpoint가 안정적일 때 사용한다.
풀파인튜닝을 바로 하기보다 LoRA/짧은 SFT로 먼저 확인한다.
```

## SFT를 할지 RL을 할지 판단

SFT가 필요한 경우:

```text
한국어 존댓말이 불안정하다.
tool-call JSON이 자주 깨진다.
터미널 명령을 설명만 하고 실제 명령 형식으로 못 낸다.
코딩 작업에서 patch/test 루프가 어색하다.
불필요한 장문 reasoning을 노출한다.
```

RL로 바로 넘어가도 되는 경우:

```text
기본 답변 형식은 이미 좋다.
tool-call JSON validity가 충분히 높다.
터미널 action이 대체로 맞다.
남은 문제는 선호도, 성공률, reward ranking으로 더 잘 잡힌다.
```

실무 판단:

```text
1. pretraining checkpoint를 평가한다.
2. behavior_mini_v1로 LoRA/짧은 SFT를 해본다.
3. 개선폭이 크면 terminal_tool_core_v1 또는 behavior_core_v1로 확장한다.
4. 개선폭이 작고 기본 형식이 이미 좋으면 RL 중심으로 간다.
```

## LoRA vs Full SFT

LoRA가 좋은 경우:

```text
빠르게 여러 버전 비교
터미널/툴콜 형식만 살짝 보정
pretraining 능력 훼손 위험을 줄이고 싶을 때
checkpoint 용량을 작게 유지하고 싶을 때
```

Full SFT가 좋은 경우:

```text
LoRA가 부족하다.
행동 변화가 전역적으로 필요하다.
최종 배포 모델에 adapter 의존성을 줄이고 싶다.
terminal/tool/code 전반이 크게 흔들린다.
```

현재 추천 순서:

```text
LoRA smoke on behavior_mini_v1
-> Korean/terminal/tool eval
-> 필요하면 LoRA on korean_domain_core_v1
-> 필요하면 LoRA on terminal_tool_core_v1
-> 그래도 부족하면 short full SFT on behavior_core_v1
-> 마지막에 RL
```

## 학습 config 방향

기존 `config/cfg_sft.yaml`는 full SFT용 기본값입니다.

현재 기본값:

```text
global_batch_size: 32768 token slots
epochs: 5
lr: 3.0e-5
lr_warmup_steps: 0
ema: 0.999
```

후보별 시작점:

```text
behavior_mini_v1:
  epochs 1~3
  global_batch_size 32768~65536
  LoRA 우선

terminal_tool_core_v1:
  epochs 1~2
  global_batch_size 32768~65536
  LoRA 또는 짧은 full SFT

korean_domain_core_v1:
  epochs 1
  다른 행동 세트와 섞거나 보조 SFT

behavior_core_v1:
  epochs 1
  full SFT는 평가 후 결정
```

현재 wrapper 기본값:

```text
global_batch_size:          32768 token slots
lr:                         8.0e-5
checkpoint_step_interval:   1000
checkpoint_keep_last:       2
lora.rank:                  16
lora.alpha:                 32.0
lora.dropout:               0.0
```

별도 HF dataset repo:

```text
https://huggingface.co/datasets/LLM-OS-Models/KoHRM-Text-1.4B-sft-lora-data
```

## 생성 명령 요약

component subset:

```bash
python scripts/sample_prepared_v1_dataset.py \
  --input /home/work/.data/hrm_text_prepared/local_terminal_conversations_ctx9k_resp6k_v1 \
  --output /home/work/.data/hrm_text_prepared/kohrm_sft_comp_terminal_80m_v1 \
  --target-tokens 80000000 \
  --epochs 5 \
  --seed 2801 \
  --copy-tokenizer
```

mix merge:

```bash
python scripts/merge_prepared_sft_data.py \
  --inputs \
    /home/work/.data/hrm_text_prepared/kohrm_sft_comp_terminal_80m_v1 \
    /home/work/.data/hrm_text_prepared/kohrm_sft_comp_toolbench_30m_v1 \
    /home/work/.data/hrm_text_prepared/kohrm_sft_comp_swe_zero_30m_v1 \
    /home/work/.data/hrm_text_prepared/kohrm_sft_comp_glm_reasoning_20m_v1 \
    /home/work/.data/hrm_text_prepared/kohrm_sft_comp_agent_reasoning_25m_v1 \
    /home/work/.data/hrm_text_prepared/kohrm_sft_comp_korean_legal_50m_v1 \
    /home/work/.data/hrm_text_prepared/kohrm_sft_comp_finance_50m_v1 \
  --output /home/work/.data/hrm_text_prepared/kohrm_sft_behavior_core_v1 \
  --epochs 5 \
  --seed 2813 \
  --copy-tokenizer
```

## 현재 상태

```text
전처리 완료.
V1Dataset 무결성 검사 완료.
학습은 아직 시작하지 않음.
현재 장기 pretraining run을 방해하지 않도록 SFT/RL 실행은 별도 판단 후 시작.
HF SFT/LoRA dataset repo 업로드 완료:
  https://huggingface.co/datasets/LLM-OS-Models/KoHRM-Text-1.4B-sft-lora-data
```
