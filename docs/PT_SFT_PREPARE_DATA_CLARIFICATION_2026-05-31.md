# PT/SFT And `prepare_sft_data.py` Clarification

기준일: 2026-05-31

이 문서는 HRM-Text 논문, upstream `sapientinc/HRM-Text`, KoHRM fork에서 `pretraining`, `SFT`, `prepare_sft_data.py`가 각각 무엇을 뜻하는지 정리합니다. 결론은 다음입니다.

```text
논문:
  별도 SFT 단계가 아니라 single-stage task-completion pretraining을 주장한다.

upstream repo:
  pretraining framework가 있고, 추가로 full-parameter SFT 경로도 있다.
  SFT도 별도 trainer가 아니라 pretrain.py + cfg_sft.yaml로 continue-train한다.

prepare_sft_data.py:
  upstream 이름/README 기준으로는 SFT 전처리 스크립트다.
  실제 산출물은 더 일반적인 HRM V1Dataset instruction-response binary layout이다.
  그래서 KoHRM에서는 PT용 instruction-response corpus 전처리에도 재사용한다.
```

## 논문 기준

HRM-Text 논문은 기존 LLM의 “raw text PT -> mid-training/SFT/post-training” 파이프라인을 그대로 따르지 않습니다. 논문 핵심은 처음부터 instruction-response pair로 scratch pretraining을 하는 것입니다.

논문 요지:

- broad raw-text causal LM pretraining을 생략한다.
- instruction-response pair만 사용한다.
- loss는 response token에만 둔다.
- instruction/prefix 영역은 PrefixLM mask로 양방향 attention을 허용한다.
- response 영역은 causal generation을 유지한다.
- `<think>...</think>` 같은 명시적 긴 reasoning trace는 제거해 internal hierarchical computation에 의존하게 한다.

따라서 논문에서 말하는 “pretraining”은 일반 raw LM pretraining이 아닙니다.

```text
일반 raw LM PT:
  raw text stream -> next-token loss over most tokens

HRM-Text paper PT:
  instruction -> response
  prefix bidirectional attention
  response causal attention
  response-only NLL
  scratch training
```

이 때문에 KoHRM에서 “PT인데 왜 SFT처럼 보이냐”는 질문의 답은 다음입니다.

```text
HRM-Text 논문식 PT 자체가 instruction-response response-only 학습이다.
수식은 SFT와 비슷하지만, 역할은 scratch pretraining이다.
```

## upstream repo 기준

upstream README에는 `Fine-Tuning (SFT)` 섹션이 있습니다. 이 SFT는 별도 trainer가 아니라 기존 `pretrain.py`를 SFT config로 다시 실행하는 방식입니다.

upstream SFT 경로:

```text
instruction/response JSONL
  -> scripts/prepare_sft_data.py
  -> V1Dataset binary layout
  -> pretrain.py --config-name cfg_sft
  -> resume_from=/path/to/pretrain_ckpt
  -> full-parameter continue-training
```

즉 upstream에 있는 것은 다음입니다.

```text
있음:
  full-parameter SFT path
  cfg_sft.yaml
  prepare_sft_data.py
  pretrain.py를 재사용한 continue-train

없음:
  별도 SFT trainer
  LoRA trainer
  RL trainer
  DPO/PPO/GRPO pipeline
```

## `prepare_sft_data.py`의 정확한 성격

파일 이름과 upstream docstring은 SFT입니다.

하지만 실제로 이 스크립트가 하는 일은 “SFT만”이 아닙니다. JSONL/parquet row를 아래 구조의 V1Dataset으로 바꿉니다.

```text
input row:
  {
    "instruction": "...",
    "response": "...",
    "condition": "direct"
  }

token layout:
  <|im_start|><condition_token>instruction<|im_end|>response<|box_end|>

output:
  metadata.json
  tokens.npy
  tokenizer_info.json
  tokenizer.json
  epoch_0/inst_start.npy
  epoch_0/inst_len.npy
  epoch_0/resp_start.npy
  epoch_0/resp_len.npy
```

`dataset_new.py`는 이 layout을 읽어 instruction span과 response span을 분리하고, `target_only=True`일 때 response token에만 loss를 겁니다.

따라서 `prepare_sft_data.py`의 더 정확한 설명은 다음입니다.

```text
upstream naming:
  SFT data preparation script

actual output:
  HRM V1Dataset instruction-response binary layout

KoHRM usage:
  1. SFT/LoRA data preparation
  2. instruction-response pretraining corpus preparation
```

즉 `prepare_sft_data.py`를 PT에 쓰는 것은 개념적으로 이상한 일이 아닙니다. 논문식 PT가 이미 task-completion instruction-response objective이기 때문입니다.

## PT와 SFT의 실제 차이

수식은 거의 같습니다.

```text
L(theta) = - sum_t m_t log p_theta(y_t | prefix, y_<t)

m_t = 0 for instruction/prefix tokens
m_t = 1 for response tokens
```

차이는 “objective 수식”보다 “학습 regime”입니다.

```text
PT / instruction pretraining:
  - scratch model에서 시작
  - 수십 B token 규모
  - broad coverage가 중요
  - 한국어/영어/코드/터미널/툴콜/법률/금융/wiki를 넓게 학습
  - lr 높음
  - optimizer/EMA trajectory를 길게 유지

SFT / post-training:
  - pretrained checkpoint에서 시작
  - 작고 고품질인 subset
  - JSON validity, command-only, tool-call, 존댓말, formatting을 강하게 보정
  - lr 낮음
  - EMA weight에서 optimizer reset하는 선택이 자연스러움
```

## KoHRM에 어떻게 적용할지

### 1. 현재 대규모 학습은 SFT가 아니라 PT로 부른다

KoHRM의 현재 pass 1/2/3/4 장기 run은 SFT가 아닙니다. 정확한 명칭은 다음 중 하나입니다.

```text
KoHRM instruction pretraining
single-stage task-completion pretraining
PrefixLM response-only pretraining
```

SFT 후보 데이터가 섞여 있어도, scratch model의 broad pretraining mix에 들어가는 순간 역할은 PT입니다.

### 2. `prepare_sft_data.py`는 이름은 유지하되 설명을 보정한다

파일명을 바꾸면 upstream과의 diff가 커지고 기존 문서/스크립트 링크가 깨질 수 있습니다. 따라서 파일명은 유지합니다.

대신 문서와 docstring에서 다음을 명확히 합니다.

```text
This script is named after upstream SFT preparation,
but it prepares the generic HRM V1Dataset instruction-response layout.
KoHRM uses the same layout for both instruction pretraining and later SFT/LoRA.
```

### 3. PT용 데이터는 SFT처럼 “정답을 맞히는 응답”이어야 한다

논문 원칙상 raw text를 그냥 response에 넣는 것보다 instruction-response task 형태가 맞습니다.

좋은 PT row 예:

```text
instruction:
  다음 한국 법령 문서를 원문 의미가 유지되도록 정리하세요.

response:
  제5조 ...
```

```text
instruction:
  In bash, continue this terminal task.

response:
  find . -type f ...
```

나쁜 PT row 예:

```text
instruction:
  빈 문자열 또는 의미 없는 wrapper

response:
  긴 raw dump 전체
```

완전 raw dump는 논문식 objective와 멀어질 수 있습니다. 다만 지식 주입을 위해 원문 continuation 스타일이 필요하면 instruction을 명시해서 “문서 복원/요약/발췌/설명” task로 만드는 편이 낫습니다.

### 4. SFT/LoRA는 PT 완료 후 별도 행동 보정으로 유지한다

현재 모델이 PT 중간 checkpoint에서 JSON-only, command-only, chat instruction following을 못하는 것은 이상하지 않습니다. 논문식 PT는 task-completion 능력을 만들지만, 우리가 원하는 엄격한 형식 준수는 후처리에서 더 강하게 잡아야 합니다.

권장 순서:

```text
1. pass 4까지 instruction pretraining 완료
2. raw checkpoint inference와 HF export inference 대조
3. long continuation probe로 지식/반복/문체 확인
4. behavior_mini_v1 LoRA smoke
5. terminal_tool_core_v1 LoRA
6. korean_domain_core_v1 LoRA
7. 필요하면 behavior_core_v1 short full SFT
8. verifier가 있는 작업부터 RL
```

### 5. full SFT는 upstream 방식대로 가능하지만 기본 1순위는 아니다

upstream의 full-parameter SFT 경로는 유효합니다.

```bash
torchrun --nproc_per_node=8 pretrain.py \
  --config-name cfg_sft \
  arch/size@arch=XL \
  data.path=/path/to/prepared_sft_data \
  resume_from=/path/to/pretrain_ckpt \
  +checkpoint_path=/path/to/sft_out
```

하지만 KoHRM에서는 먼저 LoRA가 낫습니다.

이유:

- PT final checkpoint를 크게 망가뜨릴 위험이 낮습니다.
- 후보 subset별 효과 비교가 빠릅니다.
- 한국어 도메인/터미널/툴콜을 따로 나눠볼 수 있습니다.
- 충분하면 full SFT 없이 adapter variant로 운영할 수 있습니다.

full SFT는 아래 경우에만 진행합니다.

```text
LoRA로 형식/언어 보정이 충분하지 않다.
adapter 없이 base weight 자체를 공개해야 한다.
여러 LoRA 효과를 하나의 dense checkpoint로 합치고 싶다.
```

## 문서/코드 표현 정책

앞으로 repo에서는 용어를 이렇게 씁니다.

```text
instruction pretraining:
  현재 장기 PT run. 논문 방식의 scratch task-completion training.

SFT:
  pretrained checkpoint 이후 고품질 subset으로 행동을 보정하는 후속 단계.

prepare_sft_data.py:
  이름은 upstream과 호환 유지.
  설명은 "generic HRM V1Dataset preparation for instruction-response data"로 보정.

SFT data in PT:
  SFT 후보 원천 데이터라도 PT mix에 들어가면 역할은 instruction pretraining data.
```

