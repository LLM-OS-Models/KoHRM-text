# HRM-Text Post-Training Guide - 2026-05-30

이 문서는 KoHRM-Text pretraining 이후 SFT, LoRA, RL을 어떻게 적용해야 하는지 정리합니다. 핵심은 이 모델이 일반 raw-text PT 모델도 아니고, 일반 chat SFT 모델도 아니라는 점입니다. HRM-Text 논문식 single-stage instruction pretraining 위에 KoHRM 데이터와 tokenizer를 얹은 구조이므로, post-training도 같은 포맷과 추론 전제를 유지해야 합니다.

Sources:
- HRM-Text paper: https://arxiv.org/html/2605.20613
- Upstream repo: https://github.com/sapientinc/HRM-Text
- KoHRM fork: https://github.com/LLM-OS-Models/KoHRM-text

## 한 줄 결론

지금은 full SFT를 바로 크게 돌리기보다, pretraining을 끝까지 안정적으로 이어간 뒤 같은 V1Dataset/PrefixLM 포맷으로 작은 LoRA SFT를 먼저 검증하는 것이 맞습니다. RL은 나중에 JSON validity, command correctness, unit tests 같은 verifier가 준비된 작업부터 붙이는 것이 안전합니다.

## 모델이 학습한 형식

학습 sample은 다음 구조입니다.

```text
<|im_start|><condition_token>instruction<|im_end|>response<|box_end|>
```

현재 tokenizer condition mapping:

```text
direct -> <|object_ref_start|>
cot    -> <|object_ref_end|>
noisy  -> <|quad_start|>
synth  -> <|quad_end|>
```

기본 post-training과 inference는 `direct`를 씁니다. `direct`는 answer-only, JSON-only, command-only, code-only 결과를 기대할 때 가장 안전합니다.

`dataset_new.py`는 instruction span과 response span을 분리해 읽습니다. `target_only=True`이면 instruction token의 label은 `IGNORE_LABEL_ID`가 되고, loss는 response token에만 걸립니다. 따라서 모델은 입력 instruction을 조건으로 보고, response를 생성하는 방향으로 학습됩니다.

## 왜 PT도 SFT도 아닌 중간인가

일반 raw-text PT:

```text
raw text stream -> next-token loss over almost all tokens
```

전통적인 SFT:

```text
already pretrained base -> instruction/response data -> behavior alignment
```

KoHRM/HRM-Text식 instruction pretraining:

```text
scratch model -> instruction/response data -> response-only loss at pretraining scale
```

수식만 보면 SFT와 비슷합니다.

```text
L(theta) = - sum_t m_t log p_theta(y_t | prefix, y_<t)

m_t = 0 for instruction/prefix tokens
m_t = 1 for response tokens
```

하지만 역할은 다릅니다. SFT는 이미 언어를 배운 base의 행동을 고치는 단계이고, KoHRM pretraining은 처음부터 언어/지식/도구/명령 수행 능력을 instruction-response 형태로 형성하는 단계입니다.

## PrefixLM 추론 전제

HRM-Text 논문은 response-only task-completion objective와 PrefixLM mask를 함께 씁니다. PrefixLM은 instruction/prefix 구간에서는 양방향 attention을 허용하고, response 구간은 causal generation을 유지합니다.

```text
instruction / prefix             response
[ bidirectional context ]  ->  [ causal generation ]
```

따라서 추론 코드도 prompt wrapper를 정확히 써야 합니다.

```python
wrapped = f"<|im_start|><|object_ref_start|>{prompt}<|im_end|>"
```

이 구조에서 한국어 법률/위키/금융 지식형 prompt는 한국어로 테스트하는 것이 맞고, 터미널/코딩/툴콜은 현재 학습 mix상 영어 prompt가 더 안정적일 가능성이 큽니다. 한국어로 터미널 명령을 요구하면 모델이 영어 추론문이나 설명으로 흔들릴 수 있으므로, post-training 전 smoke test에서는 영어 명령형 prompt를 먼저 기준으로 둡니다.

## HRM 아키텍처가 post-training에 주는 의미

KoHRM-Text는 단순 decoder-only Transformer가 아닙니다. HRM core는 L/H 두 수준의 recurrent module을 반복 실행합니다.

```text
input embedding
   |
   v
z_H initial = input
z_L initial = learned latent state
   |
   |  H cycle 1
   |    L module repeated L_cycles times
   |    H module once
   |
   |  H cycle 2
   |    L module repeated L_cycles times
   |    H module once
   v
LM head -> response logits
```

로컬 코드 기준 핵심 파일:

- `models/baselines/hrm_nocarry_bp_warmup.py`: H/L recurrent execution, `H_cycles`, `L_cycles`, BP warmup.
- `models/flash_attention_prefixlm_v2.py`: PrefixLM attention path.
- `dataset_new.py`: instruction/response packing과 response-only label 생성.
- `train_lora.py`: KoHRM용 LoRA SFT.

논문은 명시적 long CoT를 학습에서 제거하고 내부 hierarchical computation에 의존하도록 설계했다고 설명합니다. 그래서 post-training에서도 긴 visible CoT를 강제로 학습시키는 것은 기본값으로 두지 않는 편이 맞습니다.

## SFT 시 주의사항

1. 같은 tokenizer를 유지합니다.

   현재 KoHRM tokenizer는 131,072 vocab입니다. tokenizer를 바꾸면 embedding/lm_head 크기와 token 분포가 달라져 기존 checkpoint와 호환되지 않습니다.

2. 같은 V1Dataset 포맷을 씁니다.

   `scripts/prepare_sft_data.py` 또는 prepared subset을 사용해 `tokens.npy`, `metadata.json`, `epoch_*/*.npy` 구조를 만들어야 합니다. plain JSONL을 임의 trainer에 바로 넣으면 PrefixLM/condition token/response-only loss가 깨집니다.

3. `target_only=True`를 유지합니다.

   SFT에서도 instruction을 출력 대상으로 학습시키면 모델이 prompt를 반복하거나 설명문을 앞에 붙이는 방향으로 망가질 수 있습니다.

4. `direct` condition을 기본으로 씁니다.

   JSON, bash command, code, 짧은 한국어 답변은 `direct=<|object_ref_start|>`가 맞습니다. `cot`는 visible reasoning 응답을 의도적으로 만들 때만 제한적으로 씁니다.

5. full SFT보다 LoRA smoke를 먼저 합니다.

   이미 `train_lora.py`가 같은 loader, 같은 response-only objective, 같은 HRM carry/PrefixLM 경로를 사용합니다. 따라서 작은 행동 보정에는 LoRA가 우선입니다.

6. prompt를 과하게 꾸미지 않습니다.

   학습 sample은 대체로 짧은 instruction과 response를 특수 토큰으로 감싼 형태입니다. post-training data도 불필요한 system preamble, 긴 메타 지시, 과도한 markdown 설명을 줄이고 task instruction을 직접적으로 씁니다.

7. strict format task는 validator로 봅니다.

   JSON-only는 `json.loads`, command-only는 한 줄 여부와 금지 문자열, code-only는 syntax/unit test로 평가해야 합니다. 겉보기 fluency보다 구조 준수가 우선입니다.

## 현재 SFT/LoRA 후보

이미 prepared V1Dataset 형태로 만든 후보입니다.

```text
behavior_mini_v1          60.0M tokens   빠른 LoRA/SFT smoke
terminal_tool_core_v1    165.0M tokens   bash/tool-call/code 형식 보정
korean_domain_core_v1    100.0M tokens   한국어 법률/금융/wiki 응답 보정
behavior_core_v1         285.0M tokens   전체 행동 보정 후보
```

권장 순서:

```text
1. finish pretraining continuation without GPU idle
2. evaluate deterministic notebook probes
3. LoRA on behavior_mini_v1
4. if terminal/tool format is weak: LoRA on terminal_tool_core_v1
5. if Korean domain repetition/hallucination remains: LoRA on korean_domain_core_v1
6. if both are still weak: short full SFT on behavior_core_v1
7. only then add RL/verifier optimization
```

## LoRA 코드 적용 방식

현재 `train_lora.py`는 다음 module suffix에 LoRA를 삽입합니다.

```text
gqkv_proj
o_proj
gate_up_proj
down_proj
lm_head
```

`lm_head`는 131K vocab 때문에 adapter 용량이 커질 수 있지만, JSON bracket, bash token, 한국어 종결 표현 같은 출력 분포를 바로 고치는 데 의미가 있습니다. 작은 smoke에서는 `lm_head` 포함을 유지하고, VRAM/속도 문제가 생기면 `lm_head` 제외 실험을 별도로 비교합니다.

기본 config:

```text
config/cfg_lora.yaml
global_batch_size: 32768 token slots
epochs: 1
lr: 8e-5
lr_warmup_steps: 100
rank: 16
alpha: 32
checkpoint_keep_last: 2
```

pretraining final checkpoint에서 시작할 때는 `weights_only_resume_from_ema=true`를 유지하는 것이 맞습니다. pretraining optimizer state를 그대로 이어받아 작은 SFT를 하는 것보다, EMA weight를 base로 쓰고 fresh optimizer로 행동 보정을 하는 편이 안정적입니다.

## RL은 바로 가능한가

가능은 하지만, 지금 repo에는 PPO/GRPO/DPO 같은 RL trainer가 없습니다. 논문과 upstream README도 pretraining framework, evaluation, conversion, basic full-parameter SFT를 중심으로 설명하고, 구체적인 RL post-training recipe는 제공하지 않습니다.

RL을 붙인다면 response-token objective 위에서 별도 loop를 만들어야 합니다.

```text
same wrapped prefix
 -> generate response
 -> verifier / reward
 -> policy update on response tokens only
```

주의점:

- preference pair는 같은 instruction/prefix를 공유하고 response만 달라야 합니다.
- PrefixLM prefill에서 instruction 양방향 attention mask를 보존해야 합니다.
- reward는 visible CoT 길이가 아니라 결과 형식과 정답성을 봐야 합니다.
- terminal/tool RL은 model-graded reward만 쓰지 말고 sandbox command check, JSON schema check, unit test를 씁니다.
- 한국 법률 추출은 원문 span 보존, 조문명 정확도, JSON validity 같은 deterministic reward가 좋습니다.
- coding은 unit tests와 lint/type check를 reward로 쓰는 편이 맞습니다.

현재 모델이 command-only, JSON-only, 한국어 반복 억제를 안정적으로 못 하면 RL만으로 바로 해결하기 어렵습니다. 먼저 작은 LoRA/SFT로 출력 manifold를 좁힌 뒤 RL을 붙이는 것이 더 안전합니다.

## 논문과 upstream에 있는 것 / 없는 것

논문에 있는 것:

- instruction-response pair from scratch.
- response-only task-completion NLL.
- PrefixLM attention.
- direct/cot/synth/noisy condition tags.
- `<think>...</think>` 제거 정책과 내부 hierarchical computation 강조.
- PrefixLM inference framework/vLLM 통합 시 custom attention mask와 KV-cache 주의가 필요하다는 논의.

upstream README에 있는 것:

- full-parameter SFT 절차.
- JSONL input format: `instruction`, `response`, optional `condition`.
- `prepare_sft_data.py`로 V1Dataset 준비.
- `cfg_sft`로 checkpoint continue training.
- native vLLM support는 in progress라고 명시.
- `simple_inference_engine.py` 기반 compiled generation engine.

현재 KoHRM fork에 추가된 것:

- 131K Korean/terminal tokenizer.
- KoHRM staged pretraining data mix.
- checkpoint chain/watchers.
- HF upload scripts.
- LoRA SFT code and prepared SFT/LoRA subsets.
- Colab T4 lightweight safetensors inference notebook.

없는 것:

- production-grade vLLM serving implementation.
- Transformers `AutoModelForCausalLM`로 바로 안정 동작하는 일반 runtime.
- PPO/GRPO/DPO/RLVR trainer.
- final chat template 기반 assistant serving recipe.

## 현재 노트북 해석 기준

`notebooks/KoHRM_Text_1_4B_Colab_T4_Smoke_Test.ipynb`는 최종 벤치마크가 아니라 중간 checkpoint sanity check입니다.

나쁜 출력의 원인 후보:

```text
1. checkpoint가 아직 pretraining 중간이라 지시 준수/반복 억제가 덜 됨
2. terminal/coding을 한국어 prompt로 물어 training distribution과 어긋남
3. direct condition과 stop token을 정확히 쓰지 않음
4. decoding이 너무 greedy라 반복 local optimum에 빠짐
5. 아직 LoRA/SFT/RL post-training 전이라 JSON-only/command-only alignment가 약함
```

수정된 노트북은 다음 원칙을 따릅니다.

```text
Korean legal/wiki/finance -> Korean prompts
Terminal/coding/tool-call -> English prompts
condition -> direct
first decode -> deterministic
second decode -> optional low-temperature retry
validation -> JSON parse / one-line command / code shape
```

## 운영 우선순위

지금 가장 중요한 것은 GPU가 놀지 않게 pretraining chain을 끝까지 이어가는 것입니다. SFT/LoRA/RL 준비와 문서화는 training process를 방해하지 않는 범위에서 병렬로 진행합니다.

업로드 watcher는 stage1c/2c/3c/4c step checkpoint도 감시해야 합니다. 이전에는 stage1c 이후 목록 누락으로 자동 업로드가 빠졌고, 지금은 `scripts/watch_chain_step_checkpoints_upload.py`에 continuation stage를 추가했습니다.
