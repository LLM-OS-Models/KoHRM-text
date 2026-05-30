# KoHRM-Text Codebase Guide

기준일: 2026-05-31

이 문서는 `KoHRM-text` 저장소의 폴더와 핵심 코드가 각각 무엇을 담당하는지 정리한 운영용 가이드입니다. 이 repo는 [`sapientinc/HRM-Text`](https://github.com/sapientinc/HRM-Text)를 fork한 뒤, KoHRM-Text-1.4B scratch pretraining, 한국어/터미널 tokenizer, staged training, Hugging Face 업로드, SFT/LoRA 후보 전처리를 추가한 작업 저장소입니다.

## 전체 구조

큰 흐름은 아래 순서입니다.

```text
raw sources
  -> builder scripts
  -> scripts/prepare_sft_data.py
  -> V1Dataset arrays
  -> pretrain.py
  -> FSDP2 checkpoints
  -> watcher scripts
  -> conversion/convert_to_hf.py
  -> HF model / raw checkpoint / prepared data repos
  -> notebooks, evaluation, SFT/LoRA
```

학습 데이터는 일반 raw text를 그대로 이어 붙이는 방식이 아니라 HRM-Text 방식의 instruction-response 형태로 정리됩니다.

```text
<boq> condition instruction <eoq> response <eoa>
```

학습 중 attention/loss 관점은 다음과 같습니다.

```text
instruction / prefix
  - 모델이 읽는 입력 컨텍스트
  - PrefixLM에서 양방향 attention
  - loss 없음

response
  - 모델이 맞혀야 하는 출력 영역
  - causal attention
  - response-only CE loss
```

따라서 이 repo의 코드는 크게 네 축으로 나뉩니다.

```text
1. 데이터 준비 코드
2. HRM/PrefixLM 학습 코드
3. 체크포인트 변환/업로드/운영 watcher
4. 추론, 평가, SFT/LoRA 후처리 코드
```

## 최상위 폴더

### `config/`

Hydra 설정 파일입니다. 어떤 모델 구조를 쓸지, 어떤 데이터 경로를 볼지, batch/lr/checkpoint 정책을 어떻게 둘지 결정합니다.

주요 파일:

- `config/cfg_pretrain.yaml`

  scratch pretraining 기본 설정입니다. HRM XL, 데이터 `hlm`, `epochs: 4`, `checkpoint_keep_last: 2`를 기본값으로 둡니다. 실제 KoHRM 장기 run에서는 CLI override로 `global_batch_size=180224`, stage별 `data.path`, `resume_from`, `resume_step_offset`, `checkpoint_step_interval` 등을 바꿔 실행합니다.

- `config/cfg_sft.yaml`

  full SFT용 설정입니다. pretrained checkpoint를 `resume_from`으로 받고, SFT prepared dataset을 `data.path`로 받습니다. 기본 `global_batch_size`는 pretraining보다 작은 32768 token slots입니다.

- `config/cfg_lora.yaml`

  LoRA adapter 학습용 설정입니다. `weights_only_resume_from_ema: true`, LoRA rank/alpha/target suffix가 들어 있습니다. full weight를 직접 바꾸지 않고 adapter만 학습할 때 사용합니다.

- `config/arch/net/hrm.yaml`

  KoHRM 기본 네트워크입니다. `models/baselines/hrm_nocarry_bp_warmup.py`의 `HierarchicalReasoningModel`을 사용하고, H/L recurrent block, `H_cycles`, `L_cycles`, BP warmup 설정을 지정합니다.

- `config/arch/size/XL.yaml`

  현재 KoHRM-Text-1.4B의 backbone 크기입니다. 32 layers, hidden size 1536, 12 attention heads를 사용합니다. 131K vocab 때문에 총 파라미터 수는 upstream 1B보다 커집니다.

- `config/data/hlm.yaml`, `config/data/sft.yaml`

  학습 데이터 설정입니다. `dataset_new.py`의 V1Dataset을 사용하며, 실제 데이터 경로는 대부분 CLI override로 넣습니다.

### `models/`

모델 아키텍처, attention, optimizer, LoRA 모듈이 들어 있습니다. HRM-Text의 핵심 구현이 이 폴더에 있습니다.

주요 파일:

- `models/baselines/hrm_nocarry_bp_warmup.py`

  KoHRM pretraining의 핵심 HRM 모델입니다. 입력 embedding을 만든 뒤 H module과 L module을 recurrent하게 왕복시킵니다. `compute_train_extra_args()`는 학습 step에 따라 recurrent backprop depth를 warmup합니다. 즉 초반에는 짧은 depth로 안정화하고, 이후 `bp_max_steps`까지 늘립니다.

- `models/transformer.py`

  Transformer block의 공통 구현입니다. attention, MLP, norm 구조를 감싼 기본 블록입니다.

- `models/layers.py`

  RoPE, attention projection, SwiGLU, cache 구조 같은 저수준 layer primitives입니다.

- `models/flash_attention_prefixlm_v2.py`

  PrefixLM용 custom FlashAttention 경로입니다. prefix는 양방향으로, response는 causal하게 처리하기 위해 varlen prefix/causal 길이를 같이 넘깁니다. KoHRM 학습 성능과 메모리 효율에 직접 영향을 줍니다.

- `models/lm_head.py`

  backbone 출력 위에 LM head를 붙이고 response-only loss/metric을 계산합니다. instruction 영역은 입력으로만 쓰이고 loss 대상이 아닙니다.

- `models/adam_atan2.py`

  HRM-Text에서 쓰는 Adam-atan2 optimizer입니다. EMA swap 기능도 포함되어 checkpoint 저장/평가 시 사용됩니다.

- `models/lora.py`

  LoRA adapter 삽입과 저장 유틸리티입니다. `LoRALinear`, `inject_lora`, `mark_only_lora_trainable`, `save_lora_adapter`가 있습니다.

- `models/baselines/*.py`

  RINS, TRM, UT, plain Transformer wrapper 같은 비교/대체 구조입니다. 현재 KoHRM 장기 pretraining의 주 경로는 HRM입니다.

### `pretrain.py`

메인 학습 진입점입니다. torchrun으로 실행되는 분산 학습 스크립트입니다.

실행 흐름:

```text
Hydra config load
  -> distributed init
  -> V1Dataset + Multipack sampler 생성
  -> HRM model + LMHead 생성
  -> FSDP2 적용
  -> optimizer / EMA / train state 생성
  -> checkpoint resume
  -> train loop
  -> step checkpoint / epoch checkpoint 저장
```

중요 함수:

- `create_dataloader()`

  `dataset_new.V1Dataset`과 `multipack_sampler.MultipackDistributedBatchSampler`를 연결합니다. global batch는 샘플 수가 아니라 token slot 기준입니다.

- `apply_fsdp()`

  모델 모듈을 FSDP2로 감쌉니다. H200 8장 장기 학습에서 각 rank별 shard를 나눠 들게 합니다.

- `create_model_and_carry()`

  metadata의 vocab/context 정보를 바탕으로 HRM model, LM head, carry를 만듭니다.

- `init_train()`

  model, optimizer, train state, dataloader를 초기화합니다.

- `train_batch()`

  한 batch의 forward/backward/update를 수행합니다. `model.compute_train_extra_args()`에서 넘어온 BP warmup 관련 인자도 여기로 전달됩니다.

- `save_training_checkpoint()`

  FSDP2 shard checkpoint를 저장합니다. `checkpoint_step_interval`이 있으면 step checkpoint, epoch 종료 시에는 `fsdp2_epoch_*` 형태로 저장됩니다.

- `prune_old_checkpoints()`

  `checkpoint_keep_last` 정책에 따라 오래된 checkpoint를 정리합니다. 현재 운영은 로컬 최신 2개 중심입니다.

- `load_checkpoint()`

  stage continuation의 핵심입니다. 이전 stage의 model/optimizer/train state를 이어받고, `resume_step_offset`으로 전체 진행 step을 이어 붙입니다.

주의할 점:

- `total_steps_override`는 보통 LR/progress 계산 기준으로 쓰이며, 데이터 자체를 안전하게 자르는 기능으로만 이해하면 안 됩니다.
- stage를 이어갈 때는 checkpoint metadata의 실제 `global_step`과 watcher가 넘기는 `resume_step_offset`이 맞아야 합니다.
- 학습 중인 checkpoint 폴더를 임의로 삭제하면 continuation과 upload watcher가 깨질 수 있습니다.

### `dataset_new.py`

HRM V1Dataset reader입니다. `scripts/prepare_sft_data.py`나 `data_io/sample_tokenized.py`가 만든 tokenized prepared dataset을 읽습니다.

기대하는 구조:

```text
prepared_dataset/
  metadata.json
  tokens.npy
  epoch_0/
    inst_start.npy
    inst_len.npy
    resp_start.npy
    resp_len.npy
```

역할:

- instruction/prefix 구간과 response 구간을 slice합니다.
- context length에 맞는 batch tensor를 만듭니다.
- PrefixLM attention에 필요한 prefix length, response length, sequence metadata를 제공합니다.
- response-only loss가 가능하도록 label 위치를 맞춥니다.

### `multipack_sampler.py`

token-based batch packing sampler입니다. 샘플 길이가 서로 다르기 때문에, 단순히 N개씩 묶으면 GPU token slot이 낭비됩니다. 이 sampler는 longest processing time 방식으로 rank별 batch에 샘플을 채워 넣어 `global_batch_size`에 가까운 token utilization을 만듭니다.

중요한 의미:

- KoHRM의 batch size는 “문서 개수”가 아니라 “token slot 수”입니다.
- 데이터 길이 분포가 바뀌면 step 수와 throughput도 달라집니다.
- SFT와 pretraining batch 수치가 다르게 보이는 이유도 대부분 이 token-based packing 때문입니다.

### `scripts/`

데이터 다운로드, 원문 변환, tokenizer 학습, prepared dataset 생성, stage chain, 업로드를 담당합니다.

#### 데이터 다운로드/원문 구성

- `scripts/download_extra_training_data.py`

  Hugging Face 후보 데이터셋을 제한 용량으로 받습니다. 대형 데이터는 일부만 가져오거나 streaming/cap 방식으로 저장합니다.

- `scripts/build_korean_legal_raw_corpus.py`

  `legalize-kr`, `ordinance-kr`, `admrule-kr`, `precedent-kr` 등 한국 법률/행정/판례 원문을 raw corpus 스타일 JSONL로 만듭니다. 긴 문서는 chunking하고 metadata를 붙입니다.

- `scripts/build_korean_legal_sft_data.py`

  한국 법률 원문에서 QA/extraction 스타일 instruction-response row를 만듭니다. SFT 후보로도 쓰지만, KoHRM 정책에서는 pretraining에도 먼저 포함합니다.

- `scripts/build_kowiki_raw_corpus.py`

  Korean Wikipedia XML dump를 읽고 redirect/template 등을 정리한 뒤 chunk row로 만듭니다.

- `scripts/build_terminal_conversation_sft_data.py`

  terminal/code/tool 대화 parquet에서 instruction-response 예시를 만듭니다. 긴 터미널 로그는 trim/chunk합니다.

- `scripts/build_toolbench_sft_data.py`

  ToolBench 스타일 history/message를 KoHRM 학습 row로 변환합니다. tool-call 행동 후보 데이터에 해당합니다.

- `scripts/build_hf_extra_sft_data.py`

  angrygiraffe, DeepSeek agent, Open-MM-RL 같은 추가 HF snapshot에서 SFT 후보 row를 구성합니다.

#### tokenizer/전처리

- `scripts/train_koterm_tokenizer.py`

  KoHRM용 131K byte-level BPE tokenizer를 학습합니다. 한국어, 법률, 터미널, 코드, tool JSON 효율을 맞추는 핵심 코드입니다.

- `scripts/check_tokenizer_efficiency.py`

  한국어/법률/터미널/코드/영어 샘플에서 chars/token을 확인합니다. tokenizer가 한국어를 지나치게 잘게 쪼개는지 보는 sanity check입니다.

- `scripts/prepare_sft_data.py`

  가장 중요한 전처리 스크립트입니다. JSONL/parquet row를 KoHRM V1Dataset 배열로 바꿉니다.

  핵심 특수 토큰:

  ```text
  <|im_start|> = boq
  <|im_end|>   = eoq
  <|box_end|>  = eoa
  ```

  기본 condition mapping:

  ```text
  direct -> <|object_ref_start|>
  cot    -> <|object_ref_end|>
  noisy  -> <|quad_start|>
  synth  -> <|quad_end|>
  ```

  출력은 `tokens.npy`, `metadata.json`, `epoch_0/*.npy`입니다. 학습 중 `dataset_new.py`가 이 구조를 그대로 읽습니다.

- `scripts/build_hrm_extra_sample_epochs.py`

  이미 tokenized된 HRM cleaned task를 epoch별 V1Dataset으로 sampling합니다. upstream HRM data를 KoHRM tokenizer/format에 맞게 여러 logical pass로 구성할 때 씁니다.

- `scripts/merge_prepared_sft_data.py`

  여러 prepared V1Dataset을 하나로 합칩니다. SFT/LoRA 후보 subset을 묶을 때 사용합니다.

- `scripts/sample_prepared_v1_dataset.py`

  prepared dataset을 사람이 읽을 수 있게 몇 개 decode해 확인합니다. prompt/response 경계가 깨졌는지 볼 때 중요합니다.

- `scripts/audit_data_status.py`

  데이터 폴더별 용량, row 수, metadata, 샘플을 요약합니다. 사용자가 “뭐가 몇 GB/몇 토큰인지” 물을 때 근거를 만드는 스크립트입니다.

#### 학습 stage/watcher

아래 watcher들은 GPU 학습을 직접 대신하는 코드가 아니라, 이전 stage가 끝났는지 보고 다음 `torchrun pretrain.py`를 시작하거나 upload 작업을 붙이는 운영 코드입니다.

- `scripts/schedule_kohrm_stage_chain.py`

  초기 staged 학습 아이디어를 자동화한 scheduler입니다. 전처리 완료 데이터 snapshot을 만들고 stage별 학습을 순서대로 실행하는 역할입니다.

- `scripts/watch_stage1_then_train_next.py`

  stage1 완료 후 다음 stage 학습과 checkpoint upload를 시작하는 초기 watcher입니다. small mix 구성, checkpoint staging, converted model upload helper도 포함합니다.

- `scripts/watch_stage2_then_two_pass_chain.py`

  stage2 이후 두 번째 pass를 이어가기 위한 watcher입니다.

- `scripts/watch_stage1b_then_finish_chain.py`

  pass 2 recovery/continuation watcher입니다. stage1b 이후 남은 stage를 이어갑니다.

- `scripts/watch_stage2b_then_finish_chain.py`

  stage2b 이후 stage3b/4b 및 다음 pass 연결에 관여한 watcher입니다.

- `scripts/watch_stage1c_then_finish_chain.py`

  현재 pass 3 continuation의 핵심 watcher입니다. stage1c/2c/3c/4c 흐름을 이어가도록 설계되어 있습니다.

- `scripts/watch_stage4c_then_epoch4_chain.py`

  pass 4 continuation watcher입니다. stage4c가 끝나면 stage1d -> stage2d -> stage3d -> stage4d를 자동으로 실행하도록 추가되었습니다. GPU idle gap을 줄이기 위한 코드입니다.

- `scripts/watch_stage3_then_finish_chain.py`, `scripts/watch_manual_stage2_then_continue.py`

  중간에 수동으로 복구해야 할 때 쓰는 recovery watcher입니다.

- `scripts/guard_stage1b_handoff.py`

  stage handoff가 멈췄는지 감시하고 watcher를 재시작하는 보조 guard입니다.

#### 체크포인트/HF 업로드

- `scripts/watch_chain_step_checkpoints_upload.py`

  현재 운영에서 중요한 step checkpoint upload watcher입니다. stage별 checkpoint 폴더를 감시하고, 준비된 step checkpoint를 raw checkpoint repo와 converted model repo로 올립니다. marker file로 중복 업로드를 피합니다.

- `scripts/watch_epoch2_final_upload.py`

  epoch 2 완료본을 별도 표시/업로드하기 위한 watcher입니다. “epoch2 final”처럼 사람이 찾기 쉬운 산출물을 만드는 목적입니다.

- `scripts/watch_and_upload_hrm_checkpoints.py`

  초기 raw checkpoint upload helper입니다. epoch checkpoint를 staging해서 HF에 올립니다.

- `scripts/upload_folder_to_hf.py`

  일반 폴더를 HF model/dataset repo에 올리는 공용 uploader입니다. `.env`의 HF token을 읽고 large upload 옵션을 지원합니다.

- `scripts/upload_sft_lora_data_to_hf.py`

  SFT/LoRA prepared subset을 별도 dataset repo로 올리는 uploader입니다.

- `scripts/schedule_followup_prepared_uploads.sh`

  full Korean legal data, HRM full/no-cap prepared data 같은 후속 prepared dataset 업로드를 예약하는 shell scheduler입니다.

- `scripts/report_kohrm_status.sh`

  프로세스, 학습 로그, scheduler 로그, GPU 상태, 전처리 개수, checkpoint, 디스크 상태를 한 번에 출력하는 상태 보고 스크립트입니다.

#### LoRA 실행

- `scripts/run_kohrm_lora_experiments.sh`

  SFT/LoRA 후보 subset별 adapter 학습 wrapper입니다. `RESUME_FROM`으로 base checkpoint를 받고, `behavior-mini`, `terminal-tool`, `korean-domain`, `behavior-core`, `phase1`, `all` 같은 실행 모드를 제공합니다.

### `conversion/`

raw FSDP2 checkpoint를 Hugging Face에서 쓰기 쉬운 safetensors export로 바꿉니다.

- `conversion/convert_to_hf.py`

  핵심 변환 스크립트입니다.

  역할:

  ```text
  FSDP2 shard checkpoint
    -> state_dict key remap
    -> config.json 생성
    -> model.safetensors 저장
    -> tokenizer files 복사/정리
    -> HF repo 업로드 대상 폴더 생성
  ```

  중요한 함수:

  - `remap_key()`: training checkpoint의 module key를 HF runtime이 읽는 key로 바꿉니다.
  - `convert_state_dict()`: sharded/raw weight를 export용 dict로 정리합니다.
  - `build_hf_config()`: `model_type`, `architectures`, vocab size, hidden size, heads, context length, `prefix_lm: true` 같은 config를 만듭니다.
  - `set_tokenizer_special_tokens()`: tokenizer special tokens를 HF export에 맞게 보정합니다.

### `notebooks/`

Colab/운영 확인용 노트북과 lightweight generation runtime이 있습니다.

- `notebooks/kohrm_colab_generate.py`

  Colab T4에서도 public `model.safetensors`를 직접 로드하기 위한 경량 runtime입니다. Transformers auto model 경로가 HRM custom architecture를 모르면 깨질 수 있기 때문에, KoHRM 구조를 직접 구현해 생성합니다.

  주요 기능:

  - safetensors weight 로드
  - KoHRM attention/MLP/HRM H-L recurrence 구현
  - tokenizer special token 기반 prompt formatting
  - `condition="direct"` 등 HRM condition token 적용
  - 반복 억제, min_new_tokens, temperature/top-p decoding

- `notebooks/KoHRM_Text_1_4B_Colab_T4_Long_Knowledge_Probe.ipynb`

  현재 pretraining checkpoint의 긴 생성 신호를 보기 위한 Colab 노트북입니다. 아직 SFT/chat checkpoint가 아니므로 JSON-only, command-only 같은 strict instruction following 평가보다, 학습 데이터 분포와 비슷한 긴 prompt/긴 continuation을 확인하는 목적입니다.

- `notebooks/KoHRM_Text_1_4B_Colab_T4_Smoke_Test.ipynb`

  기존 링크 호환용입니다. 현재 내용은 long knowledge probe와 같은 방향으로 유지합니다.

- `notebooks/KoHRM_SFT_LoRA_Data_Runbook.ipynb`

  SFT/LoRA prepared dataset repo 확인과 subset별 실행 명령을 정리한 노트북입니다.

### `evaluation/`

벤치마크 실행 harness입니다.

- `evaluation/main.py`

  benchmark config를 읽고 engine을 선택해 generation/evaluation을 실행합니다.

- `evaluation/engines.py`

  VLLM engine과 simple inference engine 경로를 제공합니다.

- `evaluation/benchmarks.py`

  GSM8K, MATH, DROP, MMLU-Pro, MMLU, ARC, HellaSwag, Winogrande, BoolQ, AIME majority voting 같은 benchmark wrapper가 있습니다.

- `evaluation/config/*.yaml`

  benchmark 실행 설정입니다.

주의:

- 현재 KoHRM pretraining checkpoint는 SFT/chat checkpoint가 아니므로 strict instruction benchmark 결과가 낮게 나올 수 있습니다.
- benchmark는 post-training 이후 품질 확인용으로 쓰는 것이 맞고, PT 중간 checkpoint는 loss/accuracy와 long continuation probe를 같이 봐야 합니다.

### `simple_inference_engine.py`

raw training checkpoint용 간단 추론 엔진입니다. HF safetensors export가 아니라 FSDP2 checkpoint와 training model code를 직접 쓰는 쪽에 가깝습니다.

중요 포인트:

- `InferenceCheckpoint.tokenize_prompt(condition, prompt)`는 upstream HRM 방식과 맞게 `<boq> condition prompt <eoq>`를 만듭니다.
- `_prefill()`과 `_batched_decode()`는 cache를 쓰며 생성합니다.
- raw checkpoint와 public HF export의 결과가 다를 때, 변환 문제인지 모델 문제인지 구분하는 대조군으로 쓸 수 있습니다.

### `train_lora.py`

KoHRM checkpoint 위에 LoRA adapter를 학습하는 진입점입니다.

흐름:

```text
base checkpoint load
  -> HRM model 생성
  -> LoRA module inject
  -> LoRA parameter만 trainable
  -> V1Dataset SFT subset 학습
  -> adapter checkpoint 저장
```

SFT/LoRA 후보는 pretraining과 같은 V1Dataset 구조를 쓰되, 목적은 다릅니다.

```text
pretraining:
  broad corpus + task data
  from scratch
  지식/언어/코드/터미널 분포 학습

LoRA/SFT:
  curated subset
  pretrained checkpoint에서 시작
  형식 준수, 명령 따르기, tool-call, 한국어 응답 스타일 보정
```

### `docs/`

운영 문서입니다. 현재 README는 첫 화면이고, 긴 설명은 docs 아래에 둡니다.

중요 문서:

- `MODEL_TRAINING_ARCHITECTURE_GUIDE_2026-05-28.md`: 모델 구조와 학습 방식 큰 그림
- `POST_TRAINING_HRM_TEXT_GUIDE_2026-05-30.md`: SFT/LoRA/RL 적용 시 주의사항
- `PRETRAINING_SFT_DATA_MIX_2026-05-23.md`: 데이터 구성과 비중
- `TRAINING_OPERATIONS_LOG_2026-05-26.md`: 장기 운영 로그
- `EPOCH_PASS_CHECKPOINT_MAP_2026-05-28.md`: pass/stage/checkpoint 찾는 법
- `HF_UPLOAD_AND_COLAB_NOTES_2026-05-28.md`: HF 업로드와 Colab 실행 메모
- `AVAILABLE_DATA.md`: 로컬 데이터 인벤토리

### `legalize-kr/`, `ordinance-kr/`

한국 법률/자치법규 원문 repo입니다. 학습 코드가 아니라 raw source입니다.

주의:

- `.git`을 가진 외부 source checkout입니다.
- 대형 raw source 성격이 있으므로 일반 코드 변경 대상이 아닙니다.
- 법률 전처리 스크립트가 이 폴더를 읽어 JSONL/prepared dataset을 만듭니다.

### `ToolBench/`, `data_toolbench/`

ToolBench 원본/보조 데이터입니다. tool-call, tool-use, terminal-style post-training 후보를 만드는 원천입니다.

주의:

- evaluation split이나 오염 위험이 있는 부분은 train에서 제외해야 합니다.
- KoHRM prepared dataset으로 바로 쓰는 것이 아니라 builder script를 거쳐야 합니다.

### `outputs/`

Hydra 실행 output 디렉터리입니다. 날짜별 실행 산출물/로그가 들어갑니다. 재현/디버깅에는 유용하지만, 대형 checkpoint나 데이터의 주 저장 위치는 아닙니다.

### `.github/`

GitHub workflow 폴더입니다. 현재 워크플로우 권한 문제 때문에 Docker build workflow는 제거한 상태입니다. 이 repo에서는 push는 사용자가 직접 합니다.

### `docker/`

원본/보조 Docker 환경 파일입니다. 현재 운영 환경에서는 Docker를 쓰지 않는다는 제약이 있으므로, 실학습은 host Python/torchrun으로 진행합니다.

### `assets/`

README/model card용 이미지입니다. `banner.png`, `benchmark_scatter.png` 등이 있습니다.

### `utils/`

작은 공용 helper입니다. 현재 핵심 학습 흐름은 `pretrain.py`, `dataset_new.py`, `models/`, `scripts/`가 담당합니다.

## 핵심 실행 경로

### Pretraining

대표 실행 형태:

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  arch/size@arch=XL \
  data.path=/home/work/.data/hrm_text_prepared/<prepared_dataset> \
  checkpoint_path=/home/work/.data/hrm_text_checkpoints/<stage_name> \
  run_name=<stage_name> \
  global_batch_size=180224 \
  checkpoint_step_interval=10000 \
  checkpoint_keep_last=2
```

resume stage는 여기에 `resume_from`, `resume_step_offset`, 필요 시 `skip_batches`가 붙습니다.

### Prepared data build

대표 흐름:

```bash
python scripts/build_korean_legal_raw_corpus.py ...
python scripts/build_korean_legal_sft_data.py ...
python scripts/build_kowiki_raw_corpus.py ...
python scripts/build_terminal_conversation_sft_data.py ...

python scripts/prepare_sft_data.py \
  --train /path/to/source.jsonl \
  --tokenizer /home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1/tokenizer.json \
  --output /home/work/.data/hrm_text_prepared/<name> \
  --epochs 1 \
  --context-size 4097
```

`context-size 4097`은 special token/boundary 처리를 포함한 전처리 측 값이고, 모델의 실질 컨텍스트는 4096 token 기준으로 운영합니다.

### HF export

```bash
python conversion/convert_to_hf.py \
  --checkpoint /home/work/.data/hrm_text_checkpoints/<stage>/fsdp2_epoch_* \
  --output /home/work/.data/hrm_text_hf_exports/<export_name>
```

실제 장기 운영에서는 watcher가 이 작업을 감시/실행/업로드합니다.

### SFT/LoRA

```bash
export RESUME_FROM=/home/work/.data/hrm_text_checkpoints/<final_stage>/fsdp2_epoch_*
bash scripts/run_kohrm_lora_experiments.sh phase1
```

`phase1`은 behavior mini, Korean domain, terminal/tool core를 순서대로 실행합니다.

## 현재 운영에서 특히 중요한 파일

장기 학습이 돌고 있을 때 우선순위가 높은 파일은 아래입니다.

```text
pretrain.py
dataset_new.py
multipack_sampler.py
models/baselines/hrm_nocarry_bp_warmup.py
models/flash_attention_prefixlm_v2.py
models/lm_head.py
scripts/watch_stage1c_then_finish_chain.py
scripts/watch_stage4c_then_epoch4_chain.py
scripts/watch_chain_step_checkpoints_upload.py
conversion/convert_to_hf.py
notebooks/kohrm_colab_generate.py
```

운영 원칙:

```text
1. 학습 중인 pretrain/model/dataset 코드는 임의 수정하지 않는다.
2. watcher는 GPU를 거의 쓰지 않지만 handoff와 upload에 중요하므로 죽이지 않는다.
3. checkpoint 삭제는 HF 업로드 확인 후 최신 2개 정책 안에서만 한다.
4. Colab 결과가 이상하면 public HF export와 raw checkpoint inference를 비교한다.
5. PT checkpoint에 SFT/chat 형식 검사를 과하게 적용하지 않는다.
```

## 디버깅 기준

### GPU가 놀 때

확인 순서:

```bash
nvidia-smi
pgrep -af "torchrun|pretrain.py|watch_stage|watch_chain"
tail -100 /home/work/.data/hrm_text_logs/<current_log>
```

볼 것:

- 현재 `torchrun pretrain.py`가 살아 있는지
- watcher가 다음 stage를 기다리고 있는지
- checkpoint final marker가 생겼는데 다음 stage가 시작되지 않았는지
- upload만 돌고 학습이 없는 상태인지

### 출력 품질이 이상할 때

확인 순서:

```text
1. 해당 checkpoint가 PT 중간본인지, SFT/LoRA 이후본인지 확인
2. Colab lightweight runtime과 raw checkpoint inference 결과 비교
3. prompt가 학습 데이터 포맷과 맞는지 확인
4. repetition penalty, min_new_tokens, stop token 설정 확인
5. 변환된 HF config/tokenizer special token 확인
```

PT 중간 checkpoint는 지식/언어 분포를 배우는 단계입니다. JSON-only, command-only, exact tool-call 같은 행동 정렬은 LoRA/SFT/RL 단계에서 보정해야 합니다.

### 데이터가 깨졌는지 볼 때

확인 순서:

```bash
python scripts/sample_prepared_v1_dataset.py --path /home/work/.data/hrm_text_prepared/<name> --count 5
python scripts/audit_data_status.py --root /home/work/.data/hrm_text_prepared
```

볼 것:

- instruction과 response 경계가 맞는지
- response가 비어 있거나 너무 짧지 않은지
- `<|im_start|>`, `<|im_end|>`, `<|box_end|>`가 의도대로 들어갔는지
- 법률/위키/터미널 원문이 과도한 wrapper 없이 들어갔는지

## 코드별 빠른 찾기

```text
학습을 시작한다
  -> pretrain.py
  -> config/cfg_pretrain.yaml

데이터를 읽는다
  -> dataset_new.py
  -> multipack_sampler.py

모델 구조를 본다
  -> models/baselines/hrm_nocarry_bp_warmup.py
  -> models/transformer.py
  -> models/layers.py
  -> models/flash_attention_prefixlm_v2.py
  -> models/lm_head.py

tokenizer를 만든다
  -> scripts/train_koterm_tokenizer.py
  -> scripts/check_tokenizer_efficiency.py

원문 데이터를 KoHRM row로 만든다
  -> scripts/build_korean_legal_raw_corpus.py
  -> scripts/build_korean_legal_sft_data.py
  -> scripts/build_kowiki_raw_corpus.py
  -> scripts/build_terminal_conversation_sft_data.py
  -> scripts/build_toolbench_sft_data.py

row를 prepared dataset으로 바꾼다
  -> scripts/prepare_sft_data.py
  -> scripts/build_hrm_extra_sample_epochs.py
  -> scripts/merge_prepared_sft_data.py

stage를 이어간다
  -> scripts/watch_stage1c_then_finish_chain.py
  -> scripts/watch_stage4c_then_epoch4_chain.py
  -> scripts/watch_stage2b_then_finish_chain.py

checkpoint를 업로드한다
  -> scripts/watch_chain_step_checkpoints_upload.py
  -> scripts/watch_epoch2_final_upload.py
  -> scripts/upload_folder_to_hf.py

HF 모델로 변환한다
  -> conversion/convert_to_hf.py

Colab에서 생성해 본다
  -> notebooks/kohrm_colab_generate.py
  -> notebooks/KoHRM_Text_1_4B_Colab_T4_Long_Knowledge_Probe.ipynb

LoRA/SFT 후보를 돌린다
  -> train_lora.py
  -> config/cfg_lora.yaml
  -> scripts/run_kohrm_lora_experiments.sh

벤치마크를 돌린다
  -> evaluation/main.py
  -> evaluation/benchmarks.py
  -> evaluation/engines.py
```

