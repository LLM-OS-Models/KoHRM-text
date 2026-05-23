# KoHRM-Text

2026-05-23 기준 로컬 작업 저장소입니다. 목표는 `sapientinc/HRM-Text` 구조와 PrefixLM 학습 코드를 유지하면서, 한국어/영어/코딩/터미널/툴콜을 잘 처리하는 새 `KoHRM-Text` 계열 모델을 처음부터 학습하는 것입니다.

원본 HRM-Text README는 `UPSTREAM_README.md`에 보존했습니다.

## 모델 이름

| 항목 | 값 |
|---|---|
| 표준 모델명 | `KoHRM-Text-1.4B` |
| HF model repo | `LLM-OS-Models/KoHRM-Text-1.4B` |
| GitHub repo | `https://github.com/LLM-OS-Models/KoHRM-text.git` |
| base code | `sapientinc/HRM-Text` |
| arch | `XL` |
| 추정 params | 1,384,120,320 |

`KoHRM-Text-1.4B`는 새 131K tokenizer를 쓰는 scratch pretraining 모델입니다. 기존 `sapientinc/HRM-Text-1B` 가중치를 이어 쓰는 모델이 아닙니다.

## 현재 결론

- 기존 `sapientinc/HRM-Text-1B` 평가는 완료했고, 터미널/툴콜 기준으로는 그대로 쓰기 어렵다는 판단입니다.
- 새 tokenizer를 만들었기 때문에 기존 가중치를 이어 학습하기보다 새 pretraining으로 가는 것이 맞습니다.
- HRM 기존 cleaned pretraining 데이터 328G는 사용합니다. 다만 기존 tokenizer의 token id를 그대로 섞지 않고, JSONL/parquet 원문에서 새 tokenizer로 다시 패킹합니다.
- SFT 후보 데이터도 사전학습 mix에 먼저 넣습니다. 이후 같은 계열의 고품질 subset으로 SFT를 한 번 더 합니다.
- `tb2_lite`, Terminal Bench 2, ToolBench eval, chi-bench 같은 평가 성격 데이터는 train에서 제외합니다.

## 핵심 문서

| 문서 | 내용 |
|---|---|
| `PRETRAINING_SFT_DATA_MIX_2026-05-23.md` | 사전학습/SFT 데이터 구성, 비중, 제외 기준 |
| `TRAINING_PLAN_2026-05-23.md` | 전체 학습 전략, tokenizer, 실행 정책 |
| `STAGED_TRAINING_RUNBOOK_2026-05-23.md` | 완료된 전처리 데이터부터 학습하고 새 데이터가 생기면 이어 학습하는 실행 절차 |
| `MODEL_CARD_KoHRM-Text-1.4B.md` | HF model card 초안 |
| `AVAILABLE_DATA.md` | 로컬 데이터 인벤토리와 용량 |
| `PROGRESS_2026-05-23.md` | 실제 진행 로그 |
| `UPSTREAM_README.md` | 원본 HRM-Text README |

## 새 토크나이저

| 항목 | 값 |
|---|---|
| 위치 | `/home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1` |
| HF repo | `LLM-OS-Models/HRM-Text-Ko-Terminal-Tokenizer-131K` |
| 방식 | byte-level BPE |
| vocab | 131,072 |
| normalization | NFC |

검증된 chars/token:

| 샘플 | chars/token |
|---|---:|
| 한국어 일반 | 2.60 |
| 한국어 법률 | 2.36 |
| 한국어 터미널 지시 | 2.18 |
| shell command | 2.68 |
| tool JSON | 3.32 |
| Python code | 3.37 |
| 영어 | 4.40 |

## 학습 데이터

현재 실제 pretraining mix v1까지 생성했습니다.

| 데이터 | 상태 | token |
|---|---|---:|
| HRM cleaned base sample | 새 tokenizer로 재패킹 완료 | 250.0M |
| SWE-ZERO + GLM pilot mix | 전처리 완료 | 251.2M |
| 한국어 법률/조례/행정규칙/판례 task | 전처리 완료 | 83.1M |
| ToolBench train tool-call task | 전처리 완료 | 127.0M |
| `koterm_pretrain_mix_v1` | 병합 완료 | 711.3M |

주요 경로:

```text
/home/work/.data/hrm_text_prepared/hrm_cleaned_base_sample_v1
/home/work/.data/hrm_text_prepared/sft_swe_glm_mix_v1
/home/work/.data/hrm_text_prepared/sft_korean_legal_v1
/home/work/.data/hrm_text_prepared/sft_toolbench_v1
/home/work/.data/hrm_text_prepared/koterm_pretrain_mix_v1
```

`koterm_pretrain_mix_v1` 구성:

| 항목 | 값 |
|---|---:|
| samples | 1,176,723 |
| tokens | 711,277,327 |
| avg sample length | 604.5 |
| max sample length | 4,096 |
| disk size | 약 2.8G |

## 현재 파일럿 결과

`sft_swe_glm_mix_v1`로 B size end-to-end pilot을 완료했습니다.

| 항목 | 결과 |
|---|---:|
| arch | B |
| params | 435,159,040 |
| GPUs | 8 x H200 |
| global batch | 262,144 tokens |
| wall time | 약 7분 38초 |
| peak VRAM | 장당 약 38GB |
| final loss | 3.00653 |
| final token accuracy | 0.46379 |
| checkpoint | `/home/work/.data/hrm_text_checkpoints/koterm_b_swe_glm_pilot_v1` |
| HF repo | `LLM-OS-Models/HRM-Text-Ko-Terminal-B-SWE-GLM-Pilot` |

이 pilot은 학습 코드, FA3, FSDP2, tokenizer, V1Dataset 포맷 검증용입니다. 최종 데이터 mix는 아닙니다.

## 현재 실행 방식

전처리와 학습은 병렬로 진행합니다.

1. 이미 전처리된 `koterm_pretrain_mix_v1` 711.3M tokens로 `KoHRM-Text-1.4B` stage-0 학습을 먼저 수행합니다.
2. HRM cleaned 원본 fast-cap 전처리는 별도 CPU 프로세스로 계속 진행합니다.
3. 새 V1Dataset이 완성되면 기존 checkpoint에서 resume해서 stage-1, stage-2로 이어 학습합니다.
4. checkpoint 업로드는 학습 프로세스 안에서 하지 않고 watcher 프로세스로 분리해 epoch 단위로만 HF에 업로드합니다.

stage-0 실행 기준:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/HRM-Text

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline \
WANDB_DIR=/home/work/.data/wandb \
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NCCL_DEBUG=WARN \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
taskset -c 0-31 torchrun --standalone --nproc_per_node=8 pretrain.py \
  arch/size@arch=XL \
  data.path=/home/work/.data/hrm_text_prepared/koterm_pretrain_mix_v1 \
  +checkpoint_path=/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage0-available-mix-gbs172 \
  +project_name=KoHRM-Text \
  +run_name=KoHRM-Text-1.4B-stage0-available-mix-gbs172 \
  epochs=1 \
  global_batch_size=172032 \
  lr_warmup_steps=2000 \
  +log_interval=5 \
  checkpoint_interval=1
```

`global_batch_size=196608`은 논문 기본값이지만, 우리 tokenizer는 vocab 131,072라 final logits 메모리가 더 큽니다. 장기 학습에서는 OOM 여유를 위해 `172032`을 기본값으로 둡니다.

## 로컬 데이터 주의

이 git repo에는 원문 데이터와 체크포인트를 커밋하지 않습니다. 큰 데이터는 `.data` 또는 로컬 디렉터리에 두고, 이 저장소에는 재현 가능한 코드와 문서만 남깁니다.

대표적으로 제외하는 항목:

- `data.zip`
- `data_toolbench/`
- `legalize-kr/`
- `ordinance-kr/`
- `outputs/`
- `__pycache__/`
- `tea_debug.log`

## 다음 작업

1. stage-0 학습 완료 시 raw FSDP2 checkpoint를 HF `LLM-OS-Models/KoHRM-Text-1.4B`에 epoch 단위 업로드합니다.
2. HRM cleaned fast-cap 전처리가 끝나면 `sample_tokenized.py`로 V1Dataset을 만들고 stage-1로 resume합니다.
3. local terminal dataset `swe/math/code.parquet`를 V1Dataset으로 변환해 stage-2에 추가합니다.
4. full training용 45B~52B token mix를 확정한 뒤 장기 pretraining을 이어갑니다.
5. 최종 checkpoint를 선택하면 `conversion/convert_to_hf.py`로 model-only artifact를 따로 변환합니다.
