# KoHRM-Text Training Operations Log - 2026-05-26

이 문서는 2026-05-25 밤부터 2026-05-26 새벽까지의 실제 학습 운영, 장애, 수정, 업로드 상태를 기록합니다. 기준 시간대는 KST입니다.

## Current Snapshot

기준 시각: 2026-05-26 02:02 KST

| 항목 | 값 |
|---|---:|
| 현재 stage | `stage3-local-terminal` |
| torchrun PID | `1660234` |
| GPU | 8 x NVIDIA H200 |
| GPU util | 8장 모두 99% |
| VRAM | GPU0 약 129.9GB, GPU1-7 약 127.6GB |
| current step | 189,425 |
| stage3 start step | 165,617 |
| stage3 end step | 217,699 |
| stage3 total steps | 52,082 |
| completed stage3 steps | 23,808 |
| remaining stage3 steps | 28,274 |
| stage3 progress | 45.71% |
| measured speed | 0.9096 step/s |
| tokens/step | 180,224 |
| measured throughput | 약 0.590B tokens/hour |
| stage3 ETA | 약 8.63시간 |
| stage3 expected finish | 2026-05-26 10:40 KST 전후 |

## Running Chain

현재 목표 체인은 다음 순서입니다.

```text
stage3-local-terminal
-> stage4-korean-tool-finance
-> stage1b-hrm-fastcap-repeat
-> stage2b-hrm-full-nocap-extra-epoch1
-> stage3b-local-terminal-repeat
-> stage4b-korean-tool-finance-repeat
```

남은 stage별 대략 분량은 다음과 같습니다.

| Stage | Tokens | Steps |
|---|---:|---:|
| stage3 remaining | 약 5.10B | 28,274 |
| stage4-korean-tool-finance | 3.020B | 16,759 |
| stage1b-hrm-fastcap-repeat | 14.554B | 80,756 |
| stage2b-hrm-full-nocap-extra-epoch1 | 14.554B | 80,753 |
| stage3b-local-terminal-repeat | 9.387B | 52,082 |
| stage4b-korean-tool-finance-repeat | 3.020B | 16,759 |
| Total remaining | 약 49.6B | 275,383 |

현재 실측 속도 `0.9096 step/s`를 그대로 적용하면 전체 체인 종료 예상은 2026-05-29 14:00 KST 전후입니다. 실제 종료 시각은 checkpoint 저장 시간, HF upload 병행 부하, stage별 데이터 I/O 차이, 시스템 부하에 따라 변동될 수 있습니다.

## Completed Work

### Stage-2 Finalization

`stage2-hrm-full-nocap`는 최종 epoch checkpoint까지 완료되었습니다.

| 항목 | 값 |
|---|---:|
| final global step | 165,617 |
| data | `koterm_hrm_cleaned_full_nocap_v1` |
| stage tokens | 약 14.554B |
| checkpoint | `fsdp2_epoch_1` |
| raw checkpoint upload | 완료 |
| converted safetensors upload | 완료 |

업로드된 항목:

```text
LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints
  stage2-hrm-full-nocap-step150000-160000/
  stage2-hrm-full-nocap-final-epoch1/

LLM-OS-Models/KoHRM-Text-1.4B
  model.safetensors
  config.json
  tokenizer.json
  README.md
```

### Stage-3 Start

Stage-2 종료 후 GPU idle 구간이 발생했습니다. 원인은 기존 watcher가 stage2 final checkpoint를 감지한 뒤 안정화 대기와 작은 mix 준비를 먼저 수행했기 때문입니다.

수정 사항:

- `scripts/watch_stage2_then_two_pass_chain.py`에서 stage2 checkpoint 감지 후 안정화 대기를 제거했습니다.
- stage4용 small mix 준비는 stage3 학습과 병렬로 돌도록 thread 처리했습니다.
- 기존 watcher가 upload staging directory race로 중단될 가능성이 있어 복구용 watcher를 추가했습니다.

추가된 복구 watcher:

```text
scripts/watch_stage3_then_finish_chain.py
```

역할:

- 이미 시작된 `stage3-local-terminal`을 중복 실행하지 않습니다.
- stage3 final `epoch_1` checkpoint를 기다립니다.
- stage3가 끝나면 raw checkpoint와 converted model upload를 백그라운드로 시작합니다.
- 이후 `stage4 -> stage1b -> stage2b -> stage3b -> stage4b`를 순차 실행합니다.

현재 후속 watcher:

| 항목 | 값 |
|---|---:|
| PID | 1672885 |
| script | `scripts/watch_stage3_then_finish_chain.py` |
| 상태 | 실행 중 |
| 감지 대상 | `KoHRM-Text-1.4B-stage3-local-terminal-gbs180/fsdp2_epoch_1` |

### Step Checkpoint Upload

기존 자동 업로드는 stage 완료/epoch checkpoint 중심이었습니다. 그래서 `stage3 step_180000` 같은 중간 checkpoint가 자동으로 올라가지 않았습니다.

수정 사항:

```text
scripts/watch_chain_step_checkpoints_upload.py
```

역할:

- continuation chain의 checkpoint directory를 polling합니다.
- `fsdp2_step_*`, carry files, info json이 모두 준비된 step checkpoint만 업로드합니다.
- raw FSDP2 checkpoint는 raw checkpoint repo로 업로드합니다.
- 변환한 EMA `safetensors` 모델은 main model repo로 업로드합니다.
- `.step_upload_markers`로 중복 업로드를 막습니다.

현재 step upload watcher:

| 항목 | 값 |
|---|---:|
| PID | 1997999 |
| script | `scripts/watch_chain_step_checkpoints_upload.py` |
| min step | 190,000 |
| poll interval | 120초 |
| 상태 | 실행 중 |

`stage3 step_170000`은 수동 업로드 완료했습니다. `stage3 step_180000`도 수동 업로드 완료했습니다. `190000` 이후부터는 watcher가 자동으로 업로드합니다.

## Hugging Face Upload State

### Main Model Repo

```text
https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B
```

현재 main repo는 최신 공개 변환본을 덮어쓰는 방식입니다. raw FSDP2 checkpoint를 main model repo에 직접 올리지 않습니다. 이렇게 분리해야 Hugging Face unsafe scan 경고를 피할 수 있습니다.

현재까지 업로드 완료:

| Artifact | Status |
|---|---|
| stage2 final converted model | 완료 |
| stage3 step_170000 converted model | 완료 |
| stage3 step_180000 converted model | 완료 |

### Raw Checkpoint Repo

```text
https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints
```

현재까지 업로드 완료:

| Artifact | Size | Status |
|---|---:|---|
| stage2 step_150000/160000 raw | 약 44.3GB | 완료 |
| stage2 final epoch_1 raw | 약 22.2GB | 완료 |
| stage3 step_170000 raw | 약 22.2GB | 완료 |
| stage3 step_180000 raw | 약 22.2GB | 완료 |

### Prepared Dataset Repo

```text
https://huggingface.co/datasets/LLM-OS-Models/KoHRM-Text-1.4B-prepared-data
```

주요 prepared data upload는 완료 상태입니다.

| Dataset | Status |
|---|---|
| `koterm_hrm_cleaned_full_nocap_v1` | 업로드 완료 |
| `koterm_hrm_cleaned_fastcap_stage1_v1` | 업로드 완료 |
| Korean legal/admin datasets | 업로드 완료 |
| BCAI Finance Korean | 업로드 완료 |
| local terminal conversations | 업로드 완료 |
| tokenizer/docs | 업로드 완료 |

`koterm_hrm_cleaned_full_nocap_v1/tokens.npy`는 Hugging Face의 대형 단일 파일 제한 때문에 원본을 유지한 채 업로드 staging copy를 part 파일로 쪼개서 올렸습니다. 원본 파일은 로컬 prepared dataset에 그대로 남아 있습니다.

## Incident Log

### 1. Stage-2 종료 후 GPU idle

현상:

- Stage-2 final checkpoint는 생성됐지만 GPU가 비었습니다.
- 기존 watcher가 checkpoint 안정화와 stage4 small mix 준비를 먼저 수행해 stage3 실행이 지연됐습니다.

조치:

- stage2 watcher에서 안정화 대기를 제거했습니다.
- small mix 준비를 stage3 학습과 병렬 처리하도록 변경했습니다.
- stage3가 이미 시작된 이후에도 이후 chain을 보장하기 위해 recovery watcher를 추가했습니다.

결과:

- `stage3-local-terminal`이 시작됐고 GPU 8장이 다시 99%로 복귀했습니다.

### 2. Upload staging race

현상:

- 기존 raw checkpoint upload staging directory를 upload process가 읽는 중 다시 `rmtree`하려 하면서 `No such file or directory` 계열 오류가 발생했습니다.

조치:

- 수동 업로드는 unique timestamp staging directory를 사용했습니다.
- step upload watcher도 staging directory 이름에 stage/step/timestamp를 포함하도록 구현했습니다.

결과:

- stage2 final raw, stage3 step170000 raw, stage3 step180000 raw upload가 완료됐습니다.

### 3. Converted model upload failure

현상:

- conversion 과정에서 `NUMEXPR_MAX_THREADS` 제한으로 실패한 적이 있었습니다.

조치:

- 변환/업로드 프로세스 환경에 다음 값을 넣었습니다.

```text
NUMEXPR_MAX_THREADS=256
NUMEXPR_NUM_THREADS=64
OMP_NUM_THREADS=32
```

결과:

- stage2 final, stage3 step170000, stage3 step180000 변환 모델 업로드가 완료됐습니다.

### 4. Intermediate checkpoint auto upload gap

현상:

- 기존 자동 업로드는 stage 완료 단위라 중간 `fsdp2_step_*` checkpoint를 바로 업로드하지 않았습니다.

조치:

- `scripts/watch_chain_step_checkpoints_upload.py`를 추가했습니다.
- `--min-step 190000 --poll-seconds 120`으로 실행했습니다.

결과:

- `190000` 이후 checkpoint는 생성되면 자동 업로드됩니다.
- `170000`, `180000`은 수동 업로드 완료했습니다.

## Speed Analysis

논문 HRM-Text의 공개 기준은 대략 `60B tokens / 46 hours / 16 x H100`로 볼 수 있습니다. 따라서 논문 처리량은 다음과 같습니다.

```text
60B tokens / 46h = 약 1.30B tokens/hour
```

우리 현재 실측은 다음과 같습니다.

```text
180,224 tokens/step * 0.9096 step/s * 3600s
= 약 590M tokens/hour
= 약 0.590B tokens/hour
```

비율:

```text
1.30 / 0.590 = 약 2.2x
```

즉 논문이 token/hour 기준 약 2.2배 빠릅니다. 우리 환경에서 같은 60B tokens를 처리하면 단순 환산으로 약 102시간이 걸립니다.

### Why This Is Slower

하드웨어만 보면 8 x H200은 16 x H100과 같지 않습니다.

| 항목 | 16 x H100 SXM | 8 x H200 SXM | 우리/논문 |
|---|---:|---:|---:|
| GPUs | 16 | 8 | 0.50x |
| VRAM/GPU | 80GB | 141GB | 1.76x |
| Total VRAM | 1,280GB | 1,128GB | 0.88x |
| HBM bandwidth/GPU | 3.35TB/s | 4.8TB/s | 1.43x |
| Total HBM bandwidth | 53.6TB/s | 38.4TB/s | 0.72x |

H200은 GPU당 VRAM과 memory bandwidth가 더 좋지만, GPU 수가 절반입니다. 총 memory bandwidth 기준으로는 논문 대비 약 72% 수준이고, 총 VRAM도 약간 적습니다.

추가로 우리 쪽은 다음 차이가 있습니다.

- 모델이 `KoHRM-Text-1.4B`로 논문 1B급보다 큽니다.
- 131K tokenizer라 vocab embedding/lm head 부담이 큽니다.
- 한국어/터미널/툴콜 mix는 JSON 구조와 긴 instruction/response가 많아 packing과 PrefixLM mask 처리 부하가 큽니다.
- 10,000 step checkpoint 저장 시점마다 약간의 GPU idle/I/O stall이 생깁니다.
- HF upload와 CPU conversion을 학습과 병렬로 돌려 CPU/NFS/I/O contention이 생길 수 있습니다.

### Is The Speed Reasonable?

현재 속도는 기대보다 빠르지는 않지만, 비정상적으로 느리다고 보기는 어렵습니다.

판단:

- 논문 대비 `2.2x` 느린 것은 GPU 수가 절반이고 모델이 더 큰 점을 감안하면 설명 가능합니다.
- H200 8대가 H100 16대와 동급 속도를 낸다고 보면 안 됩니다. VRAM은 크지만 총 GPU 수와 총 HBM bandwidth가 적습니다.
- 다만 하드웨어 aggregate만 보면 논문 대비 약 0.5~0.72x 처리량은 기대할 수 있는데, 현재는 약 0.45x입니다.
- 나머지 차이는 모델 크기, tokenizer vocab, checkpoint/I/O, upload 병행, 구현/compile overhead로 설명 가능합니다.

결론:

```text
합리적 범위 안이지만 최적화 여지는 있습니다.
```

우선순위는 GPU를 계속 돌리는 것입니다. 현재처럼 학습을 멈추지 않으면서 다음 항목을 조정하는 것이 좋습니다.

1. 중간 checkpoint upload는 watcher로 분리하고 학습 프로세스에는 넣지 않습니다.
2. checkpoint interval은 너무 촘촘하게 줄이지 않습니다. 현재 10,000 step은 공간/복구 균형상 적절합니다.
3. stage3가 끝난 뒤 stage4 진입 속도를 확인합니다. stage4는 데이터가 작아 상대적으로 빠르게 끝날 가능성이 큽니다.
4. 다음 장기 stage에서 `global_batch_size=180224`를 유지합니다. 지금 VRAM은 높지만 안정적이고, 이전 OOM 이력이 있어 공격적으로 올리는 것은 위험합니다.
5. 전체 chain이 끝난 뒤 speed/profile을 보고 batch 재조정 여부를 판단합니다.

## Files Added Or Changed For Operations

| File | Purpose |
|---|---|
| `scripts/watch_stage2_then_two_pass_chain.py` | stage2 종료 후 `3 -> 4 -> 1 -> 2 -> 3 -> 4` chain orchestration |
| `scripts/watch_stage3_then_finish_chain.py` | stage3가 이미 시작된 경우 이후 chain을 복구/연결 |
| `scripts/watch_stage1b_then_finish_chain.py` | stage1b 이후 실제 checkpoint global_step 기준으로 `stage2b -> stage3b -> stage4b`를 이어가는 handoff watcher |
| `scripts/watch_chain_step_checkpoints_upload.py` | `fsdp2_step_*` 중간 checkpoint 자동 업로드 |
| `scripts/build_hrm_extra_sample_epochs.py` | HRM full/no-cap extra epoch dataset 구성 |
| `conversion/convert_to_hf.py` | `--ckpt_step` 변환 지원 |
| `simple_inference_engine.py` | step checkpoint load와 tokenizer path 처리 보강 |
| `MODEL_CARD_KoHRM-Text-1.4B.md` | 최신 public artifact 설명 갱신 |

## Stage1b Handoff Fix

기준 시각: 2026-05-26 17:14 KST

`stage3 -> stage4 -> stage1b` 자동 시작은 정상 동작했습니다. 다만 기존 `scripts/watch_stage3_then_finish_chain.py`는 stage 종료 후 다음 offset을 실제 checkpoint의 `epoch_1_info.json`에서 읽지 않고, metadata token count로 계산한 예상 step 수를 더하는 방식이었습니다.

이 방식은 대부분의 경우 충분히 가깝지만, 실제 `pretrain.py`가 마지막 batch/checkpoint를 처리하면서 `skip_batches_hint`와 metadata floor 값 사이에 수십 step 차이가 생길 수 있습니다. 예를 들어 `stage4-korean-tool-finance`는 watcher 계산상 `16,759` steps였지만, checkpoint metadata에는 `global_step=237,257`, `skip_batches_hint=16,824`로 기록됐습니다.

조치:

1. 현재 학습 중인 `stage1b-hrm-fastcap-repeat` torchrun은 그대로 유지했습니다.
2. 기존 watcher PID `1672885`만 `SIGSTOP`으로 멈춰서 stage1b 종료 후 중복 stage2b를 시작하지 못하게 했습니다.
3. 새 `scripts/watch_stage1b_then_finish_chain.py`를 실행했습니다.
4. 새 watcher는 stage1b final checkpoint가 생성되면 `epoch_1_info.json`의 실제 `global_step`을 읽고 그 값을 다음 stage의 `resume_step_offset`으로 사용합니다.
5. 기존 `scripts/watch_stage3_then_finish_chain.py`도 stage 종료 후 실제 checkpoint `global_step`을 우선 사용하도록 수정했습니다.

현재 handoff watcher:

```text
python scripts/watch_stage1b_then_finish_chain.py --retire-pid 1672885
```

첫 handoff watcher process는 로그 두 줄만 남기고 내려갔습니다. 학습 프로세스는 계속 살아 있었고 GPU 사용률도 유지됐습니다. watcher만 `setsid`로 완전히 분리해 재기동했습니다.

현재 재기동된 watcher:

```text
PID 2713801
setsid -f bash -c 'cd /home/work/.projects/LLM-OS-Models/Terminal/HRM-Text && exec python scripts/watch_stage1b_then_finish_chain.py --retire-pid 1672885 >> /home/work/.data/hrm_text_logs/watch_stage1b_then_finish_chain_20260526.log 2>&1 < /dev/null'
```

남은 stage 이름은 다음처럼 고정합니다.

| 순서 | Stage name | Data |
|---:|---|---|
| 1 | `stage1b-hrm-fastcap-repeat` | `koterm_hrm_cleaned_fastcap_stage1_v1` |
| 2 | `stage2b-hrm-full-nocap-extra-epoch1` | `koterm_hrm_cleaned_full_nocap_extra_epochs_1_3_v1` |
| 3 | `stage3b-local-terminal-repeat` | `local_terminal_conversations_ctx9k_resp6k_v1` |
| 4 | `stage4b-korean-tool-finance-repeat` | `koterm_korean_tool_finance_mix_v1` |

데이터/시간 기준:

| Stage | Disk | Metadata tokens | Planned steps | ETA at 1.02 step/s |
|---|---:|---:|---:|---:|
| stage1/stage1b fastcap | 148GB | 14.554B | 80,756 | 21.99h full stage |
| stage2 full/no-cap | 633GB | 14.554B | 80,753 | 21.99h |
| stage2b extra epoch source | 637GB | 14.554B | 80,753 | 21.99h |
| stage3 terminal | 36GB | 9.387B | 52,082 | 14.18h |
| stage4 Korean/tool/finance | 12GB | 3.021B | 16,759 | 4.56h |

실측:

| 항목 | 값 |
|---|---:|
| stage4 start | 2026-05-26 11:30:35 KST |
| stage4 finish | 2026-05-26 16:13:17 KST |
| stage4 elapsed | 4h 42m 42s |
| stage1b current step at check | 240,760 |
| stage1b current progress | 3,568 / 80,756 = 4.42% |
| stage1b expected finish | 2026-05-27 14:16 KST |
| remaining chain expected finish | 2026-05-29 07:00 KST |

해석:

```text
모델 weight resume 자체는 stage4 checkpoint에서 정상적으로 이어졌습니다. 이번 수정의 핵심은 다음 stage들의 global step 이름과 resume offset을 실제 checkpoint 기준으로 맞춰, 이후 stage2b/stage3b/stage4b가 중복 실행되거나 잘못된 step label로 이어지지 않게 하는 것입니다.
```
