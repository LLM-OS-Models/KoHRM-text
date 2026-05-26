# KoHRM Chain Handoff Status - 2026-05-26

기준 시각: 2026-05-26 17:17 KST

이 문서는 `stage3 -> stage4 -> stage1b -> stage2b -> stage3b -> stage4b` 이어학습 체인의 현재 상태, stage 이름, 데이터 용량, watcher 동작 방식을 한 곳에 정리합니다.

## Current State

| 항목 | 값 |
|---|---:|
| 현재 stage | `stage1b-hrm-fastcap-repeat` |
| 현재 global step | 240,905 |
| stage1b 시작 step | 237,192 |
| stage1b 진행 steps | 3,713 / 80,756 |
| stage1b 진행률 | 4.60% |
| global batch | 180,224 tokens |
| 실측 속도 | 약 1.02 step/s |
| GPU 사용률 | 8장 모두 99% |
| VRAM | GPU0 약 129.9GB, GPU1-7 약 127.6GB |
| stage1b 예상 종료 | 2026-05-27 14:16 KST 전후 |
| 남은 체인 예상 종료 | 2026-05-29 07:00 KST 전후 |

`stage3-local-terminal`과 `stage4-korean-tool-finance`는 이미 완료됐습니다. 현재는 stage4 final checkpoint에서 이어받은 `stage1b`가 돌고 있습니다.

## Stage Inventory

| Stage | Data path | Disk | Metadata tokens | Planned steps |
|---|---|---:|---:|---:|
| `stage1-hrm-fastcap` / `stage1b-hrm-fastcap-repeat` | `koterm_hrm_cleaned_fastcap_stage1_v1` | 148GB | 14.554B | 80,756 |
| `stage2-hrm-full-nocap` | `koterm_hrm_cleaned_full_nocap_v1` | 633GB | 14.554B | 80,753 |
| `stage2b-hrm-full-nocap-extra-epoch1` | `koterm_hrm_cleaned_full_nocap_extra_epochs_1_3_v1` | 637GB | 14.554B | 80,753 |
| `stage3-local-terminal` / `stage3b-local-terminal-repeat` | `local_terminal_conversations_ctx9k_resp6k_v1` | 36GB | 9.387B | 52,082 |
| `stage4-korean-tool-finance` / `stage4b-korean-tool-finance-repeat` | `koterm_korean_tool_finance_mix_v1` | 12GB | 3.021B | 16,759 |

## Completed Stages

| Stage | Status | Final checkpoint |
|---|---|---|
| `stage3-local-terminal` | 완료 | `KoHRM-Text-1.4B-stage3-local-terminal-gbs180/fsdp2_epoch_1` |
| `stage4-korean-tool-finance` | 완료 | `KoHRM-Text-1.4B-stage4-korean-tool-finance-gbs180/fsdp2_epoch_1` |
| `stage1b-hrm-fastcap-repeat` | 진행 중 | latest step checkpoint: `fsdp2_step_240000` |

`stage4`는 2026-05-26 11:30:35 KST에 시작해 2026-05-26 16:13:17 KST에 끝났습니다. 실제 소요 시간은 4시간 42분 42초입니다.

## Remaining ETA

현재 속도 1.02 step/s 기준 단순 ETA입니다. checkpoint 저장, HF upload, CPU conversion, NFS I/O 상황에 따라 몇 분에서 수십 분 흔들릴 수 있습니다.

| Stage | 예상 시작 | 예상 소요 | 예상 종료 |
|---|---:|---:|---:|
| `stage1b-hrm-fastcap-repeat` | 진행 중 | 남은 약 21.0h | 2026-05-27 14:16 KST |
| `stage2b-hrm-full-nocap-extra-epoch1` | 2026-05-27 14:16 KST | 약 22.0h | 2026-05-28 12:15 KST |
| `stage3b-local-terminal-repeat` | 2026-05-28 12:15 KST | 약 14.2h | 2026-05-29 02:26 KST |
| `stage4b-korean-tool-finance-repeat` | 2026-05-29 02:26 KST | 약 4.6h | 2026-05-29 07:00 KST |

## Handoff Correction

기존 recovery watcher는 stage 종료 뒤 다음 `resume_step_offset`을 metadata token count에서 계산한 step 수로 더했습니다. 이 방식은 대략 맞지만 실제 epoch checkpoint의 `global_step`과 몇십 step 차이가 날 수 있습니다.

확인된 예:

| 항목 | 값 |
|---|---:|
| stage4 metadata 계획 step | 16,759 |
| stage4 checkpoint `skip_batches_hint` | 16,824 |
| stage4 checkpoint `global_step` | 237,257 |

따라서 이어학습은 실제 checkpoint의 `epoch_1_info.json` 값을 우선해야 합니다.

수정 사항:

1. `scripts/watch_stage3_then_finish_chain.py`가 stage 종료 후 실제 checkpoint `global_step`을 우선 사용하도록 수정했습니다.
2. `scripts/watch_stage1b_then_finish_chain.py`를 추가했습니다.
3. 기존 watcher PID `1672885`는 `SIGSTOP` 상태로 멈췄습니다.
4. 새 watcher는 `stage1b` final checkpoint를 기다린 뒤 실제 `global_step`으로 `stage2b -> stage3b -> stage4b`를 이어갑니다.
5. 첫 handoff watcher 프로세스는 로그 두 줄만 남기고 내려갔기 때문에, `setsid`로 완전히 분리해 다시 실행했습니다.
6. 2026-05-27 06:24 KST 기준 watcher를 보강했습니다. final checkpoint 파일이 보인 직후 다음 stage를 바로 띄우지 않고, `stage1b` torchrun/pretrain 프로세스가 완전히 종료된 것을 확인한 뒤 `stage2b`를 시작합니다. 이 대기는 GPU overlap/OOM을 막기 위한 것이며, 정상 종료 시 지연은 매우 짧습니다.

현재 watcher 상태:

| 역할 | 상태 |
|---|---|
| 기존 recovery watcher `1672885` | `SIGSTOP`, 중복 stage 시작 방지 |
| 현재 torchrun `2655770` | stage1b 학습 중 |
| handoff watcher | stage1b final checkpoint와 stage1b process exit 대기 후 stage2b 시작 |
| checkpoint upload watcher `1997999` | 10,000 step 단위 raw/converted checkpoint 자동 업로드 |

## Upload State

완료 확인된 중간 checkpoint 업로드:

| Stage | Step |
|---|---:|
| `stage3-local-terminal` | 190000 |
| `stage3-local-terminal` | 200000 |
| `stage3-local-terminal` | 210000 |
| `stage3-local-terminal` | 220000 |
| `stage4-korean-tool-finance` | 230000 |
| `stage1b-hrm-fastcap-repeat` | 240000 |

정책:

- raw FSDP2 checkpoint는 `LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints`에 올립니다.
- 변환된 EMA `safetensors` 모델은 `LLM-OS-Models/KoHRM-Text-1.4B`에 최신 public artifact로 올립니다.
- 로컬 checkpoint는 `checkpoint_keep_last=2`로 최근 2개만 유지합니다.
- `resume_step_offset`은 stage 종료 후 반드시 checkpoint metadata의 실제 `global_step`을 기준으로 이어갑니다.

## Operational Rule

현재 최우선 원칙은 학습을 끊지 않는 것입니다. batch size를 더 키우거나 watcher를 교체하는 작업은 현재 torchrun을 건드리지 않는 방식으로만 수행합니다.

다음 확인 지점:

1. `250000` step checkpoint 생성 및 자동 업로드 확인
2. `stage1b` final checkpoint 생성 확인
3. handoff watcher가 `stage2b-hrm-full-nocap-extra-epoch1`를 실제 stage1b `global_step`으로 시작했는지 확인
4. `stage2b` 시작 직후 GPU 8장 99% 사용률 확인
