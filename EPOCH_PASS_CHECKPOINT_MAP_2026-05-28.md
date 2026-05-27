# Epoch, Pass, And Checkpoint Map - 2026-05-28

기준 시각: 2026-05-28 04:53 KST

이 문서는 KoHRM-Text 장기 학습에서 “epoch”, “pass”, “stage”, “checkpoint”가 각각 무엇을 뜻하는지 정리합니다. 현재 가장 헷갈리는 부분은 `extra_epochs_1_3_v1` 이름과 실제 실행의 `epochs=1`이 다르다는 점입니다.

## 결론

현재 실행은 `pretrain.py epochs=3`로 한 번에 3 epoch를 돌리는 구조가 아닙니다.

현재 구조는 다음입니다.

```text
각 stage는 epochs=1
-> stage final checkpoint는 항상 fsdp2_epoch_1
-> 다음 stage가 그 checkpoint를 resume_from으로 이어받음
```

따라서 “1 epoch 완료”, “2 epoch 완료” 같은 표식은 전역 이름으로 따로 있는 것이 아니라, stage directory 안의 `epoch_1_info.json`과 `fsdp2_epoch_1`로 확인합니다. 구분은 `epoch_1`이라는 tag가 아니라 parent stage 이름으로 합니다.

중요한 현재 상태:

| 항목 | 상태 |
|---|---|
| `stage2b` 현재 실행 | 진행 중 |
| 현재 step | `370,995` |
| `stage2b` 진행률 | 약 `65.86%` |
| 마지막 완료/업로드 checkpoint | `step_370000` |
| `stage2b` final epoch checkpoint | 아직 없음 |
| `stage3b/4b/1c/2c/3c/4c` | 예약만 되어 있고 아직 시작 전 |

## 용어 구분

| 용어 | 의미 | 현재 코드에서의 표식 |
|---|---|---|
| code epoch | `pretrain.py`의 `for epoch in range(1, epochs + 1)` 반복 | `fsdp2_epoch_1`, `epoch_1_info.json` |
| logical data epoch | prepared dataset 안의 `epoch_0`, `epoch_1`, `epoch_2` index shuffle | `metadata.json`와 `epoch_*` directory |
| stage | checkpoint를 이어받아 실행하는 한 덩어리의 학습 run | `KoHRM-Text-1.4B-stage...` directory |
| pass | 사람이 이해하기 위한 데이터 노출 회차 | stage/data 조합으로 해석 |
| global step | 전체 장기 run에서 이어지는 step 번호 | `*_info.json`의 `global_step` |

## 왜 전부 `epoch_1`로 보이나

각 stage를 `epochs=1`로 실행하기 때문입니다.

예를 들어 `stage1b`, `stage2b`, 나중의 `stage3b`가 모두 끝나도 각 stage 내부 final checkpoint 이름은 다음처럼 됩니다.

```text
stage1b/.../fsdp2_epoch_1
stage2b/.../fsdp2_epoch_1
stage3b/.../fsdp2_epoch_1
```

이것은 모두 “각 stage 안에서 첫 번째 epoch를 끝냈다”는 뜻입니다. 전역 1/2/3 epoch라는 뜻은 아닙니다.

정확한 완료 여부는 다음 파일을 봅니다.

```text
/home/work/.data/hrm_text_checkpoints/<stage-dir>/epoch_1_info.json
```

이 파일 안의 핵심 값:

| 필드 | 의미 |
|---|---|
| `tag` | checkpoint tag, 보통 `epoch_1` 또는 `step_N` |
| `global_step` | 전체 장기 run 기준 완료 step |
| `stage_start_step` | 해당 stage 시작 global step |
| `skip_batches_hint` | 해당 stage 안에서 처리한 batch 수 |
| `data_path` | 어떤 prepared dataset을 학습했는지 |
| `global_batch_size` | token slots/step |

## 현재 완료된 stage final checkpoints

로컬 기준으로 final epoch checkpoint가 존재하는 stage는 다음입니다.

| Stage | Data | Final checkpoint | Final global step | 의미 |
|---|---|---|---:|---|
| `stage1-hrm-fastcap` | `koterm_hrm_cleaned_fastcap_stage1_v1` | `fsdp2_epoch_1` | `88,387` | fast-cap HRM pass |
| `stage2-hrm-full-nocap` | `koterm_hrm_cleaned_full_nocap_v1` | `fsdp2_epoch_1` | `165,617` | HRM full/no-cap base pass |
| `stage3-local-terminal` | `local_terminal_conversations_ctx9k_resp6k_v1` | `fsdp2_epoch_1` | `220,433` | terminal/code/tool-heavy pass |
| `stage4-korean-tool-finance` | `koterm_korean_tool_finance_mix_v1` | `fsdp2_epoch_1` | `237,257` | Korean/tool/legal/wiki/finance pass |
| `stage1b-hrm-fastcap-repeat` | `koterm_hrm_cleaned_fastcap_stage1_v1` | `fsdp2_epoch_1` | `317,814` | fast-cap repeat pass |

현재 진행 중인 stage:

| Stage | Data | Current checkpoint | Current state |
|---|---|---|---|
| `stage2b-hrm-full-nocap-extra-epoch1` | `koterm_hrm_cleaned_full_nocap_extra_epochs_1_3_v1` | latest `step_370000` | 진행 중, final `epoch_1`은 아직 없음 |

## `extra_epochs_1_3_v1`의 정확한 의미

`koterm_hrm_cleaned_full_nocap_extra_epochs_1_3_v1` prepared dataset은 이름 그대로 추가 logical epochs를 담고 있습니다.

metadata 확인값:

```json
{
  "discarded_logical_epochs": 1,
  "written_logical_epochs": [1, 2, 3],
  "per_epoch_total_lengths": [
    14554584999,
    14553327840,
    14553002678
  ]
}
```

directory 구조:

| Directory | Logical HRM sampled epoch | 현재 소비 여부 |
|---|---:|---|
| `epoch_0` | logical epoch `1` | `stage2b`에서 현재 소비 중 |
| `epoch_1` | logical epoch `2` | 아직 소비 안 함 |
| `epoch_2` | logical epoch `3` | 아직 소비 안 함 |

중요: 현재 `stage2b` 명령은 `epochs=1`입니다. 따라서 이 stage는 `extra_epochs_1_3_v1` 안의 `epoch_0`, 즉 logical epoch 1만 학습합니다.

## 그러면 epoch 3까지 돌리는가

엄밀히 말하면 현재 실행 중인 `stage2b` 하나만으로는 logical epoch 3까지 돌리지 않습니다.

현재 watcher 예약은 다음과 같습니다.

```text
stage2b(active)
-> stage3b-local-terminal-repeat
-> stage4b-korean-tool-finance-repeat
-> stage1c-hrm-fastcap-repeat2
-> stage2c-hrm-full-nocap-repeat2
-> stage3c-local-terminal-repeat2
-> stage4c-korean-tool-finance-repeat2
```

이 예약은 “전체 데이터 노출을 더 반복한다”는 의미입니다. 하지만 `extra_epochs_1_3_v1` 안의 남은 `epoch_1`, `epoch_2`를 그대로 이어 소비하는 구조는 아닙니다. 특히 현재 `stage2c`는 다음 data를 쓰도록 예약되어 있습니다.

```text
koterm_hrm_cleaned_full_nocap_v1
```

즉 `stage2c`는 extra logical epoch 2/3이 아니라 base full/no-cap dataset을 다시 도는 repeat입니다.

## 1 epoch 완료, 2 epoch 완료 표식은 있는가

있습니다. 다만 전역 `epoch_1_complete`, `epoch_2_complete` 같은 이름이 아니라 stage별로 있습니다.

로컬에서 찾는 법:

```bash
find /home/work/.data/hrm_text_checkpoints \
  -maxdepth 2 \
  -name 'epoch_1_info.json' \
  -print
```

특정 stage 확인:

```bash
cat /home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage2-hrm-full-nocap-gbs180/epoch_1_info.json
cat /home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage1b-hrm-fastcap-repeat-gbs180/epoch_1_info.json
```

현재 `stage2b`는 아직 final epoch checkpoint가 없으므로 다음 파일이 생기면 `stage2b` 완료입니다.

```text
/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1-gbs180/fsdp2_epoch_1
/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1-gbs180/epoch_1_info.json
```

## Hugging Face에서 찾는 법

raw FSDP2 checkpoint는 다음 repo에 올라갑니다.

```text
https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints
```

현재 확인된 `stage2b` 업로드:

| Checkpoint | HF raw path | Main model repo |
|---|---|---|
| `step_320000` | `stage2b-hrm-full-nocap-extra-epoch1-step320000/` | converted `model.safetensors` 업로드 완료 |
| `step_330000` | `stage2b-hrm-full-nocap-extra-epoch1-step330000/` | converted `model.safetensors` 업로드 완료 |
| `step_340000` | `stage2b-hrm-full-nocap-extra-epoch1-step340000/` | converted `model.safetensors` 업로드 완료 |
| `step_350000` | `stage2b-hrm-full-nocap-extra-epoch1-step350000/` | converted `model.safetensors` 업로드 완료 |
| `step_360000` | `stage2b-hrm-full-nocap-extra-epoch1-step360000/` | converted `model.safetensors` 업로드 완료 |
| `step_370000` | `stage2b-hrm-full-nocap-extra-epoch1-step370000/` | converted `model.safetensors` 업로드 완료 |

main model repo는 다음입니다.

```text
https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B
```

주의: main model repo는 checkpoint별 directory를 보존하는 repo가 아니라 최신 변환본을 `model.safetensors`로 덮어쓰는 rolling latest repo입니다. 따라서 checkpoint label을 명확히 찾으려면 raw checkpoint repo의 directory 이름과 commit 시간을 봐야 합니다.

## 추천 정리 방식

앞으로 혼동을 줄이려면 stage 이름을 다음처럼 더 명확히 두는 것이 좋습니다.

| 목적 | 권장 stage name |
|---|---|
| extra logical epoch 1 | `stage2b-hrm-full-nocap-extra-logical-epoch1` |
| extra logical epoch 2 | `stage2c-hrm-full-nocap-extra-logical-epoch2` |
| extra logical epoch 3 | `stage2d-hrm-full-nocap-extra-logical-epoch3` |
| terminal repeat | `stage3b-local-terminal-repeat` |
| Korean/tool/finance repeat | `stage4b-korean-tool-finance-repeat` |

현재 이미 `stage2b`가 돌고 있으므로 중간에 바꾸면 안 됩니다. `stage2b`가 끝난 뒤 남은 extra logical epoch 2/3을 정확히 학습하려면 다음 둘 중 하나가 필요합니다.

1. `extra_epochs_1_3_v1`에서 `epoch_1`, `epoch_2`만 각각 `epoch_0`으로 보이게 하는 split prepared dataset을 만든다.
2. `V1Dataset` 또는 launcher에 시작 epoch offset 기능을 추가한다.

현재 코드에는 시작 epoch offset 기능이 없습니다. 따라서 가장 안전한 방식은 hardlink 기반 split prepared dataset을 만드는 것입니다.

## 권장 후속 조치

목표가 “진짜 logical epoch 3까지 HRM full/no-cap을 돌리는 것”이면, `stage2b` 완료 후 바로 terminal로 넘어가기 전에 다음 두 stage를 추가하는 것이 맞습니다.

```text
stage2b: extra logical epoch 1, 현재 진행 중
stage2c-extra-logical-epoch2: extra_epochs_1_3_v1/epoch_1 사용
stage2d-extra-logical-epoch3: extra_epochs_1_3_v1/epoch_2 사용
then stage3b -> stage4b -> ...
```

이렇게 해야 “epoch 3까지 돌렸다”고 정확히 말할 수 있습니다.

반대로 현재 예약대로 가면 의미는 다음에 가깝습니다.

```text
HRM full/no-cap 계열을 여러 번 더 노출한다.
다만 extra_epochs_1_3_v1의 logical epoch 2/3을 정확히 소비하는 것은 아니다.
```

## 현재 답변

질문: “에폭 3까지 돌려보는 거냐?”

답:

```text
현재 실행 중인 stage2b 하나만 보면 아닙니다.
stage2b는 extra logical epoch 1만 소비합니다.
extra logical epoch 2/3은 prepared dataset 안에 존재하지만, 현재 watcher 예약만으로는 그대로 소비되지 않습니다.
정말 epoch 3까지 정확히 돌리려면 stage2b 완료 후 split prepared dataset 또는 epoch offset 기능으로 stage2c-extra/stage2d-extra를 추가해야 합니다.
```
