# Epoch, Pass, And Checkpoint Map - 2026-05-28

기준 시각: 2026-05-28 04:59 KST

이 문서는 KoHRM-Text 장기 학습에서 말하는 “1에폭/2에폭/3에폭”을 **데이터 묶음 1/2/3/4를 한 바퀴 도는 pass** 기준으로 정리합니다.

여기서 말하는 에폭은 `pretrain.py epochs=N`의 code epoch가 아니라, 사람이 운영상 부르는 전체 데이터 pass입니다.

## 결론

네. 현재 의도는 다음이 맞습니다.

```text
데이터 1/2/3/4를 학습한다.
1회차는 이미 완료했다.
현재는 2회차의 2번 데이터, 즉 stage2b를 돌리는 중이다.
3회차까지 돌리는 계획도 잡아두었다.
```

현재 상태:

| 항목 | 값 |
|---|---:|
| 현재 stage | `stage2b-hrm-full-nocap-extra-epoch1` |
| 현재 pass 기준 | 2회차의 데이터 2 |
| 현재 step | `371,355` |
| stage2b 진행률 | 약 `66.30%` |
| 마지막 업로드 checkpoint | `step_370000` |
| 다음 checkpoint | `step_380000` |

## 데이터 1/2/3/4 정의

운영상 데이터 묶음은 다음처럼 봅니다.

| 번호 | 데이터 묶음 | 실제 prepared dataset | 대략 토큰 |
|---:|---|---|---:|
| 1 | HRM fast-cap | `koterm_hrm_cleaned_fastcap_stage1_v1` | 14.55B |
| 2 | HRM full/no-cap | `koterm_hrm_cleaned_full_nocap_v1` 또는 extra HRM no-cap view | 14.55B |
| 3 | local terminal/code/tool trajectory | `local_terminal_conversations_ctx9k_resp6k_v1` | 9.39B |
| 4 | Korean/tool/legal/wiki/finance mix | `koterm_korean_tool_finance_mix_v1` | 3.02B |

이 1/2/3/4를 한 번 도는 것을 운영상 “1에폭” 또는 “1회차 pass”라고 부르는 것이 현재 대화의 의미입니다.

## 1회차 완료 여부

1회차는 완료됐습니다.

| Pass | 번호 | Stage | Data | Final checkpoint | Final global step | 상태 |
|---:|---:|---|---|---|---:|---|
| 1 | 1 | `stage1-hrm-fastcap` | 데이터 1 | `fsdp2_epoch_1` | `88,387` | 완료 |
| 1 | 2 | `stage2-hrm-full-nocap` | 데이터 2 | `fsdp2_epoch_1` | `165,617` | 완료 |
| 1 | 3 | `stage3-local-terminal` | 데이터 3 | `fsdp2_epoch_1` | `220,433` | 완료 |
| 1 | 4 | `stage4-korean-tool-finance` | 데이터 4 | `fsdp2_epoch_1` | `237,257` | 완료 |

즉 “1에폭 완료 checkpoint”는 단일 파일 하나가 아니라 위 네 stage의 final checkpoint 묶음입니다.

## 현재 2회차 진행

2회차는 이미 시작됐고, 현재 2번 데이터를 돌리는 중입니다.

| Pass | 번호 | Stage | Data | Checkpoint | 상태 |
|---:|---:|---|---|---|---|
| 2 | 1 | `stage1b-hrm-fastcap-repeat` | 데이터 1 | `fsdp2_epoch_1`, global step `317,814` | 완료 |
| 2 | 2 | `stage2b-hrm-full-nocap-extra-epoch1` | 데이터 2 | latest `step_370000` | 진행 중 |
| 2 | 3 | `stage3b-local-terminal-repeat` | 데이터 3 | 아직 없음 | 예약 |
| 2 | 4 | `stage4b-korean-tool-finance-repeat` | 데이터 4 | 아직 없음 | 예약 |

따라서 사용자가 말한 “지금 1에폭 완료했고 2에폭의 2 돌리고 있지?”는 맞습니다.

정확한 표현은 다음입니다.

```text
데이터 1/2/3/4 기준 1회차 pass는 완료.
현재는 2회차 pass의 데이터 2(stage2b)를 학습 중.
```

## 3회차 계획

3회차 계획도 잡혀 있습니다. 현재 watcher는 `stage2b`가 끝나면 남은 2회차의 데이터 3/4를 실행하고, 그 다음 3회차의 데이터 1/2/3/4를 이어서 실행하도록 되어 있습니다.

현재 watcher 예약:

```text
stage2b(active, pass 2 data 2)
-> stage3b-local-terminal-repeat       (pass 2 data 3)
-> stage4b-korean-tool-finance-repeat  (pass 2 data 4)
-> stage1c-hrm-fastcap-repeat2         (pass 3 data 1)
-> stage2c-hrm-full-nocap-repeat2      (pass 3 data 2)
-> stage3c-local-terminal-repeat2      (pass 3 data 3)
-> stage4c-korean-tool-finance-repeat2 (pass 3 data 4)
```

표로 쓰면 다음입니다.

| 예정 순서 | Pass | 번호 | Stage | Data | 상태 |
|---:|---:|---:|---|---|---|
| 현재 | 2 | 2 | `stage2b-hrm-full-nocap-extra-epoch1` | 데이터 2 | 진행 중 |
| 1 | 2 | 3 | `stage3b-local-terminal-repeat` | 데이터 3 | 예약 |
| 2 | 2 | 4 | `stage4b-korean-tool-finance-repeat` | 데이터 4 | 예약 |
| 3 | 3 | 1 | `stage1c-hrm-fastcap-repeat2` | 데이터 1 | 예약 |
| 4 | 3 | 2 | `stage2c-hrm-full-nocap-repeat2` | 데이터 2 | 예약 |
| 5 | 3 | 3 | `stage3c-local-terminal-repeat2` | 데이터 3 | 예약 |
| 6 | 3 | 4 | `stage4c-korean-tool-finance-repeat2` | 데이터 4 | 예약 |

따라서 전체 데이터 pass 기준으로는 **3회차까지 돌리는 계획이 맞습니다.**

## 왜 stage final은 전부 `epoch_1`인가

각 stage는 코드상 `epochs=1`로 실행됩니다. 그래서 stage가 끝날 때마다 final checkpoint 이름은 전부 `fsdp2_epoch_1`입니다.

예:

```text
stage2/fsdp2_epoch_1   = pass 1 data 2 완료
stage2b/fsdp2_epoch_1  = pass 2 data 2 완료
stage2c/fsdp2_epoch_1  = pass 3 data 2 완료 예정
```

즉 `epoch_1`이라는 파일명만 보면 안 되고, 반드시 parent stage 이름을 같이 봐야 합니다.

## 체크포인트 찾는 법

로컬 final checkpoint 찾기:

```bash
find /home/work/.data/hrm_text_checkpoints \
  -maxdepth 2 \
  -name 'epoch_1_info.json' \
  -print
```

현재 진행 중인 stage2b 중간 checkpoint:

```text
/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1-gbs180/fsdp2_step_370000
/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1-gbs180/step_370000_info.json
```

stage2b 완료 표식은 아직 없고, 완료되면 다음이 생깁니다.

```text
/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1-gbs180/fsdp2_epoch_1
/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage2b-hrm-full-nocap-extra-epoch1-gbs180/epoch_1_info.json
```

HF raw checkpoint repo:

```text
https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B-raw-checkpoints
```

HF main rolling model repo:

```text
https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B
```

주의: main model repo는 최신 변환본을 `model.safetensors`로 덮어씁니다. checkpoint별 label을 정확히 찾으려면 raw checkpoint repo의 directory 이름을 봐야 합니다.

## 현재 업로드된 stage2b checkpoint

| Checkpoint | Raw HF path | Main model repo |
|---|---|---|
| `step_320000` | `stage2b-hrm-full-nocap-extra-epoch1-step320000/` | 업로드 완료 |
| `step_330000` | `stage2b-hrm-full-nocap-extra-epoch1-step330000/` | 업로드 완료 |
| `step_340000` | `stage2b-hrm-full-nocap-extra-epoch1-step340000/` | 업로드 완료 |
| `step_350000` | `stage2b-hrm-full-nocap-extra-epoch1-step350000/` | 업로드 완료 |
| `step_360000` | `stage2b-hrm-full-nocap-extra-epoch1-step360000/` | 업로드 완료 |
| `step_370000` | `stage2b-hrm-full-nocap-extra-epoch1-step370000/` | 업로드 완료 |

## 현재 답변

질문: “데이터 1234까지 우리 학습하지? 지금 1에폭 완료했고 2에폭의 2 돌리고 있지? 그리고 3에폭까지 돌리는 계획을 짠 거지?”

답:

```text
네, 그 의미라면 맞습니다.
데이터 1/2/3/4 기준으로 1회차는 완료됐습니다.
현재는 2회차의 2번 데이터(stage2b)를 돌리고 있습니다.
그리고 watcher에 2회차 3/4 이후 3회차 1/2/3/4까지 이어서 돌리는 계획을 걸어놨습니다.
```
