# KoHRM-Text Loss Analysis - 2026-05-26

이 문서는 W&B offline run 파일에서 직접 추출한 `train/loss`, `train/accuracy`, `train/exact_accuracy`를 기준으로 현재 학습이 정상적으로 진행 중인지 판단한 기록입니다. 기준 시간대는 KST입니다.

## Data Source

분석에 사용한 run 파일:

| Stage | W&B offline run |
|---|---|
| `stage1-fastcap` | `/home/work/.data/wandb/wandb/offline-run-20260523_223905-4zackn4q/run-4zackn4q.wandb` |
| `stage2-full-nocap` | `/home/work/.data/wandb/wandb/offline-run-20260524_203813-5oi4k2si/run-5oi4k2si.wandb` |
| `stage3-local-terminal` | `/home/work/.data/wandb/wandb/offline-run-20260525_184535-k4883gbd/run-k4883gbd.wandb` |

각 run은 `pretrain.py`에서 rank 0이 `wandb.log()`로 남긴 train metrics입니다. 로그 간격은 `log_interval=5`라서 5 step마다 하나의 metric record가 남습니다.

주의:

- 아래 값은 train loss/accuracy입니다. validation loss가 아니므로 일반화 성능을 직접 보장하지는 않습니다.
- stage마다 데이터 분포가 다르기 때문에 loss를 stage 간 절대값으로만 비교하면 안 됩니다.
- `exact_accuracy`는 전체 response span이 모두 맞아야 올라가는 매우 엄격한 metric입니다. 긴 instruction/response 데이터에서는 낮게 나와도 비정상이라고 단정할 수 없습니다.

## Current Stage Snapshot

기준 시각: 2026-05-26 02:08 KST

| 항목 | 값 |
|---|---:|
| 현재 stage | `stage3-local-terminal` |
| current step | 189,585 |
| metric records | 4,794 |
| stage progress | 약 46% |
| current loss | 0.7509 |
| current token accuracy | 0.7840 |
| current exact accuracy | 0.0000 |
| current LR | 2.2e-4 |

## Stage-1: HRM Fast-Cap

| Metric | Early | Recent | 변화 |
|---|---:|---:|---:|
| first 20 avg loss | 2.0080 | - | - |
| last 20 avg loss | - | 1.0295 | -0.9785 |
| first 100 avg loss | 1.8766 | - | - |
| last 100 avg loss | - | 1.0259 | -0.8507 |
| first 20 avg token accuracy | 0.5892 | - | - |
| last 20 avg token accuracy | - | 0.7450 | +0.1558 |
| first 100 avg token accuracy | 0.6081 | - | - |
| last 100 avg token accuracy | - | 0.7462 | +0.1381 |

대표 값:

| 항목 | 값 |
|---|---:|
| first step | 7,770 |
| last step | 88,385 |
| first loss | 2.1317 |
| last loss | 1.0962 |
| best observed loss | 0.7943 at step 58,050 |
| first token accuracy | 0.5752 |
| last token accuracy | 0.7271 |
| last exact accuracy | 0.3293 |

해석:

- stage1은 loss가 크게 내려갔고 token accuracy가 크게 올랐습니다.
- 새 tokenizer와 scratch model이 HRM fast-cap 데이터에서 정상적으로 언어 패턴을 학습했다는 신호입니다.
- best loss 이후 후반 loss가 약간 높은 구간도 있지만, 데이터 sampling과 batch 난이도 차이로 볼 수 있는 범위입니다.

## Stage-2: HRM Full/No-Cap

| Metric | Early | Recent | 변화 |
|---|---:|---:|---:|
| first 20 avg loss | 1.0506 | - | - |
| last 20 avg loss | - | 0.9960 | -0.0545 |
| first 100 avg loss | 1.0372 | - | - |
| last 100 avg loss | - | 0.9884 | -0.0488 |
| first 20 avg token accuracy | 0.7400 | - | - |
| last 20 avg token accuracy | - | 0.7508 | +0.0107 |
| first 100 avg token accuracy | 0.7433 | - | - |
| last 100 avg token accuracy | - | 0.7524 | +0.0091 |

대표 값:

| 항목 | 값 |
|---|---:|
| first step | 85,005 |
| last step | 165,615 |
| first loss | 0.9963 |
| last loss | 0.9782 |
| best observed loss | 0.7334 at step 143,905 |
| first token accuracy | 0.7515 |
| last token accuracy | 0.7582 |
| last exact accuracy | 0.3643 |

해석:

- stage2는 stage1에서 이미 학습한 HRM 계열 full/no-cap 데이터로 이어졌기 때문에 시작 loss가 낮습니다.
- 큰 폭의 개선보다는 안정적인 유지와 소폭 개선이 관측됩니다.
- loss가 폭주하거나 accuracy가 무너지는 현상은 없습니다.
- stage2 후반 last 100 loss가 first 100보다 낮고 token accuracy도 조금 높아졌으므로, continuation 자체는 정상입니다.

## Stage-3: Local Terminal

| Metric | Early | Recent | 변화 |
|---|---:|---:|---:|
| first 20 avg loss | 1.3616 | - | - |
| last 20 avg loss | - | 0.7188 | -0.6428 |
| first 100 avg loss | 1.1249 | - | - |
| last 100 avg loss | - | 0.7240 | -0.4010 |
| first 20 avg token accuracy | 0.6839 | - | - |
| last 20 avg token accuracy | - | 0.7909 | +0.1069 |
| first 100 avg token accuracy | 0.7167 | - | - |
| last 100 avg token accuracy | - | 0.7895 | +0.0728 |

대표 값:

| 항목 | 값 |
|---|---:|
| first step | 165,620 |
| current/latest metric step | 189,585 |
| first loss | 2.0405 |
| latest loss | 0.7509 |
| best observed loss | 0.5549 at step 183,585 |
| first token accuracy | 0.6088 |
| latest token accuracy | 0.7840 |
| latest exact accuracy | 0.0000 |

최근 10개 metric:

| Step | Loss | Token Acc | Exact Acc |
|---:|---:|---:|---:|
| 189,540 | 0.6803 | 0.8030 | 0.0000 |
| 189,545 | 0.8057 | 0.7707 | 0.0000 |
| 189,550 | 0.6970 | 0.7951 | 0.0204 |
| 189,555 | 0.6928 | 0.7920 | 0.0000 |
| 189,560 | 0.7260 | 0.7887 | 0.0000 |
| 189,565 | 0.7219 | 0.7895 | 0.0000 |
| 189,570 | 0.6723 | 0.8009 | 0.0000 |
| 189,575 | 0.8214 | 0.7703 | 0.0000 |
| 189,580 | 0.7444 | 0.7854 | 0.0000 |
| 189,585 | 0.7509 | 0.7840 | 0.0000 |

해석:

- stage3는 HRM general corpus에서 local terminal conversation 데이터로 domain이 바뀌었기 때문에 초반 loss가 높게 튀는 것이 자연스럽습니다.
- 그럼에도 first loss 2.04에서 최근 평균 0.72 수준까지 빠르게 내려왔습니다.
- token accuracy도 약 0.61에서 0.78~0.80대로 올라왔습니다.
- best loss 0.55 이후 최근 loss가 0.7대에서 흔들리지만, 이는 batch 난이도와 데이터 분포 차이로 볼 수 있습니다.
- exact accuracy가 대부분 0인 것은 긴 response 전체가 완전히 일치해야 하는 metric 특성 때문입니다. terminal/code/tool 데이터에서는 token accuracy와 loss를 더 중요하게 봐야 합니다.

## Overall Interpretation

현재 학습은 좋은 방향입니다.

근거:

1. Stage1에서 loss가 크게 감소했고 token accuracy가 크게 상승했습니다.
2. Stage2에서 loss/accuracy가 무너지지 않고 안정적으로 이어졌습니다.
3. Stage3에서 domain shift가 있었음에도 loss가 빠르게 내려가고 token accuracy가 0.79 전후까지 올라왔습니다.
4. 현재 LR `2.2e-4`에서 발산 징후가 없습니다.
5. OOM 없이 `global_batch_size=180224`가 안정적으로 유지되고 있습니다.

현재까지 보이는 리스크:

1. Validation metric이 없어서 일반화 성능 판단은 아직 제한적입니다.
2. Stage3 local terminal 데이터는 구조가 반복적인 부분이 있어 train loss가 낮아지는 속도만 보고 과신하면 안 됩니다.
3. Exact accuracy는 stage3에서 의미가 낮습니다. 별도 terminal/tool-call eval이 필요합니다.
4. Checkpoint save와 HF upload가 병행되면서 I/O stall이 생길 수 있습니다.

## Continue Or Change?

결론:

```text
지금은 그대로 진행하는 것이 맞습니다.
```

바꾸지 말아야 할 것:

- 현재 학습 중에는 batch size를 올리지 않는 것이 좋습니다.
- LR을 중간에 바꾸지 않는 것이 좋습니다.
- checkpoint interval을 더 촘촘하게 줄이지 않는 것이 좋습니다.
- stage chain을 멈추고 재구성하지 않는 것이 좋습니다.

유지할 것:

- `global_batch_size=180224`
- `checkpoint_step_interval=10000`
- `checkpoint_keep_last=2`
- stage 전환 watcher
- step checkpoint upload watcher
- HF raw checkpoint와 main safetensors repo 분리

다음에 볼 지표:

- stage3 종료 시 last 100 loss/accuracy
- stage4 시작 직후 loss spike 크기
- stage4 1,000~2,000 step 이후 loss 회복 속도
- stage1b로 HRM data repeat에 돌아갔을 때 loss가 stage2 후반과 비슷하게 시작하는지
- stage2b에서 full/no-cap extra epoch loss가 발산 없이 유지되는지

변경을 고려할 조건:

| 조건 | 해석 | 대응 |
|---|---|---|
| loss가 여러 천 step 동안 계속 상승 | LR/데이터/체크포인트 resume 문제 가능 | 학습 유지보다 원인 확인 우선 |
| token accuracy가 0.65 아래로 급락 후 회복 안 됨 | domain/data formatting 문제 가능 | 해당 stage 데이터 샘플 점검 |
| loss가 0.3 이하로 너무 낮고 acc가 0.9 이상 장기간 유지 | 반복/중복 데이터 과다 가능 | 다음 epoch 반복 비중 조정 |
| OOM 또는 VRAM 지속 상승 | batch/carry/cache 문제 | batch 유지 또는 소폭 하향 |
| checkpoint save 후 GPU idle이 장시간 지속 | I/O/upload contention | upload worker 수 줄이기 |

## Speed Context

현재 실측 속도:

```text
0.9096 step/s
180,224 tokens/step
약 0.590B tokens/hour
```

논문 기준 추정:

```text
60B tokens / 46 hours / 16 x H100
= 약 1.30B tokens/hour
```

따라서 논문 대비 token throughput은 약 45%이고, 시간 기준으로 약 2.2배 느립니다.

이는 다음 이유로 합리적인 범위입니다.

- 우리는 8 x H200이고 논문은 16 x H100입니다.
- H200은 GPU당 VRAM과 bandwidth가 크지만 GPU 수가 절반입니다.
- 모델은 `KoHRM-Text-1.4B`로 더 큽니다.
- tokenizer vocab이 131K라 embedding/lm head 부담이 큽니다.
- terminal/tool/code 데이터는 PrefixLM mask와 response span 구조가 복잡합니다.
- checkpoint와 HF upload를 병행하고 있습니다.

속도는 최적은 아니지만 현재 목표인 "안 끊기는 장기 학습"에는 적절합니다. 지금은 안정성을 우선하고, 전체 chain 종료 후 batch/profile 재검토가 맞습니다.

## Action Items

현재 즉시 변경:

- 없음. 학습을 계속 진행합니다.

운영 중 확인:

- `stage3` 종료 시 final loss/accuracy를 다시 기록합니다.
- `stage4` 시작 후 첫 1,000~2,000 step loss 회복을 확인합니다.
- `190000` 이후 중간 checkpoint 자동 업로드 watcher가 정상 동작하는지 확인합니다.

후속 평가:

- 모델이 충분히 저장된 뒤 terminal/tool-call smoke eval을 수행합니다.
- 학습 loss만으로 "실제 terminal tool-call 성능"을 확정하지 않습니다.
- stage4 이후 모델을 기준으로 짧은 한국어/터미널/툴콜 생성 샘플을 확인합니다.

