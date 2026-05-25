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

## Paper Loss Reporting

참고 논문:

```text
HRM-Text: Efficient Pretraining Beyond Scaling
https://arxiv.org/html/2605.20613
```

논문은 우리 W&B 로그처럼 step별 train loss 숫자표나 최종 train loss 값을 공개하지 않습니다. 따라서 `우리 loss 0.72`와 `논문 loss x.xx`를 직접 비교하는 방식은 불가능합니다.

대신 논문에서 loss와 관련해 확인 가능한 내용은 다음입니다.

| Paper item | 내용 | 우리와의 관계 |
|---|---|---|
| Objective | instruction-response pair에서 response token NLL만 최적화 | 우리 `train/loss`도 response mask가 적용된 LM loss이므로 방향성은 동일 |
| PrefixLM | instruction/prefix는 bidirectional attention, response는 causal generation | 우리도 HRM-Text PrefixLM 구조를 유지 |
| Figure 3(a) | full causal LM 대비 response-only objective가 response-token NLL을 낮추고, PrefixLM이 response loss를 추가 개선한다고 설명 | 우리도 stage3에서 response-token loss가 빠르게 감소 중 |
| Optimization | Adam-atan2, 2,000 step warmup 후 constant LR, no grad clipping, EMA 사용 | 우리도 같은 계열의 optimizer/LR/EMA 정책 |
| Stability appendix | full BPTT보다 truncated/warmup deep credit assignment가 gradient spike를 줄이는 안정화 방향을 제시 | 우리 run에서 loss 발산이나 accuracy collapse가 없다는 점과 정합적 |
| Infrastructure | 논문은 single continuous run, intermediate checkpointing/crash recovery/skip loss spike 없음 | 우리는 안정성을 위해 checkpoint/recovery/upload를 추가했으므로 loss curve에 checkpoint I/O stall이 섞일 수 있음 |

논문에서 직접 인용할 핵심 포인트는 다음입니다.

- HRM-Text는 broad raw-text pretraining이 아니라 instruction-response pair에서 response-only NLL을 학습합니다.
- Figure 3은 task-completion objective와 PrefixLM이 response modeling에 유리하다고 설명합니다.
- 논문은 1B 모델을 40B unique tokens, 16 x H100, 약 46시간에 학습했다고 밝힙니다.
- 논문은 train loss 숫자 자체보다 downstream benchmark와 안정성 분석으로 성공 여부를 보여줍니다.

## Paper Comparison

논문과 우리 run의 loss 비교는 "절대값 비교"가 아니라 "학습 동역학 비교"로 해야 합니다.

| 항목 | HRM-Text paper | KoHRM-Text run | 판단 |
|---|---|---|---|
| Train loss 공개 | step별 숫자 미공개 | W&B offline에 step별 기록 있음 | 직접 수치 비교 불가 |
| Loss target | response-only NLL | response-only masked LM loss | 목적 함수 방향 일치 |
| PrefixLM | 사용 | 사용 | 구조 일치 |
| Stability signal | gradient spike 억제, HRM 안정성 분석 | loss 발산 없음, token accuracy 유지/상승 | 안정성 방향 일치 |
| Training continuity | single continuous run, no intermediate checkpointing | staged continuation, checkpoint/upload watcher 사용 | 우리는 안정성/복구 우선 |
| Token budget | 40B unique, 60B total duration로 해석 | stage chain 기준 더 많은 반복/추가 도메인 포함 | loss 절대값 비교 곤란 |
| Model/tokenizer | 1B, 65,536 BPE | 1.384B, 131,072 BPE | 우리 쪽이 더 큼 |
| Dataset | task-formatted instruction-response mixture | HRM data + Korean/legal/finance/terminal/tool/code mix | stage별 domain shift 큼 |

우리 loss 흐름이 논문 방법론과 맞는 부분:

1. Stage1에서 response-token loss가 빠르게 내려갔습니다.
2. Stage2에서 HRM full/no-cap continuation이 안정적으로 유지됐습니다.
3. Stage3에서 local-terminal domain shift가 있었지만 loss가 빠르게 회복됐습니다.
4. Token accuracy가 stage3에서 0.79 전후까지 올라왔습니다.
5. 현재까지 gradient/optimization instability로 보이는 loss 폭주는 없습니다.

논문과 다르게 봐야 할 부분:

1. 논문은 downstream benchmark 중심이고 train loss curve를 공개하지 않았습니다.
2. 우리는 중간 checkpoint와 upload watcher를 사용하므로 I/O stall이 있습니다.
3. 우리는 한국어/터미널/툴콜 목적이 강해서 stage별 loss 변동이 더 큽니다.
4. 131K tokenizer는 loss scale과 token accuracy 해석에 영향을 줍니다.
5. 현재 loss가 좋아도 실제 terminal/tool-call 성능은 별도 평가가 필요합니다.

결론:

```text
논문과 loss 숫자를 직접 비교할 수는 없지만, 논문이 강조한 response-only NLL 안정 학습과 PrefixLM 효과라는 방향에서는 우리 학습 흐름이 정상입니다.
```

현재까지의 우리 loss는 논문 방법론과 충돌하지 않습니다. 오히려 stage3 domain shift 이후 loss가 빠르게 낮아지고 token accuracy가 상승했다는 점은 continuation이 잘 작동하고 있다는 강한 신호입니다.

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
