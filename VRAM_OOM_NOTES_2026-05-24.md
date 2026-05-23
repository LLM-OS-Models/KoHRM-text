# KoHRM-Text VRAM / OOM Notes

작성일: 2026-05-24

이 문서는 `KoHRM-Text-1.4B` stage-1 학습 중 VRAM이 시간이 지나며 증가하는 이유, 이전 OOM 원인, 현재 운영 기준을 기록합니다.

## 현재 관측 상태

현재 stage-1 run은 다음 설정으로 정상 학습 중입니다.

| 항목 | 값 |
|---|---:|
| GPU | 8 x NVIDIA H200 |
| GPU utilization | 8장 모두 99% |
| global batch | 180,224 tokens |
| local token slots/GPU | 22,528 |
| context | 4,096 |
| VRAM | GPU0 약 129.9GB, 나머지 약 127.6GB / 143.8GB |
| speed | 약 1.02 step/sec |
| checkpoint interval | 5,000 steps |

현재 설정은 빠르지만 여유 VRAM이 아주 넓은 편은 아닙니다. H200 장당 약 144GB 중 127-130GB를 사용하므로, NCCL/allocator/compiler/cache/checkpoint 순간 피크가 겹치면 OOM 위험이 다시 생길 수 있습니다.

## 왜 학습 중 VRAM이 점점 올라가나

VRAM 증가가 곧바로 “메모리 누수”라는 뜻은 아닙니다. 대형 PyTorch/FSDP/compile 학습에서는 다음 요인이 겹치면서 초반보다 뒤에서 VRAM이 더 높아지는 패턴이 흔합니다.

### 1. torch.compile / CUDA graph / kernel cache

HRM-Text 코드는 여러 forward/backward path를 compile합니다. 초반 몇 step에서는 모든 shape/path가 아직 compile되지 않았고, 학습이 진행되며 추가 graph, Triton kernel, CUDA kernel cache가 만들어집니다.

특히 HRM 구조는 H/L recurrent cycle과 PrefixLM loss가 있어 단순 decoder-only Transformer보다 compile path가 더 복잡합니다. 초반 VRAM만 보고 batch를 크게 잡으면 후속 graph가 생성될 때 추가 메모리를 못 받아 OOM이 날 수 있습니다.

### 2. final logits buffer 크기

이번 모델은 vocab이 131,072입니다. upstream HRM-Text 논문 설정의 65,536 vocab보다 두 배입니다.

batch token slots가 커질수록 final logits 또는 loss 계산 쪽 임시 버퍼가 매우 커집니다.

예를 들어 local token slots/GPU가 32,768이면 `32768 x 131072` bf16 logits 계열 버퍼가 필요할 수 있습니다. 이론상 단일 bf16 dense buffer만 잡아도 약 8GB 이상이고, 실제 backward/temporary/parallel buffer까지 합치면 훨씬 커집니다.

이 때문에 처음에는 `global_batch_size=262144` 또는 `229376`이 잠깐 돌아가도, 뒤에서 compile graph와 logits/loss 임시 버퍼가 겹치는 순간 OOM이 날 수 있습니다.

### 3. FSDP2 / optimizer / EMA 상태

현재 학습은 model weights만 들고 있는 것이 아닙니다.

- model parameters
- gradients
- optimizer state
- Adam-atan2 state
- EMA state
- FSDP shard/all-gather/reduce-scatter buffers
- recurrent carry 관련 state

이 상태들이 step마다 항상 같은 순간에 같은 크기로 보이는 것은 아닙니다. 특정 backward path, optimizer step, checkpoint save 시점에 피크가 올라갈 수 있습니다.

### 4. NCCL communication buffers

8 GPU 분산 학습에서는 NCCL 통신 버퍼가 필요합니다. all-gather/reduce-scatter 타이밍, bucket 크기, compile된 그래프 실행 순서에 따라 GPU별 피크가 다르게 보일 수 있습니다.

GPU0이 다른 GPU보다 더 높게 보이는 것도 일반적으로 가능합니다. rank0가 로깅, 일부 metadata, checkpoint coordination, dataloader/host interaction을 더 맡는 경우가 있기 때문입니다.

### 5. CUDA caching allocator

`nvidia-smi`의 used memory는 “현재 텐서가 실제로 쓰는 메모리”만 뜻하지 않습니다. PyTorch CUDA allocator가 한 번 확보한 블록을 재사용하려고 캐시에 잡고 있으면 `nvidia-smi`에는 계속 사용 중처럼 보입니다.

따라서 step이 진행될수록 used memory가 올라가고 잘 내려가지 않는 것은 정상일 수 있습니다. 중요한 것은 reserved가 계속 무한 증가하는지, 또는 특정 step 이후 안정 plateau를 만드는지입니다.

### 6. checkpoint 저장 시 순간 피크

FSDP2 checkpoint 저장 시 `.distcp` shard, metadata, state_dict materialization, host/device transfer가 겹칩니다. 저장 자체는 주로 CPU/disk 작업이지만, 저장 직전/직후 모델 state 접근 때문에 GPU/CPU 메모리 피크가 생길 수 있습니다.

그래서 너무 잦은 checkpoint 저장은 다음 문제를 만듭니다.

- step 처리 지연
- 디스크 사용량 급증
- HF upload 및 scan 비용 증가
- 저장 시점 피크 메모리 증가

현재 5,000 step마다 약 21GB급 FSDP2 checkpoint가 생깁니다. 500 step마다 저장하면 stage-1 기준으로 체크포인트 수와 저장 부하가 10배 늘어 과합니다.

## 이전 OOM 원인

이전 OOM은 batch를 크게 잡았을 때 초반 관측 VRAM만 보고 “괜찮다”고 판단한 것이 원인입니다.

핵심은 다음입니다.

1. vocab 131K라 logits/loss 관련 임시 버퍼가 큽니다.
2. HRM recurrent compile path가 초반 몇 step 뒤 추가 메모리를 요구합니다.
3. H200 8장이라 compute는 충분하지만, 1.4B + 131K vocab + EMA + optimizer + FSDP2 조합에서는 batch를 너무 크게 잡으면 후반 피크가 걸립니다.
4. `global_batch_size=262144`, `229376`은 초반에는 가능해 보였지만 안정 마진이 부족했습니다.

현재는 `global_batch_size=180224`로 내려 안정 진행 중입니다.

## 운영 기준

현재 stage-1에서는 GPU를 놀리지 않는 것이 우선이지만, OOM으로 run이 죽으면 재시작/검증/체크포인트 정리 비용이 더 큽니다.

권장 기준:

| 항목 | 기준 |
|---|---|
| primary batch | `global_batch_size=180224` |
| 저장 주기 | `checkpoint_step_interval=5000` |
| 로컬 보관 | 최신 2-3개 checkpoint만 유지 |
| HF main repo | 최신 safetensors export 중심 |
| HF raw repo | resume가 필요한 FSDP2 checkpoint만 별도 보관 |
| OOM 재발 시 | batch를 5-10% 낮추고 같은 resume checkpoint에서 재시작 |

## 500 step checkpoint가 과한 이유

500 step마다 저장하면 다음 문제가 생깁니다.

- 현재 FSDP2 checkpoint 하나가 약 21GB입니다.
- 500 step 간격이면 10,000 step마다 약 20개, 즉 약 420GB가 생깁니다.
- stage-1 전체 88,522 step 기준으로는 단순 계산상 170개 이상이 생겨 수 TB가 됩니다.
- 저장 자체가 학습 루프를 방해하고, HF 업로드/스캔도 커집니다.

따라서 현재처럼 5,000 step 간격으로 저장하고, 로컬은 최신 2-3개만 남기는 편이 맞습니다.

## 다음 batch 조정 판단

현재 VRAM 사용량은 높지만 학습 속도는 안정적입니다.

다음 stage에서 batch를 올리고 싶으면 한 번에 크게 올리지 말고 다음 순서가 낫습니다.

1. `global_batch_size=180224`로 안정 완료 확인
2. 다음 dataset stage에서 `196608` 테스트
3. 2-3천 step 이상 VRAM plateau 확인
4. checkpoint 저장 시점까지 통과하면 유지
5. OOM 또는 피크 불안정 시 즉시 `180224` 또는 `172032`로 복귀

논문 설정과 비교하면 H200 8장은 강하지만, 이번 모델은 vocab이 131K라 upstream과 메모리 구조가 다릅니다. 따라서 “H200이니까 무조건 H100 16장 batch를 넘긴다”는 식으로 잡으면 안정성이 떨어집니다.

## 결론

현재 VRAM 상승은 torch compile/cache, 131K vocab logits buffer, FSDP2/optimizer/EMA/NCCL buffer, checkpoint 순간 피크가 겹친 결과로 보는 것이 맞습니다.

현재 `global_batch_size=180224`, 5,000 step checkpoint, 최신 2-3개 보관 정책은 빠른 학습과 OOM 회피 사이의 현실적인 균형입니다. 학습이 완전히 안정 plateau를 보이면 다음 stage에서만 소폭 증량을 검토합니다.

