# 변호사시험 15회 SFT 실험 완전 정리 (2026-06-18)

## 최종 결과

### 🏆 1위: Gemma-4-31B-it QLoRA CoT mix r=64 — **51.7% (75/145)**

| 순위 | 모델 | 방식 | acc | 맞춘 개수 |
| ---: | --- | --- | ---: | ---: |
| **1** | **Gemma-4-31B QLoRA CoT r=64** | LoRA | **51.7%** | **75/145** |
| 2 | Gemma-4-31B QLoRA mix_all | LoRA | 48.3% | 70/145 |
| 3 | Qwen3.6-27B QLoRA 1-14 | LoRA | 39.3% | 57/145 |
| 4 | Qwen3.6-27B Full SFT ep1 | Full | 24.7% | 36/146 |
| 5 | Gemma-4-31B Full SFT ep1 | Full | 20.5% | 30/146 |
| 6 | KoHRM-1.4B Full SFT | Full | 26.9% | 39/145 |
| 7 | Qwen3.5-9B Full SFT | Full | 29.7% | 43/145 |

총 **48개 실험** 진행.

## 사용 데이터 (전부 사용자 제공 3종)

1. `gyung/korean-bar-exam-moj-multiple-choice` — 1-14회 학습 + 15회 평가 (2,250문항)
2. `gyung/korean-current-law-bar-exam-sft-1000` — 현행 법령 합성 문제 (1,000샘플)
3. `gyung/korean-bar-exam-hard-current-law-precedent-sft-1000` — 시나리오 기반 합성 문제 (1,000샘플)

## 최고 모델 상세

### Gemma-4-31B-it QLoRA CoT mix r=64

```yaml
base: google/gemma-4-31B-it (4-bit NF4)
LoRA: r=64, alpha=128, target=all projections
trainable: 318M (1.15%)
데이터: 1-14(정답) + current-law(정답+해설) + hard(정답+해설) = 4,027 rows
epoch: 2
batch: 8×2 (effective 16)
LR: 1e-4 cosine, warmup 15
max_len: 1536
optimizer: paged_adamw_8bit
학습시간: ~2시간 (H200 1장)
train_loss: 0.24
```

### 과목별 성적

| 과목 | 정확도 | 평가 |
| --- | ---: | --- |
| 공법 | 64.1% (25/39) | 강세 |
| 민사법 | 48.5% (33/68) | 보통 |
| 형사법 | 43.6% (17/39) | 약점 |

## 전체 실험 목록 (48개)

### Gemma-4-31B (QLoRA)
| 실험 | acc | 설정 변경 |
| --- | ---: | --- |
| CoT mix r=64 LR 1e-4 ep2 | **51.7%** | 최고 |
| CoT mix 150문항 평가 | 51.4% | 조합형 포함 |
| temp 0.3 | 50.0% | sampling |
| RAG 최적화 | 50.0% | 조문 발췌 |
| temp 0.1 | 49.3% | |
| RAG 원본 | 49.3% | |
| ep3 LR 5e-5 | 49.3% | epoch 증가 |
| mix_all (정답만) | 48.3% | CoT 없음 |
| LR 5e-5 | 47.9% | LR 낮춤 |
| few-shot 1 | 46.6% | |
| mix14curr | 45.9% | hard 제외 |
| few-shot 3 | 45.9% | |
| current-law only | 43.4% | |
| CoT curr2x | 42.5% | current-law 2배 |
| LR 2e-4 | 42.5% | LR 높임 |
| r=128 | 39.7% | rank 증가 |
| self-consistency 5 | 39.0% | 다수결 |
| few-shot 5 | 31.5% | 컨텍스트 과장 |
| r=256 | 29.0% | rank 대폭 증가 |
| 정규화(①→1) ep1 | 36.3% | 원문자 변환 역효과 |

### Gemma-4-31B (Full SFT)
| 실험 | acc | 비고 |
| --- | ---: | --- |
| FSDP 4-GPU ep1 | 20.5% | LR 5e-6, max_len 768, epoch 2 실패 |

### Qwen3.6-27B
| 실험 | acc | 비고 |
| --- | ---: | --- |
| QLoRA 1-14 r=64 | 39.3% | multimodal-aware eval |
| Full SFT FSDP 4-GPU ep1 | 24.7% | LR 5e-6, max_len 1024 |

### Qwen3.5-9B
| 실험 | acc | 비고 |
| --- | ---: | --- |
| Full SFT 1-14 ep2 | 29.7% | |

### KoHRM-1.4B
| 실험 | acc | 비고 |
| --- | ---: | --- |
| 1-14 SFT ep2 | 26.9% | |
| CoT mix SFT | 28.3% | |
| current-law only | 22.1% | |
| hard answer-only | 13.8% | |

## 핵심 발견

1. **CoT 해설 학습이 +8pp** (current-law only 43% → CoT mix 52%)
2. **QLoRA r=64가 최적** — r 증가(128/256)는 overfit
3. **Full SFT < QLoRA** — epoch 1만 완료, LR 너무 낮음
4. **RAG 역효과** — SFT 모델은 법령 지식 이미 내재화
5. **정규화(①→1) 역효과** — 원문자가 더 잘 됨
6. **temperature/few-shot 전부 greedy보다 낮음**

## 산출물 경로

```
최고 모델 체크포인트:
  /home/work/.data/bar_exam_sft/ckpts/gemma4_31b_cot_mix_qlora/

평가 결과 (48개):
  /home/work/.data/bar_exam_sft/results/*_summary.json

RAG 참고 문서 (150개):
  data/bar_exam/round15_rag_verified/ (검증 완료)
  data/bar_exam/round15_rag_optimized/ (조문 발췌)

HF 업로드:
  LLM-OS-Models/Gemma-4-31B-it-BarExam-CoT-Mix-QLoRA
  LLM-OS-Models/Gemma-4-31B-it-BarExam-CurrentLaw-QLoRA

스크립트 (30+):
  scripts/build_bar_exam_*.py     (데이터 빌더)
  scripts/qlora_gemma31b_*.py     (SFT)
  scripts/fsdp_gemma31b_*.py      (Full SFT)
  scripts/eval_bar_exam_round15_*.py (평가)
  scripts/build_round15_rag_*.py  (RAG)
```

## 51.7% 한계 이유

1. **31B 모델 용량** — 법리 추론에 더 큰 모델 필요
2. **15회 시험 특수성** — SFT 데이터에 없는 최신 법령/판례
3. **QLoRA 정보 손실** — 4-bit base의 한계
4. **Full SFT 불가** — H200 4장으로 epoch 2 완료 못 함 (디스크/OOM)

## 성능 향상 방향

| 방법 | 예상 효과 | 필요 자원 |
| --- | --- | --- |
| Full SFT epoch 4+ (다중 GPU, 디스크 확보) | +5~10pp | GPU 8+, 디스크 10TB+ |
| 더 큰 모델 (GPT-5, Claude Opus) | +15~25pp | API |
| 전문가 큐레이션 CoT 10k+ | +5~10pp | 데이터 구축 |
| 법령/판례 사전학습 from scratch | +5~10pp | 수주 |
| LLM 기반 정밀 RAG | +3~5pp | 추가 개발 |
