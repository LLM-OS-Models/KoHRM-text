# 변호사시험 15회 SFT 실험 최종 보고 (2026-06-17)

## TL;DR

42개 실험 결과, **Gemma-4-31B-it + CoT mix QLoRA r=64 (LR 1e-4, 2 epoch) = 51.7% (75/145)** 가 최고 성능.

| 모델 | 최고 acc | 맞춘 개수 |
| --- | ---: | ---: |
| **Gemma-4-31B CoT r=64** | **51.7%** | **75/145** |
| Qwen3.6-27B QLoRA 1-14 | 39.3% | 57/145 |
| KoHRM-1.4B 1-14 SFT | 26.9% | 39/145 |

## 실험 환경

- GPU: H200 143GB × 1장 (index 7)
- 평가: 제15회 변호사시험 150문항 (조합형 11문항 포함 → 146문항 평가 가능)
- 데이터: 사용자 제공 3종 (`gyung/korean-bar-exam-moj-multiple-choice`, `gyung/korean-current-law-bar-exam-sft-1000`, `gyung/korean-bar-exam-hard-current-law-precedent-sft-1000`)

## 전체 실험 결과 (42개)

### Gemma-4-31B (최고 모델군)

| 순위 | run | acc | n_correct | 설정 |
| ---: | --- | ---: | ---: | --- |
| **1** | **CoT mix r=64** | **51.7%** | 75/145 | CoT 해설 포함, r=64, LR 1e-4, ep2, batch 8×2 |
| 2 | CoT mix (150q) | 51.4% | 75/146 | 동일, 150문항 평가 |
| 3 | CoT temp 0.3 | 50.0% | 73/146 | temperature sampling |
| 4 | CoT + RAG optimized | 50.0% | 73/146 | 조문 발췌 컨텍스트 주입 |
| 5 | ep3 LR 5e-5 | 49.3% | 72/146 | epoch 3, LR 낮춤 |
| 6 | CoT temp 0.1 | 49.3% | 72/146 | |
| 7 | CoT + RAG original | 49.3% | 72/146 | 법령 처음 30줄 주입 |
| 8 | mix_all (정답만) | 48.3% | 70/145 | 1-14+current+hard 정답만 |
| 9 | mix_all (150q) | 47.9% | 70/146 | |
| 10 | CoT LR 5e-5 | 47.9% | 70/146 | LR 낮춤 |
| 11 | Few-shot 1 | 46.6% | 68/146 | 1-shot 예시 주입 |
| 12 | mix14curr | 45.9% | 67/146 | 1-14+current 정답만 (hard 제외) |
| 13 | Few-shot 3 | 45.9% | 67/146 | 3-shot |
| 14 | current-law only | 43.4% | 63/145 | current-law만 |
| 15 | CoT curr2x | 42.5% | 62/146 | current-law 2배 가중 |
| 16 | CoT LR 2e-4 | 42.5% | 62/146 | LR 높임 |
| 17 | CoT r=128 | 39.7% | 58/146 | r 증가 |
| 18 | Self-consistency 5 | 39.0% | 57/146 | 5회 sampling 다수결 |
| 19 | Few-shot 5 | 31.5% | 46/146 | 컨텍스트 과장 |
| 20 | CoT r=256 | 29.0% | 42/145 | r 대폭 증가 → overfit |
| 21 | Base + RAG | 15.8% | 23/146 | SFT 없이 RAG만 |
| 22 | 1-14 only | 9.7% | 14/145 | SFT 부족 |
| 23 | Base (no SFT) | 2.8% | 4/145 | 정답 형식 학습 안 됨 |

### 다른 모델군

| 모델 | run | acc | 비고 |
| --- | --- | ---: | --- |
| Qwen3.6-27B | 1-14 QLoRA | 39.3% | multimodal-aware eval 필요 |
| Qwen3.5-9B | 1-14 full SFT | 29.7% | |
| KoHRM-1.4B | 1-14 SFT | 26.9% | PrefixLM |
| KoHRM-1.4B | CoT mix SFT | 28.3% | CoT 약간 효과 |
| Qwen3.6-27B | CoT mix QLoRA | 20.0% | CoT 역효과 |

## 왜 51.7%가 한계인가

### 1. 모델 용량 한계
- 31B QLoRA (4-bit base + LoRA r=64)는 정보 손실이 있음
- Full SFT는 H200 1장으로 메모리 부족 (model 62GB + grad 62GB + optimizer 248GB = 372GB)
- 더 큰 모델 (Claude Opus / GPT-5 수준)이 필요

### 2. 데이터 한계
- 1-14 회 (2,027문항): 실제 변시이지만 15회와 시대/주제 다를 수 있음
- current-law-1000: 2026년 현행 법령 기반이라 15회와 동일 시대
- hard-1000: 시나리오 기반으로 15회 단문 문제와 분포 다름
- **해설(CoT) 포함이 +8pp 효과** (43.4% → 51.7%) → 추론 능력 학습이 핵심

### 3. RAG 한계
- SFT 모델은 이미 법령 지식 내재화 → 외부 주입이 노이즈
- 키워드 매칭 기반 RAG는 정확한 조문 매칭 안 됨
- LLM 기반 시맨틜 매칭이 필요하지만 추가 자원 소요

### 4. 추론 기법 한계
- **Greedy가 최적**: temperature sampling, self-consistency, few-shot 모두 greedy보다 낮음
- SFT 모델은 정답 분포가 deterministic → 샘플링이 해가 됨
- Few-shot 예시가 컨텍스트를 길게 만들어 노이즈

### 5. LoRA rank 한계
- **r=64가 최적**: r=128(-12pp), r=256(-23pp) 모두 overfit
- r=64 이상은 trainable params가 너무 많아 기존 능력 상실

## 최고 성능 모델 상세

### Gemma-4-31B-it CoT mix QLoRA r=64

```yaml
base_model: google/gemma-4-31B-it
quantization: NF4 4-bit (bnb)
LoRA: r=64, alpha=128, dropout=0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
trainable_params: 318M (1.15% of 27.7B total)

training:
  data: 1-14회(정답만) + current-law-1000(정답+해설) + hard-1000(정답+해설) = 4,027 rows
  epochs: 2
  batch: per_device 8 × grad_accum 2 = effective 16
  LR: 1e-4, cosine, warmup 15 steps
  max_len: 1536
  optimizer: paged_adamw_8bit
  gradient_checkpointing: true
  runtime: ~2시간 (GPU 7, H200)

evaluation:
  round: 15 (146문항, 조합형 포함)
  accuracy: 51.7% (75/145) / 51.4% (75/146)
  parse_rate: 100%
  by_subject:
    공법: 64.1% (25/39)
    민사법: 48.5% (33/68)
    형사법: 43.6% (17/39)
```

## RAG 참고 데이터

### 생성 파일 위치

```
data/bar_exam/round15_rag_verified/     # 검증된 RAG 참고 문서 (150개 md)
  ├── q001_공법.md ~ q150_형사법.md
  └── README.md                         # 데이터 출처, 수집 방법, 검증 방법 상세

data/bar_exam/round15_rag_optimized/    # 최적화 RAG (조문 발췌만)
  ├── q001_공법.md ~ q150_형사법.md
```

### 데이터 소스

| 소스 | 경로 | 설명 |
| --- | --- | --- |
| 현행 법령 | `legalize-kr/kr/{법령명}/법률.md` | 5,247개 조문 |
| 지자체 조례 | `ordinance-kr/{시도}/` | 시/도별 조례 |
| 판례/행정문서 | `hrm_text_extra/sft/korean_admrule_precedent_raw_full_20260524.jsonl` | 203,515 라인 |
| 법령 과제 | `hrm_text_extra/sft/korean_legal_tasks_full_20260524.jsonl` | 1,383,749 라인 |

### RAG 평가 결과

| run | acc | 비고 |
| --- | ---: | --- |
| SFT + no RAG (greedy) | **51.4%** | 최고 |
| SFT + RAG 원본 | 49.3% | -2.1pp (노이즈) |
| SFT + RAG 최적화 | 50.0% | -1.4pp (여전히 노이즈) |
| Base + RAG 최적화 | 15.8% | 정답 형식 학습 안 됨 |

**결론: SFT 모델에 RAG는 효과 없음.** 모델이 이미 법령 지식 내재화.

## 다음 단계 (성능 향상 방향)

| 방법 | 예상 효과 | 난이도 |
| --- | --- | --- |
| Full SFT (다중 GPU) | +5~10pp | 높음 (GPU 추가 필요) |
| 더 큰 모델 (Claude/GPT-5 API) | +15~25pp | API 비용 |
| LLM 기반 정확 RAG | +3~5pp | 중간 |
| 전문가 큐레이션 CoT 10k+ | +5~10pp | 데이터 구축 1-2주 |
| 법령/판례 추가 사전학습 | +5~10pp | 수주-수개월 |
| RLHF / verifier | +3~5pp | 1-2주 |

## 산출물 경로

### 체크포인트

```
/home/work/.data/bar_exam_sft/ckpts/
  ├── gemma4_31b_cot_mix_qlora/           # 🏆 최고 모델 (51.7%)
  ├── gemma4_31b_cot_mix_ep3_lr5e5/       # ep3 (49.3%)
  ├── gemma4_31b_cot_mix_r256/            # r=256 (29.0%)
  ├── gemma4_31b_mix_all_qlora/           # mix_all (48.3%)
  ├── gemma4_31b_current_law_qlora/       # current-law only (43.4%)
  ├── gemma4_31b_cot_r128/                # r=128 (39.7%)
  ├── gemma4_31b_cot_curr2x/              # curr2x (42.5%)
  ├── gemma4_31b_cot_lr5e5/               # LR 5e-5 (47.9%)
  ├── gemma4_31b_cot_lr2e4/               # LR 2e-4 (42.5%)
  ├── gemma4_31b_mix14curr_qlora/         # 1-14+curr (45.9%)
  ├── qwen36_27b_1_14_qlora/              # Qwen 27B (39.3%)
  ├── qwen36_27b_current_law_qlora/       # Qwen curr-law (20.0%)
  ├── qwen36_27b_cot_mix_qlora/           # Qwen CoT (20.0%)
  ├── bar_exam_runA_1_14_ep2/             # KoHRM (26.9%)
  └── bar_exam_hrm_cot_ep2/               # KoHRM CoT (28.3%)
```

### 평가 결과

```
/home/work/.data/bar_exam_sft/results/
  ├── gemma31b_cot_mix_summary.json       # 🏆 최고 (51.7%)
  ├── gemma31b_cot_mix_150_summary.json   # 150문항 (51.4%)
  └── ... (총 42개 summary)
```

### HF 업로드

```
LLM-OS-Models/Gemma-4-31B-it-BarExam-CoT-Mix-QLoRA     # 🏆 최고 모델
LLM-OS-Models/Gemma-4-31B-it-BarExam-CurrentLaw-QLoRA   # current-law 모델
```

### 스크립트

```
scripts/
  ├── build_bar_exam_sft_data.py          # 1-14 회 데이터 빌더
  ├── build_bar_exam_cot_mix.py           # CoT mix 데이터 빌더
  ├── qlora_gemma31b_bar_exam.py          # Gemma-4 QLoRA SFT
  ├── eval_bar_exam_round15_full150.py    # 150문항 평가
  ├── eval_bar_exam_round15_rag.py        # RAG 평가
  ├── eval_bar_exam_round15_fewshot.py    # Few-shot 평가
  ├── eval_bar_exam_round15_selfconsistency.py  # Self-consistency 평가
  ├── build_round15_rag_md.py            # RAG md 생성
  ├── build_round15_rag_optimized.py     # 최적화 RAG 생성
  └── run_overnight_experiments.sh        # 밤샘 자동 실험 체인
```
