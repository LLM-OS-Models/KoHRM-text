# HRM-Text 사용 가능 학습 데이터

> 2026-05-23 검토 반영: 구체적인 학습 계획은
> `HRM-Text/TRAINING_PLAN_2026-05-23.md`에 정리했다. 이 문서는 보유 데이터
> 인벤토리이고, 실제 학습에서는 raw text를 그대로 섞지 말고 HRM-Text 논문 방식의
> instruction-response / PrefixLM 포맷으로 변환한다.
>
> 사전학습/SFT 데이터 mix의 최신 기준은
> [PRETRAINING_SFT_DATA_MIX_2026-05-23.md](PRETRAINING_SFT_DATA_MIX_2026-05-23.md)에 별도로 정리했다.

## 다운로드 현황

| 데이터 | 상태 | 절대경로 | 크기 |
|---|---|---|---|
| HRM-Text Pre-cleaned | **완료** | `/home/work/.data/huggingface/hub/datasets--sapientinc--HRM-Text-data-io-cleaned-20260515/` | 328G |
| 한국어 위키백과 | **완료** | `/home/work/.data/huggingface/kowiki-20260501-pages-articles-multistream.xml.bz2`, `/home/work/.data/huggingface/kowiki-20260501-pages-articles-multistream.xml` | 1.3G bz2 / 5.9G xml |
| ToolBench (data.zip) | **완료** | `/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text/data_toolbench/data/` | 21G |
| BCAI Finance Kor | **완료** | `/home/work/.data/huggingface/hrm_text_extra/finance/BCAI-Finance-Kor-1862K`, `/home/work/.data/huggingface/hrm_text_extra/finance/bcai_finance_kor_hrm_20260524.jsonl` | 원본 5.2G / HRM JSONL 5.3G |

## SFT/RL 후보 prepared 데이터

세부 정책과 사용 순서는 [SFT_RL_CANDIDATE_PREP_2026-05-28.md](SFT_RL_CANDIDATE_PREP_2026-05-28.md)에 정리했습니다.

| 데이터 | 상태 | 절대경로 | 토큰 |
|---|---|---|---:|
| behavior mini | **완료** | `/home/work/.data/hrm_text_prepared/kohrm_sft_behavior_mini_v1` | 60.0M |
| terminal/tool core | **완료** | `/home/work/.data/hrm_text_prepared/kohrm_sft_terminal_tool_core_v1` | 165.0M |
| Korean domain core | **완료** | `/home/work/.data/hrm_text_prepared/kohrm_sft_korean_domain_core_v1` | 100.0M |
| behavior core | **완료** | `/home/work/.data/hrm_text_prepared/kohrm_sft_behavior_core_v1` | 285.0M |

## 한국어 데이터

### 한국어 위키백과 — 1.3G bz2 / 5.9G 해제
- 압축: `/home/work/.data/huggingface/kowiki-20260501-pages-articles-multistream.xml.bz2`
- 해제: `/home/work/.data/huggingface/kowiki-20260501-pages-articles-multistream.xml`
- 출처: https://dumps.wikimedia.org/kowiki/20260501/
- 토크나이저 학습 코퍼스 + 사전학습 한국어 일반 지식

### 법령 (legalize-kr) — 675M
- `/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text/legalize-kr/`
- 한국 현행 법령 전체 (마크다운)

### 조례 (ordinance-kr) — 3.2G
- `/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text/ordinance-kr/`
- 전국 17개 지자체 조례 (서울, 경기, 부산 등)

### 행정규칙 (admrule-kr) — 523M
- `/home/work/.projects/LLM-OS-Models/Terminal/admrule-kr/`
- 부처별 행정규칙 (고용노동부, 국방부, 법무부 등 20+ 부처)

### 판례 (precedent-kr) — 3.0G
- `/home/work/.projects/LLM-OS-Models/Terminal/precedent-kr/`
- 분류: 가사, 민사, 형사, 특허, 세무, 선거·특별, 일반행정, 기타

### 한국어 총합: ~12.4G (텍스트 기준)

## 터미널/툴 학습 데이터

### ToolBench (ToolLLaMA) — 21G
- `/home/work/.projects/LLM-OS-Models/Terminal/HRM-Text/data_toolbench/data/`
- `toolllama_G123_dfs_train.json` (2.0G) — 학습용
- `toolllama_G123_dfs_eval.json` (7.9M) — 평가용

### NVIDIA Terminal Dataset — 5.0G
- `/home/work/.projects/LLM-OS-Models/Terminal/dataset/`
- dataset_adapters, synthetic_tasks 포함

### Terminal Bench 2 — 124M
- `/home/work/.projects/LLM-OS-Models/Terminal/terminal-bench-2/`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/`
- 평가용 벤치마크

## 학습 적용 방법

### 사전학습 (Pretraining)
- HRM-Text pre-cleaned 데이터 (328G) 기본
- 한국어 12.4G + ToolBench 4.1G + Terminal 5.0G 추가로 도메인 강화
- 단, 한국어 법률/판례 raw text를 그대로 많이 넣는 방식은 비추천
- 법령/판례는 QA, 요약, 근거 추출, 조항 변환, 형식 검증 task로 변환
- 터미널/툴콜은 다음 명령 예측, 명령 출력 해석, JSON argument 생성 task로 변환
- data_io 파이프라인 또는 동일한 V1Dataset 변환기로 토크나이징 후 학습 경로에 로드
- 현재 이 환경의 `/dev/shm`는 2GB뿐이므로 `/dev/shm/sampled` 기본값은 그대로 사용 불가

### SFT (Fine-tuning)
- ToolBench → tool use 능력
- NVIDIA Terminal → 터미널 명령어 능력
- 한국 법률 → instruction-response 쌍 생성 후 법률 QA 능력
- 입력 포맷: `{"instruction": "...", "response": "...", "condition": "direct"}`

---

# 토크나이저 재학습 — 상세 가이드

## 1. 현황 및 문제

| 항목 | 현재 HRM-Text | 목표 |
|---|---|---|
| 어휘 크기 | 65,536 | 131,072~151,936 |
| 한국어 토큰 | **0개** | 15,000~30,000+ |
| 한국어 char/token | 0.45 | 2.0+ |
| 영어 char/token | 2.5 | 2.5+ (유지) |
| 특수 토큰 | 19개 (비전/COT용) | 40~60개 (툴콜/터미널/포맷) |

한국어가 char/token 0.45 → 한 글자당 2개 이상의 토큰 소비. "대한민국"이 10토큰으로 인코딩됨.

## 2. 2026년 토크나이저 트렌드

### 주요 모델 어휘 크기 비교

| 모델 | 어휘 크기 | 방식 | 한국어 지원 |
|---|---|---|---|
| Qwen3 | 151,936 | tiktoken BPE | 우수 (CJK 대폭 포함) |
| gemma-4 | 256,000 | SentencePiece | 우수 (256K로 전 음절 커버 가능) |
| DeepSeek-V4 | 129,280 | BPE | 양호 (CJK 기반) |
| Llama 4 | 128,256 | tiktoken BPE | 보통 (Llama3 기반 확장) |
| HRM-Text (현) | 65,536 | Rust BPE | **없음** |

### 핵심 트렌드
1. **대어휘 (128K~256K)** 가 표준. 65K는 2024년 기준으로도 작음
2. **tiktoken BPE** (Qwen, GPT-4, Llama) 또는 **SentencePiece BPE** (gemma, DeepSeek) 가 양대 산맥
3. **다국어 우선 설계** — 훈련 코퍼스부터 다국어 포함
4. 특수 토큰을 높은 ID 영역에 예약하여 일반 어휘와 충돌 방지

### 국가대표 AI 토크나이저 방식
- **Naver HyperCLOVAX**: SentencePiece 기반, 한국어 대규모 코퍼스 학습, 한국어 압축률 최고 수준
- **Kakao Kanana**: 한국어 형태론적 분해 중시, 한국어 패턴 최적화
- **Upstage Solar**: Llama 기반 어휘 확장, 한국어 토큰화 효율 개선
- **KT MiDM**: 한국어 대화/기술 텍스트 최적화

**공통점**: 한국어 토큰화에서는 음절 단위("한")가 아닌 **형태소/서브워드 단위**("하", "는", "법", "률") 분해가 효율적. BPE가 이를 자동 학습.

## 3. 권장 재학습 방안

### 방안 A: 코퍼스 확장 BPE 재학습 (권장)

**목표 어휘**: 131,072 (기존 65K의 2배)
**토크나이저 학습 코퍼스의 한국어 비중**: 35~45%
**실제 pre-training token 비중**: 한국어 20~30% 권장

**코퍼스 구성**:
```
한국어 위키백과 (5G 텍스트)           ← 일반 한국어, 15%
한국 법령 legalize-kr (675M)          ← 법률 한국어, 2%
한국 조례 ordinance-kr (3.2G)         ← 법률 한국어, 10%
한국 행정규칙 admrule-kr (523M)       ← 법률 한국어, 2%
한국 판례 precedent-kr (3.0G)         ← 법률 한국어, 10%
원본 영어 코퍼스 (FLAN, SYNTH)         ← 기존 영어, 40%
ToolBench + Terminal 데이터            ← 터미널/툴, 5%
코드/프로그래밍 코퍼스                  ← 코드 심볼, 16%
```

**비중을 분리하는 이유**:
- 토크나이저는 한국어를 oversampling해야 조사/어미/한자어/법률 용어를 효율적으로 압축함
- 하지만 모델 pre-training에서 법률 raw text 비중을 과도하게 높이면 일반성, 코딩, 터미널 능력이 떨어질 수 있음
- 목표는 한국어 법률 전용 모델이 아니라 한국어 터미널/툴콜/코딩 모델이므로 code/terminal/tool 데이터 비중을 별도 확보해야 함

**실행**:
```bash
# 1. 코퍼스 준비 (JSONL 또는 평문 텍스트)
mkdir -p /home/work/.data/huggingface/tokenizer_corpus/
# 위키, 법률, 영어 코퍼스를 하나의 디렉토리에 모으기

# 2-A. data_io Rust 토크나이저 사용
cd data_io/tokenizer
cargo run --release --bin train_tokenizer -- \
  /path/to/corpus_korean /path/to/corpus_english \
  --vocab-size 131072 \
  -o /path/to/trained_tokenizers/bpe-ko-131k/tokenizer.json

# 2-B. 또는 HuggingFace tokenizers 라이브러리 사용
python3 train_tokenizer.py \
  --corpus /home/work/.data/huggingface/tokenizer_corpus/ \
  --vocab-size 131072 \
  --output /home/work/.data/huggingface/trained_tokenizers/hrm-ko-131k/
```

### 방안 B: 기존 다국어 토크나이저 교체

**Qwen3 토크나이저 채택 (권장)**:
- 어휘: 151,936 / tiktoken BPE
- 한국어 이미 잘 지원, CJK 토큰 대량 포함
- 툴콜 특수 토큰 이미 내장

```bash
# Qwen3 토크나이저 복사
cp /home/work/.data/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*/tokenizer.json \
   /path/to/hrm-text/tokenizer.json
```

**장점**: 즉시 사용, 검증됨, 한국어 char/token 2.0+
**단점**: vocab 65K→152K 변경으로 임베딩/출력 레이어 재학습 필요

### 방안 C: 기존 65K에 한국어 토큰 추가 (최소 변경)

기존 영어 BPE 어휘 유지 + 한국어高频 토큰 추가 확장:
```python
from tokenizers import Tokenizer, models, trainers
base = Tokenizer.from_file("original_65k.json")
# 한국어 코퍼스로 추가 학습
trainer = trainers.BpeTrainer(vocab_size=131072, ...)
base.train(trainer, files=["korean_corpus.txt"])
```

**장점**: 영어 성능 완전 보존
**단점**: 기존 토큰 ID 유지 불가능할 수 있음 (BPE 병합 순서 변경)

## 4. 필수 특수 토큰 설계

### 터미널/툴콜 특화 토큰

HRM-Text 훈련용으로 다음 특수 토큰을 어휘에 포함해야 함:

```
=== 시스템/지시 ===
<|im_start|>          — 메시지 시작 (ChatML)
<|im_end|>            — 메시지 종료
<|system|>            — 시스템 프롬프트
<|user|>              — 사용자 메시지
<|assistant|>         — 어시스턴트 응답

=== 툴/함수 호출 ===
<|tool_call|>         — 툴 호출 시작
<|/tool_call|>        — 툴 호출 종료
<|tool_response|>     — 툴 응답
<|function|>          — 함수 정의
<|/function|>         — 함수 정의 종료
<|execute|>           — 실행 명령
<|result|>            — 실행 결과

=== 터미널 ===
<|terminal|>          — 터미널 명령 시작
<|/terminal|>         — 터미널 명령 종료
<|command|>           — 쉘 명령
<|output|>            — 명령 출력
<|error|>             — 에러 출력
<|exit_code|>         — 종료 코드

=== 구조화 출력 ===
<|json_start|>        — JSON 블록
<|json_end|>
<|xml_start|>         — XML 블록
<|xml_end|>
<|code_start|>        — 코드 블록
<|code_end|>

=== 추론 ===
<think>               — 사고 과정 시작 (pre-training에서는 제거하거나 cot bucket으로 분리)
</think>              — 사고 과정 종료

=== 기존 HRM-Text 토큰 (유지) ===
<|PAD|>, <|direct|>, <|cot|>, <|noisy|>, <|synth|>,
<|im_start|>, <|im_end|>, <|object_ref_start|>, <|object_ref_end|>,
<|box_start|>, <|box_end|>, <|quad_start|>, <|quad_end|>,
<|vision_start|>, <|vision_end|>, <|vision_pad|>, <|image_pad|>,
<|video_pad|>, <|fim_prefix|>
```

### 필수 일반 토큰 보장 항목

토크나이저가 반드시 효율적으로 처리해야 할 문자/패턴:

```
=== 숫자 ===
0-9, 소수점, 쉼표 구분자 (1,000,000), 마이너스 (-1)
→ 숫자가 과도하게 byte 단위로 쪼개지지 않는지 진단. 개별 digit 고정은 필수 아님

=== 코드/터미널 기호 ===
< > ( ) [ ] { } | & ; $ # @ ! ~ ` \ / " '
→ 모든 ASCII 기호가 단일 토큰 또는 짧은 토큰으로 처리되어야 함

=== JSON/XML 구조 ===
{:,}", [{,}], <tag>, </tag>
→ 툴콜 JSON이 효율적으로 토큰화되어야 함

=== 한국어 특수 ===
조사: 은/는, 이/가, 을/를, 에, 에서, 로/으로
어미: 다, 한다, 했다, 되다, 되었다
법률 용어: 조, 항, 호, 부, 위원회, 법률, 규정, 시행령
한자: 法, 規, 條, 項 (법률 문서에 빈번)
```

## 5. 토크나이저 학습 도구 비교 (2026)

| 도구 | 방식 | 장점 | 단점 | 추천 |
|---|---|---|---|---|
| **tiktoken** | BPE | Qwen/GPT/Llama 표준, 빠름 | Python 전용, 학습 기능 제한적 | 프로덕션 |
| **HuggingFace Tokenizers** | BPE/WordPiece/Unigram | Rust 백엔드, 빠름, 유연함 | 설정 복잡 | **학습용 (권장)** |
| **SentencePiece** | BPE/Unigram | gemma/DeepSeek 표준, 언어 독립적 | 속도 느림 | 대안 |
| **data_io Rust** | BPE | HRM-Text 기본, 통합 | 커스텀, 문서 부족 | 기존 호환 |

**권장**: HuggingFace Tokenizers (Rust 백엔드)로 학습 → JSON 내보내기 → HRM-Text/data_io에서 로드

## 6. 검증 체크리스트

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("trained_tokenizers/hrm-ko-131k/tokenizer.json")

# 1. 한국어 기본
enc = tok.encode("대한민국 헌법 제1조 대한민국은 민주공화국이다.")
assert len(enc.tokens) <= 15, f"한국어 토큰 과다: {enc.tokens}"
print(f"KO char/token: {len('대한민국 헌법 제1조') / len(enc.tokens):.2f}")  # 1.5+

# 2. 법률 용어
enc = tok.encode("제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다.")
print(f"법률 토큰: {enc.tokens}")

# 3. 터미널 명령어
enc = tok.encode("ls -la /home/work && grep -r 'error' *.log | sort | uniq -c")
terminal_text = "ls -la /home/work && grep -r 'error' *.log | sort | uniq -c"
print(f"터미널 char/token: {len(terminal_text) / len(enc.tokens):.2f}")  # 2.0+

# 4. 툴콜 JSON
enc = tok.encode('{"name": "get_weather", "arguments": {"city": "Seoul"}}')
print(f"JSON 토큰: {enc.tokens}")

# 5. 숫자 보존
enc = tok.encode("1234567890")
print(f"숫자 토큰: {enc.tokens}")  # 과도한 byte fallback 여부 확인

# 6. 특수 기호
enc = tok.encode("<|tool_call|> <|terminal|> <|execute|>")
print(f"특수 토큰: {enc.tokens}")  # 1토큰씩 인코딩되어야 함

# 7. 영어 성능 유지
enc = tok.encode("The quick brown fox jumps over the lazy dog.")
english_text = "The quick brown fox jumps over the lazy dog."
print(f"EN char/token: {len(english_text) / len(enc.tokens):.2f}")  # 3.0+
```

## 7. 실행 계획 요약

1. **위키 압축 해제** → XML에서 텍스트 추출 (wikiextractor 사용)
2. **코퍼스 병합** — 한국어(위키+법률) 12G + 영어(FLAN/SYNTH 샘플) + 코드/터미널
3. **HuggingFace Tokenizers로 BPE 학습** — vocab 131K, 한국어 40~50%
4. **특수 토큰 주입** — 터미널/툴콜/포맷 토큰 40~60개
5. **검증** — 위 체크리스트 통과
6. **HRM-Text 적용** — vocab_size 재설정, 임베딩/lm_head 새 학습

## 8. 2026-05-23 실제 산출물

현재 실제로 확보/생성된 산출물:

| 항목 | 위치 | 크기/규모 | 비고 |
|---|---|---:|---|
| extra raw/sample data | `/home/work/.data/huggingface/hrm_text_extra` | 2.8GB | SWE-ZERO 1GB, GLM 1GB, structured wiki 256MB 등 |
| tokenizer v1 | `/home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1` | 131,072 vocab | HF 업로드 완료 |
| SWE prepared | `/home/work/.data/hrm_text_prepared/sft_swe_zero_v1` | 53,868 samples / 182.7M tokens | long instruction middle truncation |
| GLM prepared | `/home/work/.data/hrm_text_prepared/sft_glm_reasoning_v1` | 56,021 samples / 68.5M tokens | `<think>` 제거 |
| SWE+GLM mix | `/home/work/.data/hrm_text_prepared/sft_swe_glm_mix_v1` | 109,889 samples / 251.2M tokens | B pilot 학습에 사용 |
| B pilot checkpoint | `/home/work/.data/hrm_text_checkpoints/koterm_b_swe_glm_pilot_v1` | 6.6GB | HF 업로드 완료 |

업로드:

- tokenizer: `https://huggingface.co/LLM-OS-Models/HRM-Text-Ko-Terminal-Tokenizer-131K`
- raw FSDP2 pilot checkpoint: `https://huggingface.co/LLM-OS-Models/HRM-Text-Ko-Terminal-B-SWE-GLM-Pilot`
- current model repo: `https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B`
- prepared dataset repo: `https://huggingface.co/datasets/LLM-OS-Models/KoHRM-Text-1.4B-prepared-data`

주의:

- 현재 mix는 token 기준 SWE terminal/code가 약 72.7%라, terminal smoke에는 좋지만 최종 한국어 모델용 비중으로는 과하다.
- 장기 학습 전 한국어 일반 instruction, 한국어 법률/행정/판례 task, tool-call/JSON task를 추가해 비중을 조정해야 한다.

## 9. 2026-05-24 전처리/업로드 현황

완료된 prepared dataset:

| dataset | tokens | size |
|---|---:|---:|
| `koterm_hrm_cleaned_fastcap_stage1_v1` | 14.55B | 148G |
| `local_terminal_conversations_ctx9k_resp6k_v1` | 9.39B | 36G |
| `koterm_pretrain_mix_v1` | 711.3M | 2.8G |
| `kowiki_raw_full_v1` | 462.5M | 1.8G |
| `korean_legal_raw_full_v1` | 308.9M | 1.2G |
| `korean_admrule_precedent_raw_full_v1` | 271.7M | 1.1G |
| `hrm_cleaned_base_sample_v1` | 250.0M | 994M |
| `sft_swe_glm_mix_v1` | 251.2M | 990M |
| `sft_swe_zero_v1` | 182.7M | 720M |
| `sft_toolbench_v1` | 127.0M | 500M |
| `hf_extra_reasoning_agent_mm_v1` | 112.6M | 444M |
| `sft_korean_legal_v1` | 83.1M | 336M |
| `sft_glm_reasoning_v1` | 68.5M | 282M |
| `sft_bcai_finance_kor_v1` | 857.7M | 3.3G |

진행/예약:

| 항목 | 상태 |
|---|---|
| HRM 328G cleaned full/no-cap 재토큰화 | 진행 중, tokenized root 600G 이상, metadata 5221개 생성 |
| HRM full/no-cap V1Dataset 패킹 | 재토큰화 종료 후 `koterm_hrm_cleaned_full_nocap_v1`로 예약 |
| 한국어 법률/조례/행정규칙/판례 task full nocap | 생성, V1Dataset 전처리, HF 업로드 완료 |
| BCAI Finance Kor | 다운로드, HRM JSONL 변환, V1Dataset 전처리, HF 업로드 완료 |
| prepared dataset HF 업로드 | 1차 업로드와 legal/finance 후속 업로드 완료, HRM full/no-cap만 예약 |
