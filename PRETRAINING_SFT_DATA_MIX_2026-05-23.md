# HRM-Text 한국어/터미널 모델 데이터 구성 계획

작성일: 2026-05-23

이 문서는 새 `HRM-Text` 계열 한국어/영어/코딩/터미널/툴콜 모델을 만들 때 실제로 어떤 데이터를 사전학습과 SFT에 넣을지 정리한 기준 문서입니다.

## 결론

SFT 후보 데이터도 사전학습에 모두 넣습니다. 여기서 말하는 사전학습은 HRM-Text 논문 방식에 맞춘 `instruction -> response` PrefixLM 학습입니다. 즉 일반 raw LM만 하는 것이 아니라, 모든 데이터를 가능한 한 `{"instruction": ..., "response": ..., "condition": ...}` 형식으로 바꿔 response-only loss로 학습합니다.

SFT는 그 뒤에 한 번 더 합니다. 같은 계열의 데이터를 다시 쓰되, SFT 단계에서는 품질이 높은 subset, 포맷이 엄격한 tool-call/terminal trajectory, 한국어 응답 스타일, 실패 복구 루프를 더 강하게 가중합니다.

현재 완료된 B pilot은 최종 mix가 아닙니다. pilot에는 `/home/work/.data/hrm_text_prepared/sft_swe_glm_mix_v1`만 들어갔고, HRM 기존 328G cleaned pretraining 데이터는 아직 실제 학습 입력으로 쓰지 않았습니다. 앞으로는 기존 HRM cleaned 328G를 기본 축으로 반드시 포함합니다.

## 토크나이저

현재 새 토크나이저는 이미 만들어졌습니다.

| 항목 | 값 |
|---|---|
| 위치 | `/home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1` |
| 방식 | byte-level BPE |
| vocab | 131,072 |
| Unicode normalization | NFC |
| HF 업로드 | `https://huggingface.co/LLM-OS-Models/HRM-Text-Ko-Terminal-Tokenizer-131K` |

성격:

- 한국어, 영어, 코드, 터미널, JSON/tool-call을 모두 처리하는 범용 tokenizer입니다.
- 한국어 법률/행정 문서와 터미널/코드 텍스트를 tokenizer corpus에서 의도적으로 과대표집했습니다.
- `<|im_start|>`, `<|im_end|>`, `<|assistant|>`, `<|tool_call|>`, `<|terminal|>`, `<|box_end|>` 같은 포맷 토큰을 단일 토큰으로 유지합니다.
- 기존 HRM tokenizer로 이미 토큰화된 바이너리는 그대로 섞지 않습니다. 새 tokenizer와 token id가 다르므로, 기존 HRM cleaned 데이터도 원문 JSONL/parquet에서 다시 토큰화해야 합니다.

검증된 압축률:

| 샘플 | chars/token |
|---|---:|
| 한국어 일반 | 2.60 |
| 한국어 법률 | 2.36 |
| 한국어 터미널 지시 | 2.18 |
| shell command | 2.68 |
| tool JSON | 3.32 |
| Python code | 3.37 |
| 영어 | 4.40 |

Tokenizer corpus 목표 비중:

| bucket | 목표 비중 | 이유 |
|---|---:|---|
| 한국어 일반/법률 | 35~40% | 한국어 조사, 어미, 법률/행정 용어 압축률 확보 |
| 영어 instruction/general | 20~25% | HRM 기본 영어 능력 유지 |
| 코드/터미널/SWE | 20~25% | CLI, shell, traceback, patch, 테스트 출력 표현 보존 |
| tool-call/JSON/API | 10~15% | JSON argument, schema, API 이름을 깨지지 않게 유지 |
| 수학/STEM/reasoning | 5~10% | 수식, 논리 기호, 풀이 텍스트 보존 |

## 사전학습 전체 비중

최종 사전학습 token mix의 1차 목표입니다.

| bucket | 목표 token 비중 | 포함 데이터 |
|---|---:|---|
| HRM 기존 영어/general/reasoning base | 35~40% | HRM cleaned `flan`, `SYNTH`, `tasksource`, 일반 instruction |
| 수학/STEM/reasoning | 10~15% | HRM math 계열, GLM, Claude reasoning 일부 |
| 한국어 일반/법률/행정/위키 | 20~25% | kowiki, 법령, 조례, 행정규칙, 판례 |
| 코드/터미널/SWE | 15~20% | local terminal dataset, SWE-ZERO, code parquet |
| tool-call/API/JSON | 8~12% | ToolBench, DeepSeek agent traces, 한국어 tool-call 합성 |

터미널/툴콜 성능을 더 공격적으로 밀 때는 코드/터미널/SWE를 20~25%까지 올리고 HRM general을 35% 근처로 낮춥니다. 한국어 법률 데이터는 중요하지만 법률 전용 모델이 목표가 아니므로 전체를 과도하게 지배하게 두지 않습니다.

## 사전학습에 넣을 데이터

### 1. HRM-Text 기존 cleaned pretraining 데이터

| 항목 | 위치 | 크기 | 사용 여부 |
|---|---|---:|---|
| 전체 HF cache | `/home/work/.data/huggingface/hub/datasets--sapientinc--HRM-Text-data-io-cleaned-20260515/` | 328G cache, 실제 파일 합 약 325.9GiB | 사용 |
| snapshot | `snapshots/f033bc0e1a81634385093afc60445a52a7ade64a/` | 원문 JSONL/parquet symlink | 사용 |

이 데이터는 HRM-Text 논문 방식에 맞춰 이미 `instruction`, `response`, `condition` 구조를 갖고 있습니다. 새 tokenizer로 다시 V1Dataset 형태로 패킹해서 사용합니다.

세부 구성:

| 그룹 | 크기 | 설명 | 처리 |
|---|---:|---|---|
| `data_clustered/flan` | 약 271.0GiB | 가장 큰 일반 instruction/translation/classification/QA 계열 | 사용하되 task별 cap 적용. Amazon polarity, translation 같은 반복 task가 과도하게 지배하지 않게 제한 |
| `data_clustered/SYNTH` | 약 36.5GiB | synthetic rewritten knowledge/instruction | 적극 사용 |
| `data_clustered/openmathinstruct2` | 약 8.2GiB | 수학 direct/cot | 사용 |
| `data_clustered/dmmath` | 약 3.8GiB | 수학/추론 | 사용 |
| `data_clustered/acereason` | 약 2.1GiB | reasoning | 사용 |
| `data_clustered/openthoughts2` | 약 665MiB | reasoning | `<think>` 성격이면 분리/제거 후 사용 |
| `data_clustered/textbookreasoning` | 약 556MiB | textbook reasoning | 사용 |
| `data_clustered/ampsmathematica` | 약 411MiB | 수학/Mathematica | 사용 |
| `data_clustered/tasksource` | 약 391MiB | task instruction | 사용 |
| `data_clustered/sudoku_extreme` | 약 385MiB | symbolic puzzle | 낮은 비중으로 사용 |
| `data/*.jsonl` | 약 2.1GiB | GSM8K, MATH, NuminaMath, Natural Reasoning, Principia 등 | 사용 |

주의:

- 기존 HRM 328G는 “쓴다”가 맞습니다.
- 다만 기존 tokenized binary를 그대로 섞는 것이 아니라 새 tokenizer로 다시 토큰화합니다.
- `flan`이 271GiB로 너무 크기 때문에 단순 전체 균등 샘플링을 하면 번역/감성분류가 과대표집될 수 있습니다. 파일/task 단위 cap이 필요합니다.

### 2. 한국어 일반/법률/행정 데이터

| 데이터 | 위치 | 크기 | 사용 여부 |
|---|---|---:|---|
| 한국어 위키백과 | `/home/work/.data/huggingface/kowiki-20260501-pages-articles-multistream.xml` | 5.9G 해제, 1.3G bz2 | 사용 |
| 한국 법령 | `HRM-Text/legalize-kr/` | 675M | 사용 |
| 한국 조례 | `HRM-Text/ordinance-kr/` | 3.2G | 사용 |
| 행정규칙 | `admrule-kr/` | 523M | 사용 |
| 판례 | `precedent-kr/` | 3.0G | 사용 |

처리 방식:

- 위키는 단순 raw continuation만 쓰지 않고 제목/문단 기반 설명, 요약, 키워드 추출, 근거 문장 추출 task로 변환합니다.
- 법령/조례/행정규칙은 조항 검색, 조항 요약, 적용 범위 설명, 용어 정의, 인용 형식 변환, 근거 추출 task로 만듭니다.
- 판례는 `판시사항`, `판결요지`, `판례내용`을 분리해서 쟁점 요약, 결론 추출, 근거 문단 찾기, 사건 유형 분류 task로 만듭니다.
- 법률 데이터는 한국어 문체와 장문 구조 학습에 중요하지만, 전체 모델을 법률 모델로 만들 정도로 과대가중하지 않습니다.

### 3. 코드/터미널/SWE 데이터

| 데이터 | 위치 | 크기/상태 | 사용 여부 |
|---|---|---:|---|
| local terminal dataset | `dataset/` | 5.0G | 사용 |
| SWE-ZERO sample | `/home/work/.data/huggingface/hrm_text_extra/sft/swe_zero_terminal_sft_sample.jsonl` | 1.0G, 53,868 rows | 사용 |
| SWE-ZERO prepared | `/home/work/.data/hrm_text_prepared/sft_swe_zero_v1` | 182.7M tokens | 이미 사용 가능 |

처리 방식:

- 터미널 명령 제안, 실행 결과 해석, 실패 로그 분석, 다음 patch/command 예측으로 만듭니다.
- SWE-ZERO는 assistant turn별 다음 행동 예측 데이터로 사용합니다.
- 이 데이터는 사전학습에도 넣고, SFT에서도 다시 강하게 사용합니다.
- 현재 pilot에서 SWE 비중이 token 기준 약 72.7%라 최종 mix로는 과합니다. 최종 사전학습에서는 15~20%, 터미널 특화 run에서는 20~25%를 목표로 합니다.

### 4. Tool-call/API/JSON 데이터

| 데이터 | 위치 | 크기/상태 | 사용 여부 |
|---|---|---:|---|
| ToolBench 전체 추출본 | `HRM-Text/data_toolbench/data/` | 21G | 사용 |
| ToolBench train trajectory | `HRM-Text/data_toolbench/data/toolllama_G123_dfs_train.json` | 1.9G | 사용 |
| ToolBench instruction queries | `HRM-Text/data_toolbench/data/instruction/*.json` | 약 1.6G | 사용 |
| ToolBench eval | `HRM-Text/data_toolbench/data/toolllama_G123_dfs_eval.json` | 7.6M | 학습 제외 |
| DeepSeek-v4-Pro-Agent partial | `/home/work/.data/huggingface/hrm_text_extra/raw/TeichAI__DeepSeek-v4-Pro-Agent/` | raw 일부, extra raw 전체 506M 안에 포함 | 제한 사용 |

처리 방식:

- ToolBench train은 tool 선택, function argument JSON 생성, tool result 이후 최종 답변 생성 task로 변환합니다.
- toolenv metadata는 API 이름/설명/schema를 prompt에 넣는 방식으로 일부 사용합니다.
- eval split은 contamination 방지를 위해 학습에서 제외합니다.
- DeepSeek agent trace는 형식은 유용하지만 라이선스 확인 전 공개 모델 학습에는 보수적으로 다룹니다. 내부 research run에는 작은 비중으로 넣고, 공개 배포용 최종 run에서는 라이선스가 불명확하면 제외합니다.

### 5. 추가 reasoning/SFT 후보 데이터

| 데이터 | 위치 | 크기/상태 | 사용 여부 |
|---|---|---:|---|
| GLM-5.1 Reasoning sample | `/home/work/.data/huggingface/hrm_text_extra/sft/glm_5_1_reasoning_sft_sample.jsonl` | 1.0G, 57,916 rows | 사용 |
| GLM prepared | `/home/work/.data/hrm_text_prepared/sft_glm_reasoning_v1` | 68.5M tokens | 이미 사용 가능 |
| Claude Opus 4.6/4.7 reasoning | `/home/work/.data/huggingface/hrm_text_extra/raw/angrygiraffe__claude-opus-4.6-4.7-reasoning-8.7k/` | 약 240M급 파일 묶음 | 사용 |
| structured-wikipedia EN sample | `/home/work/.data/huggingface/hrm_text_extra/tokenizer_corpus/structured_wikipedia_en_sample.jsonl` | 256M, 18,070 rows | 낮은 비중 사용 |

처리 방식:

- GLM/Claude의 긴 private reasoning은 그대로 대량 투입하지 않습니다.
- pretraining에서는 `<think>...</think>`를 제거한 final answer 중심 샘플을 기본으로 씁니다.
- 명시적 추론을 배우게 할 필요가 있는 샘플만 `cot` condition bucket으로 분리합니다.
- SFT 단계에서는 reasoning style이 모델 출력에 과하게 새지 않도록 no-reasoning 버전과 final answer 버전을 우선합니다.

### 6. 현재 prepared dataset

| prepared dataset | 위치 | 샘플/token | 상태 |
|---|---|---:|---|
| SWE-ZERO | `/home/work/.data/hrm_text_prepared/sft_swe_zero_v1` | 53,868 samples / 182.7M tokens | 사용 가능 |
| GLM reasoning | `/home/work/.data/hrm_text_prepared/sft_glm_reasoning_v1` | 56,021 samples / 68.5M tokens | 사용 가능 |
| SWE+GLM pilot mix | `/home/work/.data/hrm_text_prepared/sft_swe_glm_mix_v1` | 109,889 samples / 251.2M tokens | pilot 완료 |

이 prepared dataset들은 모두 사전학습 mix에 포함합니다. 단, 최종 장기 학습에서는 HRM 328G, 한국어, ToolBench, terminal dataset까지 포함한 새 balanced dataset으로 다시 merge합니다.

## SFT 데이터도 사전학습에 넣는 원칙

다음 데이터는 “SFT용이라서 사전학습에서 제외”하지 않습니다. 전부 사전학습에 먼저 들어갑니다.

| 데이터 | 사전학습 사용 | SFT 재사용 | 이유 |
|---|---|---|---|
| SWE-ZERO | 예 | 예 | 터미널/코딩 다음 행동 패턴 자체가 핵심 능력 |
| ToolBench | 예 | 예 | tool selection과 JSON argument는 사전학습부터 익혀야 함 |
| GLM/Claude reasoning | 예 | 예 | final answer 중심 추론 패턴 보강 |
| DeepSeek agent traces | 제한 사용 | 제한 사용 | agent trajectory 형식 학습에 유용하나 라이선스 확인 필요 |
| 한국어 법률 task | 예 | 예 | 한국어 장문 지시, 근거 기반 응답, 공문체 학습 |
| local terminal dataset | 예 | 예 | 실제 목표인 한국어/터미널 모델의 핵심 |

차이는 weighting과 품질 기준입니다.

- 사전학습: 넓게 많이 사용합니다. 약간 noisy한 데이터도 dataset cap과 dedup으로 조절해 넣습니다.
- SFT: 좁고 깨끗하게 사용합니다. 포맷 오류, 잘못된 tool call, 불완전한 reasoning, evaluation contamination 위험 샘플은 제거합니다.

## SFT 단계 구성

사전학습 후 SFT는 다음 순서로 한 번 더 합니다.

| SFT bucket | 목표 비중 | 포함 데이터 |
|---|---:|---|
| 터미널/코딩/SWE | 30~40% | local terminal dataset, SWE-ZERO 고품질 subset |
| tool-call/API/JSON | 20~25% | ToolBench train, 한국어 function calling 합성 |
| 한국어 일반/법률 QA | 15~20% | 위키/법령/판례 기반 고품질 instruction |
| 일반 assistant/reasoning | 15~20% | HRM cleaned high-quality, GLM/Claude final answer |
| safety/format repair | 3~5% | JSON repair, command correction, refusal/clarification 패턴 |

SFT에서는 특히 다음을 강제합니다.

- 한국어 존댓말 응답
- 터미널 명령과 설명의 분리
- tool call JSON의 strict validity
- 실패한 명령 출력 이후 다음 행동 제안
- 긴 `<think>` 노출 방지
- 평가셋 contamination 방지

## 학습에서 제외하거나 분리할 데이터

| 데이터 | 처리 |
|---|---|
| `tb2_lite`, Terminal Bench 2 | 평가용. 학습 제외 |
| ToolBench eval split | 평가용. 학습 제외 |
| `actava/chi-bench` | benchmark 성격. 기본 학습 제외, 필요하면 별도 평가용 |
| Open-MM-RL multimodal 원본 | text-only 모델에는 낮은 우선순위. 텍스트 instruction만 검토 후 극소량 |
| 라이선스 불명확 agent trace | 내부 research run만 제한 사용. 공개 모델에는 보류 |

## 현재 상태

| 작업 | 상태 |
|---|---|
| 새 tokenizer 학습 | 완료 |
| tokenizer HF 업로드 | 완료 |
| SWE-ZERO 1G 샘플 다운로드 | 완료 |
| GLM 1G 샘플 다운로드 | 완료 |
| structured-wikipedia 256M 샘플 다운로드 | 완료 |
| SWE-ZERO V1Dataset 전처리 | 완료 |
| GLM V1Dataset 전처리 | 완료 |
| SWE+GLM B pilot | 완료 |
| HRM 328G cleaned 데이터 확인 | 완료 |
| HRM 328G 새 tokenizer 재패킹 | 진행 필요 |
| 한국어 위키/법률/판례 task 변환 | 진행 필요 |
| ToolBench task 변환 | 진행 필요 |
| terminal dataset 변환 | 진행 필요 |
| 최종 balanced pretraining dataset merge | 진행 필요 |

## 다음 실행 순서

1. HRM cleaned 328G의 JSONL/parquet를 새 tokenizer로 재토큰화합니다.
2. `flan`은 파일/task cap을 둬서 271GiB가 전체를 지배하지 않게 합니다.
3. 한국어 위키/법률/조례/행정규칙/판례를 instruction-response task로 변환합니다.
4. ToolBench train과 local terminal dataset을 tool/terminal SFT 형식으로 변환합니다.
5. SWE/GLM prepared dataset까지 모두 merge해서 balanced pretraining dataset을 만듭니다.
6. H200 8장으로 L 또는 XL batch probe를 돌려 최대 global batch를 찾습니다.
7. 안정 batch 확인 후 장기 pretraining을 시작하고, checkpoint는 너무 자주 올리지 않고 의미 있는 간격으로 Hugging Face에 업로드합니다.

