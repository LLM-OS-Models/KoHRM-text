# Text2SQL LoRA Data Guide - 2026-06-03

이 문서는 KoHRM-Text에 영어 Text2SQL 능력을 붙이기 위한 데이터 후보, 전처리 방식, LoRA 학습 실행법을 정리한다.

## 결론

한국어 Text2SQL은 우선순위가 아니다. 영어 SQL을 잘하게 만드는 것이 목표다.

후보 우선순위는 다음 기준을 따른다.

```text
1. Hugging Face 다운로드 수
2. Hugging Face likes
3. license가 학습/배포에 명확한지
4. schema + question + SQL 형태가 깨끗한지
5. 평가셋 오염을 관리할 수 있는지
```

1차 LoRA 후보는 `core-clean + duckdb`다. `SQaLe`는 인기가 있고 최신 데이터지만 schema가 매우 길어서 4K context에서는 전량 core에 섞기보다 별도 schema-heavy 후보로 다루는 것이 낫다.

```text
core-clean:
  gretelai/synthetic_text_to_sql
  b-mc2/sql-create-context
  Clinton/Text-to-sql-v1

schema-heavy:
  trl-lab/SQaLe-text-to-SQL-dataset

mix-v2:
  DanielRegaladoCardoso/text-to-sql-mix-v2

duckdb:
  motherduckdb/duckdb-text2sql-25k
```

`SynSQL-2.5M`과 `NumbersStation/NSText2SQL`은 큰 후보로 남긴다. 바로 전량 학습하지 않고, 별도 `large` profile에서 capped subset으로 먼저 검증한다.

## 인기 후보

2026-06-03 기준 Hugging Face API로 확인한 수치다.

| Dataset | Downloads | Likes | License | Role |
|---|---:|---:|---|---|
| `xlangai/spider` | 11,586 | 174 | cc-by-sa-4.0 | benchmark/train split seed |
| `b-mc2/sql-create-context` | 3,911 | 499 | cc-by-4.0 | core SFT |
| `gretelai/synthetic_text_to_sql` | 3,176 | 661 | apache-2.0 | core SFT |
| `Salesforce/wikisql` | 1,931 | 125 | unknown on HF | simple benchmark/basic SQL |
| `Clinton/Text-to-sql-v1` | 906 | 73 | apache-2.0 | core SFT |
| `trl-lab/SQaLe-text-to-SQL-dataset` | 901 | 16 | mit | schema-heavy core SFT |
| `DanielRegaladoCardoso/text-to-sql-mix-v2` | 697 | 0 | apache-2.0 | deduped mix |
| `seeklhy/SynSQL-2.5M` | 546 | 28 | apache-2.0 | large filtered SFT |
| `NumbersStation/NSText2SQL` | 539 | 90 | other | large candidate, license review |
| `motherduckdb/duckdb-text2sql-25k` | 81 | 43 | cc-by-sa-4.0 | DuckDB practical SQL |

`motherduckdb/duckdb-text2sql-25k`는 인기순 최상위는 아니지만 DuckDB 기능, PRAGMA, CSV/Parquet/JSON, extension 관련 SQL을 다루므로 실무성은 높다.

## 왜 벤치마크를 전부 학습하면 안 되는가

정답 SQL이 있는 벤치마크도 SFT 데이터로 쓸 수 있다. 다만 전량 학습하면 그 데이터로는 평가를 못 한다.

정책:

```text
benchmark train split:
  SFT에 사용 가능

benchmark validation/test split:
  가능하면 보류
  SFT 데이터와 별도 holdout/eval로 유지

DuckDB 25K 같은 단일 split 데이터:
  train/holdout을 자체 분리하거나
  일부를 평가용으로 남긴다
```

## 학습 포맷

KoHRM은 아직 instruction pretraining 이후 모델이고, chat-template SFT 모델이 아니다. 따라서 Text2SQL SFT prompt는 짧고 규칙적이어야 한다.

기본 포맷:

```text
Dialect: SQL

Schema:
CREATE TABLE ...

Question:
How many orders were placed in 2024?

SQL:
```

response:

```sql
SELECT COUNT(*) FROM orders WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01';
```

DuckDB 전용 포맷:

```text
Dialect: DuckDB
Category: sql/pragmas

Schema:
CREATE TABLE ...

Question:
What are the tables available in the current database?

SQL:
```

response:

```sql
PRAGMA show_tables;
```

피해야 할 것:

```text
복잡한 system prompt
JSON-only 강제
chat template
assistant: 같은 역할 라벨
긴 chain-of-thought 노출
불필요한 설명문
```

## 추가된 코드

```text
scripts/build_text2sql_sft_data.py
  Hugging Face Text2SQL 데이터를 통합 JSONL로 정규화한다.

scripts/prepare_text2sql_lora_data.sh
  raw JSONL 생성, tokenizer 기반 V1Dataset 전처리, prepared dataset merge를 실행한다.

scripts/run_kohrm_lora_experiments.sh
  text2sql-core, text2sql-duckdb, text2sql-core-duckdb, text2sql-large, text2sql-all 실행 타겟을 추가했다.
```

## 산출물 경로

2026-06-03 20:14 KST 기준 완료된 1차 산출물:

```text
kohrm_sft_text2sql_core_clean_v1:
  samples:      440,783
  tokens:       104,366,382
  max sample:   2,788 tokens
  truncation:   0
  dropped:      0

kohrm_sft_text2sql_duckdb_v1:
  samples:      24,498
  tokens:       10,680,836
  max sample:   1,434 tokens
  truncation:   0
  dropped:      0

kohrm_sft_text2sql_core_clean_duckdb_v1:
  samples:      465,281
  tokens:       115,047,218
  max sample:   2,788 tokens
```

처음 시도한 full `core`에는 `SQaLe` 전량이 포함되어 일부 schema가 너무 길었다. 4K context에서 truncation 비율이 커져 1차 LoRA용으로는 부적절하다고 판단하고, `core-clean`을 먼저 완료했다. `SQaLe`는 `schema-heavy` 후보로 별도 전처리한다.

raw JSONL:

```text
/home/work/.data/hrm_text_raw/text2sql/text2sql_core_clean_sft.jsonl
/home/work/.data/hrm_text_raw/text2sql/text2sql_core_sft.jsonl
/home/work/.data/hrm_text_raw/text2sql/text2sql_duckdb_sft.jsonl
/home/work/.data/hrm_text_raw/text2sql/text2sql_schema_heavy_sft.jsonl
/home/work/.data/hrm_text_raw/text2sql/text2sql_mix_v2_sft.jsonl
/home/work/.data/hrm_text_raw/text2sql/text2sql_large_sft.jsonl
```

prepared V1Dataset:

```text
/home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_core_clean_v1
/home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_core_clean_duckdb_v1
/home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_core_v1
/home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_duckdb_v1
/home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_core_duckdb_v1
/home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_schema_heavy_v1
/home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_mix_v2_v1
/home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_large_v1
/home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_all_v1
```

## 전처리 실행

clean core + DuckDB:

```bash
HF_HOME=/home/work/.data/hf_cache \
EPOCHS=2 \
STREAMING=false \
scripts/prepare_text2sql_lora_data.sh core-clean-duckdb
```

DuckDB만:

```bash
scripts/prepare_text2sql_lora_data.sh duckdb
```

schema-heavy 후보:

```bash
MAX_SCHEMA_CHARS=8000 \
scripts/prepare_text2sql_lora_data.sh schema-heavy
```

큰 후보 subset:

```bash
scripts/prepare_text2sql_lora_data.sh large \
  --max-rows synsql=250000 \
  --max-rows nstext2sql=250000
```

전체 후보:

```bash
scripts/prepare_text2sql_lora_data.sh all
```

## LoRA 실행

pretraining 최종 checkpoint를 지정한다.

```bash
export RESUME_FROM=/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage4d-korean-tool-finance-repeat3-gbs180
```

clean core + DuckDB LoRA:

```bash
EPOCHS=1 \
GBS=32768 \
LR=8.0e-5 \
bash scripts/run_kohrm_lora_experiments.sh text2sql-core-clean-duckdb
```

개별 실험:

```bash
bash scripts/run_kohrm_lora_experiments.sh text2sql-core-clean
bash scripts/run_kohrm_lora_experiments.sh text2sql-core
bash scripts/run_kohrm_lora_experiments.sh text2sql-duckdb
bash scripts/run_kohrm_lora_experiments.sh text2sql-large
```

## 추천 순서

```text
1. core-clean-duckdb prepared 생성
2. 샘플 20개 눈검수
3. text2sql-core-clean-duckdb LoRA 1 epoch
4. Spider/WikiSQL/DuckDB holdout prompt로 quick eval
5. 부족하면 large subset 250K + 250K를 추가
6. 그래도 부족하면 large cap을 키운다
```

전량 large SFT를 바로 돌리면 SQL 형식은 늘 수 있지만, 일반 행동 LoRA와 충돌하거나 특정 synthetic style로 기울 수 있다. 먼저 작은 실험으로 SQL 정확도와 일반 응답 손상을 확인하는 것이 맞다.
