---
license: other
language:
- ko
- en
tags:
- hrm-text
- korean
- terminal
- tool-use
- code
- pretraining
pipeline_tag: text-generation
---

# KoHRM-Text-1.4B

`KoHRM-Text-1.4B`는 `sapientinc/HRM-Text`의 PrefixLM 학습 구조를 기반으로, 한국어/영어/코딩/터미널/툴콜 사용성을 목표로 scratch pretraining하는 모델입니다.

이 카드는 2026-05-23 기준 작업 중인 모델 카드 초안입니다. 현재 업로드되는 epoch artifact는 raw HRM-Text FSDP2 checkpoint이며, 바로 Transformers에서 로드하는 최종 배포 형식이 아닙니다.

## 모델 정보

| 항목 | 값 |
|---|---|
| model id | `LLM-OS-Models/KoHRM-Text-1.4B` |
| base code | `sapientinc/HRM-Text` |
| training from | scratch |
| architecture | HRM-Text `XL` |
| params | 1,384,120,320 |
| context | 4096 tokens |
| dtype | bfloat16 |
| tokenizer | byte-level BPE, NFC normalization |
| vocab | 131,072 |

## 토크나이저

새 tokenizer는 한국어, 영어, 코드, shell, terminal instruction, JSON tool-call을 함께 고려해 학습했습니다.

| 샘플 | chars/token |
|---|---:|
| 한국어 일반 | 2.60 |
| 한국어 법률 | 2.36 |
| 한국어 터미널 지시 | 2.18 |
| shell command | 2.68 |
| tool JSON | 3.32 |
| Python code | 3.37 |
| 영어 | 4.40 |

Tokenizer repo: `LLM-OS-Models/HRM-Text-Ko-Terminal-Tokenizer-131K`

## 학습 데이터

stage-0/stage0b 입력은 전처리 완료된 711.3M token mix입니다.

| 데이터 | token |
|---|---:|
| HRM cleaned base sample | 250.0M |
| SWE-ZERO + GLM reasoning mix | 251.2M |
| 한국어 법률/조례/행정규칙/판례 task | 83.1M |
| ToolBench train tool-call task | 127.0M |
| 합계 | 711.3M |

현재 stage-1은 HRM cleaned fast-cap V1Dataset 14.55B tokens로 학습 중입니다. 이후 stage는 local terminal dataset, 추가 한국어/코딩/툴콜 데이터를 순차적으로 포함합니다. 평가 데이터 성격의 `tb2_lite`, Terminal Bench 2, ToolBench eval, chi-bench는 train에서 제외합니다.

## 학습 방식

- Objective: PrefixLM style response-only loss
- Optimizer: HRM-Text upstream Adam-atan2
- Context: 4096 tokens
- Hardware: 8 x NVIDIA H200
- Current stage-1 global batch: 262,144 tokens
- Checkpoint policy: epoch-level raw FSDP2 checkpoint upload

stage-1은 8 x H200에서 `global_batch_size=262144`로 실행 중이며, 관측 VRAM은 GPU0 약 118GB, 나머지 약 116GB입니다. 안정 속도는 약 `1.09-1.10 sec/step`, 약 238k-240k tokens/sec입니다. 문제가 생기면 `196608` batch로 되돌려 resume합니다.

Staged pretraining에서는 checkpoint의 model/optimizer/EMA/carry를 이어받고, `resume_step_offset`과 `total_steps_override`로 LR schedule을 전체 pretraining 기준에 맞춥니다. 즉, 새 데이터가 준비될 때마다 학습을 재시작하되 optimizer와 schedule을 끊지 않는 방향으로 운용합니다.

## 현재 상태

- stage-0/stage0b training: complete
- stage0b HF upload: complete
- stage-1 HRM fast-cap training: in progress
- final Transformers conversion: not yet produced
- public benchmark score: not yet evaluated for this model

## 제한사항

현재 checkpoint artifact는 중간 학습 산출물입니다. 안전성 정렬, 최종 instruction tuning, 최종 benchmark, 배포용 변환이 끝난 모델이 아닙니다. 한국어 터미널/툴콜 능력은 목표 영역이지만, stage-0만으로는 완성된 성능을 보장하지 않습니다.
