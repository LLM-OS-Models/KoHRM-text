# KoHRM-Text Docs

이 폴더는 KoHRM-Text 학습/운영 문서를 모아둔 곳입니다. 루트 [README.md](../README.md)는 프로젝트 첫 화면이고, 세부 문서는 여기에서 관리합니다.

## 먼저 읽을 문서

- [MODEL_TRAINING_ARCHITECTURE_GUIDE_2026-05-28.md](MODEL_TRAINING_ARCHITECTURE_GUIDE_2026-05-28.md)

  모델 구조, PrefixLM, response-only loss, PT/SFT 관계, staged continuation을 한 번에 설명합니다.

- [METHODOLOGY_ARCHITECTURE_NOTES_2026-05-24.md](METHODOLOGY_ARCHITECTURE_NOTES_2026-05-24.md)

  HRM-Text 논문 방식과 KoHRM 적용 차이를 정리합니다.

- [BATCH_AND_CONTEXT_LENGTH_NOTES_2026-05-27.md](BATCH_AND_CONTEXT_LENGTH_NOTES_2026-05-27.md)

  pretraining/SFT batch size, token-based batch, context length 4096의 의미를 설명합니다.

## 학습 계획과 데이터

- [PRETRAINING_SFT_DATA_MIX_2026-05-23.md](PRETRAINING_SFT_DATA_MIX_2026-05-23.md)

  사전학습/SFT 데이터 구성, 비중, 제외 기준입니다.

- [TRAINING_PLAN_2026-05-23.md](TRAINING_PLAN_2026-05-23.md)

  전체 학습 전략, tokenizer, 실행 정책입니다.

- [STAGED_TRAINING_RUNBOOK_2026-05-23.md](STAGED_TRAINING_RUNBOOK_2026-05-23.md)

  준비된 데이터부터 학습하고, 새 데이터가 생기면 이어 학습하는 운영 절차입니다.

- [EPOCH_PASS_CHECKPOINT_MAP_2026-05-28.md](EPOCH_PASS_CHECKPOINT_MAP_2026-05-28.md)

  데이터 1/2/3/4 pass 기준으로 stage와 checkpoint를 찾는 문서입니다.

## 운영과 상태

- [TRAINING_OPERATIONS_LOG_2026-05-26.md](TRAINING_OPERATIONS_LOG_2026-05-26.md)

  장기 학습 운영 로그, stage chain, 업로드 watcher, 속도 분석입니다.

- [CHAIN_HANDOFF_STATUS_2026-05-26.md](CHAIN_HANDOFF_STATUS_2026-05-26.md)

  stage handoff 상태, stage 이름, 용량, ETA, watcher 보정 기록입니다.

- [TRAINING_LOSS_ANALYSIS_2026-05-26.md](TRAINING_LOSS_ANALYSIS_2026-05-26.md)

  train loss와 token accuracy 해석, 계속 진행 여부 판단입니다.

- [VRAM_OOM_NOTES_2026-05-24.md](VRAM_OOM_NOTES_2026-05-24.md)

  VRAM 증가/OOM 원인과 batch 정책입니다.

- [PROGRESS_2026-05-23.md](PROGRESS_2026-05-23.md)

  실제 진행 로그입니다.

## 공개 카드와 인벤토리

- [MODEL_CARD_KoHRM-Text-1.4B.md](MODEL_CARD_KoHRM-Text-1.4B.md)

  Hugging Face model card 초안입니다.

- [HF_DATASET_CARD_KoHRM-Text-Prepared-Data.md](HF_DATASET_CARD_KoHRM-Text-Prepared-Data.md)

  Hugging Face prepared dataset card 초안입니다.

- [AVAILABLE_DATA.md](AVAILABLE_DATA.md)

  로컬 데이터 인벤토리와 용량입니다.

- [UPSTREAM_README.md](UPSTREAM_README.md)

  원본 `sapientinc/HRM-Text` README 보존본입니다.

## 정리 기준

루트에는 프로젝트 첫 화면인 `README.md`만 둡니다. 운영 기록, 방법론, 모델 카드, 데이터 카드, checkpoint map 같은 긴 문서는 모두 `docs/` 아래에 둡니다.
