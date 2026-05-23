# KoHRM-Text Staged Training Runbook

작성일: 2026-05-23

## 목적

GPU를 전처리 완료까지 기다리게 두지 않고, 준비된 V1Dataset부터 바로 pretraining에 투입합니다. 새 데이터 전처리가 끝나는 순서대로 checkpoint에서 resume해서 stage를 이어갑니다.

## 표준 이름

| 항목 | 값 |
|---|---|
| 모델명 | `KoHRM-Text-1.4B` |
| HF model repo | `LLM-OS-Models/KoHRM-Text-1.4B` |
| arch | `XL` |
| params | 1,384,120,320 |
| tokenizer | `/home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1` |
| vocab | 131,072 |
| context | 4096 tokens |

## 현재 stage 계획

| stage | 입력 데이터 | token | 상태 |
|---|---|---:|---|
| stage-0 | `koterm_pretrain_mix_v1` | 711,277,327 | 완료 |
| stage0b | `koterm_pretrain_mix_v1` resume pass | 711,277,327 | 완료 |
| stage-1 | `koterm_hrm_cleaned_fastcap_stage1_v1` | 14,554,291,763 | 실행 중 |
| stage-2 | local terminal `swe/math/code.parquet` 변환본, wiki/legal/tool-call prepared data | 10B+ | 대기 |
| stage-3 | full HRM cleaned no-cap + 추가 한국어/터미널/툴콜 balanced mix | 45B~52B 목표 | 계획 |

stage-0는 이미 전처리가 끝난 HRM sample, SWE/GLM, 한국어 법률, ToolBench train tool-call 데이터를 합친 mix입니다. stage-1부터는 새 전처리 산출물을 추가하고, 이전 checkpoint를 `resume_from`으로 지정해 계속 학습합니다.

## 현재 실행 중인 프로세스

PID는 재시작 시 바뀔 수 있으므로 로그 경로를 기준으로 추적합니다.

| 작업 | 기준 경로 |
|---|---|
| stage-1 training | `/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage1-hrm-fastcap-gbs180` |
| HRM full/no-cap tokenizer | `/home/work/.data/hrm_text_tokenized/koterm_hrm_cleaned_fastcap_v1` |
| prepared dataset HF upload | `/home/work/.data/hrm_text_hf_upload_stage/LLM-OS-Models__KoHRM-Text-1.4B-prepared-data` |
| follow-up preprocessing/upload scheduler | `/home/work/.data/hrm_text_logs/followup_prepared_uploads_20260524.log` |
| stage-0 checkpoint | `/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage0-available-mix-gbs172` |
| stage0b checkpoint | `/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage0b-debug-launch2` |

## batch 정책

| global batch | per GPU tokens | 판단 |
|---:|---:|---|
| 262,144 | 32,768 | XL에서 OOM 발생 |
| 196,608 | 24,576 | 논문 global batch와 동일하지만 장기 run에는 VRAM 여유가 너무 얇음 |
| 180,224 | 22,528 | 현재 stage-1 장기 run 기본값 |
| 172,032 | 21,504 | fallback 후보 |

논문 설정은 vocab 65,536 기준입니다. 우리는 vocab 131,072라 final logits buffer가 두 배 커집니다. H200 8장이라 compute는 충분하지만, HRM bp warmup 이후 backward depth와 final logits 메모리가 겹칩니다. stage-1에서는 `262144`, `229376` probe에서 OOM 위험을 확인했고, 장기 run은 `180224`로 안정화했습니다.

## 업로드 정책

- raw FSDP2 checkpoint는 epoch 단위로만 업로드합니다.
- 업로드는 `scripts/watch_and_upload_hrm_checkpoints.py`가 처리합니다.
- 학습 중 step 단위 업로드는 하지 않습니다. 네트워크 I/O가 학습을 방해할 수 있습니다.
- 최종 배포용 모델은 raw checkpoint가 아니라 `conversion/convert_to_hf.py` 변환 결과를 별도 업로드합니다.

## stage 이어 학습 명령 패턴

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  arch/size@arch=XL \
  data.path=/path/to/next_v1dataset \
  resume_from=/home/work/.data/hrm_text_checkpoints/previous_stage \
  +checkpoint_path=/home/work/.data/hrm_text_checkpoints/next_stage \
  +project_name=KoHRM-Text \
  +run_name=next_stage_name \
  epochs=1 \
  global_batch_size=172032 \
  lr_warmup_steps=2000 \
  resume_step_offset=4134 \
  total_steps_override=302000 \
  +log_interval=5 \
  checkpoint_interval=1
```

resume 시 `resume_from`에는 `fsdp2_epoch_*` 디렉터리가 들어 있는 checkpoint root를 지정합니다. `resume_epoch`을 생략하면 가장 큰 epoch 번호를 자동 선택합니다.

중요: upstream HRM-Text 코드는 model/optimizer/EMA/carry는 저장하고 불러오지만, global step/LR schedule은 기본적으로 새 run마다 0부터 다시 시작합니다. KoHRM-Text 쪽에는 staged pretraining용으로 `resume_step_offset`과 `total_steps_override`를 추가했습니다. 다음 stage부터는 이전 stage의 누적 step을 `resume_step_offset`에 넣고, 전체 계획 step을 `total_steps_override`에 넣어 논문식 long pretraining schedule에 더 가깝게 이어갑니다.

## 모니터링 명령

```bash
pgrep -af "torchrun|pretrain.py|watch_and_upload|target/release/tokenizer"
tail -80 /home/work/.data/hrm_text_logs/KoHRM-Text-1.4B-stage0-available-mix-gbs172.log
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
find /home/work/.data/hrm_text_tokenized/koterm_hrm_cleaned_fastcap_v1 -mindepth 1 -maxdepth 1 -type d | wc -l
du -sh /home/work/.data/hrm_text_tokenized/koterm_hrm_cleaned_fastcap_v1
```

## 다음 전처리 산출물 처리

HRM cleaned full/no-cap tokenizer가 끝나면:

```bash
python sample_tokenized.py \
  tokenized_path=/home/work/.data/hrm_text_tokenized/koterm_hrm_cleaned_fastcap_v1 \
  output_path=/home/work/.data/hrm_text_prepared/koterm_hrm_cleaned_fastcap_sampled_v1 \
  epochs=4 \
  context_size=4097 \
  > /home/work/.data/hrm_text_logs/koterm_hrm_cleaned_fastcap_sampled_v1_analytics.md
```

그 다음 `scripts/merge_prepared_sft_data.py`로 다음 stage 입력 mix를 만들고, 최신 checkpoint에서 resume합니다.

## 후속 전처리/업로드 예약

`scripts/schedule_followup_prepared_uploads.sh`는 현재 대용량 prepared dataset 업로드와 충돌하지 않도록 순차 실행되는 예약 스크립트입니다.

처리 순서:

1. `legalize-kr`, `ordinance-kr`, `admrule-kr`, `precedent-kr` 전체에서 uncapped 한국어 법률 task JSONL을 생성합니다.
2. 이 JSONL을 KoHRM 131K tokenizer로 V1Dataset 전처리합니다.
3. 기존 prepared-data HF 업로드가 끝날 때까지 기다린 뒤 legal full task 산출물을 dataset repo에 추가 업로드합니다.
4. HRM 328G full/no-cap tokenizer 프로세스가 끝날 때까지 기다립니다.
5. tokenized root를 `koterm_hrm_cleaned_full_nocap_v1` V1Dataset으로 패킹합니다.
6. full/no-cap HRM prepared dataset을 같은 HF dataset repo에 추가 업로드합니다.

로그:

```bash
tail -80 /home/work/.data/hrm_text_logs/followup_prepared_uploads_20260524.log
pgrep -af "schedule_followup|build_korean_legal_sft_data|sample_tokenized.py|upload_folder_to_hf.py"
```

## data_io tokenizer 로컬 패치

HRM cleaned 원본은 `sapientinc/data_io`의 Rust tokenizer 경로를 기준으로 처리합니다. 현재 로컬 clone은 `/home/work/.projects/LLM-OS-Models/Terminal/data_io`입니다.

현재 적용한 로컬 변경:

- `WalkDir::follow_links(true)` 적용. Hugging Face snapshot 내부 symlink를 따라가야 실제 parquet/jsonl 파일을 스캔할 수 있습니다.
- `read_any_stream` callback이 `bool`을 반환하도록 변경. row cap에 도달하면 파일 전체를 끝까지 읽지 않고 조기 종료합니다.
- `--epochs-for-caps`와 `--cap prefix=max_rows` 옵션 추가. 너무 큰 source는 prefix별로 제한해 fast-cap 전처리를 먼저 만들 수 있습니다.
- condition이 `synth,direct`, `noisy,cot`처럼 복합으로 들어오면 쉼표 기준으로 분리합니다.
- condition token을 하나도 쓰지 못한 row는 `direct` condition으로 fallback합니다.
- metadata에 `row_limit`을 저장합니다. 같은 input 파일이라도 cap이 바뀌면 다시 처리됩니다.

현재 기본 cap:

| prefix | rows per epoch |
|---|---:|
| `SYNTH__` | 20,000 |
| `flan__cot_` | unlimited |
| `flan__` | 5,000 |
| `dmmath__` | 100,000 |
| `ampsmathematica__` | 10,000 |
| `tasksource__` | 10,000 |
| `openmathinstruct2__` | 2,000,000 |
| `acereason__` | 2,000,000 |
| `openthoughts2__` | 500,000 |
| `sudoku_extreme__` | 1,000,000 |

이 fast-cap은 full 328G 전처리를 포기한다는 뜻이 아닙니다. GPU를 놀리지 않기 위한 stage-1 입력을 먼저 만들고, full/large stratified retokenization은 별도 stage로 이어갑니다.
