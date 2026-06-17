# KoHRM-Text-1.4B 변호사시험 SFT 런북 (2026-06-15)

KoHRM-Text-1.4B (`LLM-OS-Models/KoHRM-Text-1.4B`) 베이스에서 변호사시험 선택형 기출문제로 Full-SFT 하고, 15회 변호사시험으로 평가한 결과를 정리한다. 이어서 `gyung/korean-bar-exam-hard-current-law-precedent-sft-1000` 로 추가 Full-SFT 하고 15회로 다시 평가한 결과까지 포함한다.

## TL;DR

| run | base | SFT 데이터 | epoch | 최종 train loss | condition | 15회 정확도 | 파싱률 |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| Base | KoHRM-Text-1.4B (stage4d PT) | — | — | — | direct | **13.1 %** (19/145) | 61.4 % |
| Run A | KoHRM-Text-1.4B | 1–14회 변호사시험 (2,027문항) | 2 | 0.324 | direct | **26.9 %** (39/145) | 100 % |
| Run B | Run A ckpt | hard-current-law-precedent-sft-1000 | 2 | 0.243 | cot | **20.0 %** (29/145) | 100 % |
| Run B' | Run A ckpt | hard-current-law-precedent-sft-1000 | 2 | 0.243 | direct | **22.1 %** (32/145) | 100 % |

무작위 기준 약 20 %.

- Base는 프롬프트 형식을 못 알아들어서 정답 번호가 아니라 해설/조문 텍스트를 그대로 출력 → 파싱 실패 38.6 %, 정확도 13.1 %.
- Run A는 SFT만으로 정답 형식을 학습해 **26.9 %**까지 오름. 모든 과목에서 random(20 %)을 넘김.
- Run B (hard-1000 추가 SFT)는 cot/direct 모두에서 Run A 대비 하락. "정답: X" 다음 바로 eos가 와서 해설까지 가지 못하고 정답 라벨 분포만 모방.

자세한 원인과 개선 방향은 [결과 분석](#결과-분석)에서 다룬다.

## 환경

```text
machine:    H200 1장 (index 7)
framework:  pretrain.py (torchrun --nproc_per_node=1) + simple_inference_engine.py
config:     config/cfg_sft.yaml  (ema=0.999, lr_min_ratio=0.1, bp_warmup_ratio=0.0)
tokenizer:  /home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1/tokenizer.json
chat layout (KoHRM PrefixLM, ChatML-ish special tokens):
    <boq><condition_token><instruction><eoq><response><eoa>
    boq        = <|im_start|>   (id 2)
    eoq        = <|im_end|>     (id 3)
    eoa        = <|box_end|>    (id 35, eos)
    direct ->  <|object_ref_start|>   (id 32)
    cot    ->  <|object_ref_end|>     (id 33)
    noisy   -> <|quad_start|>        (id 36)
    synth   -> <|quad_end|>          (id 37)
```

KoHRM은 별도 chat template 없이 PrefixLM 방식으로 동작한다. instruction 영역은 양방향 attention, response 영역만 causal + loss. tokenizer_config.json의 chat_template 필드는 비어있고, `simple_inference_engine.tokenize_prompt()`가 boq + condition 토큰 + instruction + eoq 순서로 token id를 만든다.

## 데이터

### 1. 변호사시험 기출 (train/eval)

- 원본: `huggingface.co/datasets/gyung/korean-bar-exam-moj-multiple-choice` 의 `data/questions.csv` (8.8MB, 2,250문항)
- 로컬 복사: `Terminal/HRM-Text/data/bar_exam/processed_multiple_choice/questions.csv` (md5 동일)
- 분포: 15개 회차 × 회차당 150문항. 과목 = 공법 / 민사법 / 형사법
- 회차 분할
  - train: 1–14 회 → 2,027 문항 (조합형/결측 73건 제외)
  - eval: 15 회 → 145 문항 (조합형/결측 5건 제외)
- 정답 형식: 단일 보기는 `1..5`, 드물게 `10/20/30/40/50/60/70` (두 자리 조합), `정답없음` 드물음

### 2. hard-current-law-precedent-sft-1000 (추가 SFT용)

- 원본: `huggingface.co/datasets/gyung/korean-bar-exam-hard-current-law-precedent-sft-1000` 의 `sft/train.jsonl` (5.97MB, 1,000샘플)
- 구조: ChatML-style `messages` (system + user + assistant) + `references` 메타데이터
- assistant 응답 순서 (변경 금지 — 첫 토큰이 정답 번호):
  ```text
  정답: <번호>

  해설: 정답은 <번호>번이다. ㄱ은 옳다/옳지 않다. ... ㄴ은 ... ㄷ은 ... ㄹ은 ...

  참고 법령: <법령1>(url); <법령2>(url); ...
  ```
- 과목 분포: 공법 270 / 민사법 460 / 형사법 270

### SFT JSONL 빌드

두 데이터 모두 KoHRM 표준 instruction/response JSONL로 변환한다 (`condition` 필드 포함).

```bash
# 1) 변호사시험 1-14회 train / 15회 eval
python scripts/build_bar_exam_sft_data.py \
  --input data/bar_exam/processed_multiple_choice/questions.csv \
  --output /home/work/.data/bar_exam_sft/raw/bar_exam_train_1_14.jsonl \
  --rounds 1-14

python scripts/build_bar_exam_sft_data.py \
  --input data/bar_exam/processed_multiple_choice/questions.csv \
  --output /home/work/.data/bar_exam_sft/raw/bar_exam_eval_15.jsonl \
  --rounds 15-15

# 2) hard-1000 (messages -> instruction/response flatten)
python scripts/build_bar_exam_hard_sft_data.py \
  --input /home/work/.data/bar_exam_sft/raw_hard/sft/train.jsonl \
  --output /home/work/.data/bar_exam_sft/raw/bar_exam_hard_1000.jsonl
```

빌드 결과:

| 파일 | rows | 비고 |
| --- | ---: | --- |
| `bar_exam_train_1_14.jsonl` | 2,027 | condition=direct, response="정답: X" |
| `bar_exam_eval_15.jsonl` | 146 | 평가 전용 (SFT 입력에 미사용) |
| `bar_exam_hard_1000.jsonl` | 1,000 | condition=cot, response="정답: X / 해설 / 참고 법령" |

### V1Dataset 변환 (pretrain.py 입력 포맷)

`scripts/prepare_sft_data.py` 로 tokenized binary 포맷을 만든다. 이 파일이 pretrain.py의 `data.path`로 들어간다.

```bash
TOKENIZER=/home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1/tokenizer.json

python scripts/prepare_sft_data.py \
  --train /home/work/.data/bar_exam_sft/raw/bar_exam_train_1_14.jsonl \
  --tokenizer "$TOKENIZER" \
  --output /home/work/.data/bar_exam_sft/prepared/bar_exam_1_14 \
  --epochs 2 \
  --seed 20260615 \
  --context-size 4097 \
  --overflow-policy truncate-instruction-middle \
  --truncate-head-tokens 1024 \
  --condition-override direct

python scripts/prepare_sft_data.py \
  --train /home/work/.data/bar_exam_sft/raw/bar_exam_hard_1000.jsonl \
  --tokenizer "$TOKENIZER" \
  --output /home/work/.data/bar_exam_sft/prepared/bar_exam_hard_1000 \
  --epochs 2 \
  --seed 20260615 \
  --context-size 4097 \
  --overflow-policy truncate-instruction-middle \
  --truncate-head-tokens 1024 \
  --strip-think-blocks
```

| prepared dir | rows | tokens | avg | max |
| --- | ---: | ---: | ---: | ---: |
| `bar_exam_1_14` | 2,027 | 778,769 | 384 | 1,160 |
| `bar_exam_hard_1000` | 1,000 | 594,803 | 595 | 865 |

## SFT 실행 (GPU 7 단일 H200)

`scripts/run_bar_exam_sft_1gpu.sh` 가 `torchrun --nproc_per_node=1` 로 `cfg_sft.yaml` 을 로드한다. 기존 `run_kohrm_full_sft_dual_4gpu.sh`와 같은 hydra 인터페이스에 GPU 수만 1로 줄인 형태.

```bash
# Run A: base -> 1-14 SFT
BAR_RUN_NAME=bar_exam_runA_1_14_ep2 \
BAR_RESUME_FROM=/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-stage4d-korean-tool-finance-repeat3-gbs180 \
BAR_PREPARED=/home/work/.data/bar_exam_sft/prepared/bar_exam_1_14 \
BAR_OUT=/home/work/.data/bar_exam_sft/ckpts/bar_exam_runA_1_14_ep2 \
BAR_EPOCHS=2 BAR_GBS=4096 BAR_LR=3.0e-5 \
BAR_LR_WARMUP_STEPS=20 BAR_SAVE_STEPS=200 BAR_KEEP_LAST=2 \
BAR_GPU=7 BAR_PORT=29677 \
bash scripts/run_bar_exam_sft_1gpu.sh

# Run B: Run A ckpt -> hard-1000 추가 SFT
BAR_RUN_NAME=bar_exam_runB_hard_1000_ep2 \
BAR_RESUME_FROM=/home/work/.data/bar_exam_sft/ckpts/bar_exam_runA_1_14_ep2 \
BAR_PREPARED=/home/work/.data/bar_exam_sft/prepared/bar_exam_hard_1000 \
BAR_OUT=/home/work/.data/bar_exam_sft/ckpts/bar_exam_runB_hard_1000_ep2 \
BAR_EPOCHS=2 BAR_GBS=4096 BAR_LR=2.0e-5 \
BAR_LR_WARMUP_STEPS=10 BAR_SAVE_STEPS=100 BAR_KEEP_LAST=2 \
BAR_GPU=7 BAR_PORT=29678 \
BAR_WEIGHTS_ONLY_RESUME_FROM_EMA=false \
bash scripts/run_bar_exam_sft_1gpu.sh
```

Run B에서 `BAR_WEIGHTS_ONLY_RESUME_FROM_EMA=false` 로 한 이유: Run A가 이미 `weights_only_resume_from_ema=true` 로 EMA 가중치를 model state로 치환해서 저장했기 때문에, Run B에서는 그 model state를 그대로 로드하고 EMA 버퍼를 새로 초기화한다.

### 하이퍼파라미터 요약

| run | base | epochs | GBS (tokens) | LR | warmup | 샘플 수 | 학습 시간 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Run A | stage4d PT | 2 | 4,096 | 3.0e-5 | 20 step | 2,027 | ~5 분 |
| Run B | Run A | 2 | 4,096 | 2.0e-5 | 10 step | 1,000 | ~7 분 |

## 평가 (15회 변호사시험)

`scripts/eval_bar_exam_round15.py` 가 `simple_inference_engine.inference_load_checkpoint` + `inference_generate` 로 greedy 생성을 돌리고, `정답: <번호>` 패턴에서 1..5 값을 파싱해서 gold와 비교한다.

**중요**: `simple_inference_engine` 은 `torch.compile` + `flash_attn` 조합으로 KV append가 현재 hopper 빌드에서 호환되지 않으므로, 평가 시 환경변수로 compile을 끈다.

```bash
# Run A 평가 (direct)
KOHRM_DISABLE_INFERENCE_COMPILE=1 CUDA_VISIBLE_DEVICES=7 python scripts/eval_bar_exam_round15.py \
  --ckpt-path /home/work/.data/bar_exam_sft/ckpts/bar_exam_runA_1_14_ep2 \
  --ckpt-epoch 2 --no-ema \
  --questions-csv /home/work/.data/bar_exam_sft/raw/data/questions.csv \
  --round 15 \
  --output /home/work/.data/bar_exam_sft/results/runA_pred.jsonl \
  --summary-out /home/work/.data/bar_exam_sft/results/runA_summary.json \
  --max-tokens 2048 --max-generation 128 --batch-size 8 --temp 0.0 --condition direct

# Run B 평가 (cot, 해설 응답을 학습했으므로 cot condition 사용)
KOHRM_DISABLE_INFERENCE_COMPILE=1 CUDA_VISIBLE_DEVICES=7 python scripts/eval_bar_exam_round15.py \
  --ckpt-path /home/work/.data/bar_exam_sft/ckpts/bar_exam_runB_hard_1000_ep2 \
  --ckpt-epoch 2 --no-ema \
  --questions-csv /home/work/.data/bar_exam_sft/raw/data/questions.csv \
  --round 15 \
  --output /home/work/.data/bar_exam_sft/results/runB_pred.jsonl \
  --summary-out /home/work/.data/bar_exam_sft/results/runB_summary.json \
  --max-tokens 2048 --max-generation 256 --batch-size 8 --temp 0.0 --condition cot
```

`--no-ema` 를 쓰는 이유: SFT 체크포인트는 이미 `weights_only_resume_from_ema=true` 로 EMA 가중치를 model state_dict로 치환해 저장한다. 평가에서 다시 `ckpt_use_ema=True` 로 불러오면 빈 EMA 버퍼를 swap 하려고 해서 가중치가 깨진다.

`--max-tokens` 는 prompt + generation 총 시퀀스 길이다. 15회 question_text 의 최대 토큰 길이가 약 1,500이므로 2,048 정도면 안전하다.

### 정확도 결과

| run | condition | n_total | n_parsed | n_correct | accuracy | 공법 | 민사법 | 형사법 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | direct | 145 | 89  | 19 | 13.1 % | 17.9 % (7/39)  | 11.9 % (8/67)  | 10.3 % (4/39)  |
| Run A | direct | 145 | 145 | 39 | 26.9 % | 30.8 % (12/39) | 22.4 % (15/67) | 30.8 % (12/39) |
| Run B | cot    | 145 | 145 | 29 | 20.0 % | 20.5 % (8/39)  | 19.4 % (13/67) | 20.5 % (8/39)  |
| Run B' | direct | 145 | 145 | 32 | 22.1 % | 20.5 % (8/39)  | 20.9 % (14/67) | 25.6 % (10/39) |

무작위 기준(5지선다 단일 정답) = 20 %.

Base 샘플 생성 (왜 파싱이 안 되는지 보여주는 예):

```text
gold=4  raw='정답: ㄷ'                                  <- 보기 번호가 아니라 ㄱ/ㄴ/ㄷ 글자
gold=2  raw='[정답: 1] ① 방송통신위원회는 ...'           <- 정답 번호 뒤에 해설이 바로 이어짐
gold=4  raw='[1] 대체복무요원의 정당가입을 금지하는 ...'  <- 정답 표시 없이 판결요지 스타일
```

Run A 샘플 생성 (정답 형식은 맞지만 분포 편향):

```text
gold=3  raw='정답: 40'
gold=1  raw='정답: 40'
```

Run B 샘플 생성 (Run A의 짧은 패턴이 지속):

```text
gold=3  raw='정답: 20'
gold=1  raw='정답: 30'
gold=4  raw='정답: 5'
gold=2  raw='정답: 30'
```

Run B' (direct) 샘플 생성:

```text
gold=3  raw='정답: 5'
gold=4  raw='정답: 5'
```

## 결과 분석

두 run 모두 정확도가 random 수준에 머문다. 생성 응답을 보면 원인이 보인다.

```text
Run A sample:
  raw: '정답: 40'   <- question별로 다른 답이 아니라 특정 패턴 반복
  raw: '정답: 40'

Run B sample:
  raw: '정답: 20'
  raw: '정답: 30'
  raw: '정답: 5'
  raw: '정답: 30'
```

Run A는 "정답: 40" 을 반복 생성하고, Run B는 "정답: 20/30" 같은 두 자리 조합형 정답을 자꾸 출력한다. 즉 모델이 question 본문을 읽고 정답을 추론한다기보다, 정답 라벨의 주변 분포를 모방(imitate)하는 수준에 그친 것이다.

원인 후보:

1. **데이터 부족 + 단순한 응답 형식**
   - 1-14회 데이터는 응답이 `정답: X` 한 줄뿐. 학습 신호가 너무 sparse해서 모델이 입력을 읽을 유인이 적다.
   - hard-1000은 해설이 길지만 1,000샘플로는 기존 분포를 override 하기 부족.
2. **GBS가 너무 작음**
   - 4,096 tokens/step로 2,027샘플 / 778k 토큰을 2 epoch 돌리면 step 수는 380. cosine LR이 충분히 닿기 전에 종료.
3. **KoHRM-Text-1.4B 베이스의 법률 지식 부족**
   - 사전학습 말뭉치에 현행 법령/판례가 제한적일 가능성. 모델이 새 법령 내용을 in-context로 읽고 추론하는 능력이 부족.
4. **응답 길이가 너무 짧게 잘림**
   - hard-1000 학습 데이터의 assistant 응답은 평균 800자 이상이지만, 생성 시 `정답: 20` 다음에 바로 eos 토큰이 와서 해설까지 가지 못한다. Run A에서 학습한 "정답: X 만 출력" 패턴이 Run B에서도 지속됨.

개선 방향 (후속 실험 후보):

- **응답 형식 확장**: 1-14 회 SFT 데이터도 `정답: X\n근거: 조문/사실관계` 형식으로 빌드. 정답 한 글자보다 긴 응답이 모델에게 question 이해를 강제.
- **epoch/GBS 키우기**: GBS=8192 이상, epoch=4~8. 특히 hard-1000.
- **데이터 순서 섞기**: Run B를 Run A 이어서가 아니라 1-14 + hard-1000을 동시 mix 해서 SFT.
- **베이스 교체**: 사전학습에 법령/판례 코퍼스 추가, 또는 더 큰 베이스(3B 이상) 사용.

## 산출물 위치

```text
데이터
  /home/work/.data/bar_exam_sft/raw/
    bar_exam_train_1_14.jsonl             # 2,027 rows (KoHRM SFT format)
    bar_exam_eval_15.jsonl                # 146 rows (15회, eval-only)
    bar_exam_hard_1000.jsonl              # 1,000 rows
    *.stats.json
  /home/work/.data/bar_exam_sft/raw/data/questions.csv     # 원본
  /home/work/.data/bar_exam_sft/raw_hard/sft/train.jsonl   # hard-1000 원본

V1Dataset (pretrain.py 입력)
  /home/work/.data/bar_exam_sft/prepared/bar_exam_1_14/
  /home/work/.data/bar_exam_sft/prepared/bar_exam_hard_1000/

체크포인트
  /home/work/.data/bar_exam_sft/ckpts/bar_exam_runA_1_14_ep2/
    fsdp2_epoch_2/   carry_epoch_2.0.pt   all_config.yaml
  /home/work/.data/bar_exam_sft/ckpts/bar_exam_runB_hard_1000_ep2/
    fsdp2_epoch_2/   carry_epoch_2.0.pt   all_config.yaml

평가 결과
  /home/work/.data/bar_exam_sft/results/
    base_pred.jsonl            base_summary.json
    runA_pred.jsonl            runA_summary.json
    runB_pred.jsonl            runB_summary.json            (cot condition)
    runB_pred_direct.jsonl     runB_summary_direct.json     (direct condition)

학습 로그
  /home/work/.data/hrm_text_logs/bar_exam_runA_1_14_ep2.log
  /home/work/.data/hrm_text_logs/bar_exam_runB_hard_1000_ep2.log

스크립트
  scripts/build_bar_exam_sft_data.py          # 1-14 / 15 jsonl 빌더
  scripts/build_bar_exam_hard_sft_data.py     # hard-1000 jsonl 빌더
  scripts/run_bar_exam_sft_1gpu.sh            # 단일 GPU용 SFT 러너
  scripts/eval_bar_exam_round15.py            # 15회 평가 스크립트
  scripts/debug_inference_smoke.py            # 단일 프롬프트 디버그

러너 설정 (cfg_sft.yaml 기본값 + CLI override)
  config/cfg_sft.yaml                         # ema=0.999, lr_min_ratio=0.1
  config/cfg_pretrain.yaml
  pretrain.py                                 # hydra entry (--config-name=cfg_sft)
  simple_inference_engine.py                  # KoHRM PrefixLM greedy generator
```
