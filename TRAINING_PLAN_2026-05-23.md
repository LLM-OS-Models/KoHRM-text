# HRM-Text 한국어 터미널 모델 학습 계획

작성일: 2026-05-23

최신 데이터 mix 기준 문서: `HRM-Text/PRETRAINING_SFT_DATA_MIX_2026-05-23.md`

## 현재 진행 상태

- 추가 데이터는 전체 다운로드가 아니라 `.data` 아래에 부분 샘플 중심으로 확보했다.
- 현재 추가 데이터 위치는 `/home/work/.data/huggingface/hrm_text_extra`이고 총 약 2.8G다.
- `SWE-ZERO-12M-trajectories`는 터미널/코딩 궤적용으로 1.0G, 53,868 rows를 SFT JSONL로 변환했다.
- `GLM-5.1-Reasoning-1M-Cleaned`는 reasoning SFT 후보로 1.0G, 57,916 rows를 SFT JSONL로 변환했다.
- `structured-wikipedia`는 tokenizer/general knowledge 보조용으로 256M, 18,070 rows를 샘플링했다.
- `claude-opus-4.6-4.7-reasoning-8.7k`는 작아서 raw snapshot 일부/전체 파일을 확보했다.
- `DeepSeek-v4-Pro-Agent`는 파일 수가 4,000개 이상이라 HF 비인증 rate limit에 걸렸다. 일부 파일은 받아졌지만, 전체 강행 대신 샘플링 또는 HF token 사용 후 재시도한다.
- 2026-05-23 현재 GPU는 H200 8장, 각 약 143GB VRAM free 상태다.
- Docker는 사용할 수 없다는 조건으로 진행한다. 따라서 Docker `--shm-size` 확장 대신 `.data` 경로를 학습/전처리 working directory로 쓴다.
- `/dev/shm`는 2GB뿐이므로 HRM-Text README의 `/dev/shm/sampled` 기본값은 사용하지 않는다.

## 결론

목표는 `HRM-Text` 구조를 유지하되, 한국어와 터미널/툴콜/코딩을 잘 처리하는 새 모델을 처음부터 학습하는 것이다. 기존 `sapientinc/HRM-Text-1B` 가중치를 그대로 이어 쓰는 방식은 추천하지 않는다. 토크나이저를 바꾸면 embedding/lm head가 바뀌므로 실질적으로 새 pre-training이다. 대신 HRM-Text의 아키텍처, PrefixLM objective, FSDP2 학습 코드, SFT 전처리 코드는 그대로 재사용한다.

가장 중요한 판단은 raw text를 많이 붓는 것이 아니라, HRM-Text 논문 방식에 맞춰 `instruction -> response` 형식의 task-completion 데이터로 바꾸는 것이다. 한국어 법령/판례도 단순 원문 예측보다 요약, 조항 검색, 근거 추출, 질의응답, 형식 변환 작업으로 만들고, 터미널/코딩 데이터도 한 턴 또는 다중 턴 실행 궤적을 다음 행동 예측 형태로 만들어야 한다.

## 근거

### HRM-Text 논문에서 따라야 할 점

- HRM-Text는 일반적인 raw text next-token pretraining이 아니라 instruction-response pair만 사용한다.
- loss는 response 토큰에만 걸고, instruction 구간은 PrefixLM attention으로 양방향 참조한다.
- 데이터는 pooled uniform sampling이 아니라 dataset/task 단위 stratum으로 cap을 둔다.
- explicit `<think>...</think>` 긴 추론문은 pre-training에서 제거해 내부 recurrent computation을 쓰도록 했다.
- 공개 1B 기준은 BPE 65,536 vocab, context 4096, global batch 196,608 tokens, Adam-atan2, EMA 0.9999, 16 H100 약 46시간이다.

### 한국어 국가대표 모델/리포트에서 배울 점

- HyperCLOVA X는 한국어, 영어/다국어, 코드가 거의 균형을 이루도록 구성했고, 한국어 비중을 약 1/3까지 올렸다. 또한 한국어의 교착어 특성을 반영해 morpheme-aware byte-level BPE 100K를 사용했다.
- Kanana는 고품질 필터링, staged pre-training, depth up-scaling, pruning/distillation을 강조했고, function calling은 domain pre-training 후 Korean-specific SFT를 거치는 2단계 방식을 썼다.
- Mi:dm K 2.5 Pro는 한국어 신뢰성, 코드/agentic 데이터, multi-turn tool-use를 별도 축으로 만들고, MCP JSON Schema 형태의 도구 명세와 다양한 system prompt 포맷을 섞었다.
- K-EXAONE은 tokenizer를 100K에서 150K로 키우고, 기존 고빈도 어휘를 유지하면서 한국어/STEM/code에 capacity를 재배분했다. Unicode는 NFKC보다 NFC를 사용해 code/STEM 기호의 의미 손상을 줄였다.
- EEVE-Korean은 영어 중심 tokenizer의 한국어 비효율을 vocabulary expansion으로 줄였지만, 우리는 tokenizer를 바꾸고 처음부터 학습하므로 확장보다 새 BPE 학습이 더 깔끔하다.

## 현재 보유 데이터

| 분류 | 위치 | 용량/상태 | 용도 |
|---|---|---:|---|
| HRM-Text cleaned base | `/home/work/.data/huggingface/hub/datasets--sapientinc--HRM-Text-data-io-cleaned-20260515/` | 약 326GiB | 영어 instruction/reasoning base |
| HRM base 중 FLAN | 위 경로 `data_clustered/flan` | 약 271GiB | 일반 instruction |
| HRM base 중 SYNTH | 위 경로 `data_clustered/SYNTH` | 약 36GiB | rewritten knowledge |
| 한국어 위키 | `/home/work/.data/huggingface/kowiki-20260501-pages-articles-multistream.xml*` | 압축 1.3G, 해제 약 5G 예상 | 한국어 일반 지식/tokenizer |
| 한국 법령 | `HRM-Text/legalize-kr/` | 약 675M | 법률 QA/요약/근거추출 |
| 한국 조례 | `HRM-Text/ordinance-kr/` | 약 3.2G | 행정/지역 법규 |
| 행정규칙 | `../admrule-kr/` | 약 523M | 행정 문서 |
| 판례 | `../precedent-kr/` | 약 3.0G | 판례 QA/요약 |
| ToolBench | `HRM-Text/data_toolbench/data/` | 약 4.1G | tool-use SFT/pretrain |
| NVIDIA Terminal Dataset | `../dataset/` | 약 5.0G | terminal/coding adapter data |
| Terminal Bench 2 / tb2_lite | `../terminal-bench-2/`, `../tb2_lite/` | 평가용 | train 금지, contamination 방지 |

현재 `/dev/shm`는 2GB뿐이다. Docker도 사용할 수 없으므로 README의 `/dev/shm/sampled` 기본값은 쓰지 않는다. 실제 학습에서는 `/home/work/.data/hrm_text_prepared/...` 같은 `.data` 경로를 `data.path`로 지정한다.

## 추가 데이터 다운로드 정책

작은 데이터는 전체 다운로드하고, 큰 데이터는 먼저 100MB~1GB 샘플만 받는다. 전체 다운로드는 품질이 확인된 뒤에만 한다.

| 데이터셋 | 전체 크기 | 판단 | 다운로드 정책 |
|---|---:|---|---|
| `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k` | 0.24GB | 작고 품질 높은 reasoning/coding SFT 후보 | 전체 다운로드 |
| `TeichAI/DeepSeek-v4-Pro-Agent` | 0.26GB | agent trace 후보. 라이선스 미기재라 공개 모델 학습에는 주의 | 전체 다운로드, research-only로 표시 |
| `actava/chi-bench` | 0.17GB | healthcare agent benchmark 성격 | 전체 다운로드, 기본은 평가용 |
| `TuringEnterprises/Open-MM-RL` | 0.03GB | multimodal 중심. text-only 모델에는 낮은 우선순위 | 전체 다운로드하되 학습 비중 낮게 |
| `AlienKevin/SWE-ZERO-12M-trajectories` | 33.5GB | 터미널/코딩 궤적 핵심 후보 | 우선 1GB SFT JSONL 샘플 |
| `Jackrong/GLM-5.1-Reasoning-1M-Cleaned` | 29.6GB | reasoning distillation 후보 | 우선 1GB SFT JSONL 샘플 |
| `wikimedia/structured-wikipedia` | 67.6GB | structured knowledge. 이미 한국어 위키가 있어 낮은 우선순위 | tokenizer용 256MB 샘플 |

다운로드 스크립트:

```bash
python HRM-Text/scripts/download_extra_training_data.py \
  --out-dir /home/work/.data/huggingface/hrm_text_extra \
  --swe-mib 1024 \
  --glm-mib 1024 \
  --wiki-mib 256
```

현재 확보된 샘플:

| 파일 | 크기 | rows | 역할 |
|---|---:|---:|---|
| `/home/work/.data/huggingface/hrm_text_extra/sft/swe_zero_terminal_sft_sample.jsonl` | 1.0G | 53,868 | 터미널/코딩 다음 행동 SFT |
| `/home/work/.data/huggingface/hrm_text_extra/sft/glm_5_1_reasoning_sft_sample.jsonl` | 1.0G | 57,916 | reasoning SFT/검증용 |
| `/home/work/.data/huggingface/hrm_text_extra/tokenizer_corpus/structured_wikipedia_en_sample.jsonl` | 256M | 18,070 | tokenizer/general knowledge 보조 |

## 토크나이저 계획

권장 1차 목표는 custom BPE 131,072 vocab이다. 1B 모델에서 151K도 가능하지만 embedding/lm head가 커져 메모리와 학습량을 더 먹는다. 131K로 한국어/터미널 효율이 충분한지 먼저 검증하고, 부족하면 151,936으로 올린다.

설계 원칙:

- Unicode normalization은 NFC를 쓴다. 코드, 수식, 법률 문서의 기호 의미를 보존한다.
- byte-level BPE를 사용해 unknown 문자를 안전하게 처리한다.
- 한국어 법률/일반어, 영어, 코드, 터미널, JSON/tool-call 포맷을 모두 tokenizer corpus에 넣는다.
- tokenizer corpus의 비중과 model pre-training 비중은 반드시 분리한다. tokenizer는 한국어와 코드/터미널을 일부러 과대표집해 압축률을 확보하고, 모델 학습은 HRM-Text general/reasoning 기반을 유지하면서 도메인 능력을 얹는다.
- HRM-Text의 기존 condition/control token은 반드시 유지한다. `prepare_sft_data.py` 기본값이 `direct=<|object_ref_start|>`, `cot=<|object_ref_end|>`, `noisy=<|quad_start|>`, `synth=<|quad_end|>`에 의존한다.

1차 tokenizer corpus 이상 비중:

| bucket | 비중 | 예시 |
|---|---:|---|
| 한국어 일반/법률 | 35~40% | kowiki, 법령, 조례, 행정규칙, 판례 |
| 영어/general instruction | 20~25% | HRM FLAN/SYNTH 일부, Claude/GLM 일부 |
| 코드/터미널/SWE | 20~25% | NVIDIA Terminal, SWE-ZERO, shell logs, Python/code |
| tool-call/JSON/API | 10~15% | ToolBench, DeepSeek agent traces, function schema |
| 수학/STEM/reasoning | 5~10% | HRM math, GLM reasoning 일부 |

이 비중은 “학습 텍스트 양”의 이상값이지 최종 vocab allocation을 강제하는 값은 아니다. BPE가 실제 빈도에 따라 병합을 만들기 때문에, 소스별 cap과 샘플링 순서가 중요하다. 단순히 `legalize-kr`, `ordinance-kr`부터 2GB를 읽으면 한국어 법률이 tokenizer를 지배하므로, 현재 스크립트는 `--max-mib-per-input`으로 top-level input별 cap을 둔다.

훈련 스크립트:

```bash
python HRM-Text/scripts/train_koterm_tokenizer.py \
  --input \
    HRM-Text/legalize-kr \
    HRM-Text/ordinance-kr \
    admrule-kr \
    precedent-kr \
    dataset \
    HRM-Text/data_toolbench/data \
    /home/work/.data/huggingface/hrm_text_extra/sft \
    /home/work/.data/huggingface/hrm_text_extra/tokenizer_corpus \
    /home/work/.data/huggingface/hrm_text_extra/raw/angrygiraffe__claude-opus-4.6-4.7-reasoning-8.7k \
  --output-dir /home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1 \
  --vocab-size 131072 \
  --max-gib 2.5 \
  --max-mib-per-input 256
```

검증:

```bash
python HRM-Text/scripts/check_tokenizer_efficiency.py \
  /home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1/tokenizer.json
```

## Pre-training 데이터 구성

HRM-Text 방식에 맞춘 40B unique token급 이상 비중:

| bucket | 권장 비중 | 설명 |
|---|---:|---|
| HRM 원본 general/reasoning | 40~50% | 일반 instruction, rewritten knowledge, math/symbolic 유지 |
| 한국어 일반/법률 task | 20~25% | 원문 그대로가 아니라 QA/요약/근거추출/조항변환 |
| 코드/터미널/SWE trajectory | 15~20% | 다음 명령 예측, 테스트 실패 수정, shell 출력 해석 |
| tool/function calling | 7~12% | ToolBench, DeepSeek agent, 한국어 function calling |
| extra reasoning/STEM | 5~8% | GLM/Claude reasoning 일부. `<think>`는 pretrain에서는 제거하거나 `cot` bucket으로 분리 |

터미널/툴콜 특화가 더 중요하면 코드/터미널을 25%까지 올리고 HRM 원본을 40% 근처로 낮춘다. 한국어 법률만 과하게 올리는 것은 피한다. 모델 목표가 “한국어 터미널 툴콜”이지 “법률 전용 QA”가 아니기 때문이다.

pre-training에서의 구체적 해석:

- 한국어 20~30%는 괜찮지만, 법률/판례 raw text만으로 채우면 안 된다.
- 한국어 bucket 안에서도 일반 위키/상식, 법령, 판례, 행정 문서, 한국어 터미널/툴콜 지시를 나눠야 한다.
- 코드/터미널 15~20%는 최소치다. 터미널 모델을 명확히 목표로 하면 20~25%까지 올린다.
- tool/function calling 7~12%는 JSON 구조 정확도와 도구 선택 정확도를 위한 별도 bucket으로 유지한다.
- GLM/Claude식 긴 reasoning은 그대로 대량 투입하지 않는다. HRM-Text 논문처럼 pre-training에서는 `<think>`를 제거하거나, 명시적 reasoning이 필요한 샘플만 `cot` condition에 둔다.
- Terminal Bench 2, `tb2_lite`, chi-bench 같은 평가 데이터는 train에 섞지 않는다.

## SFT 구성

SFT는 pre-training 뒤에 별도로 한다.

| bucket | 우선순위 | 처리 |
|---|---:|---|
| NVIDIA Terminal Dataset | 최상 | 터미널 명령, 출력 해석, 수정 루프 |
| SWE-ZERO sample | 최상 | assistant turn별 다음 행동 예측 샘플 |
| ToolBench | 높음 | API 선택/argument JSON 생성 |
| DeepSeek agent traces | 중상 | 라이선스 확인 전 research-only |
| 한국어 function calling 합성 | 높음 | Kanana/Mi:dm 방식처럼 한국어 tool-use 부족분 보강 |
| 한국 법률 QA/요약 | 중간 | 도메인 신뢰성 보강 |
| Claude/GLM reasoning | 중간 | 과다 사용 금지. 긴 `<think>`는 분리 |

SFT 전처리:

```bash
python HRM-Text/scripts/prepare_sft_data.py \
  --train /home/work/.data/huggingface/hrm_text_extra/sft/swe_zero_terminal_sft_sample.jsonl \
  --tokenizer /home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1/tokenizer.json \
  --output /home/work/.data/hrm_text_prepared/sft_swe_zero_v0 \
  --epochs 5 \
  --context-size 4097
```

## 학습 실행 판단

현재 GPU는 H200 8장, 각 약 143GB VRAM이다. 메모리만 보면 L(0.6B)은 충분하고, XL(1B)도 단일 8장 FSDP2로 시도할 수 있다. 다만 논문 기준 XL 1B는 16 H100 46시간이므로, 8 H200에서는 같은 40B급 학습을 할 때 대략 70~100시간 범위를 예상하는 것이 보수적이다. L 0.6B는 8 H100 50시간 기준이므로 H200 8장에서는 35~55시간 정도가 현실적이다. Docker는 사용하지 않고, 모든 전처리 산출물과 체크포인트는 `.data` 아래에 둔다.

배치 크기 방침:

- H200 8장의 VRAM을 최대한 쓰되, 처음부터 무리한 global batch로 장기 학습을 시작하지 않는다.
- L 크기는 논문 기준 `global_batch_size=172032`에서 시작하고, OOM이 없으면 `196608`, `229376`, `262144` 순서로 올려본다.
- XL 크기는 논문 기준 `global_batch_size=196608`에서 시작하고, OOM이 없으면 `229376`, `262144`를 probe한다.
- batch probe는 짧은 파일럿 데이터로 먼저 수행한다. 안정 batch가 정해진 뒤 장기 학습에 들어간다.
- `checkpoint_interval=1`을 유지해 epoch마다 저장한다. step마다 저장하거나 업로드하지 않는다.

추천 순서:

1. `B` 또는 `L` 크기로 1B~3B token 파일럿을 돌려 데이터/토크나이저/PrefixLM 포맷을 검증한다.
2. TB2-lite, 한국어 JSON/tool-call smoke eval, tokenizer efficiency를 본다.
3. 문제가 없으면 L 0.6B 40B급으로 확장한다.
4. L에서 터미널/한국어 지표가 충분히 나오면 XL 1B로 간다.

실행 예시:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun --nproc_per_node=8 HRM-Text/pretrain.py \
  arch/size@arch=L \
  data.path=/home/work/.data/hrm_text_prepared/pretrain_koterm_v0 \
  +checkpoint_path=/home/work/.data/hrm_text_checkpoints/koterm_l_v0 \
  global_batch_size=172032
```

주의: 현재 `data_io` companion pipeline이 이 저장소 안에 포함되어 있지 않다. full pre-training용 binary `tokens.npy + epoch_* indices + metadata.json`를 만들려면 `sapientinc/data_io` 또는 동일 포맷 변환기가 필요하다. SFT는 `scripts/prepare_sft_data.py`로 바로 만들 수 있다.

### 2026-05-23 파일럿 결과 반영

실제로 아래 end-to-end pilot이 완료됐다.

| 항목 | 결과 |
|---|---:|
| dataset | `/home/work/.data/hrm_text_prepared/sft_swe_glm_mix_v1` |
| samples | 109,889 |
| tokens | 251,170,780 |
| arch | B |
| params | 435,159,040 |
| GPUs | 8 x H200 |
| global batch | 262,144 tokens |
| wall time | 약 7분 38초 |
| peak VRAM | 장당 약 38GB |
| final loss | 3.00653 |
| final token accuracy | 0.46379 |
| checkpoint | `/home/work/.data/hrm_text_checkpoints/koterm_b_swe_glm_pilot_v1` |
| upload | `https://huggingface.co/LLM-OS-Models/HRM-Text-Ko-Terminal-B-SWE-GLM-Pilot` |

판단:

- 데이터 포맷, tokenizer, PrefixLM/response-only loader, FA3, FSDP2 save path가 모두 동작한다.
- B 크기에서는 H200 VRAM이 많이 남으므로 실제 학습은 L 또는 XL로 올려야 한다.
- pilot mix는 SWE가 token 기준 약 72.7%라 terminal/code smoke 용도로는 좋지만 최종 pretraining mix로는 과하다. 장기 학습 전 한국어 일반/법률 instruction과 tool-call/JSON 데이터를 더 섞어야 한다.
- `scripts/merge_prepared_sft_data.py`를 추가했으므로, 여러 prepared V1Dataset을 재토큰화 없이 합칠 수 있다.

## Hugging Face 업로드 정책

`.env`에는 `HF_TOKEN`이 있다. 토큰 값은 로그에 출력하지 않는다.

업로드는 너무 자주 하지 않는다:

- tokenizer 검증이 통과하면 tokenizer artifact를 1회 업로드한다.
- 파일럿 학습은 완료 checkpoint만 업로드한다.
- 장기 학습은 epoch checkpoint 단위로만 업로드한다.
- 최종 선택 checkpoint는 `conversion/convert_to_hf.py`로 model-only HF 형식으로 변환한 뒤 별도 업로드한다.
- network/HF 오류가 학습 프로세스를 죽이지 않도록 업로드는 별도 watcher 프로세스에서 수행한다.

사용 스크립트:

```bash
# 작은 artifact/tokenizer 업로드
python HRM-Text/scripts/upload_folder_to_hf.py \
  --folder /home/work/.data/huggingface/trained_tokenizers/hrm-ko-terminal-131k-v1 \
  --repo-id LLM-OS-Models/HRM-Text-Ko-Terminal-Tokenizer-131K

# epoch checkpoint watcher
python HRM-Text/scripts/watch_and_upload_hrm_checkpoints.py \
  --checkpoint-root /home/work/.data/hrm_text_checkpoints/koterm_l_v0 \
  --repo-id LLM-OS-Models/HRM-Text-Ko-Terminal-L-Pilot \
  --stage-root /home/work/.data/hrm_text_hf_upload_stage \
  --poll-seconds 300 \
  --stable-seconds 120
```

raw FSDP2 checkpoint 업로드는 복구/모니터링용이다. 실제 배포용 모델은 최종 checkpoint를 HF 형식으로 변환한 산출물만 업로드한다.

## 평가

평가는 train contamination을 피해서 분리한다.

- `tb2_lite`: 지금까지 쓰던 terminal JSON replay score.
- `terminal-bench-2`: 실제 터미널 과제.
- `chi-bench`: healthcare agent/tool benchmark라 train이 아니라 eval 후보.
- 한국어 tool-call: FunctionChat-Bench류 또는 자체 한국어 MCP JSON schema benchmark.
- tokenizer: 한국어/터미널/JSON/code chars-per-token, special token 1-token 여부.

## 바로 할 일

1. HRM cleaned 328G를 새 tokenizer로 재토큰화해서 사전학습 mix에 포함.
2. 한국어 위키/법령/조례/행정규칙/판례를 instruction-response task로 변환.
3. ToolBench와 local terminal dataset을 tool/terminal task로 변환.
4. SWE/GLM prepared dataset까지 포함한 balanced pretraining dataset 생성.
5. H200 8장 기준 L/XL batch size probe.
6. 안정 batch 확인 후 장기 학습 및 epoch 단위 HF 업로드 watcher 실행.
