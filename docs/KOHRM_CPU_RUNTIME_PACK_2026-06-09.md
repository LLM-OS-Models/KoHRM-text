# KoHRM-Text CPU Runtime Pack

작성일: `2026-06-09`

## 결론

`LLM-OS-Models/KoHRM-Text-1.4B`는 현재 GGUF로 바로 만들 수 없다.

이유는 모델 구조가 일반 Llama/Qwen/Gemma 계열이 아니라 아래 전용 구조이기 때문이다.

```text
model_type: hrm_text
architectures: HrmTextForCausalLM
H_cycles: 2
L_cycles: 3
prefix_lm: true
```

llama.cpp 변환기로 직접 시도하면 다음 지점에서 막힌다.

```text
ERROR:hf-to-gguf:Model HrmTextForCausalLM is not supported
```

따라서 지금 현실적인 CPU 경로는 GGUF가 아니라 PyTorch 전용 runtime이다.

## 추가한 파일

```text
HRM-Text/inference/kohrm_cpu_runtime.py
HRM-Text/inference/requirements-cpu.txt
HRM-Text/scripts/upload_kohrm_cpu_runtime_pack.py
```

이 runtime은 기존 `HRM-Text/notebooks/kohrm_colab_generate.py`의 safetensors 직접 로딩 경로를 재사용하고, CPU용 양자화와 H/L cycle override를 추가한다.

## 사용법

기본 권장값은 `dynamic-int8`이다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 \
python HRM-Text/inference/kohrm_cpu_runtime.py \
  --model LLM-OS-Models/KoHRM-Text-1.4B \
  --quant dynamic-int8 \
  --prompt "리눅스에서 현재 디렉토리 파일 목록을 보는 명령어는?" \
  --max-new-tokens 128 \
  --max-seq-len 768 \
  --temperature 0
```

16GB CPU RAM 환경에서는 아래 순서로 쓰면 된다.

```text
1순위: dynamic-int8
2순위: none
3순위: weight-int4
```

`dynamic-int8`은 PyTorch CPU dynamic quantization을 사용한다. 일반적으로 메모리와 속도 균형이 가장 낫다.

`weight-int4`는 직접 구현한 portable 4bit weight-only fallback이다. 메모리는 줄지만 매 forward마다 unpack/dequantize가 들어가서 매우 느리다. “반드시 작은 메모리로 돌아가야 한다”는 경우에만 쓴다.

## H/L cycle override

KoHRM은 같은 H/L module을 반복 적용한다. 기본은 `H=2`, `L=3`이다.

CPU에서는 아래처럼 반복 횟수를 줄여 속도를 올릴 수 있다.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 \
python HRM-Text/inference/kohrm_cpu_runtime.py \
  --model LLM-OS-Models/KoHRM-Text-1.4B \
  --quant dynamic-int8 \
  --h-cycles 1 \
  --l-cycles 1 \
  --prompt "리눅스에서 현재 디렉토리 파일 목록을 보는 명령어는?" \
  --max-new-tokens 128 \
  --max-seq-len 768 \
  --temperature 0
```

주의할 점은 명확하다.

- `H=2,L=3`: 원래 품질 경로.
- `H=1,L=1`: CPU 속도 우선 경로.
- cycle을 줄이면 품질이 떨어질 수 있다.

## Smoke test 결과

같은 짧은 prompt, `max_new_tokens=4`, `max_seq_len=128`, `OMP_NUM_THREADS=8` 기준이다.

```text
none:
  elapsed: 1.48s
  speed:   2.69 tok/s
  cycles:  H=2, L=3

dynamic-int8:
  elapsed: 0.53s
  speed:   7.59 tok/s
  cycles:  H=2, L=3

dynamic-int8 + H=1,L=1:
  elapsed: 0.24s
  speed:   8.18 tok/s
  cycles:  H=1, L=1

weight-int4:
  elapsed: 23.25s
  speed:   0.17 tok/s
  cycles:  H=2, L=3
```

짧은 smoke test라 절대 성능 숫자는 참고용이다. 하지만 방향은 분명하다.

```text
실사용: dynamic-int8
메모리 강제 절약: weight-int4
품질 유지: H=2,L=3
속도 우선: H=1,L=1
```

## 왜 GGUF가 어려운가

GGUF 파일은 단순히 weight를 담는 포맷이 아니다. llama.cpp가 해당 architecture의 forward pass를 알아야 한다.

KoHRM은 일반 Transformer block을 한 번씩 쌓는 모델이 아니다.

- H module과 L module이 있다.
- `H_cycles`, `L_cycles`만큼 recurrent하게 반복한다.
- PrefixLM formatting과 stop token 처리가 다르다.
- KV cache 구조도 일반 chat causal LM과 다르다.

따라서 GGUF를 제대로 만들려면 다음 작업이 필요하다.

```text
1. llama.cpp MODEL_ARCH에 HRM_TEXT 추가
2. H/L recurrent forward 구현
3. gqkv gated attention 구현
4. PrefixLM prompt/token boundary 처리
5. tokenizer pre-tokenizer hash 등록
6. quantized tensor name mapping 작성
7. llama-cli generation smoke test
```

단순 converter patch로 끝나는 문제가 아니다.

## HF CPU pack

HF에는 가중치를 중복 업로드하지 않고 CPU runtime pack을 따로 올린다.

대상 repo:

```text
LLM-OS-Models/KoHRM-Text-1.4B-CPU-Runtime
```

이 repo에는 다음만 들어간다.

```text
README.md
inference/kohrm_cpu_runtime.py
inference/requirements-cpu.txt
notebooks/kohrm_colab_generate.py
```

가중치는 실행 시 원본 repo에서 받는다.

```text
LLM-OS-Models/KoHRM-Text-1.4B
```

공용컴 기준으로 HF token은 `.env`에서 읽되 출력하지 않는다.
