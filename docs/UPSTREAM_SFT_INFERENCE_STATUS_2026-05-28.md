# Upstream SFT / vLLM / CPU Support Status - 2026-05-28

이 문서는 현재 repo, 원본 `sapientinc/HRM-Text`, 논문 기준으로 SFT, vLLM, CPU 실행 방법이 있는지 정리합니다.

## 요약

```text
Full SFT 방법:
  있음.

LoRA SFT 방법:
  upstream에는 없음.
  KoHRM fork에서 별도로 추가.

HRM native vLLM 추론:
  아직 없음.

vLLM baseline 평가:
  있음.
  하지만 HRM checkpoint serving이 아니라 HF/vLLM 호환 모델 baseline 평가용.

CPU 실사용 generation:
  없음.
  CPU 변환/무결성 확인은 가능.
```

## SFT

원본 repo에는 full-parameter SFT 방법이 있습니다.

구성:

```text
scripts/prepare_sft_data.py
config/cfg_sft.yaml
pretrain.py --config-name cfg_sft
```

원본 README의 요지는 다음입니다.

```text
1. instruction/response JSONL을 V1Dataset으로 전처리한다.
2. pretrain checkpoint에서 cfg_sft로 continue-train한다.
3. weights_only_resume_from_ema=true를 주면 EMA weight에서 optimizer를 새로 시작한다.
```

KoHRM에서도 이 경로는 유효합니다. 다만 지금은 full SFT보다 LoRA/짧은 SFT smoke를 먼저 권장합니다.

## LoRA

원본 repo에는 LoRA SFT가 없습니다.

KoHRM fork에서 추가한 파일:

```text
models/lora.py
train_lora.py
config/cfg_lora.yaml
scripts/run_kohrm_lora_experiments.sh
```

관련 실행법은 [LORA_TRAINING_GUIDE_2026-05-28.md](LORA_TRAINING_GUIDE_2026-05-28.md)를 봅니다.

## vLLM

원본 repo에는 vLLM baseline 평가 config가 있습니다.

```text
evaluation/config/vllm_benchmarking.yaml
evaluation/engines.py
evaluation/README.md
```

하지만 이것은 Qwen/Ouro/Llama 같은 HF/vLLM 호환 baseline 모델을 평가하기 위한 경로입니다. HRM raw FSDP2 checkpoint를 vLLM으로 바로 serving하는 경로는 아닙니다.

원본 README 기준으로 native vLLM support는 아직 in progress입니다.

따라서 현재 상태:

```text
HRM-Text raw checkpoint -> vLLM serving:
  안 됨.

KoHRM converted safetensors -> vLLM serving:
  아직 안 됨.
  model_type=hrm_text custom architecture에 대한 vLLM/Transformers wrapper가 필요함.
```

## CPU 환경

CPU에서 가능한 것:

```text
conversion/convert_to_hf.py --device cpu
model.safetensors 무결성 확인
tokenizer/config 로드 확인
```

CPU에서 현실적으로 안 되는 것:

```text
HRM raw checkpoint generation
plain AutoModelForCausalLM generation
vLLM serving
```

이유:

```text
simple_inference_engine.py는 CUDA cache와 FlashAttention 기반 generation 경로를 쓴다.
public HF repo에는 아직 HrmTextForCausalLM remote-code wrapper가 없다.
CPU로 1.4B recurrent PrefixLM generation을 돌리는 경로는 구현/최적화되어 있지 않다.
```

## 현재 KoHRM 공개 모델 카드 상태

현재 모델 카드에는 CPU/Colab T4 smoke test가 있습니다.

그 smoke test의 의미:

```text
tokenizer 다운로드 확인
config 확인
model.safetensors 로드 확인
파라미터 수 확인
```

그 smoke test가 하지 않는 것:

```text
text generation
AutoModelForCausalLM 실행
vLLM 실행
```

## 결론

현재 실무적으로 믿을 수 있는 경로는 다음입니다.

```text
학습:
  pretrain.py / cfg_pretrain
  pretrain.py --config-name cfg_sft
  train_lora.py / cfg_lora

전처리:
  scripts/prepare_sft_data.py
  scripts/sample_prepared_v1_dataset.py
  scripts/merge_prepared_sft_data.py

추론:
  simple_inference_engine.py + raw FSDP2 checkpoint + CUDA

변환:
  conversion/convert_to_hf.py

아직 없는 것:
  native HRM vLLM serving
  plain Transformers AutoModel generation
  CPU generation recipe
```
