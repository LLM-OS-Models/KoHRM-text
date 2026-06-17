"""QLoRA SFT for Gemma-4-12B-it on Korean Bar Exam data.

12B uses Gemma4UnifiedForConditionalGeneration (multimodal). The class is not
exported in transformers.models.gemma4 by default, so we try multiple import
paths and fall back to Gemma4ForConditionalGeneration.
"""
from __future__ import annotations

import argparse
import json
import importlib
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


SYSTEM = "대한민국 현행 법령을 기준으로 변호사시험 선택형 문제를 푸는 법률 학습 도우미이다."


def build_user(row: dict) -> str:
    return (
        "다음 변호사시험 선택형 문제를 읽고, 가장 옳은 정답 번호만 답하시오. "
        "출력은 '정답: <번호>' 형식 한 줄로만 하시오.\n\n"
        f"[과목] {row.get('subject','')}\n"
        f"[문제]\n{row.get('question','')}"
    )


def load_model_class():
    """Try Gemma4Unified first, fall back to Gemma4ForConditionalGeneration."""
    candidates = [
        ("transformers.models.gemma4.modeling_gemma4", "Gemma4UnifiedForConditionalGeneration"),
        ("transformers", "Gemma4UnifiedForConditionalGeneration"),
        ("transformers.models.gemma4.modeling_gemma4", "Gemma4ForConditionalGeneration"),
        ("transformers", "Gemma4ForConditionalGeneration"),
        ("transformers", "AutoModelForImageTextToText"),
    ]
    for mod_path, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, cls_name):
                cls = getattr(mod, cls_name)
                print(f"using {mod_path}.{cls_name}", flush=True)
                return cls
        except Exception:
            pass
    raise RuntimeError("no Gemma4 class available")


class JsonlSFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1536):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                inst = o["instruction"]
                inst_text = inst.split("\n\n", 1)[-1] if "대한민국 현행 법령" in inst else inst
                self.rows.append({"question": inst_text, "answer": o["response"]})
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        messages = [
            {"role": "user", "content": build_user(row)},
            {"role": "assistant", "content": row["answer"]},
        ]
        full = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = self.tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full_ids = self.tok(full, truncation=True, max_length=self.max_len, add_special_tokens=False)["input_ids"]
        prompt_ids = self.tok(prompt, truncation=True, max_length=self.max_len, add_special_tokens=False)["input_ids"]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
            "mm_token_type_ids": [0] * len(full_ids),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--per-device-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--warmup-steps", type=int, default=15)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print(f"loading 4-bit model from {args.model} ...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    from transformers import AutoModelForImageTextToText
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, quantization_config=bnb,
            trust_remote_code=True, attn_implementation="sdpa",
        )
    except Exception as e:
        print(f"AutoModelForImageTextToText failed: {e}; trying direct", flush=True)
        ModelClass = load_model_class()
        model = ModelClass.from_pretrained(
            args.model, quantization_config=bnb,
            trust_remote_code=True, attn_implementation="sdpa",
        )
    model.config.use_cache = False

    # Force text-only path
    if hasattr(model.config, "language_model_only"):
        model.config.language_model_only = True
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "language_model_only"):
        model.config.text_config.language_model_only = True

    # Unwrap Gemma4ClippableLinear — but only if inner layer is bnb 4-bit
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        from bitsandbytes.nn import Linear4bit
        unwrapped = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, Gemma4ClippableLinear):
                inner = module.linear
                if isinstance(inner, Linear4bit):
                    pn, _, cn = name.rpartition('.')
                    parent = model.get_submodule(pn) if pn else model
                    setattr(parent, cn, inner)
                    unwrapped += 1
        print(f"unwrapped {unwrapped} ClippableLinear (bnb 4-bit only)", flush=True)
    except Exception as e:
        print(f"ClippableLinear unwrap skip: {e}", flush=True)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    target_modules = r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = JsonlSFTDataset(args.train_jsonl, tok, max_len=args.max_len)
    print(f"train rows: {len(ds)}", flush=True)

    pad_id = tok.pad_token_id or 0
    def collate_fn(features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        mm = [f["mm_token_type_ids"] for f in features]
        max_len = max(len(x) for x in input_ids)
        def pad(s, v): return list(s) + [v] * (max_len - len(s))
        return {
            "input_ids": torch.tensor([pad(x, pad_id)[:max_len] for x in input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([pad([1]*len(x), 0)[:max_len] for x in input_ids], dtype=torch.long),
            "labels": torch.tensor([pad(x, -100)[:max_len] for x in labels], dtype=torch.long),
            "mm_token_type_ids": torch.tensor([pad(x, 0)[:max_len] for x in mm], dtype=torch.long),
        }

    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        disable_tqdm=False,
        logging_first_step=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate_fn)
    total_steps = int(args.epochs * len(ds) / (args.per_device_batch * args.grad_accum))
    print(f"=== QLoRA SFT START total_steps={total_steps} batch={args.per_device_batch}x{args.grad_accum} r={args.lora_r} ===", flush=True)
    trainer.train()
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    print(f"=== QLoRA SFT DONE saved to {args.output} ===", flush=True)


if __name__ == "__main__":
    main()
