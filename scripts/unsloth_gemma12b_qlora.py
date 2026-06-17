"""Unsloth-based SFT for Gemma-4-12B-it (gemma4_unified) on Korean Bar Exam.

Uses Unsloth FastModel API which patches transformers to support gemma4_unified
architecture. LoRA fine-tune, 4-bit NF4 base.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


SYSTEM = "대한민국 현행 법령을 기준으로 변호사시험 선택형 문제를 푸는 법률 학습 도우미이다."


def build_user(row):
    return (
        "다음 변호사시험 선택형 문제를 읽고, 가장 옳은 정답 번호만 답하시오. "
        "출력은 '정답: <번호>' 형식 한 줄로만 하시오.\n\n"
        f"[과목] {row.get('subject','')}\n"
        f"[문제]\n{row.get('question','')}"
    )


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
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--warmup-steps", type=int, default=15)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    args = ap.parse_args()

    from unsloth import FastModel
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from trl import SFTTrainer, SFTConfig

    print(f"loading 4-bit {args.model} via Unsloth FastModel ...", flush=True)
    model, tok = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_len,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    ds = JsonlSFTDataset(args.train_jsonl, tok, max_len=args.max_len)
    print(f"train rows: {len(ds)}", flush=True)

    pad_id = tok.pad_token_id or 0
    def collate_fn(features):
        # Drop mm_token_type_ids if present (Unsloth handles it internally)
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        max_len = max(len(x) for x in input_ids)
        def pad(s, v): return list(s) + [v] * (max_len - len(s))
        return {
            "input_ids": torch.tensor([pad(x, pad_id)[:max_len] for x in input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([pad([1]*len(x), 0)[:max_len] for x in input_ids], dtype=torch.long),
            "labels": torch.tensor([pad(x, -100)[:max_len] for x in labels], dtype=torch.long),
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
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        report_to="none",
        disable_tqdm=False,
        logging_first_step=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate_fn)
    total_steps = int(args.epochs * len(ds) / (args.per_device_batch * args.grad_accum))
    print(f"=== Unsloth QLoRA START total_steps={total_steps} batch={args.per_device_batch}x{args.grad_accum} r={args.lora_r} ===", flush=True)
    trainer.train()
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    print(f"=== Unsloth QLoRA DONE saved to {args.output} ===", flush=True)


if __name__ == "__main__":
    main()
