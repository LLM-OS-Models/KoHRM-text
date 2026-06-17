"""Tiny smoke test for a KoHRM FSDP2 SFT checkpoint.

Loads via the project inference engine, runs one greedy decode on a hand
written prompt, and prints the token ids + decoded text. Helps diagnose
empty generations and EMA mismatches without a full eval run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_inference_engine import inference_load_checkpoint, inference_generate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--ckpt-epoch", type=int, default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--condition", default="direct")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-generation", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    ckpt = inference_load_checkpoint(
        ckpt_path=args.ckpt_path,
        ckpt_epoch=args.ckpt_epoch,
        ckpt_step=None,
        ckpt_use_ema=not args.no_ema,
        device="cuda",
    )
    print("eoa token:", ckpt.tokenizer_info["eoa"], "->",
          ckpt.tokenizer.convert_tokens_to_ids(ckpt.tokenizer_info["eoa"]))
    print("boq token:", ckpt.tokenizer_info["boq"], "->",
          ckpt.tokenizer.convert_tokens_to_ids(ckpt.tokenizer_info["boq"]))
    print("eoq token:", ckpt.tokenizer_info["eoq"], "->",
          ckpt.tokenizer.convert_tokens_to_ids(ckpt.tokenizer_info["eoq"]))
    print("condition mapping:", ckpt.tokenizer_info["condition_mapping"])

    ids = ckpt.tokenize_prompt(args.condition, args.prompt)
    print("prompt token ids (first 24):", list(ids[:24]), "...len", len(ids))

    items = [(0, (args.condition, args.prompt))]
    for idx, gen in inference_generate(
        ckpt, iter(items),
        max_tokens=args.max_tokens,
        max_generation=args.max_generation,
        batch_size=args.batch_size,
        temp=args.temp,
    ):
        print(f"--- generation ---")
        print(repr(gen))


if __name__ == "__main__":
    main()
