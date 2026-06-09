"""CPU-oriented KoHRM-Text inference runtime.

KoHRM-Text uses the custom ``hrm_text`` / ``HrmTextForCausalLM`` architecture,
so it cannot currently be served by llama.cpp/GGUF or ordinary vLLM paths.
This runtime wraps the existing safetensors loader and adds CPU-friendly
quantization and cycle overrides.

Recommended mode for normal CPU use:

    python HRM-Text/inference/kohrm_cpu_runtime.py \
      --model LLM-OS-Models/KoHRM-Text-1.4B \
      --quant dynamic-int8 \
      --prompt "리눅스에서 현재 디렉토리 파일 목록을 보는 명령어는?" \
      --max-new-tokens 64

Experimental memory-first mode:

    python HRM-Text/inference/kohrm_cpu_runtime.py --quant weight-int4 ...
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "notebooks" / "kohrm_colab_generate.py"
DEFAULT_REPO_ID = "LLM-OS-Models/KoHRM-Text-1.4B"


def _load_helper():
    if not HELPER_PATH.exists():
        raise FileNotFoundError(f"missing KoHRM helper: {HELPER_PATH}")
    spec = importlib.util.spec_from_file_location("kohrm_colab_generate", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("kohrm_colab_generate", module)
    spec.loader.exec_module(module)
    return module


def _read_dotenv_token() -> str | None:
    """Read a local HF token without printing it or exporting it to shell logs."""
    candidates = [
        Path.cwd() / ".env",
        REPO_ROOT.parent / ".env",
        REPO_ROOT / ".env",
        Path.home() / ".cache" / "huggingface" / "token",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.name == "token":
            token = path.read_text(encoding="utf-8").strip()
            return token or None
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key.split(None, 1)[1]
            if key in {"HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                token = value.strip().strip('"').strip("'")
                return token or None
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def resolve_model_dir(model: str, revision: str | None = None) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        return path
    token = _read_dotenv_token()
    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors", "README.md"],
            token=token,
        )
    )


@dataclass
class RuntimeStats:
    prompt_tokens: int
    generated_tokens: int
    elapsed_s: float
    tokens_per_s: float
    quantization: str
    h_cycles: int
    l_cycles: int
    dtype: str


class WeightOnlyInt8Linear(nn.Module):
    """Simple symmetric per-group int8 weight-only Linear.

    This is a portability fallback, not an optimized kernel. It reduces resident
    weight memory after conversion, but dequantizes on forward. For speed, prefer
    PyTorch dynamic int8.
    """

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, in_features: int, out_features: int, group_size: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous())

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> "WeightOnlyInt8Linear":
        weight = linear.weight.detach().to(dtype=torch.float32, device="cpu")
        out_features, in_features = weight.shape
        pad = (-in_features) % group_size
        if pad:
            weight = F.pad(weight, (0, pad))
        grouped = weight.view(out_features, -1, group_size)
        scales = grouped.abs().amax(dim=-1).clamp_min(1e-8) / 127.0
        qweight = torch.round(grouped / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        return cls(qweight=qweight, scales=scales.to(torch.float16), in_features=in_features, out_features=out_features, group_size=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (self.qweight.to(torch.float32) * self.scales.to(torch.float32).unsqueeze(-1)).view(self.out_features, -1)
        weight = weight[:, : self.in_features].to(dtype=x.dtype)
        return F.linear(x, weight)


class WeightOnlyInt4Linear(nn.Module):
    """Portable symmetric per-group int4 weight-only Linear.

    Values are stored as packed signed nibbles. Forward unpacks and dequantizes
    on CPU, so this is memory-first rather than speed-first.
    """

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor, in_features: int, out_features: int, padded_features: int, group_size: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.padded_features = int(padded_features)
        self.group_size = int(group_size)
        self.register_buffer("packed", packed.contiguous())
        self.register_buffer("scales", scales.contiguous())

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> "WeightOnlyInt4Linear":
        weight = linear.weight.detach().to(dtype=torch.float32, device="cpu")
        out_features, in_features = weight.shape
        pad_group = (-in_features) % group_size
        if pad_group:
            weight = F.pad(weight, (0, pad_group))
        if weight.shape[1] % 2:
            weight = F.pad(weight, (0, 1))
        padded_features = weight.shape[1]
        grouped = weight.view(out_features, -1, group_size)
        scales = grouped.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
        q = torch.round(grouped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int16)
        q = (q + 16).remainder(16).to(torch.uint8).view(out_features, padded_features)
        low = q[:, 0::2]
        high = q[:, 1::2] << 4
        packed = low | high
        return cls(
            packed=packed,
            scales=scales.to(torch.float16),
            in_features=in_features,
            out_features=out_features,
            padded_features=padded_features,
            group_size=group_size,
        )

    def _unpack(self) -> torch.Tensor:
        low = self.packed & 0x0F
        high = (self.packed >> 4) & 0x0F
        q = torch.empty((self.out_features, self.packed.shape[1] * 2), dtype=torch.int16, device=self.packed.device)
        q[:, 0::2] = low.to(torch.int16)
        q[:, 1::2] = high.to(torch.int16)
        q = torch.where(q >= 8, q - 16, q)
        return q[:, : self.padded_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._unpack().to(torch.float32)
        weight = (q.view(self.out_features, -1, self.group_size) * self.scales.to(torch.float32).unsqueeze(-1)).view(self.out_features, -1)
        weight = weight[:, : self.in_features].to(dtype=x.dtype)
        return F.linear(x, weight)


def _replace_linear_modules(module: nn.Module, *, quant: str, group_size: int, quantize_lm_head: bool, prefix: str = "") -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if child_prefix == "lm_head" and not quantize_lm_head:
                continue
            if child.bias is not None:
                raise ValueError(f"bias is not supported by portable weight-only quantization: {child_prefix}")
            if quant == "weight-int8":
                new_child = WeightOnlyInt8Linear.from_linear(child, group_size=group_size)
            elif quant == "weight-int4":
                new_child = WeightOnlyInt4Linear.from_linear(child, group_size=group_size)
            else:
                raise ValueError(f"unsupported weight-only quantization: {quant}")
            setattr(module, name, new_child)
            replaced += 1
        else:
            replaced += _replace_linear_modules(child, quant=quant, group_size=group_size, quantize_lm_head=quantize_lm_head, prefix=child_prefix)
    return replaced


def apply_quantization(
    model: nn.Module,
    quant: str,
    *,
    group_size: int = 128,
    quantize_lm_head: bool = False,
) -> nn.Module:
    if quant == "none":
        return model
    if quant == "dynamic-int8":
        torch.backends.quantized.engine = "fbgemm"
        return torch.ao.quantization.quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8, inplace=False)
    if quant in {"weight-int8", "weight-int4"}:
        replaced = _replace_linear_modules(model, quant=quant, group_size=group_size, quantize_lm_head=quantize_lm_head)
        if replaced == 0:
            raise RuntimeError("no Linear modules were replaced")
        gc.collect()
        return model.cpu().eval()
    raise ValueError(f"unknown quantization mode: {quant}")


def load_runtime(
    model_dir: Path,
    *,
    quant: str,
    h_cycles: int | None,
    l_cycles: int | None,
    group_size: int,
    quantize_lm_head: bool,
):
    helper = _load_helper()
    model, tokenizer, cfg = helper.load_kohrm(model_dir, device="cpu")
    if h_cycles is not None:
        cfg["H_cycles"] = int(h_cycles)
        model.cfg["H_cycles"] = int(h_cycles)
        model.model.cfg["H_cycles"] = int(h_cycles)
    if l_cycles is not None:
        cfg["L_cycles"] = int(l_cycles)
        model.cfg["L_cycles"] = int(l_cycles)
        model.model.cfg["L_cycles"] = int(l_cycles)
    model = apply_quantization(model, quant, group_size=group_size, quantize_lm_head=quantize_lm_head)
    return helper, model.eval(), tokenizer, cfg


def generate(
    model: nn.Module,
    tokenizer: Any,
    cfg: dict[str, Any],
    helper: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    min_new_tokens: int,
    max_seq_len: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    condition: str,
) -> tuple[str, RuntimeStats]:
    wrapped = helper.format_kohrm_prompt(prompt, condition=condition)
    prompt_tokens = len(tokenizer.encode(wrapped, add_special_tokens=False).ids)
    start = time.perf_counter()
    output = helper.generate_from_loaded(
        model,
        tokenizer,
        cfg,
        prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        max_seq_len=max_seq_len,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        condition=condition,
    )
    elapsed = time.perf_counter() - start
    out_tokens = len(tokenizer.encode(output, add_special_tokens=False).ids) if output else 0
    stats = RuntimeStats(
        prompt_tokens=prompt_tokens,
        generated_tokens=out_tokens,
        elapsed_s=elapsed,
        tokens_per_s=(out_tokens / elapsed if elapsed > 0 else math.nan),
        quantization="",
        h_cycles=int(cfg.get("H_cycles", 0)),
        l_cycles=int(cfg.get("L_cycles", 0)),
        dtype=str(next(model.parameters()).dtype) if any(True for _ in model.parameters()) else "unknown",
    )
    return output, stats


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run KoHRM-Text on CPU with optional quantization.")
    ap.add_argument("--model", default=DEFAULT_REPO_ID, help="HF repo id or local directory containing KoHRM HF export files.")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--quant", choices=["none", "dynamic-int8", "weight-int8", "weight-int4"], default="dynamic-int8")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--quantize-lm-head", action="store_true", help="Also quantize lm_head in portable weight-only modes. Saves memory but slows generation.")
    ap.add_argument("--h-cycles", type=int, default=None, help="Override H_cycles. Lower values trade quality for CPU speed.")
    ap.add_argument("--l-cycles", type=int, default=None, help="Override L_cycles. Lower values trade quality for CPU speed.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--min-new-tokens", type=int, default=0)
    ap.add_argument("--max-seq-len", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--condition", default="direct", choices=["direct", "cot", "noisy", "synth"])
    ap.add_argument("--json-stats", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    # Keep CPU execution predictable on shared machines.
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 8))
    model_dir = resolve_model_dir(args.model, revision=args.revision)
    helper, model, tokenizer, cfg = load_runtime(
        model_dir,
        quant=args.quant,
        h_cycles=args.h_cycles,
        l_cycles=args.l_cycles,
        group_size=args.group_size,
        quantize_lm_head=args.quantize_lm_head,
    )
    output, stats = generate(
        model,
        tokenizer,
        cfg,
        helper,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        condition=args.condition,
    )
    stats.quantization = args.quant
    print(output)
    if args.json_stats:
        print(json.dumps(stats.__dict__, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
