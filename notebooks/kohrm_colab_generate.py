"""Minimal KoHRM-Text generation runtime for Colab.

This file intentionally avoids `transformers` and FlashAttention. It loads the
public `model.safetensors` export and runs HRM-Text generation with PyTorch
scaled-dot-product attention. It is built for smoke generation tests on Colab
T4 and small CUDA machines.
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer


def _rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(position_ids: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32) / head_dim))
    freqs = torch.einsum("bt,d->btd", position_ids.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return ((x * cos.unsqueeze(-2)) + (_rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)


class KoHRMAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, device: str = "meta") -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gqkv_proj = nn.Linear(hidden_size, (4 * num_heads) * head_dim, bias=False, device=device)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False, device=device)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
        cache_pos: int,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        gqkv = self.gqkv_proj(x).view(bsz, seqlen, 4 * self.num_heads, self.head_dim)
        gate, q, k, v = gqkv.split((self.num_heads, self.num_heads, self.num_heads, self.num_heads), dim=-2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if cache is not None:
            end = cache_pos + seqlen
            cache["k"][:, cache_pos:end].copy_(k)
            cache["v"][:, cache_pos:end].copy_(v)
            k = cache["k"][:, :end]
            v = cache["v"][:, :end]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2)
        y = (torch.sigmoid(gate) * y).reshape(bsz, seqlen, self.num_heads * self.head_dim)
        return self.o_proj(y)


class KoHRMMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, device: str = "meta") -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class KoHRMBlock(nn.Module):
    def __init__(self, cfg: dict[str, Any], device: str = "meta") -> None:
        super().__init__()
        self.eps = float(cfg["rms_norm_eps"])
        self.attn = KoHRMAttention(cfg["hidden_size"], cfg["num_attention_heads"], cfg["head_dim"], device=device)
        self.mlp = KoHRMMLP(cfg["hidden_size"], cfg["intermediate_size"], device=device)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: dict[str, torch.Tensor] | None, cache_pos: int) -> torch.Tensor:
        x = x + self.attn(_rms_norm(x, self.eps), cos, sin, cache, cache_pos)
        x = x + self.mlp(_rms_norm(x, self.eps))
        return x


class KoHRMModule(nn.Module):
    def __init__(self, cfg: dict[str, Any], num_layers: int, device: str = "meta") -> None:
        super().__init__()
        self.eps = float(cfg["rms_norm_eps"])
        self.layers = nn.ModuleList([KoHRMBlock(cfg, device=device) for _ in range(num_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        caches: list[dict[str, torch.Tensor]] | None,
        cache_pos: int,
    ) -> torch.Tensor:
        x = hidden_states + input_injection
        for idx, layer in enumerate(self.layers):
            x = layer(x, cos, sin, None if caches is None else caches[idx], cache_pos)
        return _rms_norm(x, self.eps)


class KoHRMCore(nn.Module):
    def __init__(self, cfg: dict[str, Any], num_layers: int, device: str = "meta") -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding_scale = float(cfg.get("embedding_scale", 1.0))
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], device=device)
        self.register_buffer("z_L_init", torch.empty(cfg["hidden_size"], device=device), persistent=True)
        self.H_module = KoHRMModule(cfg, num_layers, device=device)
        self.L_module = KoHRMModule(cfg, num_layers, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        caches: dict[str, list[list[dict[str, torch.Tensor]]]] | None,
        cache_pos: int,
    ) -> torch.Tensor:
        x = self.embedding_scale * self.embed_tokens(input_ids)
        cos, sin = _rope_cos_sin(position_ids, self.cfg["head_dim"], float(self.cfg["rope_theta"]), x.dtype)
        z_h = x
        z_l = self.z_L_init.to(dtype=x.dtype).view(1, 1, -1).expand_as(x)

        h_cycles, l_cycles = int(self.cfg["H_cycles"]), int(self.cfg["L_cycles"])
        for h_idx in range(h_cycles):
            for l_idx in range(l_cycles):
                pass_idx = h_idx * l_cycles + l_idx
                z_l = self.L_module(z_l, z_h, cos, sin, None if caches is None else caches["L"][pass_idx], cache_pos)
            z_h = self.H_module(z_h, z_l, cos, sin, None if caches is None else caches["H"][h_idx], cache_pos)
        return z_h


class KoHRMTextForGeneration(nn.Module):
    def __init__(self, cfg: dict[str, Any], num_layers: int, device: str = "meta") -> None:
        super().__init__()
        self.cfg = cfg
        self.num_layers = num_layers
        self.model = KoHRMCore(cfg, num_layers, device=device)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        caches: dict[str, list[list[dict[str, torch.Tensor]]]] | None = None,
        cache_pos: int = 0,
    ) -> torch.Tensor:
        hidden = self.model(input_ids, position_ids, caches, cache_pos)
        return self.lm_head(hidden)

    def init_cache(self, batch_size: int, max_seq_len: int, device: torch.device, dtype: torch.dtype) -> dict[str, list[list[dict[str, torch.Tensor]]]]:
        heads, head_dim = int(self.cfg["num_attention_heads"]), int(self.cfg["head_dim"])

        def one_layer() -> dict[str, torch.Tensor]:
            shape = (batch_size, max_seq_len, heads, head_dim)
            return {
                "k": torch.empty(shape, device=device, dtype=dtype),
                "v": torch.empty(shape, device=device, dtype=dtype),
            }

        def one_pass() -> list[dict[str, torch.Tensor]]:
            return [one_layer() for _ in range(self.num_layers)]

        return {
            "H": [one_pass() for _ in range(int(self.cfg["H_cycles"]))],
            "L": [one_pass() for _ in range(int(self.cfg["H_cycles"]) * int(self.cfg["L_cycles"]))],
        }


def _module_layer_count(state: dict[str, torch.Tensor], prefix: str) -> int:
    layers = set()
    marker = f"{prefix}.layers."
    for key in state:
        if key.startswith(marker):
            layers.add(int(key[len(marker) :].split(".", 1)[0]))
    return max(layers) + 1


def load_kohrm(repo_dir: str | Path, device: str | None = None, max_gpu_memory_gib: float | None = None) -> tuple[KoHRMTextForGeneration, Tokenizer, dict[str, Any]]:
    repo_dir = Path(repo_dir)
    cfg = json.loads((repo_dir / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(repo_dir / "tokenizer.json"))

    state = load_file(str(repo_dir / "model.safetensors"), device="cpu")
    num_layers = _module_layer_count(state, "model.H_module")
    model = KoHRMTextForGeneration(cfg, num_layers=num_layers, device="meta")
    model.load_state_dict(state, strict=True, assign=True)
    del state

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.device(device)
    dtype = torch.float16 if target.type == "cuda" else torch.float32
    model = model.to(device=target, dtype=dtype).eval()
    if target.type == "cuda":
        torch.set_float32_matmul_precision("high")
    if target.type == "cuda" and max_gpu_memory_gib is not None:
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory free/total GiB: {free / 2**30:.2f}/{total / 2**30:.2f}")
    return model, tokenizer, cfg


def format_kohrm_prompt(prompt: str, condition_token: str = "<|object_ref_start|>") -> str:
    return f"<|im_start|>{condition_token}{prompt}<|im_end|>"


def _apply_repetition_penalty(logits: torch.Tensor, seen_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not seen_ids:
        return logits
    for token_id in set(seen_ids):
        value = logits[..., token_id]
        logits[..., token_id] = torch.where(value < 0, value * penalty, value / penalty)
    return logits


def _apply_no_repeat_ngram(logits: torch.Tensor, seen_ids: list[int], ngram_size: int) -> torch.Tensor:
    if ngram_size <= 0 or len(seen_ids) < ngram_size - 1:
        return logits
    prefix = tuple(seen_ids[-(ngram_size - 1):])
    blocked: set[int] = set()
    for idx in range(len(seen_ids) - ngram_size + 1):
        if tuple(seen_ids[idx:idx + ngram_size - 1]) == prefix:
            blocked.add(seen_ids[idx + ngram_size - 1])
    if blocked:
        logits[..., list(blocked)] = -torch.inf
    return logits


def _sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    seen_ids: list[int] | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> int:
    logits = logits.float()
    seen_ids = seen_ids or []
    logits = _apply_repetition_penalty(logits, seen_ids, repetition_penalty)
    logits = _apply_no_repeat_ngram(logits, seen_ids, no_repeat_ngram_size)
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        keep = torch.cumsum(sorted_probs, dim=-1) <= top_p
        keep[..., 0] = True
        sorted_probs = sorted_probs.masked_fill(~keep, 0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx.gather(-1, next_sorted).item())
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.inference_mode()
def generate_from_loaded(
    model: KoHRMTextForGeneration,
    tokenizer: Tokenizer,
    cfg: dict[str, Any],
    prompt: str,
    *,
    max_new_tokens: int = 64,
    max_seq_len: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.18,
    no_repeat_ngram_size: int = 4,
    condition_token: str = "<|object_ref_start|>",
) -> str:
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    wrapped = format_kohrm_prompt(prompt, condition_token=condition_token)
    input_ids = tokenizer.encode(wrapped, add_special_tokens=False).ids
    if len(input_ids) + max_new_tokens + 1 > max_seq_len:
        raise ValueError(f"Prompt plus generation exceeds max_seq_len={max_seq_len}: prompt_tokens={len(input_ids)}")

    caches = model.init_cache(1, max_seq_len, dev, dtype)
    ids = torch.tensor([input_ids], device=dev, dtype=torch.long)
    pos = torch.arange(ids.shape[1], device=dev, dtype=torch.long).unsqueeze(0)
    logits = model(ids, pos, caches=caches, cache_pos=0)[:, -1, :]
    cache_pos = ids.shape[1]

    eos_id = int(cfg.get("eos_token_id") or tokenizer.token_to_id("<|box_end|>"))
    stop_ids = {
        eos_id,
        tokenizer.token_to_id("<|im_end|>"),
        tokenizer.token_to_id("<|box_end|>"),
    }
    stop_ids = {int(x) for x in stop_ids if x is not None}
    out_ids: list[int] = []
    seen_ids = list(input_ids)
    next_id = _sample_next(logits, temperature, top_p, seen_ids, repetition_penalty, no_repeat_ngram_size)
    for _ in range(max_new_tokens):
        if next_id in stop_ids:
            break
        out_ids.append(next_id)
        seen_ids.append(next_id)
        token = torch.tensor([[next_id]], device=dev, dtype=torch.long)
        pos = torch.tensor([[cache_pos]], device=dev, dtype=torch.long)
        logits = model(token, pos, caches=caches, cache_pos=cache_pos)[:, -1, :]
        cache_pos += 1
        next_id = _sample_next(logits, temperature, top_p, seen_ids, repetition_penalty, no_repeat_ngram_size)

    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_text(
    repo_dir: str | Path,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    max_seq_len: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.18,
    no_repeat_ngram_size: int = 4,
    condition_token: str = "<|object_ref_start|>",
    device: str | None = None,
) -> str:
    model, tokenizer, cfg = load_kohrm(repo_dir, device=device, max_gpu_memory_gib=14.0)
    return generate_from_loaded(
        model,
        tokenizer,
        cfg,
        prompt,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        condition_token=condition_token,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small KoHRM-Text generation test without transformers.")
    parser.add_argument("repo_dir", type=Path, help="Directory containing config.json, tokenizer.json, and model.safetensors")
    parser.add_argument("--prompt", default="한국어로 현재 디렉터리에서 가장 큰 파일 10개를 찾는 bash 명령을 알려주세요.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.18)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(generate_text(
        args.repo_dir,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        device=args.device,
    ))


if __name__ == "__main__":
    main()
