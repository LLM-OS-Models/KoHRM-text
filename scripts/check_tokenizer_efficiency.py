"""Print simple tokenizer efficiency diagnostics for Korean/terminal samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer


SAMPLES = {
    "ko_general": "대한민국 헌법 제1조 대한민국은 민주공화국이다.",
    "ko_legal": "제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다.",
    "ko_terminal": "현재 디렉터리에서 로그 파일을 찾아 오류가 많은 순서대로 정렬해줘.",
    "terminal": "ls -la /home/work && grep -r 'error' *.log | sort | uniq -c",
    "tool_json": '{"name":"run_command","arguments":{"cmd":"pytest -q","cwd":"/workspace"}}',
    "code": "def solve(items):\n    return [x for x in items if x is not None]",
    "english": "The quick brown fox jumps over the lazy dog.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(Path(args.tokenizer)))
    print(f"vocab_size={tok.get_vocab_size(with_added_tokens=True)}")
    for name, text in SAMPLES.items():
        enc = tok.encode(text, add_special_tokens=False)
        token_count = max(1, len(enc.tokens))
        print(f"{name:12s} chars={len(text):4d} tokens={token_count:4d} chars/token={len(text)/token_count:.2f}")
        print("  " + " ".join(enc.tokens[:40]))
    for special in ["<|im_start|>", "<|assistant|>", "<|tool_call|>", "<|terminal|>", "<|box_end|>"]:
        enc = tok.encode(special, add_special_tokens=False)
        print(f"special {special}: ids={enc.ids} tokens={enc.tokens}")


if __name__ == "__main__":
    main()
