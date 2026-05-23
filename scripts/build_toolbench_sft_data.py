"""Build HRM-Text SFT JSONL from ToolBench/ToolLLaMA conversations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_history(messages: list[dict], upto: int, max_messages: int) -> str:
    start = max(0, upto - max_messages)
    parts = []
    for msg in messages[start:upto]:
        role = msg.get("from", "unknown")
        value = str(msg.get("value", "")).strip()
        if not value:
            continue
        parts.append(f"<|{role}|>\n{value}")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="HRM-Text/data_toolbench/data/toolllama_G123_dfs_train.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-output-mib", type=int, default=512)
    ap.add_argument("--max-history-messages", type=int, default=8)
    ap.add_argument("--max-response-chars", type=int, default=12000)
    args = ap.parse_args()

    input_path = Path(args.input)
    print(f"loading {input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = args.max_output_mib * 1024 * 1024

    rows = 0
    tasks = 0
    bytes_written = 0
    with open(out, "w", encoding="utf-8") as f:
        for item in data:
            conversations = item.get("conversations") or []
            tasks += 1
            for idx, msg in enumerate(conversations):
                if msg.get("from") != "assistant":
                    continue
                response = str(msg.get("value", "")).strip()
                if not response or len(response) > args.max_response_chars:
                    continue
                instruction = format_history(conversations, idx, args.max_history_messages)
                if not instruction:
                    continue
                row = {
                    "instruction": instruction,
                    "response": response,
                    "condition": "direct",
                    "source": "HRM-Text/ToolBench",
                    "task_id": item.get("id", ""),
                }
                line = json.dumps(row, ensure_ascii=False) + "\n"
                encoded = line.encode("utf-8")
                if bytes_written + len(encoded) > max_bytes:
                    print(f"wrote {rows:,} rows from {tasks:,} tasks, {bytes_written:,} bytes to {out}")
                    return
                f.write(line)
                rows += 1
                bytes_written += len(encoded)

    print(f"wrote {rows:,} rows from {tasks:,} tasks, {bytes_written:,} bytes to {out}")


if __name__ == "__main__":
    main()
