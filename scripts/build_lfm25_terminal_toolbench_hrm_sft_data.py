#!/usr/bin/env python
"""Convert the LFM2.5 terminal+ToolBench conversations dataset to KoHRM SFT JSONL.

The LFM dataset keeps assistant tool calls in a separate ``tool_calls`` string
column inside each message. KoHRM SFT data is plain instruction/response JSONL,
so this script renders each assistant turn as a target response and preserves
tool calls in text form.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import load_from_disk


ROLE_LABELS = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
    "terminal": "terminal",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def normalize_role(value: Any) -> str:
    role = clean_text(value).lower() or "user"
    return ROLE_LABELS.get(role, role)


def parse_tool_calls(raw: Any) -> list[dict[str, Any]]:
    text = clean_text(raw)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return [{"type": "raw", "content": text}]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return [{"type": "raw", "content": parsed}]


def render_tool_calls(tool_calls: list[dict[str, Any]], mode: str) -> str:
    if not tool_calls:
        return ""
    if mode == "json":
        return json.dumps({"tool_calls": tool_calls}, ensure_ascii=False, separators=(",", ":"))
    if mode == "marker":
        payload = json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
        return f"<tool_call>\n{payload}\n</tool_call>"
    if mode == "action":
        parts: list[str] = []
        for call in tool_calls:
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = clean_text(fn.get("name") or call.get("name") or call.get("type") or "tool")
            args = fn.get("arguments", call.get("arguments", call.get("input", {})))
            if isinstance(args, str):
                try:
                    args_obj = json.loads(args)
                except Exception:
                    args_obj = args
            else:
                args_obj = args
            args_text = json.dumps(args_obj, ensure_ascii=False, separators=(",", ":"))
            parts.append(f"Action: {name}\nAction Input: {args_text}")
        return "\n\n".join(parts)
    raise ValueError(f"unknown tool call render mode: {mode}")


def render_message(message: dict[str, Any], tool_call_format: str) -> str:
    content = clean_text(message.get("content"))
    tool_text = render_tool_calls(parse_tool_calls(message.get("tool_calls")), tool_call_format)
    if content and tool_text:
        return f"{content}\n\n{tool_text}"
    return content or tool_text


def trim_left(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return "[Earlier context omitted]\n" + text[-max_chars:]


def chunk_text(text: str, max_chars: int) -> Iterable[tuple[int, int, str]]:
    if max_chars <= 0 or len(text) <= max_chars:
        yield 0, 1, text
        return

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        cut = end
        if end < len(text):
            newline = text.rfind("\n", start + int(max_chars * 0.6), end)
            if newline > start:
                cut = newline
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = max(cut, start + 1)

    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        yield idx, total, chunk


def format_turn(role: str, content: str) -> str:
    return f"[{role}]\n{content.strip()}"


def build_instruction(
    prior_turns: list[str],
    source: str,
    max_history_messages: int,
    max_instruction_chars: int,
    chunk_index: int,
    chunk_count: int,
) -> str:
    history = prior_turns[-max_history_messages:] if max_history_messages > 0 else prior_turns
    context = "\n\n".join(history)
    context = trim_left(context, max_instruction_chars)
    chunk_note = ""
    if chunk_count > 1:
        chunk_note = f"\n\n[response_chunk]\n{chunk_index + 1}/{chunk_count}"
    return (
        "Continue the next assistant message for this terminal/tool-use conversation.\n"
        "Return only the assistant message. Preserve valid JSON, shell command, and "
        "tool-call syntax when the task requires it.\n\n"
        f"[source]\n{source}\n\n"
        f"[conversation]\n{context}"
        f"{chunk_note}"
    )


def iter_examples(
    conversations: list[dict[str, Any]],
    source: str,
    args: argparse.Namespace,
) -> Iterable[dict[str, Any]]:
    rendered_turns: list[tuple[int, str, str]] = []
    for turn_index, message in enumerate(conversations):
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("role"))
        rendered = render_message(message, args.tool_call_format)
        if not rendered:
            continue
        rendered_turns.append((turn_index, role, rendered))

    assistant_positions = [idx for idx, (_, role, _) in enumerate(rendered_turns) if role == "assistant"]
    if args.turn_selection == "final-assistant" and assistant_positions:
        target_positions = [assistant_positions[-1]]
    else:
        target_positions = assistant_positions

    example_index = 0
    for pos in target_positions:
        turn_index, _role, rendered = rendered_turns[pos]
        prior = [format_turn(role, content) for _, role, content in rendered_turns[:pos]]
        if not prior:
            continue
        for chunk_index, chunk_count, response in chunk_text(rendered, args.max_response_chars):
            if len(response.strip()) < args.min_response_chars:
                continue
            condition = "cot" if args.cot_if_think and "<think>" in response.lower() else args.condition
            yield {
                "instruction": build_instruction(
                    prior,
                    source,
                    args.max_history_messages,
                    args.max_instruction_chars,
                    chunk_index,
                    chunk_count,
                ),
                "response": response,
                "condition": condition,
                "source": f"lfm25_terminal_toolbench_{source}",
                "turn_index": turn_index,
                "example_index": example_index,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "tool_call_format": args.tool_call_format,
                "turn_selection": args.turn_selection,
            }
            example_index += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-path",
        default="/home/work/.data/liquid_cli_sft/datasets/lfm25_8b_a1b_terminal_full_toolbench_full_conversations_v1",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-filter", choices=["all", "terminal", "toolbench"], default="all")
    ap.add_argument("--tool-call-format", choices=["action", "marker", "json"], default="action")
    ap.add_argument("--turn-selection", choices=["final-assistant", "all-assistant"], default="final-assistant")
    ap.add_argument("--condition", default="direct")
    ap.add_argument("--cot-if-think", action="store_true")
    ap.add_argument("--max-history-messages", type=int, default=10)
    ap.add_argument("--max-instruction-chars", type=int, default=24000)
    ap.add_argument("--max-response-chars", type=int, default=12000)
    ap.add_argument("--min-response-chars", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--progress-interval", type=int, default=10000)
    args = ap.parse_args()

    dataset = load_from_disk(args.dataset_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "dataset_path": args.dataset_path,
        "source_filter": args.source_filter,
        "tool_call_format": args.tool_call_format,
        "turn_selection": args.turn_selection,
        "rows_seen": 0,
        "rows_used": 0,
        "examples": 0,
        "examples_by_source": Counter(),
        "rows_by_source": Counter(),
        "max_history_messages": args.max_history_messages,
        "max_instruction_chars": args.max_instruction_chars,
        "max_response_chars": args.max_response_chars,
    }

    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            stats["rows_seen"] += 1
            if args.max_rows and stats["rows_seen"] > args.max_rows:
                break
            source = clean_text(row.get("source")) or "unknown"
            stats["rows_by_source"][source] += 1
            if args.source_filter != "all" and source != args.source_filter:
                continue

            conversations = row.get("conversations") or []
            if not isinstance(conversations, list):
                continue
            stats["rows_used"] += 1
            for example in iter_examples(conversations, source, args):
                handle.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["examples"] += 1
                stats["examples_by_source"][source] += 1
                if args.max_examples and stats["examples"] >= args.max_examples:
                    break
            if args.max_examples and stats["examples"] >= args.max_examples:
                break
            if args.progress_interval and stats["rows_seen"] % args.progress_interval == 0:
                print(
                    f"rows_seen={stats['rows_seen']:,} rows_used={stats['rows_used']:,} "
                    f"examples={stats['examples']:,}",
                    flush=True,
                )

    stats["examples_by_source"] = dict(stats["examples_by_source"])
    stats["rows_by_source"] = dict(stats["rows_by_source"])
    stats["output"] = str(output)
    output.with_suffix(output.suffix + ".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
