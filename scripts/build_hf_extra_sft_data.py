"""Convert small downloaded HF auxiliary corpora to HRM-Text SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator


def trim_left(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return "[이전 대화 일부 생략]\n" + text[-max_chars:]


def format_turn(role: str, content: str) -> str:
    return f"[{role}]\n{content.strip()}"


def text_from_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            typ = str(part.get("type") or "")
            if part.get("text") is not None:
                parts.append(str(part["text"]))
            elif part.get("thinking") is not None:
                parts.append("<think>\n" + str(part["thinking"]).strip() + "\n</think>")
            elif typ in {"tool_use", "toolCall", "function_call"}:
                parts.append("[tool_call]\n" + json.dumps(part, ensure_ascii=False))
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
        return "\n\n".join(p for p in parts if p.strip())
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def normalize_role(role: object) -> str:
    value = str(role or "user").strip()
    lower = value.lower()
    if lower in {"developer", "system"}:
        return "system"
    if lower in {"toolresult", "tool", "function"}:
        return "terminal"
    if lower in {"assistant", "user"}:
        return lower
    return lower or "user"


def conversation_examples(
    turns: list[tuple[str, str]],
    source: str,
    max_instruction_chars: int,
    condition: str = "direct",
    metadata: str = "",
) -> Iterator[dict]:
    prior: list[str] = []
    ex_index = 0
    for turn_index, (role, content) in enumerate(turns):
        if role == "assistant" and prior and content.strip():
            context_items = ([metadata] if metadata else []) + prior
            context = trim_left("\n\n".join(context_items), max_instruction_chars)
            yield {
                "instruction": (
                    "다음 대화 맥락에서 assistant가 이어서 작성할 응답을 생성하십시오.\n\n"
                    f"{context}"
                ),
                "response": content.strip(),
                "condition": condition,
                "source": source,
                "turn_index": turn_index,
                "example_index": ex_index,
            }
            ex_index += 1
        prior.append(format_turn(role, content))


def iter_angrygiraffe(path: Path, max_instruction_chars: int) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="ignore") as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            turns: list[tuple[str, str]] = []
            for msg in obj.get("messages") or []:
                if not isinstance(msg, dict):
                    continue
                text = text_from_content(msg.get("content")).strip()
                if text:
                    turns.append((normalize_role(msg.get("role")), text))
            metadata = f"[metadata]\ncategory: {obj.get('category', '')}\nmodel: {obj.get('model', '')}"
            for ex in conversation_examples(
                turns,
                "angrygiraffe_claude_opus_4_6_4_7_reasoning",
                max_instruction_chars,
                condition="cot",
                metadata=metadata,
            ):
                ex["row_index"] = row_index
                yield ex


def iter_deepseek_agent(root: Path, max_instruction_chars: int) -> Iterator[dict]:
    for path in sorted(root.glob("*.jsonl")):
        turns: list[tuple[str, str]] = []
        session_id = path.stem
        with path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("type") == "session":
                    session_id = str(obj.get("id") or session_id)
                    continue
                if obj.get("type") != "message":
                    continue
                msg = obj.get("message")
                if not isinstance(msg, dict):
                    continue
                role = normalize_role(msg.get("role"))
                body = text_from_content(msg.get("content")).strip()
                if not body:
                    continue
                tool_name = msg.get("toolName")
                if tool_name and role == "terminal":
                    body = f"[tool:{tool_name}]\n{body}"
                turns.append((role, body))
        metadata = f"[metadata]\nsession_id: {session_id}\nsource_file: {path.name}"
        for ex in conversation_examples(
            turns,
            "teichai_deepseek_v4_pro_agent",
            max_instruction_chars,
            condition="cot",
            metadata=metadata,
        ):
            ex["source_file"] = path.name
            ex["session_id"] = session_id
            yield ex


def iter_open_mm_rl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="ignore") as f:
        rows = json.load(f)
    for obj in rows:
        if not isinstance(obj, dict):
            continue
        question = str(obj.get("question") or "").strip()
        answer = str(obj.get("answer") or "").strip()
        if not question or not answer:
            continue
        refs = str(obj.get("references") or "").strip()
        instruction = (
            "다음은 멀티모달 수학/추론 데이터의 텍스트화된 문제입니다. "
            "첨부 이미지는 실제 학습 입력에 포함하지 않고, 문제 텍스트와 메타데이터를 근거로 답하십시오.\n\n"
            f"[domain]\n{obj.get('domain', '')}\n\n[subDomain]\n{obj.get('subDomain', '')}\n\n"
            f"[format]\n{obj.get('format', '')}\n\n[references]\n{refs}\n\n[question]\n{question}"
        )
        yield {
            "instruction": instruction,
            "response": answer,
            "condition": "direct",
            "source": "turing_open_mm_rl_text_only",
            "conversation_id": obj.get("conversation_id"),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--angrygiraffe-full-train")
    ap.add_argument("--deepseek-agent-root")
    ap.add_argument("--open-mm-rl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-instruction-chars", type=int, default=24000)
    ap.add_argument("--progress-interval", type=int, default=5000)
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    sources: list[Iterator[dict]] = []
    if args.angrygiraffe_full_train:
        sources.append(iter_angrygiraffe(Path(args.angrygiraffe_full_train), args.max_instruction_chars))
    if args.deepseek_agent_root:
        sources.append(iter_deepseek_agent(Path(args.deepseek_agent_root), args.max_instruction_chars))
    if args.open_mm_rl:
        sources.append(iter_open_mm_rl(Path(args.open_mm_rl)))

    rows = 0
    bytes_written = 0
    by_source: dict[str, int] = {}
    with out.open("w", encoding="utf-8") as f:
        for source_iter in sources:
            for row in source_iter:
                line = json.dumps(row, ensure_ascii=False) + "\n"
                f.write(line)
                rows += 1
                bytes_written += len(line.encode("utf-8"))
                source = row.get("source", "unknown")
                by_source[source] = by_source.get(source, 0) + 1
                if args.progress_interval and rows % args.progress_interval == 0:
                    print(f"rows={rows:,} bytes={bytes_written:,} source={source}", flush=True)

    stats = {
        "rows": rows,
        "bytes": bytes_written,
        "by_source": by_source,
        "max_instruction_chars": args.max_instruction_chars,
    }
    stats_path = out.with_suffix(out.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
