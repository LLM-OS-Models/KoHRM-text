#!/usr/bin/env python3
"""Replace v5 question/choice blocks with clean 15th_split JSON text."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("round15_rag_contexts_v5_20260629")
SPLIT_ROOT = Path("15th_split")

SUBJECTS = {
    "public_law": "공법",
    "criminal_law": "형사법",
    "civil_law": "민사법",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def render_structured(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return "\n".join(f"{key}: {clean_text(val)}" for key, val in value.items())
    if isinstance(value, list):
        return "\n".join(clean_text(item) for item in value)
    return clean_text(value)


def build_problem_block(item: dict[str, Any]) -> str:
    parts: list[str] = []

    passage = render_structured(item.get("passage"))
    if passage:
        parts.append(passage)

    material = render_structured(item.get("material"))
    if material:
        parts.append("[자료]\n" + material)

    underlines = render_structured(item.get("underlines"))
    if underlines:
        parts.append("[밑줄]\n" + underlines)

    question = clean_text(item.get("question"))
    if question:
        parts.append(question)

    propositions = item.get("propositions") or {}
    if propositions:
        prop_lines = [f"{label}. {clean_text(text)}" for label, text in propositions.items()]
        parts.append("\n".join(prop_lines))

    choices = item.get("choices") or {}
    choice_lines = [f"{label} {clean_text(text)}" for label, text in choices.items()]
    parts.append("\n".join(choice_lines))

    return "## 문제\n```text\n" + "\n\n".join(parts).strip() + "\n```"


def build_choice_breakdown(item: dict[str, Any]) -> str:
    propositions = item.get("propositions") or {}
    if propositions:
        lines = [f"- {label}: {clean_text(text)}" for label, text in propositions.items()]
    else:
        choices = item.get("choices") or {}
        lines = [f"- {label}: {clean_text(text)}" for label, text in choices.items()]
    return "## 선지 분해\n" + "\n".join(lines)


def replace_or_insert_section(text: str, heading: str, section: str, after_heading: str | None = None) -> str:
    pattern = re.compile(rf"^## {re.escape(heading)}\n.*?(?=^## |\Z)", re.M | re.S)
    if pattern.search(text):
        return pattern.sub(section.rstrip() + "\n\n", text, count=1)

    if after_heading is None:
        return text.rstrip() + "\n\n" + section.rstrip() + "\n"

    after_pattern = re.compile(rf"(^## {re.escape(after_heading)}\n.*?(?=^## |\Z))", re.M | re.S)
    if after_pattern.search(text):
        return after_pattern.sub(lambda m: m.group(1).rstrip() + "\n\n" + section.rstrip() + "\n\n", text, count=1)

    return text.rstrip() + "\n\n" + section.rstrip() + "\n"


def main() -> None:
    changed: list[str] = []
    for split_subject, korean_subject in SUBJECTS.items():
        for split_path in sorted((SPLIT_ROOT / split_subject).glob("*.json")):
            item = json.loads(split_path.read_text(encoding="utf-8"))
            number = int(item["question_number"])
            target = ROOT / f"q{number:03d}_{korean_subject}.md"
            if not target.exists():
                raise FileNotFoundError(target)

            old = target.read_text(encoding="utf-8")
            new = replace_or_insert_section(old, "문제", build_problem_block(item))
            new = replace_or_insert_section(new, "선지 분해", build_choice_breakdown(item), after_heading="쟁점 키워드")

            if new != old:
                target.write_text(new, encoding="utf-8")
                changed.append(str(target))

    print(f"changed={len(changed)}")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
