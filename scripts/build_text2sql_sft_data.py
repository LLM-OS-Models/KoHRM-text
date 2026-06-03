"""Build KoHRM Text2SQL SFT JSONL from public Hugging Face datasets.

The output format is the generic input expected by scripts/prepare_sft_data.py:

    {"instruction": "...", "response": "SELECT ...", "condition": "direct"}

This script intentionally keeps prompts short and regular. KoHRM pretraining
does not use chat-template-heavy instruction wrappers, so Text2SQL SFT should
teach the direct mapping from schema/question/dialect to SQL without extra
assistant-role prose.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable, Iterator, Optional

DEFAULT_CACHE_DIR = "/home/work/.data/hf_cache"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    dataset_id: str
    split: str
    group: str
    dialect: str
    license: str
    downloads_2026_06_03: Optional[int]
    likes_2026_06_03: Optional[int]
    default_max_rows: int = 0
    trust_remote_code: bool = False


SOURCES: dict[str, SourceSpec] = {
    "gretel": SourceSpec(
        key="gretel",
        dataset_id="gretelai/synthetic_text_to_sql",
        split="train",
        group="core",
        dialect="SQL",
        license="apache-2.0",
        downloads_2026_06_03=3176,
        likes_2026_06_03=661,
    ),
    "sql_create_context": SourceSpec(
        key="sql_create_context",
        dataset_id="b-mc2/sql-create-context",
        split="train",
        group="core",
        dialect="SQLite",
        license="cc-by-4.0",
        downloads_2026_06_03=3911,
        likes_2026_06_03=499,
    ),
    "clinton": SourceSpec(
        key="clinton",
        dataset_id="Clinton/Text-to-sql-v1",
        split="train",
        group="core",
        dialect="SQLite",
        license="apache-2.0",
        downloads_2026_06_03=906,
        likes_2026_06_03=73,
    ),
    "sqale": SourceSpec(
        key="sqale",
        dataset_id="trl-lab/SQaLe-text-to-SQL-dataset",
        split="train",
        group="core",
        dialect="SQL",
        license="mit",
        downloads_2026_06_03=901,
        likes_2026_06_03=16,
    ),
    "text_to_sql_mix_v2": SourceSpec(
        key="text_to_sql_mix_v2",
        dataset_id="DanielRegaladoCardoso/text-to-sql-mix-v2",
        split="train",
        group="core",
        dialect="SQL",
        license="apache-2.0",
        downloads_2026_06_03=697,
        likes_2026_06_03=0,
    ),
    "duckdb": SourceSpec(
        key="duckdb",
        dataset_id="motherduckdb/duckdb-text2sql-25k",
        split="train",
        group="duckdb",
        dialect="DuckDB",
        license="cc-by-sa-4.0",
        downloads_2026_06_03=81,
        likes_2026_06_03=43,
    ),
    "synsql": SourceSpec(
        key="synsql",
        dataset_id="seeklhy/SynSQL-2.5M",
        split="train",
        group="large",
        dialect="SQL",
        license="apache-2.0",
        downloads_2026_06_03=546,
        likes_2026_06_03=28,
        default_max_rows=250_000,
    ),
    "nstext2sql": SourceSpec(
        key="nstext2sql",
        dataset_id="NumbersStation/NSText2SQL",
        split="train",
        group="large",
        dialect="SQLite",
        license="other",
        downloads_2026_06_03=539,
        likes_2026_06_03=90,
        default_max_rows=250_000,
    ),
}


PROFILE_SOURCES: dict[str, list[str]] = {
    "core_clean": ["gretel", "sql_create_context", "clinton"],
    "core": ["gretel", "sql_create_context", "clinton", "sqale", "text_to_sql_mix_v2"],
    "schema_heavy": ["sqale"],
    "mix_v2": ["text_to_sql_mix_v2"],
    "duckdb": ["duckdb"],
    "large": ["synsql", "nstext2sql"],
    "all": [
        "gretel",
        "sql_create_context",
        "clinton",
        "sqale",
        "text_to_sql_mix_v2",
        "duckdb",
        "synsql",
        "nstext2sql",
    ],
}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def first_present(row: dict, names: Iterable[str]) -> str:
    for name in names:
        value = clean_text(row.get(name))
        if value:
            return value
    return ""


def sql_fingerprint(schema: str, question: str, sql: str) -> str:
    payload = "\n".join([schema.strip(), question.strip(), sql.strip()])
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


SQL_TABLE_RE = re.compile(
    r"\b(?:from|join|update|into|table)\s+[`\"[]?([A-Za-z_][\w.$-]*)[`\"\]]?",
    flags=re.IGNORECASE,
)
CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?([A-Za-z_][\w.$-]*)[`\"]?",
    flags=re.IGNORECASE,
)


def normalize_identifier(name: str) -> str:
    return name.rsplit(".", 1)[-1].strip("`\"[]").lower()


def sql_table_names(sql: str) -> set[str]:
    return {normalize_identifier(match.group(1)) for match in SQL_TABLE_RE.finditer(sql)}


def split_schema_blocks(schema: str) -> list[str]:
    if not schema:
        return []
    chunks = re.split(r"(?=CREATE\s+TABLE\s+)", schema, flags=re.IGNORECASE)
    return [chunk.strip(" ;\n\t") + ";" for chunk in chunks if chunk.strip()]


def slim_schema(schema: str, sql: str, max_chars: int) -> tuple[str, bool]:
    """Keep SQL-relevant CREATE TABLE blocks when schemas are huge.

    Many Text2SQL corpora include whole database schemas with dozens of tables.
    If we later truncate the middle blindly, the target table can disappear.
    This function prefers blocks whose table names appear in the target SQL,
    then fills any remaining budget with leading context blocks.
    """
    if max_chars <= 0 or len(schema) <= max_chars:
        return schema, False

    blocks = split_schema_blocks(schema)
    if not blocks:
        return schema[: max_chars // 2] + "\n...\n" + schema[-max_chars // 2 :], True

    wanted = sql_table_names(sql)
    selected: list[str] = []
    selected_names: set[str] = set()
    for block in blocks:
        match = CREATE_TABLE_RE.search(block)
        table = normalize_identifier(match.group(1)) if match else ""
        if table and table in wanted:
            selected.append(block)
            selected_names.add(table)

    for block in blocks:
        if len("\n\n".join(selected + [block])) > max_chars:
            continue
        match = CREATE_TABLE_RE.search(block)
        table = normalize_identifier(match.group(1)) if match else ""
        if table and table in selected_names:
            continue
        selected.append(block)
        if len("\n\n".join(selected)) >= max_chars:
            break

    if not selected:
        return schema[: max_chars // 2] + "\n...\n" + schema[-max_chars // 2 :], True

    slimmed = "\n\n".join(selected)
    if len(slimmed) > max_chars:
        head = max_chars // 2
        tail = max_chars - head
        slimmed = slimmed[:head] + "\n...\n" + slimmed[-tail:]
    return slimmed, True


def normalize_row(spec: SourceSpec, row: dict, max_schema_chars: int) -> Optional[dict]:
    question = ""
    schema = ""
    sql = ""
    dialect = spec.dialect
    source_detail = ""
    extra_lines: list[str] = []

    if spec.key == "gretel":
        question = first_present(row, ["sql_prompt", "question", "prompt"])
        schema = first_present(row, ["sql_context", "schema", "context"])
        sql = first_present(row, ["sql", "query", "answer", "response"])
        source_detail = first_present(row, ["domain", "source"])
        complexity = first_present(row, ["sql_complexity"])
        task_type = first_present(row, ["sql_task_type"])
        if complexity:
            extra_lines.append(f"Complexity: {complexity}")
        if task_type:
            extra_lines.append(f"Task: {task_type}")
    elif spec.key == "clinton":
        question = first_present(row, ["instruction", "question", "prompt"])
        schema = first_present(row, ["input", "schema", "context"])
        sql = first_present(row, ["response", "sql", "query", "answer"])
        source_detail = first_present(row, ["source"])
    elif spec.key == "text_to_sql_mix_v2":
        question = first_present(row, ["instruction", "question", "prompt"])
        schema = first_present(row, ["schema_context", "schema", "context", "input"])
        sql = first_present(row, ["sql", "query", "answer", "response"])
        dialect = first_present(row, ["dialect"]) or dialect
        source_detail = first_present(row, ["source", "difficulty"])
        difficulty = first_present(row, ["difficulty"])
        if difficulty:
            extra_lines.append(f"Difficulty: {difficulty}")
    elif spec.key == "duckdb":
        question = first_present(row, ["prompt", "question", "instruction"])
        schema = first_present(row, ["schema", "context", "input"])
        sql = first_present(row, ["query", "sql", "answer", "response"])
        source_detail = first_present(row, ["category"])
        if source_detail:
            extra_lines.append(f"Category: {source_detail}")
    elif spec.key == "sqale":
        question = first_present(row, ["question", "prompt", "instruction"])
        schema = first_present(row, ["schema", "context", "input"])
        sql = first_present(row, ["query", "sql", "answer", "response"])
        joins = clean_text(row.get("num_joins"))
        tables = clean_text(row.get("num_tables"))
        if tables:
            extra_lines.append(f"Tables: {tables}")
        if joins:
            extra_lines.append(f"Joins: {joins}")
    elif spec.key == "nstext2sql":
        instruction = first_present(row, ["instruction", "question", "prompt"])
        output = first_present(row, ["output", "sql", "query", "answer", "response"])
        source_detail = first_present(row, ["source"])
        schema, question = split_instruction_schema_question(instruction)
        if not question:
            question = instruction
        sql = output
    else:
        question = first_present(row, ["question", "prompt", "instruction", "sql_prompt"])
        schema = first_present(row, ["schema", "context", "input", "sql_context", "schema_context"])
        sql = first_present(row, ["sql", "query", "answer", "response", "output"])

    if not question or not sql:
        return None

    schema, schema_slimmed = slim_schema(schema, sql, max_schema_chars)

    instruction_parts = [f"Dialect: {dialect}"]
    if extra_lines:
        instruction_parts.extend(extra_lines)
    if schema:
        instruction_parts.extend(["", "Schema:", schema])
    else:
        instruction_parts.extend(["", "Schema:", "(not provided)"])
    instruction_parts.extend(["", "Question:", question, "", "SQL:"])
    instruction = "\n".join(instruction_parts).strip()

    return {
        "instruction": instruction,
        "response": sql,
        "condition": "direct",
        "source": spec.dataset_id,
        "source_key": spec.key,
        "source_group": spec.group,
        "source_detail": source_detail,
        "dialect": dialect,
        "license": spec.license,
        "schema_slimmed": schema_slimmed,
        "fingerprint": sql_fingerprint(schema, question, sql),
    }


def split_instruction_schema_question(text: str) -> tuple[str, str]:
    marker = "-- Using valid SQLite, answer the following questions for the tables provided above."
    if marker in text:
        schema, rest = text.split(marker, 1)
        question = rest.replace("--", " ").strip()
        return schema.strip(), question
    if "### Instruction:" in text and "### Input:" in text:
        before_input, after_instruction = text.split("### Instruction:", 1)
        instruction_part, input_part = after_instruction.split("### Input:", 1)
        if "### Response:" in input_part:
            input_part = input_part.split("### Response:", 1)[0]
        return input_part.strip(), instruction_part.strip()
    return "", text.strip()


def iter_dataset_rows(spec: SourceSpec, streaming: bool) -> Iterator[dict]:
    from datasets import load_dataset

    kwargs = {
        "split": spec.split,
        "streaming": streaming,
        "trust_remote_code": spec.trust_remote_code,
    }
    dataset = load_dataset(spec.dataset_id, **kwargs)
    yield from dataset


def write_jsonl(rows: Iterable[dict], path: Path) -> tuple[int, int, Counter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    bytes_written = 0
    by_source: Counter = Counter()
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            f.write(line)
            count += 1
            bytes_written += len(line.encode("utf-8"))
            by_source[row.get("source_key", "unknown")] += 1
    return count, bytes_written, by_source


def build_sources(
    specs: list[SourceSpec],
    out_path: Path,
    streaming: bool,
    max_rows: dict[str, int],
    progress_interval: int,
    dedupe: bool,
    max_schema_chars: int,
) -> dict:
    seen: set[str] = set()
    stats: dict[str, dict] = {}

    def rows() -> Iterator[dict]:
        for spec in specs:
            kept = 0
            skipped = 0
            duplicate = 0
            cap = max_rows.get(spec.key, spec.default_max_rows)
            try:
                for raw_index, raw in enumerate(iter_dataset_rows(spec, streaming=streaming)):
                    row = normalize_row(spec, raw, max_schema_chars=max_schema_chars)
                    if row is None:
                        skipped += 1
                        continue
                    fp = row["fingerprint"]
                    if dedupe and fp in seen:
                        duplicate += 1
                        continue
                    seen.add(fp)
                    row["raw_index"] = raw_index
                    row["dataset_split"] = spec.split
                    kept += 1
                    yield row
                    if cap and kept >= cap:
                        break
                    if progress_interval and kept % progress_interval == 0:
                        print(f"{spec.key}: kept={kept:,} skipped={skipped:,} duplicate={duplicate:,}", flush=True)
            except Exception as exc:
                stats[spec.key] = {
                    "error": repr(exc),
                    "kept_rows_before_error": kept,
                    "skipped_rows_before_error": skipped,
                    "duplicate_rows_before_error": duplicate,
                }
                print(f"[WARN] {spec.key} failed: {exc!r}", flush=True)
                continue
            stats[spec.key] = {
                "kept_rows": kept,
                "skipped_rows": skipped,
                "duplicate_rows": duplicate,
                "max_rows": cap,
                "dataset_id": spec.dataset_id,
                "split": spec.split,
                "license": spec.license,
                "downloads_2026_06_03": spec.downloads_2026_06_03,
                "likes_2026_06_03": spec.likes_2026_06_03,
            }

    rows_written, bytes_written, by_source = write_jsonl(rows(), out_path)
    final_stats = {
        "output": str(out_path),
        "rows": rows_written,
        "bytes": bytes_written,
        "by_source": dict(by_source),
        "sources": stats,
        "streaming": streaming,
        "dedupe": dedupe,
        "max_schema_chars": max_schema_chars,
    }
    out_path.with_suffix(out_path.suffix + ".stats.json").write_text(
        json.dumps(final_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_stats, ensure_ascii=False, indent=2), flush=True)
    return final_stats


def resolve_sources(profile: str, source_keys: list[str]) -> list[SourceSpec]:
    keys: list[str] = []
    if profile:
        if profile not in PROFILE_SOURCES:
            raise ValueError(f"unknown profile {profile!r}; known: {sorted(PROFILE_SOURCES)}")
        keys.extend(PROFILE_SOURCES[profile])
    keys.extend(source_keys)
    if not keys:
        keys = PROFILE_SOURCES["core"]
    deduped: list[str] = []
    for key in keys:
        if key not in SOURCES:
            raise ValueError(f"unknown source {key!r}; known: {sorted(SOURCES)}")
        if key not in deduped:
            deduped.append(key)
    return [SOURCES[key] for key in deduped]


def parse_max_rows(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--max-rows entry must be key=N, got {item!r}")
        key, value = item.split("=", 1)
        if key not in SOURCES:
            raise ValueError(f"unknown source in --max-rows: {key!r}")
        out[key] = int(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILE_SOURCES), default="core")
    ap.add_argument("--source", action="append", default=[], help="Additional source key. Can be repeated.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max-rows", action="append", default=[], help="Per-source cap as key=N. 0 means no cap.")
    ap.add_argument("--progress-interval", type=int, default=10_000)
    ap.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--max-schema-chars",
        type=int,
        default=24_000,
        help="Trim very large schemas to SQL-relevant CREATE TABLE blocks before tokenization.",
    )
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(args.cache_dir) / "datasets"))
    specs = resolve_sources(args.profile, args.source)
    max_rows = parse_max_rows(args.max_rows)
    build_sources(
        specs=specs,
        out_path=Path(args.output),
        streaming=args.streaming,
        max_rows=max_rows,
        progress_interval=args.progress_interval,
        dedupe=args.dedupe,
        max_schema_chars=args.max_schema_chars,
    )


if __name__ == "__main__":
    main()
