# Korean BAR Round150 Legal RAG Workset

This folder packages 150 Korean bar-exam multiple-choice tasks into model-ready files.
Answer-free prompts and contexts are separated from scoring labels to avoid evaluation leakage.

## Files

- `questions/model_inputs.jsonl`: answer-free model inputs with context paths.
- `inference/openbook_chat_prompts.jsonl`: answer-free chat prompts with candidate legal documents.
- `inference/closedbook_chat_prompts.jsonl`: question-only chat prompts for no-retrieval baselines.
- `contexts/`: answer-free markdown context files.
- `targets/retrieval_targets.jsonl`: answer labels and gold legal document ids for scoring.
- `eval/manual_eval_tasks.jsonl`: open-book vLLM evaluation prompts.
- `eval/blind_manual_eval_tasks.jsonl`: prompt-only vLLM evaluation tasks with labels removed from rows.
- `sft/bar_exam_context_answer_sft.jsonl`: direct JSON-answer SFT rows.
- `sft/bar_exam_fts_agent_sft.jsonl`: search-action then answer-action SFT rows.
- `review/manual_review_sheet.csv`: compact manual review table.
- `review/manual_casebook.md`: human-readable scored casebook for manual audits.
- `reports/workset_quality_report.json`: row counts, answer distribution, context/gold-doc statistics.
- `manifest.json`: reproducibility metadata.

## Summary

```json
{
  "generated_utc": "2026-06-21T17:14:34Z",
  "manual_dir": "/home/work/.data/harness1/bar_exam_manual_work_20260621",
  "context_dir": "/home/work/.data/harness1/bar_exam_model_contexts_20260621/round15_rag_contexts_v3_model_ready",
  "output_dir": "HRM-Text/data/bar_exam/round150_rag_contexts_v2_20260621",
  "rows": 150,
  "subjects": {
    "공법": 40,
    "민사법": 70,
    "형사법": 40
  },
  "answer_free_files": [
    "questions/model_inputs.jsonl",
    "inference/openbook_chat_prompts.jsonl",
    "inference/closedbook_chat_prompts.jsonl",
    "eval/blind_manual_eval_tasks.jsonl",
    "contexts/*.md"
  ],
  "label_files": [
    "targets/retrieval_targets.jsonl",
    "eval/manual_eval_tasks.jsonl",
    "review/manual_casebook.md"
  ],
  "sft_files": [
    "sft/bar_exam_context_answer_sft.jsonl",
    "sft/bar_exam_fts_agent_sft.jsonl"
  ],
  "quality_report": "reports/workset_quality_report.json"
}
```
