# Prepared Training Data Samples

This folder contains small decoded samples from the tokenized KoHRM `V1Dataset`
folders used for training and SFT/LoRA preparation. The full datasets are too
large for Git, so each JSONL file stores only a few short excerpts.

## Files

- `stage1_fastcap.jsonl`
- `stage2_full_nocap.jsonl`
- `stage3_terminal.jsonl`
- `stage4_korean_tool_finance.jsonl`
- `sft_korean_legal.jsonl`
- `sft_bcai_finance.jsonl`
- `sft_toolbench.jsonl`
- `sft_swe_glm_mix.jsonl`
- `index.json`

Each JSONL row has:

- `instruction_text`: decoded prefix/instruction span.
- `response_text`: decoded response span supervised by the loss.
- `instruction_len_tokens`, `response_len_tokens`: original token lengths.
- `instruction_truncated`, `response_truncated`: whether the excerpt was cut.
- `epoch_0_index`: source sample index in the prepared dataset.

## Regenerate

```bash
python scripts/export_prepared_training_samples.py \
  --output samples/prepared_training_data \
  --samples-per-dataset 10
```

The exporter reads `metadata.json`, `tokens.npy`, and `epoch_0/*.npy` from each
prepared dataset. It does not modify the real training data.
