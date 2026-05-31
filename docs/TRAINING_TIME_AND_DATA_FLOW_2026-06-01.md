# KoHRM Training Time and Data Flow Notes

Snapshot time: `2026-06-01 07:14 KST`

This document records the current KoHRM long-run status, why it takes longer
than the upstream HRM-Text reference run, and how the prepared data flows into
training.

## Current Run

Active stage:

```text
run:          KoHRM-Text-1.4B-stage1d-hrm-fastcap-repeat3
pass:         4
stage:        1d, HRM fastcap data
resume from:  stage4c-korean-tool-finance-repeat2 epoch checkpoint
global step:  712,065 / 934,306
GPU:          8 x H200
GPU util:     99% on all 8 GPUs at the snapshot
VRAM:         about 127.6-129.9 GiB used per GPU
disk:         1.8T free on /home/work/.data at the snapshot
```

Pass 4 progress:

```text
epoch4 start offset:       702,956
epoch4 total planned:      230,350 steps
epoch4 completed:          9,109 steps
epoch4 progress:           3.95%
current stage1d progress:  9,109 / 80,756 steps = 11.28%
```

Current speed at the snapshot:

```text
observed speed:       about 1.01 steps/s
global batch:         180,224 token slots/step
throughput:           about 0.655B token slots/hour
```

## Expected Finish Times

KST estimates from the same snapshot:

```text
stage1d fastcap end:           2026-06-02 02:56
stage2d full_nocap end:        2026-06-03 01:09
stage3d terminal end:          2026-06-03 15:28
stage4d Korean/tool/finance:   2026-06-03 20:05
```

The final time can move by a few hours because checkpoint save, conversion,
upload, filesystem latency, or a stage handoff can add overhead.

## Wall-Clock Time

Using the local training start marker `2026-05-23 17:38 KST`:

```text
elapsed by snapshot:      205.6 hours = 8 days 13 hours 36 minutes
estimated final total:    265.9 hours = 11 days 1 hour 51 minutes
```

## Comparison to HRM-Text Reference

References:

- HRM-Text paper PDF: https://sapientinc.github.io/HRM-Text/assets/HRM_Text.pdf
- Upstream repo: https://github.com/sapientinc/HRM-Text

Reference run described by the paper/upstream materials:

```text
model family:        HRM-Text XL / 1B reference
hardware:            2 nodes x 8 H100 = 16 H100 GPUs
reported time:       about 46 hours
batch:               196,608 tokens
training duration:   60B token presentations
```

KoHRM run:

```text
model:               KoHRM-Text-1.4B
hardware:            8 H200 GPUs
batch:               180,224 token slots
final duration:      about 166.1B token presentations across 4 passes
tokenizer:           131,072 vocab byte-level BPE
context:             4,096 tokens
```

Numeric comparison:

```text
elapsed now vs 46h:      205.6 / 46 = 4.47x
final estimate vs 46h:   265.9 / 46 = 5.78x
KoHRM tokens vs 60B:     166.1 / 60 = 2.77x
KoHRM throughput:        0.655B token slots/hour
reference throughput:    60B / 46h = 1.304B token slots/hour
throughput ratio:        0.655 / 1.304 = 50.2%
```

The slower wall-clock time is therefore expected. It is mainly explained by:

```text
1. Token amount:
   KoHRM is scheduled for about 166.1B token presentations, 2.77x the
   60B-token reference duration.

2. Hardware count:
   The reference run uses 16 H100 GPUs. KoHRM uses 8 H200 GPUs. H200 is stronger
   per GPU, but the GPU count is half, so total throughput is still lower.

3. Model/vocab size:
   KoHRM uses a 131K tokenizer and a 1.4B-class model. The larger vocabulary
   increases embedding and LM-head work compared with the 1B reference setup.

4. Staged operation:
   KoHRM saves checkpoints, converts/upload selected checkpoints, and chains
   separate data stages. These steps are necessary for recovery and publishing,
   but add some overhead.
```

More time does not automatically guarantee a better model. The expected benefit
comes from more Korean/domain/terminal exposure and continued loss improvement.
Actual quality still has to be checked with held-out probes and later SFT/RL
experiments.

## Prepared Data Flow

The training input is not a directory of raw text files. It is a tokenized
HRM-Text `V1Dataset`.

```text
raw/source data
  -> tokenizer/preparation scripts
  -> prepared V1Dataset folder
  -> dataset_new.py
  -> pretrain.py
  -> checkpoint folder
  -> conversion/convert_to_hf.py
  -> Hugging Face model repo
```

Prepared dataset layout:

```text
prepared_dataset/
  metadata.json
  tokenizer.json                 optional if tokenizer is copied locally
  tokenizer_info.json            optional helper copy in some datasets
  tokens.npy                     contiguous int32 token array
  epoch_0/
    inst_start.npy
    inst_len.npy
    resp_start.npy
    resp_len.npy
```

`metadata.json` records the tokenizer, vocabulary size, max sequence length, and
total token count. `tokens.npy` stores all token IDs. The `epoch_0` arrays point
to instruction and response spans inside `tokens.npy`.

## What the Loader Does

`dataset_new.py` reads the prepared layout and emits PrefixLM batches.

For each sample:

```text
instruction span:
  - goes into the model as prefix/context
  - gets bidirectional prefix attention
  - labels are ignored when target_only=true

response span:
  - follows the prefix
  - gets causal output attention
  - contributes response-only cross-entropy loss
```

The loader concatenates many instruction/response samples into a token-budgeted
batch with `MultipackDistributedBatchSampler`. The configured `global_batch_size`
is token slots, not number of rows.

Current production setting:

```text
global batch:       180,224 token slots
world size:         8 GPUs
per-GPU batch:      22,528 token slots
context length:     4,096 usable tokens
metadata max len:   4,097 including autoregressive shift accounting
target_only:        true
```

## Current Major Prepared Datasets

```text
stage1_fastcap:
  path:   /home/work/.data/hrm_text_prepared/koterm_hrm_cleaned_fastcap_stage1_v1
  tokens: 14.554B

stage2_full_nocap:
  path:   /home/work/.data/hrm_text_prepared/koterm_hrm_cleaned_full_nocap_v1
  tokens: 14.554B

stage3_terminal:
  path:   /home/work/.data/hrm_text_prepared/local_terminal_conversations_ctx9k_resp6k_v1
  tokens: 9.387B

stage4_korean_tool_finance:
  path:   /home/work/.data/hrm_text_prepared/koterm_korean_tool_finance_mix_v1
  tokens: 3.021B
```

One full pass over data 1/2/3/4 is about `41.515B` token presentations. Four
passes are about `166.060B` token presentations.

## Sample Files

Small decoded samples are tracked under:

```text
samples/prepared_training_data/
```

They are generated from the real prepared datasets with:

```bash
python scripts/export_prepared_training_samples.py \
  --output samples/prepared_training_data \
  --samples-per-dataset 10
```

The sample files are for inspection only. They are not used by the production
training process and do not replace the large prepared datasets.
