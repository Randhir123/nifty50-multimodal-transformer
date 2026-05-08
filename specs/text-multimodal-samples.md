# Text Multimodal Samples

## Objective

Wire company text records into the aligned multimodal sample artifact.

This extends the existing tabular, KG, and image sample path with:

- `text_tokens`: `[num_samples, text_dim]`

Each text token row must align one-to-one with existing `stock_ids` and `end_dates`.

## Input contract

The text token builder consumes:

- aligned `stock_ids` from `MultimodalSampleArrays`;
- aligned `end_dates` from `MultimodalSampleArrays`;
- a text-record CSV/dataframe with:
  - `stock_id`
  - `event_date`
  - `source_type`
  - `title`
  - `body_text`

## Output contract

The resulting `.npz` artifact preserves existing arrays and adds:

```text
text_tokens[i] corresponds to stock_ids[i] and end_dates[i]
```

The artifact remains compatible with fusion training using `--use-text`.

## Alignment and leakage rules

- For each sample row, only records for the same `stock_id` are visible.
- Only records with `event_date <= end_date` are visible.
- Records are sorted by recency and concatenated using the existing text utility.
- Future records must not affect earlier sample tokens.

## Implementation plan

1. Add `attach_text_tokens(...)` to return a copy of `MultimodalSampleArrays` with aligned `text_tokens`.
2. Reuse `build_text_tokens_for_samples(...)` and existing text normalization utilities.
3. Extend the CLI with `--text-records-csv`, `--text-top-k`, and `--text-dim` for real text artifacts.
4. Add unit tests for row alignment, future-text cutoff, empty-text fallback, and NPZ schema.
5. Add a focused CI workflow that builds a tabular+text NPZ and runs one CPU fusion epoch with `--use-text`.

## Acceptance criteria

```bash
pytest tests/unit/test_text_multimodal_samples.py
python scripts/build_multimodal_samples.py \
  --tabular-csv /tmp/tabular_samples.csv \
  --feature-cols feature_1,feature_2 \
  --text-records-csv /tmp/text_records.csv \
  --text-dim 16 \
  --output data/processed/tabular_text_multimodal_samples.npz \
  --window-size 3
python -m src.training.train_fusion \
  --dataset data/processed/tabular_text_multimodal_samples.npz \
  --use-text \
  --epochs 1 --batch-size 2 --device cpu
```

## Non-goals

- no large pretrained text model in this slice;
- no live news scraping;
- no PDF extraction pipeline;
- no ablation runner yet.
