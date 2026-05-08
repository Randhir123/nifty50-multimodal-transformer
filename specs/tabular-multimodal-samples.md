# Tabular Multimodal Samples

## Objective

Wire real tabular rolling-window data into the aligned multimodal sample artifact introduced by `specs/true-multimodal-sample-builder.md`.

This makes the required fusion arrays real and meaningful:

- `tabular_tokens`
- `y`
- `end_dates`
- `stock_ids`

Optional `image_tokens`, `text_tokens`, and `kg_tokens` remain future slices.

## Input contract

The builder accepts a CSV or dataframe with at least:

- `stock_id` — ticker or stock identifier;
- `date` — sample date;
- one or more numeric feature columns;
- `label` — binary target for that stock/date.

The caller passes the ordered feature columns explicitly. If omitted in the CLI, the script infers numeric columns except `label`.

## Output contract

The builder returns `MultimodalSampleArrays` and can save a `.npz` artifact with:

- `tabular_tokens`: `[num_samples, window_size, num_features]`
- `y`: `[num_samples]`
- `end_dates`: `[num_samples]`
- `stock_ids`: `[num_samples]`

The artifact remains compatible with `src.training.train_fusion.load_fusion_arrays` for tabular-only training.

## Alignment and leakage rules

- Build rolling windows independently per `stock_id`.
- Sort rows chronologically within each stock.
- Never let a rolling window cross from one stock into another.
- The label for a sample is the label at the rolling window end date.
- Features must already be computed using data available at or before the sample date.
- This slice does not compute labels; it consumes an existing `label` column.

## Implementation plan

1. Add `build_tabular_multimodal_samples(...)` to `src/data/multimodal_samples.py`.
2. Extend `scripts/build_multimodal_samples.py` with CSV input flags.
3. Add tests for:
   - per-stock rolling windows;
   - chronological sample order;
   - validation of required columns;
   - saved NPZ schema.

## Acceptance criteria

```bash
pytest tests/unit/test_tabular_multimodal_samples.py
pytest tests/unit/test_multimodal_sample_builder.py
python scripts/build_multimodal_samples.py \
  --tabular-csv data/toy/tabular_samples.csv \
  --feature-cols feature_1,feature_2 \
  --output data/processed/tabular_multimodal_samples.npz \
  --window-size 3
```

If the toy CSV path does not exist in a local checkout, tests still prove the behavior using in-memory dataframes.

## Non-goals

- no label generation;
- no feature engineering;
- no real image/text/KG token wiring;
- no ablation runner;
- no change to the fusion model.
