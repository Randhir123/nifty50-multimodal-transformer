# Image Multimodal Samples

## Objective

Wire candlestick chart images into the aligned multimodal sample artifact.

This extends the existing tabular and KG sample path with:

- `image_tokens`: `[num_samples, image_dim]`

Each image token row must align one-to-one with the existing `stock_ids` and `end_dates` rows.

## Input contract

The image token builder consumes:

- aligned `stock_ids` from `MultimodalSampleArrays`;
- aligned `end_dates` from `MultimodalSampleArrays`;
- a chart directory containing deterministic chart images named by `src.viz.charts.resolve_chart_path`;
- image encoder configuration compatible with `src.models.image_transformer.ImageTransformer`.

## Output contract

The resulting `.npz` artifact preserves all existing arrays and adds:

```text
image_tokens[i] corresponds to stock_ids[i] and end_dates[i]
```

The artifact remains compatible with fusion training using `--use-image`.

## Alignment and leakage rules

- Chart paths are resolved from `(stock_id, end_date)` only.
- Chart images must have been generated from market rows where `date <= end_date`.
- This slice reads existing chart files; it does not generate charts from OHLCV rows.
- Missing chart files fail fast unless the caller explicitly chooses a future fallback mode.

## Implementation plan

1. Add chart path resolution for aligned sample arrays.
2. Add image loading and resizing into tensors.
3. Reuse `ImageTransformer.encode_images(...)` to produce CPU-friendly image embeddings.
4. Add `attach_image_tokens(...)` to return a copy of `MultimodalSampleArrays` with aligned image tokens.
5. Extend the CLI with chart directory and image encoder flags.
6. Add unit tests for path alignment, missing files, image-token shape, and NPZ schema.
7. Update CI to build a tiny chart fixture, produce image tokens, and run one CPU fusion epoch with `--use-image`.

## Acceptance criteria

```bash
pytest tests/unit/test_image_multimodal_samples.py
python scripts/build_multimodal_samples.py \
  --tabular-csv /tmp/tabular_samples.csv \
  --feature-cols feature_1,feature_2 \
  --image-chart-dir /tmp/charts \
  --output data/processed/tabular_image_multimodal_samples.npz \
  --window-size 3
python -m src.training.train_fusion \
  --dataset data/processed/tabular_image_multimodal_samples.npz \
  --use-image \
  --epochs 1 --batch-size 2 --device cpu
```

## Non-goals

- no image model training in this slice;
- no checkpoint loading;
- no chart generation from OHLCV rows;
- no text token wiring;
- no ablation runner yet.
