# Multimodal Ablation Runner

## Objective

Add a repeatable ablation runner that trains the fusion model across modality combinations and writes a compact results table for coursework evidence.

The runner should answer:

```text
What changes when we add KG, image, text, or all modalities to the same aligned sample artifact?
```

## Input contract

The runner consumes one aligned `.npz` artifact containing at least:

- `tabular_tokens`
- `y`
- `end_dates`

Optional modality keys:

- `kg_tokens`
- `image_tokens`
- `text_tokens`

## Variants

Default variants:

1. `tabular_only`
2. `tabular_kg`
3. `tabular_image`
4. `tabular_text`
5. `tabular_image_text_kg`

A variant is skipped if the required key is missing from the dataset, unless strict mode is requested.

## Output contract

The runner writes:

- `ablation_results.csv`
- `ablation_results.json`

Each row includes:

- variant name
- enabled modalities
- command used
- checkpoint path
- accuracy
- precision
- recall
- f1
- roc_auc

## Implementation plan

1. Add `scripts/run_ablation_study.py`.
2. Reuse the existing `src.training.train_fusion` CLI for each variant.
3. Load each saved checkpoint and extract `val_metrics`.
4. Write CSV and JSON outputs.
5. Add unit tests for variant selection, missing-key skip behavior, and result serialization.
6. Add a focused CI workflow that runs the ablation runner on a tiny all-modality toy `.npz`.

## Acceptance criteria

```bash
pytest tests/unit/test_ablation_runner.py
python scripts/run_ablation_study.py \
  --dataset data/processed/multimodal_samples.npz \
  --output-dir data/processed/ablations \
  --epochs 1 \
  --batch-size 2 \
  --device cpu \
  --model-dim 16 \
  --num-heads 4 \
  --num-layers 1 \
  --ff-dim 32
```

## Non-goals

- no hyperparameter search;
- no statistical significance testing;
- no claim that any modality improves performance without reviewing results;
- no live market-data download.
