# KG Multimodal Samples

## Objective

Wire lightweight knowledge graph context into the aligned multimodal sample artifact.

This extends the real tabular sample artifact with:

- `kg_tokens`: `[num_samples, kg_dim]`

The token rows must align one-to-one with existing `stock_ids` and `end_dates` from the multimodal sample contract.

## Input contract

The KG token builder consumes:

- a `networkx` market knowledge graph created by `src.kg.build_graph.build_market_knowledge_graph`;
- aligned `stock_ids` from `MultimodalSampleArrays`;
- aligned `end_dates` from `MultimodalSampleArrays`;
- optional recent-return rows with `stock_id`, `date`, and `recent_return`.

CLI support may additionally read:

- `--kg-stock-sector-csv` with `stock_id,sector_id`;
- `--kg-returns-csv` with `stock_id,date,recent_return`;
- `--kg-events-csv` with `stock_id,event_date,event_type`.

## Output contract

The resulting `MultimodalSampleArrays` preserves all existing required arrays and adds:

```text
kg_tokens[i] corresponds to stock_ids[i] and end_dates[i]
```

The output `.npz` remains compatible with fusion training using `--use-kg`.

## Alignment and leakage rules

- KG context is retrieved as of each sample end date.
- Event flags only use events visible in the configured lookback window ending at `end_date`.
- Recent-return aggregates only use rows where `date <= end_date`.
- No KG token may be created from a different stock/date row than its tabular sample.

## Implementation plan

1. Add a helper that retrieves KG contexts for aligned sample arrays.
2. Reuse existing `retrieve_kg_context` and `build_kg_tokens_from_contexts` utilities.
3. Add a helper to return a copy of `MultimodalSampleArrays` with `kg_tokens` attached.
4. Add CLI flags to build KG tokens when a tabular CSV and stock-sector CSV are provided.
5. Add unit tests for shape, row alignment, event cutoff behavior, and NPZ schema.
6. Update CI to run KG tests and a tabular+KG one-epoch smoke test.

## Acceptance criteria

```bash
pytest tests/unit/test_kg_multimodal_samples.py
python scripts/build_multimodal_samples.py \
  --tabular-csv /tmp/tabular_samples.csv \
  --feature-cols feature_1,feature_2 \
  --kg-stock-sector-csv /tmp/stock_sectors.csv \
  --kg-returns-csv /tmp/kg_returns.csv \
  --kg-events-csv /tmp/kg_events.csv \
  --output data/processed/tabular_kg_multimodal_samples.npz \
  --window-size 3
python -m src.training.train_fusion \
  --dataset data/processed/tabular_kg_multimodal_samples.npz \
  --use-kg \
  --epochs 1 --batch-size 2 --device cpu
```

## Non-goals

- no graph neural network;
- no learned graph embeddings;
- no image/text token wiring;
- no ablation runner yet.
