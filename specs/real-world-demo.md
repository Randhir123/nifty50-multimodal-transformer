# Real-World Multimodal Demo

## Objective

Add a manual demo that builds an aligned multimodal artifact from real market data and optionally runs the ablation workflow.

The demo uses:

- yfinance OHLCV snapshots for selected Indian equities;
- yfinance OHLCV snapshot for the Nifty50 benchmark;
- project technical features and outperformance labels;
- candlestick chart images generated from real OHLCV windows;
- lightweight KG context from stock-sector mappings and real recent-return aggregates;
- leakage-safe text records derived from as-of market summaries.

## Default universe

Small CPU-friendly default universe:

- `RELIANCE.NS`
- `TCS.NS`
- `INFY.NS`
- benchmark: `^NSEI`

## Label definition

For each stock/date sample:

```text
label = 1 if future_stock_return_horizon > future_index_return_horizon else 0
```

Features, chart images, KG context, and text records are constructed from data available at or before each sample date.

## Output directory

The script writes to `data/processed/real_world_demo/` by default:

- `raw/*.csv`
- `tabular_samples.csv`
- `text_records.csv`
- `stock_sectors.csv`
- `kg_returns.csv`
- `charts/*.png`
- `real_world_multimodal_samples.npz`
- `DEMO_SUMMARY.md`
- optional `ablations/ablation_results.csv`
- optional `ablations/ablation_results.json`

## Manual run

```bash
python scripts/run_real_world_demo.py \
  --output-dir data/processed/real_world_demo \
  --period 9mo \
  --window-size 20 \
  --horizon-days 3 \
  --run-ablations \
  --epochs 1 \
  --batch-size 4 \
  --device cpu
```

## Non-goals

- no CI gate requiring live network access;
- no live news/filing scraping;
- no claim that any modality improves performance without reviewing ablation results;
- no large training run by default.
