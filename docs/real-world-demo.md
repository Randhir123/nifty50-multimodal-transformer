# Real-World Multimodal Demo

This demo builds an aligned multimodal artifact from live yfinance OHLCV data and can optionally run the fusion ablation study.

It uses the same project pipeline as the rest of the repo:

```text
yfinance OHLCV snapshots
  -> technical features
  -> future outperformance labels
  -> rolling tabular windows
  -> candlestick chart images
  -> image tokens
  -> market-summary text records
  -> text tokens
  -> lightweight KG context
  -> KG tokens
  -> real_world_multimodal_samples.npz
  -> optional ablation results
```

## Quick run

```bash
python scripts/run_real_world_demo.py \
  --output-dir data/processed/real_world_demo \
  --period 9mo \
  --window-size 20 \
  --horizon-days 3
```

## Run with ablations

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

## Default universe

The default demo universe is intentionally small:

```text
RELIANCE.NS
TCS.NS
INFY.NS
benchmark: ^NSEI
```

Override it with:

```bash
python scripts/run_real_world_demo.py \
  --tickers RELIANCE.NS TCS.NS INFY.NS HDFCBANK.NS ICICIBANK.NS \
  --benchmark ^NSEI \
  --period 1y
```

## Outputs

The default output directory is `data/processed/real_world_demo/`.

Important artifacts:

```text
data/processed/real_world_demo/
├── raw/                                # local yfinance CSV snapshots
├── charts/                             # generated candlestick PNGs
├── tabular_samples.csv                 # real feature/label rows
├── text_records.csv                    # as-of market-summary text records
├── stock_sectors.csv                   # lightweight sector mapping
├── kg_returns.csv                      # recent-return features for KG context
├── event_records.csv                   # high-volume event flags
├── real_world_multimodal_samples.npz   # aligned multimodal artifact
├── DEMO_SUMMARY.md                     # run summary
└── ablations/                          # optional, when --run-ablations is set
    ├── ablation_results.csv
    └── ablation_results.json
```

## Notes and limitations

- This is a manual demo, not a CI gate, because it uses live yfinance downloads.
- The text modality in this demo uses deterministic market-summary text derived from real OHLCV features. It does not scrape live news, filings, or PDFs.
- The image modality uses real generated candlestick charts for each aligned sample date.
- The KG modality uses lightweight sector mapping, high-volume event flags, and recent-return aggregates.
- Do not interpret one-epoch ablation numbers as investment signals. They are coursework evidence that the multimodal artifact and training path run end to end.
