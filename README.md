# Nifty50 Multimodal Transformer

A coursework-scale **knowledge-augmented multimodal Transformer** for Indian equities.

The goal is to predict whether a stock will outperform the Nifty50 benchmark over a short future horizon by combining four synchronized views of the same stock/date sample:

- **Tabular market data**: OHLCV-derived technical features and relative strength versus Nifty.
- **Chart images**: Gramian Angular Field (GAF) and Markov Transition Field (MTF) representations of the close-price series, encoded by a small 3-block CNN (`ImageCNN`). This replaces the earlier candlestick PNG + ViT approach. In 3-fold walk-forward CV on a 6-stock, 1-year dataset, `tabular_image` achieved the highest mean AUC of all variants (+2.8pp over tabular-only), confirming that GAF/MTF captures temporal shape information that scalar rolling statistics discard.
- **Text records**: real external news fetched via `yfinance` and encoded using FinBERT, combined with leakage-safe market summaries.
- **Knowledge graph context**: sector, peer, event, and recent-return context.

The central object is an aligned multimodal artifact:

```text
(stock_id, end_date)
  -> tabular_tokens
  -> image_tokens
  -> text_tokens
  -> kg_tokens
  -> y
```

This lets the fusion Transformer learn from numbers, chart structure, text context, and relational context in one shared embedding space.

---

## What this repo demonstrates

This repository is no longer just a collection of modality-specific modules. It now has an end-to-end path that builds real aligned multimodal samples and trains/evaluates fusion variants.

What's implemented:

- aligned multimodal sample contract;
- real tabular rolling-window sample builder;
- KG token wiring aligned by `stock_id + end_date`;
- chart-image token wiring using GAF+MTF numpy arrays and `ImageCNN.encode_images(...)` (Session 7: replaced candlestick PNGs + ViT with mathematical time-series images + CNN);
- text token wiring with `event_date <= end_date` cutoffs;
- fusion training across modality combinations;
- ablation runner that compares tabular-only versus multimodal variants;
- simple top-k portfolio historical backtesting (`scripts/run_backtest.py`);
- manual real-world demo using yfinance OHLCV snapshots;
- CI gates for tabular, KG, image, text, and ablation smoke paths.

Not yet claimed:

- real investment performance;
- heavy external document/PDF parsing;
- a statistically meaningful backtest;
- a trained model suitable for financial decisions.

The demo is intended to prove the **pipeline and representation story** first: real market data can be transformed into aligned multimodal embeddings and evaluated through ablations.

---

## Architecture at a glance

```text
yfinance OHLCV snapshots
  -> feature engineering + labels
  -> rolling tabular windows
  -> GAF + MTF time-series images (close-price, 20-day window)
  -> as-of-date text records (FinBERT-encoded)
  -> lightweight KG context
  -> aligned multimodal NPZ
  -> FusionTransformer (ImageCNN + tabular + text + KG)
  -> ablation results / rankings / visualizations
```

The fusion model receives modality-specific tokens, projects them into a common dimension, concatenates them with modality embeddings, and applies Transformer self-attention across modalities.

```text
tabular tokens  ----\
image tokens    -----\
text tokens     ------> shared sequence -> FusionTransformer -> prediction
kg tokens       -----/
```

Alignment is the critical design rule. Every modality row must describe the same `stock_id` and `end_date`, and no modality can use future information.

---

## Real-world demo

The main demo entry point is:

```bash
python scripts/run_real_world_demo.py \
  --output-dir data/processed/real_world_demo \
  --period 9mo \
  --window-size 20 \
  --horizon-days 3
```

With one-epoch ablations:

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

Default universe:

```text
Stocks:
- RELIANCE.NS
- TCS.NS
- INFY.NS

Benchmark:
- ^NSEI
```

To keep the demo transparent, the stock universe is not hidden in code. You can override it directly from the command line:

```bash
python scripts/run_real_world_demo.py \
  --tickers RELIANCE.NS TCS.NS INFY.NS HDFCBANK.NS ICICIBANK.NS SBIN.NS \
  --benchmark ^NSEI \
  --output-dir data/processed/real_world_demo \
  --period 9mo \
  --window-size 20 \
  --horizon-days 3 \
  --run-ablations \
  --epochs 1 \
  --batch-size 4 \
  --device cpu
```

For a first run or screen recording, start with the default 3-stock universe. Once the path works, rerun with 6-10 stocks for richer charts and ablation visuals.

The run writes:

```text
data/processed/real_world_demo/
├── raw/                                # yfinance CSV snapshots
├── charts/                             # generated candlestick PNGs
├── tabular_samples.csv                 # real feature/label rows
├── text_records.csv                    # as-of market-summary text records
├── stock_sectors.csv                   # lightweight sector mapping
├── kg_returns.csv                      # recent-return features for KG context
├── event_records.csv                   # high-volume event flags
├── real_world_multimodal_samples.npz   # aligned multimodal artifact
├── DEMO_SUMMARY.md                     # run summary
├── backtest/                           # portfolio backtest metrics and plots
└── ablations/                          # optional, when --run-ablations is set
    ├── ablation_results.csv
    └── ablation_results.json
```

For the detailed walkthrough and recording checklist, see:

```text
docs/real-world-demo.md
```

---

## How to demonstrate value

A strong demo should not start with code. It should start with the value question:

> Can the model see something useful when price behavior, chart structure, text context, and peer/sector relationships are represented together?

Recommended demo sequence:

1. **Show the raw modalities**  
   Open `tabular_samples.csv`, a few files from `charts/`, `text_records.csv`, and `event_records.csv`.

2. **Show alignment**  
   Open `DEMO_SUMMARY.md` and explain that every row is keyed by `stock_id + end_date`.

3. **Show the embedding artifact**  
   Inspect `real_world_multimodal_samples.npz` and its shapes: `tabular_tokens`, `image_tokens`, `text_tokens`, `kg_tokens`, `y`.

4. **Show fusion**  
   Explain that `FusionTransformer` projects all modalities into a common dimension and uses self-attention to mix them.

5. **Show ablations**  
   Open `ablations/ablation_results.csv` and compare:

   ```text
   tabular_only
   tabular_kg
   tabular_image
   tabular_text
   tabular_image_text_kg
   ```

6. **Be honest about backtesting**  
   Show the simple top-k portfolio backtest curve, but explain that it is a conceptual simulation without slippage or transaction costs.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Run tests:

```bash
pytest
```

Targeted checks:

```bash
pytest tests/unit/test_multimodal_sample_builder.py
pytest tests/unit/test_tabular_multimodal_samples.py
pytest tests/unit/test_kg_multimodal_samples.py
pytest tests/unit/test_image_multimodal_samples.py
pytest tests/unit/test_text_multimodal_samples.py
pytest tests/unit/test_ablation_runner.py
```

---

## Key commands

Build a toy all-modality artifact:

```bash
python scripts/build_multimodal_samples.py \
  --toy-output data/processed/multimodal_samples.npz
```

Run fusion training on an all-modality artifact:

```bash
python -m src.training.train_fusion \
  --dataset data/processed/multimodal_samples.npz \
  --use-image --use-text --use-kg \
  --epochs 1 \
  --batch-size 4 \
  --device cpu
```

Run ablations:

```bash
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

Generate visualization artifacts (embedding projections + ablation bar charts):

```bash
python scripts/visualize_real_world_demo.py \
  --demo-dir data/processed/real_world_demo
```

---

## Repository layout

```text
.
├── AGENTS.md                     # agent workflow instructions
├── .agent-skills/                # local agent skills for disciplined PRs
├── config/                       # ticker lists
├── data/                         # raw/interim/processed artifacts, not committed
├── docs/                         # demo and project documentation
├── scripts/                      # runnable demos and experiment drivers
├── specs/                        # implementation specs
├── src/
│   ├── app/                      # workflow/API-style wrappers
│   ├── data/                     # downloads, features, labels, sample builders
│   ├── kg/                       # graph construction and context retrieval
│   ├── models/                   # tabular, image, text, fusion models
│   ├── training/                 # training loops and metrics
│   └── viz/                      # ranking, embedding projection, peer graph utilities
└── tests/                        # unit, smoke, integration tests
```

---

## Cross-validation

The ablation runner supports purged walk-forward cross-validation to produce more reliable mean ± std estimates across multiple time folds.

```bash
python scripts/run_ablation_study.py \
  --dataset data/processed/multimodal_samples.npz \
  --output-dir data/processed/ablations \
  --cv-splits 3 \
  --horizon-days 3 \
  --embargo-days 5 \
  --epochs 1 \
  --batch-size 2 \
  --device cpu \
  --model-dim 16 \
  --num-heads 4 \
  --num-layers 1 \
  --ff-dim 32
```

`--cv-splits N` (N > 1) divides the dataset chronologically into N+1 equal chunks and runs N expanding-window folds. Two groups are excluded from each fold's training set:

- **Purged** samples whose label window (`end_date + horizon_days` calendar days) reaches into the validation period.
- **Embargoed** samples within `embargo_days` calendar days before the validation start.

The effective exclusion cutoff is `fold_start − max(horizon_days, embargo_days)`.

Outputs:

```text
ablations/
├── ablation_results.csv          # one row per variant, mean metrics across folds
├── ablation_results_folds.csv    # one row per variant × fold
├── ablation_results.json
└── ablation_diagnostics.md
```

Use `--single-split` (or the default `--cv-splits 1`) to keep the original single-fold subprocess behaviour, which is what the CI smoke step uses.

---

## Leakage guarantees

Every multimodal sample is keyed by `(stock_id, end_date)`. The pipeline enforces that for prediction date `D` with horizon `H`: tabular windows contain only rows with `date <= D`; text records are filtered to `event_date <= D`; candlestick chart filenames encode `D` (format `{SYMBOL}_{YYYYMMDD}.png`) and are generated from OHLCV data sliced to `date <= D`; KG context carries an `as_of_date` field equal to `D`; and the label is computed from prices at `D+1` through `D+H`, with NaN-labelled rows dropped so a sample is never emitted without valid future data.

These invariants are mechanically verified in [`tests/integration/test_no_leakage.py`](tests/integration/test_no_leakage.py) using deterministic synthetic data. The test covers six positive assertions and two negative tests (future text injection is dropped; truncated future raises rather than silently falling back). It runs on every push and pull request as a required CI gate. GAF/MTF image filenames follow the same convention (`{SYMBOL}_{YYYYMMDD}.npy`) and are covered by the same gate. The test does not cover subtler issues such as train/test contamination across time folds — that requires purged walk-forward cross-validation, which is a separate concern.

---

## Current limitations

- The real-world demo uses yfinance snapshots, so runs can differ over time unless you keep the generated `raw/` CSVs.
- The text modality in the real-world demo uses recent `yfinance` news and FinBERT, falling back to deterministic summaries for older historical dates.
- Ablation metrics are classification metrics from short training runs unless you increase epochs and tune properly.
- The included portfolio backtest is a simple top-k simulation without transaction costs or slippage.

---

## Responsible use

This is an educational/coursework project. It is not financial advice, not a trading system, and not a validated investment model.
