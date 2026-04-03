# Nifty-50 Multimodal Transformer

Predict which Nifty-50 stocks will outperform the index over the next 3 trading days by fusing tabular OHLCV features, candlestick chart images, news/analyst text, and a lightweight knowledge graph.

---

## Architecture overview

```
Raw OHLCV CSV
     │
     ▼
src/data          ← feature engineering, labels, rolling-window dataset
     │
     ├──► src/models/tabular.py   ← Transformer on numeric features
     ├──► src/models/image.py     ← CNN/ViT encoder on 60-day candlestick charts
     ├──► src/models/text.py      ← pre-trained LM encoder on news/analyst text
     └──► src/models/kg.py        ← graph context tokens from src/kg
                │
                ▼
         src/models/fusion.py     ← multimodal fusion Transformer
                │
                ▼
         src/training             ← time-based split, training loop, metrics
                │
                ▼
         src/viz                  ← ranking table, peer graph, embedding projection
                │
                ▼
         src/app                  ← workflow entry points, minimal API surface
```

Each branch (tabular, image, text, KG) can be trained independently and fused later.

---

## Build order (milestones)

| # | Milestone | Module(s) |
|---|-----------|-----------|
| 1 | Repo scaffold | *(this commit)* |
| 2 | Data pipeline | `src/data` |
| 3 | Candlestick charts | `src/data` → `data/interim/charts/` |
| 4 | Tabular Transformer baseline | `src/models/tabular.py`, `src/training` |
| 5 | Image branch | `src/models/image.py` |
| 6 | Text branch | `src/models/text.py` |
| 7 | Knowledge augmentation | `src/kg` |
| 8 | Multimodal fusion Transformer | `src/models/fusion.py` |
| 9 | Visualisation | `src/viz` |
| 10 | Operationalisation | `src/app` |

---

## Repo layout

```
.
├── data/
│   ├── raw/          # source CSVs (not committed)
│   ├── interim/      # charts, intermediate artefacts (not committed)
│   └── processed/    # final datasets (not committed)
├── notebooks/        # exploratory notebooks
├── src/
│   ├── data/         # pipeline: features, labels, dataset builder
│   ├── models/       # tabular, image, text, KG, fusion branches
│   ├── training/     # loops, splits, metrics
│   ├── kg/           # graph construction and retrieval
│   ├── viz/          # visualisation utilities
│   └── app/          # entry points and API surface
├── specs/            # design and task specs
├── pyproject.toml
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## Label definition

A stock is labelled **positive** for day *t* if its 3-day forward return exceeds the equal-weighted Nifty-50 index return over the same window. No future data leaks into features computed at day *t*.


## Dataset schema (first working pipeline)

### 1) Raw CSV input (stock OHLCV)

Required columns:

- `date` (parseable date)
- `open`
- `high`
- `low`
- `close`
- `volume`

### 2) Raw CSV input (NIFTY index)

Required columns:

- `date`
- `close` (renamed to `index_close` in the merged frame)

### 3) Engineered feature columns (`src/data/features.py`)

- `log_return_1d`
- `cum_return_3d`
- `cum_return_5d`
- `cum_return_10d`
- `realized_vol_5d`
- `realized_vol_10d`
- `high_low_range_over_close`
- `close_over_10dma_minus_1`
- `close_over_20dma_minus_1`
- `volume_over_20d_avg`
- `stock_minus_index_return`

### 4) Label columns (`src/data/labels.py`)

- `stock_return_next_3d`
- `nifty_return_next_3d`
- `label` (`1` if stock next-3d return > index next-3d return, else `0`)

### 5) Rolling Transformer dataset (`src/data/dataset.py`)

`create_rolling_transformer_dataset` returns:

- `X`: shape `[num_samples, window_size, num_features]`
- `y`: shape `[num_samples]`, label at each window end date
- `end_dates`: shape `[num_samples]`, timestamp of each prediction row

This keeps the pipeline CSV-first, lightweight, and model-agnostic.

## Milestone 3: candlestick chart generation

`src/viz/charts.py` provides deterministic chart utilities for building the image branch input without introducing model logic yet.

### Chart generation assumptions

- One chart corresponds to one `(stock, prediction_date)` sample.
- Each image uses the latest **60 trading rows** where `date <= prediction_date`.
- Rendered layers are fixed to keep outputs reproducible and deployment-friendly:
  - OHLC candlesticks
  - volume bars
  - 10-day moving average
  - 20-day moving average
- Input OHLCV data must include: `date`, `open`, `high`, `low`, `close`, `volume`.

### Output path convention

- Filename format: `{SYMBOL}_{YYYYMMDD}.png`
- Resolved path format: `{output_dir}/{SYMBOL}_{YYYYMMDD}.png`
- Example: `data/interim/charts/RELIANCE_20260203.png`

### Dataset row attachment strategy

- Use `attach_chart_paths(...)` to add a `chart_path` column to tabular sample rows.
- This keeps row-level linkage deterministic before any expensive rendering job runs.
- Use `generate_or_resolve_sample_chart(...)` in batch/serving jobs to lazily generate missing chart files or reuse existing files.
