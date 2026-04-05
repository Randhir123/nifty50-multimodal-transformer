# Nifty-50 Multimodal Transformer

Predict which Nifty-50 stocks will outperform the index over the next 3 trading days by fusing tabular OHLCV features, candlestick chart images, multi-source company text (news, filings, guidance, investor materials), and a lightweight knowledge graph.

---

## Architecture overview

```
Raw OHLCV CSV
     │
     ▼
src/data          ← feature engineering, labels, rolling-window dataset
     │
     ├──► src/models/tabular_transformer.py   ← Transformer on numeric features
     ├──► src/models/image_transformer.py ← lightweight patch Transformer on candlestick charts
     ├──► src/models/text.py      ← encoder on normalized multi-source company text
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
| 4 | Tabular Transformer baseline | `src/models/tabular_transformer.py`, `src/training/train_tabular.py`, `src/training/evaluate.py` |
| 5 | Image branch | `src/models/image_transformer.py`, `src/training/train_image.py` |
| 6 | Text branch (multi-source company text) | `src/models/text.py`, `src/data/text.py` |
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


## Milestone 4: tabular Transformer baseline

### Model architecture (`src/models/tabular_transformer.py`)

- Input tensor shape: **`[batch, window_len, feature_dim]`**
- `Linear(feature_dim -> model_dim)` input projection
- Sinusoidal positional encoding (supports variable sequence lengths)
- 2-4 layer `nn.TransformerEncoder` stack
- Sequence pooling via:
  - `mean` pooling (default), or
  - learned `CLS` token pooling
- Binary classification head outputs one **logit per sample**

### Expected dataset artifact

Training expects a `.npz` file with keys:

- `X`: shape `[num_samples, window_len, feature_dim]`
- `y`: shape `[num_samples]` with binary labels `{0,1}`
- `end_dates`: shape `[num_samples]` for chronological splitting

### Training entry point

```bash
python -m src.training.train_tabular \
  --dataset data/processed/rolling_windows.npz \
  --checkpoint-path data/processed/checkpoints/tabular_transformer.pt
```

What it includes:
- time-based train/validation split from `end_dates`
- train + validation loops with BCE-with-logits
- best-checkpoint saving by validation F1

### Evaluation helper entry point

Use metric utility in `src/training/evaluate.py`:

- accuracy
- precision
- recall
- F1
- ROC-AUC

This tabular baseline is intentionally lightweight and modular so it can be reused from API workflows (`src/app`) or batch jobs and extended later for multimodal fusion.


## Milestone 5: candlestick image branch

### Image branch architecture (`src/models/image_transformer.py`)

- Input tensor shape: **`[batch, 3, image_size, image_size]`**
- Patch embedding via `Conv2d(kernel_size=patch_size, stride=patch_size)`
- Learnable `CLS` token + sinusoidal positional encoding
- Lightweight `nn.TransformerEncoder` stack (default: 2 layers)
- Binary head returns one logit per sample
- `encode_images(...)` exposes per-sample embeddings (`[batch, model_dim]`) for later fusion wiring

### Expected chart input format

Image-branch training expects a sample-level table (`.csv` or `.parquet`) with:

- `date`: sample prediction date (used for chronological splitting)
- `chart_path`: image file path generated using `src/viz/charts.py` conventions
- `label`: project binary target (`1` if stock outperforms index over next 3 days, else `0`)

Recommended chart path pattern remains:

- `{output_dir}/{SYMBOL}_{YYYYMMDD}.png`

### Image training entry point

```bash
python -m src.training.train_image \
  --samples data/processed/image_samples.csv \
  --checkpoint-path data/processed/checkpoints/image_transformer.pt
```

What it includes:
- sample-level time-based train/validation split using `date`
- image-only dataset wrapper with minimal preprocessing (RGB read + resize + float conversion)
- train + validation loops with BCE-with-logits
- metric computation via `src/training/evaluate.py`
- best-checkpoint saving by validation F1

### Planned connection to multimodal fusion

This milestone intentionally trains only the image branch. The model already exposes a clean embedding interface (`encode_images`) so fusion modules can later consume image embeddings alongside tabular/text/KG embeddings without changing chart preprocessing or training data contracts.


## Milestone 6: multi-source company text branch

### Normalized company-text schema (`src/data/text.py`)

The text modality now uses one source-agnostic record schema:

- `stock_id`
- `event_date`
- `source_type`
- `title`
- `body_text`

Supported source categories include (non-exhaustive):

- news headlines/articles
- stock exchange filings
- management guidance updates
- investor presentation text
- other company-related text records

### Per-sample text construction

For each `(stock, date)` model sample, `build_company_text_input(...)` applies:

1. filter records with `event_date <= date`
2. sort by descending recency
3. concatenate top-k records into one model-ready input string

### Text encoder expectations (`src/models/text.py`)

- The text encoder consumes normalized per-sample strings and is agnostic to the original source type.
- This keeps preprocessing coursework-scale and modular for later fusion work.
- PDF-derived text is supported through lightweight direct extraction helpers (no OCR-heavy parsing in this milestone).
