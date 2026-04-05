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
     └──► src/kg/                 ← graph construction and retrieval context for fusion
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

## Testing and CI

### Run tests locally

```bash
# run everything
pytest

# run only unit + smoke scope (same selection as CI)
pytest -m "unit or smoke"

# run integration-only checks
pytest -m integration
```

### CI coverage today

GitHub Actions runs on every `push` and `pull_request` and currently verifies:

- deterministic **unit tests** for:
  - feature generation
  - label generation
  - rolling-window dataset creation
  - KG context retrieval
- lightweight **smoke tests** for:
  - candlestick chart generation
  - tabular Transformer forward pass
  - image branch forward pass
  - text branch forward pass
  - KG context retrieval

### Verified working paths

- data pipeline (`src/data/features.py` → `src/data/labels.py` → `src/data/dataset.py`)
- chart generation (`src/viz/charts.py`)
- tabular model (`src/models/tabular_transformer.py`)
- image forward path (`src/models/image_transformer.py`)
- text forward path (`src/models/text.py`)
- KG retrieval (`src/kg/build_graph.py` + `src/kg/query_graph.py`)

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

## Milestone 7: knowledge augmentation layer

### KG schema (`src/kg/build_graph.py`)

The milestone-7 graph is deliberately lightweight and deterministic.

**Node types**
- `index` (e.g. `NIFTY50`)
- `sector` (e.g. `Financials`, `IT`)
- `stock` (ticker/symbol IDs)
- `event_type` (e.g. `earnings`, `guidance`, `regulatory_filing`)

**Edge types**
- `index_contains_sector`
- `sector_contains_stock`
- `peer_in_sector` (between stocks in the same sector)
- `stock_has_event_type` (stores dated event history for flags)

### Example node and edge IDs

- index node: `index:NIFTY50`
- sector node: `sector:IT`
- stock node: `stock:TCS`
- event node: `event_type:earnings`

Example links:
- `index:NIFTY50 -- sector:IT`
- `sector:IT -- stock:TCS`
- `stock:TCS -- stock:INFY` (peer link)
- `stock:TCS -- event_type:earnings` with `event_dates=[...]`

### KG context retrieval contract (`src/kg/query_graph.py`)

`retrieve_kg_context(...)` returns a normalized dictionary for one `(stock_id, as_of_date)` sample:

- `schema_version`
- `stock_id`
- `as_of_date`
- `index_id`
- `sector_id`
- `peer_ids`
- `peer_count`
- `peer_avg_recent_return`
- `sector_avg_recent_return`
- `event_flags` (dictionary from event type to binary flag)

This payload is deterministic, JSON-serializable, and deployment-friendly for `src/app` workflows.

### How KG context connects to multimodal fusion

Milestone 7 does **not** add graph embeddings yet. Instead, it exposes two integration paths that fusion code can consume later:

1. **Structured feature path** via `kg_context_to_feature_dict(...)` for direct numeric/categorical feature fusion.
2. **Tokenization path** by converting normalized context fields into KG tokens in a later milestone.

This keeps graph construction/query logic stable while allowing Milestone 8 fusion modules to choose the final representation strategy.


## Milestone 8: multimodal fusion Transformer

### Fusion architecture (`src/models/fusion.py`)

`FusionTransformer` is the central multimodal encoder that fuses token/embedding streams from:

- tabular branch tokens (**required**)
- image branch embeddings or tokens (**optional**)
- text branch embeddings or tokens (**optional**)
- KG context features/tokens (**optional**)

Design details:

- each modality has its own input projection to a shared `model_dim`
- learned modality embeddings mark token origin (tabular/image/text/KG)
- modality tokens are concatenated into one sequence
- Transformer encoder layers perform cross-modal interaction
- output pooling supports either:
  - learned `CLS` token pooling (default), or
  - mean pooling
- final head emits one binary logit per sample

### Supported modality combinations (`src/training/train_fusion.py`)

Fusion training keeps one stable training contract and enables/disables modalities via CLI flags:

- `--use-image` for tabular + image
- `--use-text` for tabular + text
- `--use-image --use-text` for tabular + text + image
- `--use-image --use-text --use-kg` for tabular + text + image + KG

Example:

```bash
python -m src.training.train_fusion \
  --dataset data/processed/fusion_samples.npz \
  --use-image --use-text --use-kg \
  --checkpoint-path data/processed/checkpoints/fusion_transformer.pt
```

Expected `.npz` keys:

- required: `tabular_tokens`, `y`, `end_dates`
- optional (flag-dependent): `image_tokens`, `text_tokens`, `kg_tokens`

`src/training/evaluate.py` metrics are reused directly for train/validation reporting and checkpoint selection.

## Verified working paths (stabilization pass)

### Milestone audit: runnable entry points vs module-only components

| Milestone | Status in repository | Runnable entry point today | Notes |
|---|---|---|---|
| 2. Data pipeline | Implemented (`src/data/features.py`, `src/data/labels.py`, `src/data/dataset.py`) | ✅ Via smoke tests (`pytest`) and tabular verification script | No dedicated CLI module yet. |
| 3. Candlestick charts | Implemented (`src/viz/charts.py`) | ✅ Via smoke tests (`pytest`) | Utility API exists; no standalone CLI wrapper. |
| 4. Tabular Transformer baseline | Implemented | ✅ `python -m src.training.train_tabular ...` and `python scripts/verify_tabular_baseline.py` | End-to-end toy-data verification added. |
| 5. Image branch | Implemented | ✅ `python -m src.training.train_image ...` | Forward pass and chart generation covered in smoke tests. |
| 6. Text branch | Implemented | ✅ `python -m src.training.train_text ...` | Lightweight forward-pass smoke test uses `src/models/text.py`; training entry point uses `src/models/text_encoder.py`. |
| 7. Knowledge augmentation | Implemented (`src/kg/build_graph.py`, `src/kg/query_graph.py`) | ✅ Via smoke tests (`pytest`) | Utility API exists; no standalone CLI wrapper. |
| 8. Multimodal fusion Transformer | Implemented (`src/models/fusion.py`, `src/training/train_fusion.py`) | ✅ `python -m src.training.train_fusion ...` | Supports tabular+image, tabular+text, tabular+text+image, tabular+text+image+KG. |
| 9. Visualization | Implemented (`src/viz/ranking.py`, `src/viz/embeddings.py`, `src/viz/peer_graph.py`) | ✅ Via unit tests and workflow wrappers | Ranking tables, embedding maps, and peer-graph payload/image helpers are available. |
| 10. Operationalization | Implemented (`src/app/workflows.py`, `src/app/api.py`) | ✅ Python-callable workflows and endpoint-style wrappers | Wraps model inference, KG retrieval, ranking, embeddings, and peer-graph utilities for later API/OpenClaw exposure. |

### Toy data path (no external market data needed)

The repository now includes a tiny synthetic dataset under:

- `data/toy/stock_ohlcv.csv`
- `data/toy/index_ohlcv.csv`
- `data/toy/text_records.csv`
- `data/toy/event_records.csv`

### Commands to verify each component

```bash
# 1) Run component smoke tests
pytest tests/smoke/test_model_and_chart_smoke.py

# 2) Run tabular baseline end-to-end verification on toy data
python scripts/verify_tabular_baseline.py

# 3) (Optional) Run tabular training directly on generated toy artifact
python -m src.training.train_tabular \
  --dataset data/processed/verification/toy_rolling_windows.npz \
  --checkpoint-path data/processed/verification/tabular_from_cli.pt \
  --epochs 1 --batch-size 8 --device cpu
```

### What the smoke tests cover

- feature generation
- label generation
- rolling-window dataset creation
- candlestick chart generation
- tabular Transformer forward pass
- image branch forward pass
- fusion Transformer forward pass
- text branch forward pass
- KG context retrieval
- sample text assembly

## Milestone 9: visualization layer

`src/viz` now includes reusable utilities for dashboard/API-ready visual outputs:

- **Ranking tables** from model probabilities (`src/viz/ranking.py`)
- **Embedding projections** with PCA and t-SNE (`src/viz/embeddings.py`)
- **Peer graph structures and plots** from KG outputs (`src/viz/peer_graph.py`)

### Supported visual outputs

- Ranking dataframe with columns:
  - `stock_id`
  - `date`
  - `probability`
  - `predicted_label`
  - `rank`
- Embedding projection dataframe with:
  - metadata columns (for example `sample_id`, `stock_id`)
  - projected coordinates (`proj_x`, `proj_y`)
  - method metadata (`method`, variance/perplexity fields)
- Peer graph payload dictionary:
  - `nodes`: serializable node records (`id`, `node_type`, `entity_id`)
  - `edges`: serializable edge records (`source`, `target`, `edge_type`, `event_dates`)
- Optional static peer-graph image artifact via `plot_peer_graph(...)`

### Example workflow: embedding maps

```python
import numpy as np
import pandas as pd

from src.viz.embeddings import project_embeddings

embeddings = np.random.randn(128, 64)
metadata = pd.DataFrame({
    "sample_id": range(128),
    "stock_id": [f"S{i%10}" for i in range(128)],
})

pca_map = project_embeddings(embeddings, method="pca", metadata=metadata)
tsne_map = project_embeddings(embeddings, method="tsne", metadata=metadata, random_state=42)

# pca_map / tsne_map can be passed directly to matplotlib, plotly, or API responses
```

### Example workflow: ranked stock views

```python
import numpy as np
import pandas as pd

from src.viz.ranking import build_ranked_predictions

samples = pd.DataFrame(
    {
        "stock_id": ["TCS", "INFY", "RELIANCE"],
        "date": ["2026-01-05", "2026-01-05", "2026-01-05"],
    }
)
probabilities = np.array([0.63, 0.58, 0.41])

ranked = build_ranked_predictions(samples, probabilities, threshold=0.5)
# ranked is deterministic and reusable for notebooks, dashboards, and APIs.
```

Design principles used in this milestone:
- lightweight, typed interfaces
- deterministic ordering and random seeds
- no coupling to model training loops
- serialization-friendly outputs for later app/API integration


## Milestone 10: operationalization layer

`src/app` now provides a thin operationalization layer that wraps existing model, KG, and visualization components without duplicating core logic. The functions are plain Python-callable entry points now and can be exposed later through REST/queue/agent integrations.

### Workflow entry points (`src/app/workflows.py`)

- `rank_stocks(...)`
  - Inputs:
    - `samples: pd.DataFrame` with `stock_id`, `date`
    - Either `probabilities: np.ndarray` **or** a fusion-compatible `model` + modality tensors
  - Output:
    - `RankedStocksResult` containing a ranked dataframe (`stock_id`, `date`, `probability`, `predicted_label`, `rank`)
  - Uses:
    - `src/viz/ranking.py` (`build_ranked_predictions`)
    - Fusion inference contract through `predict_fusion_probabilities(...)`

- `analyze_stock(...)`
  - Inputs:
    - `stock_id`, `as_of_date`, ranked predictions dataframe
    - Optional KG graph and returns dataframe
  - Output:
    - `StockAnalysisResult` with one ranking row and optional KG context
  - Uses:
    - `src/kg/query_graph.py` (`retrieve_kg_context`)

- `compare_stocks(...)`
  - Inputs:
    - list of stock IDs, date, ranked predictions
    - optional KG graph/returns
  - Output:
    - `StockComparisonResult` with a comparison dataframe sorted by rank
  - Uses:
    - `analyze_stock(...)` internally (single-stock workflow composition)

- `show_peer_graph(...)`
  - Inputs:
    - KG graph, optional `output_path`
  - Output:
    - `PeerGraphResult` with payload and optional rendered image path
  - Uses:
    - `src/viz/peer_graph.py` (`build_peer_graph_payload`, `plot_peer_graph`)

- `show_embedding_map(...)`
  - Inputs:
    - embedding matrix, projection method (`pca` or `tsne`), optional metadata
  - Output:
    - `EmbeddingMapResult` containing projected dataframe
  - Uses:
    - `src/viz/embeddings.py` (`project_embeddings`)

### Minimal API-like surface (`src/app/api.py`)

`src/app/api.py` adds endpoint-style wrappers (`*_endpoint`) that demonstrate exactly how each workflow would be invoked by a transport layer.

- These wrappers currently return plain Python objects / pandas dataframes, keeping behavior deterministic and demo-friendly.
- No cloud deployment or OpenClaw runtime wiring is implemented in this milestone by design.

### Intended OpenClaw integration path (later milestone)

This layer is intentionally compatible with later OpenClaw integration by keeping:

- stable, explicit workflow signatures
- typed result objects
- JSON-serializable nested payloads where needed (for graph/KG context)
- separation between orchestration (`src/app`) and core logic (`src/models`, `src/kg`, `src/viz`)

When OpenClaw adapters are added later, they can call these workflows directly instead of re-implementing ranking, KG retrieval, or visualization preparation logic.
