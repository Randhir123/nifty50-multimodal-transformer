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
