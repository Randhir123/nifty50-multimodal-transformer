# Colab Demo Notebooks

This folder contains Colab notebooks for recording and teaching the real-world multimodal pipeline.

## Recommended recording path

Use this notebook for the main demo:

```text
all_in_one_real_world_demo.ipynb
```

It runs the full flow in one Colab runtime:

```text
conceptual framing
  -> setup and yfinance download
  -> features and labels
  -> aligned multimodal tensors
  -> fusion ablations
  -> actual-data visuals
```

This is the easiest notebook to use for recording because it avoids passing state across multiple notebooks.

## Architecture reference

```text
model_and_embedding_details.ipynb
```

It documents:

- tabular feature/window shape;
- image Transformer settings;
- text token dimension;
- KG token fields and dimension;
- fusion Transformer hidden size, number of heads, number of layers, and feed-forward dimension;
- why the model uses a Transformer encoder and no decoder;
- output head: dropout, linear layer, raw logit, `BCEWithLogitsLoss`, and sigmoid at inference.

## Optional/reference split notebooks

The split notebooks are still useful for teaching individual stages:

0. `project_introduction.ipynb`  
   Conceptual introduction notebook. Explains the project goals, the multimodal workflow, and how different modalities are transformed into embeddings. It uses conceptual workflow/architecture diagrams only — no numeric result or backtest visuals.

1. `00_setup_and_data_download.ipynb`  
   Clones/installs the repo in Colab and downloads real OHLCV/index data from yfinance.

2. `01_features_labels_and_alignment.ipynb`  
   Converts raw OHLCV into technical features and future outperformance labels.

3. `02_build_multimodal_artifact.ipynb`  
   Builds the aligned multimodal NPZ artifact with tabular, image, text, and KG tokens.

4. `03_train_fusion_and_ablate.ipynb`  
   Runs fusion ablations across modality combinations and plots actual ablation metrics.

5. `04_visualize_embeddings_and_demo_story.ipynb`  
   Generates actual-data embedding projections, a sample gallery, and demo summary artifacts.

## Default universe

```text
Stocks:
- RELIANCE.NS
- TCS.NS
- INFY.NS

Benchmark:
- ^NSEI
```

Edit the `TICKERS` cell in the all-in-one notebook to use a larger universe.

## Why the all-in-one notebook?

The all-in-one notebook makes the demo easier to run live:

```text
one runtime
one stateful flow
one set of generated artifacts
```

The split notebooks are helpful for explanation, but the consolidated notebook is better for recording.

## Important limitation

These notebooks generate actual-data embeddings and ablation metrics. They do not yet implement a full portfolio backtest. Do not present ablation metrics as investment-grade backtest results.
