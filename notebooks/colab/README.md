# Colab Demo Notebooks

This folder breaks the real-world demo into one notebook per step. Use this flow for recording and teaching instead of running one giant script.

## Notebook order

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

Edit the `TICKERS` cell in notebook 00 to use a larger universe.

## Why notebooks?

The notebooks make the demo easier to explain:

```text
conceptual framing
  -> raw data
  -> features and labels
  -> aligned multimodal tensors
  -> fusion ablations
  -> actual-data visuals
```

This also makes it clear which visuals come from real run artifacts and which conceptual diagrams are only explanation aids.

## Important limitation

These notebooks generate actual-data embeddings and ablation metrics. They do not yet implement a full portfolio backtest. Do not present ablation metrics as investment-grade backtest results.
