# Notebooks

Two notebooks live here. Both run on Google Colab or locally from the repo root.

## [`colab/demo.ipynb`](colab/demo.ipynb)

A step-by-step walkthrough for external readers. It runs on pre-committed Run C cached data: no yfinance downloads, no FinBERT model loading, and no live training. Expected runtime is under 5 minutes on a fresh Colab environment, with most of that spent on package setup.

Sections covered: the four modalities with visualizations, the Transformer fusion architecture and parameter counts, walk-forward purged CV and leakage safety, saved checkpoint diagnostics, Run C ablation results, the corrected backtest headline, and an ablation-derived modality contribution view. The production Run C model uses mean pooling and does not expose attention weights, so the demo uses the contribution fallback rather than reporting misleading attention values.

All numeric claims are anchored to `colab/demo_data/run_c_summary.json`, which records the 6-stock training universe, 49-ticker OHLCV peer universe, 1,260 samples, 37-feature KG vector, and corrected backtest metrics.

## [`colab/run_experiment.ipynb`](colab/run_experiment.ipynb)

An unattended experiment runner for producing new run artifacts. It downloads OHLCV data, encodes text, builds GAF/MTF images, runs modality ablations with walk-forward purged CV, and writes a `summary.md` to Google Drive. Use this to reproduce or extend Run C, or to run new configurations with different universes, periods, or model settings.

For CLI users, the same pipeline is available via `scripts/run_real_world_demo.py` and `scripts/run_ablation_study.py`. See the top-level [README](../README.md) for reproduction commands.
