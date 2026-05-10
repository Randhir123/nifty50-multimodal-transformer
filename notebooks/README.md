# Notebooks

Two notebooks live here. Both run on Google Colab or locally from the repo root.

## [`colab/demo.ipynb`](colab/demo.ipynb)

A step-by-step walkthrough of the project, intended for an external reader. It runs on
pre-committed cached data — no yfinance downloads, no FinBERT model loading, no live
training. Expected runtime: under 5 minutes on a fresh Colab environment (most of that
is `pip install`).

Sections covered: the four modalities (tabular/text/image/KG) with visualizations for
each; the Transformer fusion architecture and parameter counts; walk-forward purged
cross-validation and leakage safety; training dynamics; modality ablation results (Run C);
the corrected backtest equity curve; and attention-based per-prediction modality attribution.

All numeric claims are anchored to the Run C artifact (`run_c_summary.json` in
`colab/demo_data/`). The notebook can be re-run from scratch and will produce the same
visualizations.

## [`colab/run_experiment.ipynb`](colab/run_experiment.ipynb)

An unattended experiment runner for producing new Run artifacts. Downloads OHLCV data,
encodes text with FinBERT, builds GAF/MTF images, runs all modality ablation variants
with walk-forward purged CV, and writes a `summary.md` to Google Drive. Use this to
reproduce or extend Run C (6-stock, 1-year, 3-fold CV) or to run new configurations
(different universe, period, or model hyperparameters).

---

For CLI users, the same pipeline is available via `scripts/run_real_world_demo.py` and
`scripts/run_ablation_study.py`. See the top-level [README](../README.md) for
reproduction commands.
