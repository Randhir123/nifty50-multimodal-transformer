The notebooks here support running and demonstrating the project on Google Colab.

`colab/run_experiment.ipynb` runs the full pipeline (data prep, ablation, backtest, modality-independence check) end-to-end on Colab compute. Configure tickers, date range, and training settings in the marked config cell, then run the notebook unattended. Results write to a timestamped subdirectory under your Google Drive.

A demo notebook with step-by-step architecture and visualization is in progress and will land separately.

For users who prefer the command line, the equivalent functionality lives under `scripts/` — see the project README's "Reproducing the results" section.
