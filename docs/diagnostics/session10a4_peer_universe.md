# Session 10a.4 Peer-Universe KG Diagnostic

Status: implementation complete, experiment pending.

The KG v2 builder now accepts separate training and peer OHLCV dictionaries. Training remains on the configured sample tickers, while sector/peer features are computed over the larger OHLCV-only peer universe. The Colab notebook defaults to the 6-stock training set plus the 43 additional peer tickers from the session 10a.4 plan.

Pre-rerun comparison targets from the previous 6-stock KG v2 run:

| Metric | Previous value | New value |
|---|---:|---:|
| `(tabular, kg)` independence | 0.130 | pending |
| `tabular_kg` ROC-AUC delta | +0.045 | pending |
| Backtest Sharpe | 1.64 | pending |

Run the Colab notebook with `FORCE_REFRESH = False` and the cached raw folder. Any peer ticker that fails yfinance download is logged in `summary.md` under `download_failures` and omitted from `peer_tickers_loaded`; training ticker failures remain fatal.
