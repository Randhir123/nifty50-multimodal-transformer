# Session 6.5 Test 3 — Tabular Feature Leakage Audit

**Feature builder**: `src/data/features.py::compute_technical_features`  
**Label builder**: `src/data/labels.py::generate_outperformance_label`  
**Window builder**: `src/data/multimodal_samples.py::build_tabular_multimodal_samples`  

## Global Normalization Check

- `build_tabular_multimodal_samples` stores raw feature values in the NPZ without any normalization.
- No `StandardScaler`, `MinMaxScaler`, or `zscore` is applied in the builder.
- The `LogisticRegression` baseline in session 6 applies `StandardScaler` fit only on the training split — correct.
- `FusionTransformer` training in `train_fusion.py` uses raw token values; `LayerNorm` inside the model is applied per-sample, not over the dataset. **No global normalization leakage.**

## Per-Feature Audit

| Feature | Computation | Timestamps used at D | Window contained in [D-W+1, D]? | Leakage? | Notes |
|---|---|---|---|---|---|
| `log_return_1d` | log(close[D] / close[D-1]) | close[D-1], close[D] | Yes | No | Single-day retrospective log return. |
| `cum_return_3d` | close[D] / close[D-3] - 1 | close[D-3], close[D] | Yes | No | 3-day trailing price return; fully within look-back. |
| `cum_return_5d` | close[D] / close[D-5] - 1 | close[D-5], close[D] | Yes | No | 5-day trailing price return. |
| `cum_return_10d` | close[D] / close[D-10] - 1 | close[D-10], close[D] | Yes | No | 10-day trailing price return. |
| `realized_vol_5d` | std(log_return_1d[D-4:D]) × sqrt(5), min_periods=5 | close[D-5], ..., close[D] | Yes | No | Rolling 5-day realised volatility of log returns; requires min 5 periods. |
| `realized_vol_10d` | std(log_return_1d[D-9:D]) × sqrt(10), min_periods=10 | close[D-10], ..., close[D] | Yes | No | Rolling 10-day realised volatility. |
| `high_low_range_over_close` | (high[D] - low[D]) / close[D] | high[D], low[D], close[D] | Yes | No | Same-day intraday range, normalised by close. No look-ahead. |
| `close_over_10dma_minus_1` | close[D] / mean(close[D-9:D]) - 1, min_periods=10 | close[D-9], ..., close[D] | Yes | No | 10-day moving average is computed on a rolling basis per stock; no cross-sample statistics. |
| `close_over_20dma_minus_1` | close[D] / mean(close[D-19:D]) - 1, min_periods=20 | close[D-19], ..., close[D] | Yes | No | 20-day moving average; same logic as 10dma. No global statistics used. |
| `volume_over_20d_avg` | volume[D] / mean(volume[D-19:D]), min_periods=20 | volume[D-19], ..., volume[D] | Yes | No | Relative volume vs 20-day rolling mean. Rolling mean uses only past volume. |
| `stock_minus_index_return` | pct_change(close[D]) - pct_change(nsei_close[D]) | close[D-1], close[D], nsei_close[D-1], nsei_close[D] | Yes | No | Same-day relative return vs NSEI. pct_change(1) uses [D-1, D] only. NOTE: this feature and the LABEL both use nsei_close, creating a structural correlation that could be picked up by logreg but whose direction (positive or negative) is dataset-period dependent. |

## Label Computation

```
stock_return_next_3d[D] = close[D+3] / close[D] - 1   ← uses D+3
nifty_return_next_3d[D] = nsei[D+3] / nsei[D] - 1     ← uses D+3
label[D] = 1 if stock_return_next_3d[D] > nifty_return_next_3d[D] else 0
```

The label is properly a forward quantity and is **not** part of the feature set. The label's `nsei[D]` component is distinct from the `stock_minus_index_return` feature (which uses `nsei[D-1]` and `nsei[D]`). No circular dependency.

## Summary

**No feature leakage found.** All 11 features use only data available at or before prediction date D. No global normalization statistics are computed over the full dataset before the train/val split. The feature engineering is clean.

### Structural Correlation Note

Although there is no leakage, `stock_minus_index_return` and the binary label share the Nifty50 index as a common reference. Specifically:

- Feature: `(close[D]-close[D-1])/close[D-1] - (nsei[D]-nsei[D-1])/nsei[D-1]` (relative to yesterday's index)
- Label: `close[D+3]/close[D] > nsei[D+3]/nsei[D]` (relative to today's index, over next 3 days)

This is not leakage, but it means the model is partially asking 'did the stock beat the index yesterday?' to predict 'will it beat the index tomorrow?' The predictive value of that question is regime-dependent: it might be positive (momentum) in some periods and negative (reversion) in others, contributing to cross-fold AUC variance.