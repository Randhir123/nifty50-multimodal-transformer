# Session 6: Corrected Backtest Results

## What Changed

The previous backtest emitted `WARNING: Could not find exact future_return columns. Falling back to y_true outperformance proxy.` This caused it to substitute `label * 0.01` as the stock return (+1% proxy for every outperformance-labeled sample) and `0.005` as a fixed benchmark return. The resulting "performance" numbers were disguised classification accuracy, not portfolio returns.

`scripts/run_backtest.py` now computes real 3-day forward returns:
- **Stock return**: `close.shift(-3) / close - 1` per stock group in `tabular_samples.csv`
- **Benchmark return**: 3-day forward return of `^NSEI` read from `raw/NSEI.csv`
- Old behavior is still accessible via `--use-y-proxy` flag

---

## 3-Stock Universe (RELIANCE.NS, TCS.NS, INFY.NS)

| Metric | Before (y-proxy) | After (real returns) |
|---|---|---|
| Model total return | +13.8% | **−13.9%** |
| Benchmark total return | +20.3% | **+8.6%** |
| Trading days | 37 | 34 |
| Rebalance dates | 37 | 37 |
| Avg position count | 1.0 | 1.0 |
| Sharpe ratio | −4.88 | **−4.14** |
| Max drawdown (model) | 0.0% | **−18.4%** |
| Return source | y_proxy | real_forward_returns |

*Note: Sharpe assumes risk-free rate = 0, 252 trading days/year.*

---

## 6-Stock Universe (RELIANCE, TCS, INFY, SBIN, ICICIBANK, HDFCBANK)

| Metric | Before (y-proxy) | After (real returns) |
|---|---|---|
| Model total return | +18.4% | **−52.1%** |
| Benchmark total return | +29.6% | **−16.1%** |
| Trading days | 52 | 49 |
| Rebalance dates | 52 | 52 |
| Avg position count | 1.0 | 1.0 |
| Sharpe ratio | −5.80 | **−6.75** |
| Max drawdown (model) | 0.0% | **−54.9%** |
| Return source | y_proxy | real_forward_returns |

*Note: Sharpe assumes risk-free rate = 0, 252 trading days/year.*

---

## Interpretation

**Backtest correction changed the headline result substantially.**

The y-proxy framing made both universes look like a modest underperformance story (model −6 to −11 pp vs benchmark). The corrected backtest reveals the model is actively losing money at scale: −13.9% on 3 stocks and −52.1% on 6 stocks against benchmarks of +8.6% and −16.1% respectively. The previous positive direction of the numbers was an artefact of the proxy always assigning a positive return to every outperformance-labelled day, regardless of what prices actually did.

This result is mechanically consistent with the logreg baseline findings: the model's ROC-AUC is ~0.44 on the validation fold (below chance), so the top-1 selection is effectively anti-predictive — the stock the model is most confident about tends to be the one that underperforms. On daily rebalancing over 50+ dates, that compounds to the −52% figure.

The backtest is now a real portfolio simulation. The signal-absence finding (documented in `session6_logreg_baseline.md`) is the primary explanation for both the sub-chance AUC and the negative real-return backtest.
