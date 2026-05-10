# Session 6.5 Test 2 — Label and Market Stationarity

## 2a. Label Base Rate (Fraction of Positives) per Fold

| Fold | Train pos% | Val pos% | Delta (pp) |
|---|---|---|---|
| 0 | 42.0% | 51.8% | +9.8pp |
| 1 | 46.4% | 48.2% | +1.8pp |
| 2 | 47.5% | 41.0% | -6.5pp |

## 2b. NIFTY 50 Daily Return Statistics per Fold Period

| Fold | Train mean | Train std | Train days | Val mean | Val std | Val days | Vol ratio (val/train) |
|---|---|---|---|---|---|---|---|
| 0 | -0.0411% | 0.5228% | 50 | 0.0793% | 0.4914% | 53 | 0.94× |
| 1 | 0.0268% | 0.5119% | 102 | -0.0320% | 0.6502% | 54 | 1.27× |
| 2 | -0.0002% | 0.5600% | 155 | -0.1054% | 1.3228% | 53 | 2.36× |

## 2c. Per-Stock Annualised Volatility (Train vs Val) — Fold 2 Only

| Stock | Ann vol train | Ann vol val | Ratio |
|---|---|---|---|
| HDFCBANK.NS | 12.2% | 32.2% | 2.64× |
| ICICIBANK.NS | 15.1% | 26.9% | 1.78× |
| INFY.NS | 24.0% | 30.2% | 1.26× |
| RELIANCE.NS | 17.7% | 26.5% | 1.50× |
| SBIN.NS | 18.5% | 32.4% | 1.75× |
| TCS.NS | 21.8% | 25.9% | 1.19× |

## 2d. All Folds — Per-Stock Volatility Summary

| Fold | Stock | Ann vol train | Ann vol val | Ratio |
|---|---|---|---|---|
| 0 | HDFCBANK.NS | 12.1% | 11.1% | 0.92× |
| 0 | ICICIBANK.NS | 11.4% | 15.3% | 1.34× |
| 0 | INFY.NS | 23.7% | 21.7% | 0.91× |
| 0 | RELIANCE.NS | 17.2% | 15.0% | 0.87× |
| 0 | SBIN.NS | 11.7% | 14.8% | 1.27× |
| 0 | TCS.NS | 17.9% | 18.9% | 1.05× |
| 1 | HDFCBANK.NS | 11.6% | 14.1% | 1.22× |
| 1 | ICICIBANK.NS | 13.6% | 17.5% | 1.28× |
| 1 | INFY.NS | 22.7% | 26.3% | 1.16× |
| 1 | RELIANCE.NS | 16.5% | 20.0% | 1.21× |
| 1 | SBIN.NS | 13.5% | 25.4% | 1.88× |
| 1 | TCS.NS | 18.5% | 27.0% | 1.46× |
| 2 | HDFCBANK.NS | 12.2% | 32.2% | 2.64× |
| 2 | ICICIBANK.NS | 15.1% | 26.9% | 1.78× |
| 2 | INFY.NS | 24.0% | 30.2% | 1.26× |
| 2 | RELIANCE.NS | 17.7% | 26.5% | 1.50× |
| 2 | SBIN.NS | 18.5% | 32.4% | 1.75× |
| 2 | TCS.NS | 21.8% | 25.9% | 1.19× |