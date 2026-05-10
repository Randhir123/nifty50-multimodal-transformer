# Session 6.5 Summary

## One-Paragraph Interpretation

**All three sub-causes contribute partially, with market volatility regime shift being the dominant factor — the signal exists in stable regimes; the model fails when volatility spikes.**

Fold 1 (Dec 2025 – Feb 2026) achieves val ROC-AUC **0.544**, demonstrating that extractable signal does exist in this dataset at the right time. Fold 2 (Feb – May 2026) collapses to **0.413**: the Nifty50 daily volatility in that val period is **2.36× higher** than the training period — a severe market regime change (consistent with global macro turbulence in early 2026) that the features, all trained on a lower-volatility regime, cannot generalize across. Fold 0 (Sep – Dec 2025) also fails (**0.443**) despite similar volatility (ratio 0.94×); the dominant driver there is a **9.8pp label base-rate shift** (42% positive in train → 52% in val), likely caused by Nifty50 underperforming its constituent stocks during that period. Feature audit found **no leakage** in any of the 11 tabular features or in global normalization. The correct ranking of causes is: **(1) market volatility regime shift** in the most recent period — this is the single largest explainer and will recur in any dataset ending in early 2026; **(2) label non-stationarity** from small-universe effects — with 3–6 stocks, the outperformance-vs-Nifty label is sensitive to sector rotation and benchmark composition, both of which drift over 3-month windows. **Proposed next session**: expand the universe to 15–20 Nifty 50 stocks with at least 3 years of price history, so training data spans multiple volatility regimes and the label distribution is averaged over a larger peer set. This addresses both ranked causes simultaneously without any architecture changes.

---

## Evidence Summary

| Test | Finding |
|---|---|
| Test 1 — Per-fold logreg AUC | Fold 0: **0.443**, Fold 1: **0.544**, Fold 2: **0.413** |
| Test 1 — Train AUC | Fold 0: 0.618, Fold 1: 0.602, Fold 2: 0.590 — all show train signal |
| Test 2 — Label base-rate shift | Fold 0: **+9.8pp** (42% → 52%); Fold 1: +1.8pp; Fold 2: −6.5pp |
| Test 2 — Nifty50 vol ratio (val/train) | Fold 0: 0.94×; Fold 1: 1.27×; **Fold 2: 2.36×** |
| Test 3 — Feature leakage | **No leakage found** in any of the 11 features; no global normalization |

The correlation between fold health and regime stability is clear:

| Fold | Vol ratio | Label shift | Val AUC | Verdict |
|---|---|---|---|---|
| 0 | 0.94× | +9.8pp | 0.443 | Label shift is the main killer |
| 1 | 1.27× | +1.8pp | **0.544** | Signal exists; both stresses are moderate |
| 2 | 2.36× | −6.5pp | 0.413 | Volatility regime shift is the main killer |

See individual files for full tables:
- [session6_5_logreg_per_fold.md](session6_5_logreg_per_fold.md)
- [session6_5_stationarity.md](session6_5_stationarity.md)
- [session6_5_feature_audit.md](session6_5_feature_audit.md)
