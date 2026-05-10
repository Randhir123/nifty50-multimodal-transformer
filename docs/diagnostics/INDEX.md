# Diagnostics Index

All diagnostic files in this directory, with one-line descriptions and headline numbers. Each entry here corresponds to a claim in [`README.md`](../../README.md) or [`docs/findings.md`](../findings.md).

---

## Session 5 — Trainer collapse investigation

| File | Description | Headline number |
|---|---|---|
| [`session5_findings.md`](session5_findings.md) | Root-cause analysis of the FusionTransformer constant-output collapse and the mean-pooling fix. Post-fix results across 5 variants. | tabular_only AUC 0.561 at 50 epochs, probability range recovered to 0.421 |
| [`session5_H1_bias.csv`](session5_H1_bias.csv) | Raw training curves for H1 (output bias initialization). Pre-fix: probability range ≤ 0.006 across all 50 epochs, all-positive prediction. | vl_prob_range max ≈ 0.008 — bias init alone does not break the collapse |
| [`session5_H1_bias_init.csv`](session5_H1_bias_init.csv) | Alternative bias-init experiment curves (same epoch format). Confirms output bias init stabilises first-epoch loss spike but does not widen the probability band. | vl_prob_range ≤ 0.008 — collapse persists |
| [`session5_H4_mean_pooling.csv`](session5_H4_mean_pooling.csv) | Training curves for H4 (CLS → mean pooling). This is the fix that broke the collapse. | Probability range widening observed; AUC recovery to >0.50 within 10 epochs |

---

## Session 6 — Real news ETL and corrected backtest

| File | Description | Headline number |
|---|---|---|
| [`session6_independence_post_news.csv`](session6_independence_post_news.csv) | Modality independence matrix after replacing price-derived text with real `yfinance` news + FinBERT. | (tabular, text) = 0.170; (tabular, image with candlestick ViT) = 0.047 ≈ noise floor |
| [`session6_logreg_baseline.md`](session6_logreg_baseline.md) | Single-fold logistic regression baseline on the session 6 artifact (before GAF/MTF). Shows sub-0.5 AUC across all variants on the held-out period. | tabular_only AUC 0.409; tabular+text AUC 0.416 — single 25% held-out split |
| [`session6_backtest_corrected.md`](session6_backtest_corrected.md) | Corrected backtest replacing the y-proxy (classification accuracy disguised as returns) with real 3-day forward returns. Both universes underperform the benchmark. | 3-stock: model −13.9% vs benchmark +8.6%; 6-stock: model −52.1% vs benchmark −16.1% |

---

## Session 6.5 — Transfer failure investigation

| File | Description | Headline number |
|---|---|---|
| [`session6_5_summary.md`](session6_5_summary.md) | One-paragraph interpretation and evidence table for the three sub-causes of train→val failure. Primary cause identified as market volatility regime shift. | Fold 1 AUC 0.544 (signal exists); Fold 2 AUC 0.413 (2.36× volatility regime shift) |
| [`session6_5_logreg_per_fold.md`](session6_5_logreg_per_fold.md) | Per-fold logistic regression AUC, sample counts, and date ranges for 3-fold walk-forward CV. Primary source for fold boundary dates. | Fold 0: 0.443, Fold 1: 0.544, Fold 2: 0.413 (tabular logreg) |
| [`session6_5_stationarity.md`](session6_5_stationarity.md) | Per-fold label base-rate and Nifty50 volatility statistics. Includes per-stock annualised volatility for fold 2. | Fold 0 label shift +9.8pp; Fold 2 Nifty50 vol ratio 2.36× |
| [`session6_5_feature_audit.md`](session6_5_feature_audit.md) | Row-by-row audit of all 11 tabular features confirming no future leakage and no global normalization before the split. | No leakage found in any feature |

---

## Session 7 — GAF/MTF image modality

| File | Description | Headline number |
|---|---|---|
| [`session7_decision.md`](session7_decision.md) | Decision document for the GAF/MTF + CNN image modality. Contains full evidence from all three diagnostics (independence, logreg, Transformer ablation) and the Outcome 1 (KEEP) decision. | tabular_image AUC 0.5242, +0.028 vs tabular_only — strongest single auxiliary modality |
| [`session7_independence_post_gaf.csv`](session7_independence_post_gaf.csv) | Modality independence matrix after replacing candlestick ViT with GAF/MTF + CNN. | (tabular, image) = 0.082, up from 0.047 — 75% increase above noise floor |
| [`session7_logreg.md`](session7_logreg.md) | Per-fold logreg AUC for 4 modality combinations on the GAF/MTF artifact. | Image adds no linear signal (delta −0.0005 vs tabular_only) — expected with random CNN |
| [`session7_logreg_with_image.md`](session7_logreg_with_image.md) | Single-split logreg with GAF/MTF image features included. Confirms marginal additive signal at the linear level. | tabular_image logreg AUC 0.559 vs tabular_only 0.557 (+0.002) |
| [`session7_ablation_results.csv`](session7_ablation_results.csv) | Intermediate Transformer ablation results from an earlier diagnostic pass (50 epochs, 3 variants, old image tokens). These numbers reflect pre-GAF/MTF performance; the canonical session 7 results are in `session7_decision.md`. | tabular_only AUC 0.561, tabular_image 0.560 — near-identical, confirming old image tokens were noise |
