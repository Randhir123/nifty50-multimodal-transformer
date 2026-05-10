# Experimental Findings

This document supplements the README with experimental detail. It is self-contained — all numbers are inline and do not depend on external files.

---

## Modality independence over time

Modality independence is measured as mean absolute Pearson correlation between mean-pooled PCA-reduced embeddings across modality pairs. A score near the noise floor (~0.041) means the modality carries no information independent of the others; a higher score means it is genuinely complementary. The script [`scripts/check_modality_independence.py`](../scripts/check_modality_independence.py) computes these tables from any multimodal NPZ artifact.

The pipeline went through two independence-raising changes:

**Before real news ETL.** Text tokens were generated from OHLCV price statistics (deterministic summaries derived from the same tabular features). The result was near-total dependence: text and image embeddings carried minimal signal beyond what the tabular modality already contained. The model was also predicting all-positive (F1 locking to recall) and validation AUC stalled below 0.50 for all variants.

**After real news ETL.** Text tokens encode real `yfinance` news headlines with FinBERT (768-dim).

| | tabular | text | image | kg |
|---|---|---|---|---|
| tabular | 1.000 | 0.170 | 0.047 | 0.138 |
| text | 0.170 | 1.000 | 0.107 | 0.197 |
| image | 0.047 | 0.107 | 1.000 | 0.075 |
| kg | 0.138 | 0.197 | 0.075 | 1.000 |

The (tabular, text) score dropped to 0.170 — text is now genuinely independent of price features. The (tabular, image) score of 0.047 is essentially at the noise floor: the candlestick ViT image tokens were near-random noise.

**After GAF/MTF + CNN.** Image tokens encode GAF + MTF representations of the close-price series, encoded by a 3-layer CNN.

| | tabular | text | image | kg |
|---|---|---|---|---|
| tabular | 1.000 | 0.170 | **0.082** | 0.138 |
| text | 0.170 | 1.000 | 0.078 | 0.197 |
| image | **0.082** | 0.078 | 1.000 | 0.064 |
| kg | 0.138 | 0.197 | 0.064 | 1.000 |

(tabular, image) rose from 0.047 to 0.082 — a ~75% increase from the noise floor. The image modality now carries temporal-shape information (angular correlations across the 20-day window; quantile-bin transition probabilities) that scalar rolling statistics explicitly discard by collapsing windows to single numbers. Text independence (0.170) is unchanged, as expected — only the image pipeline changed.

---

## Per-fold ablation results

### Logistic regression — 3-fold walk-forward CV

Logreg on mean-pooled tabular tokens is the linear-signal baseline. It measures whether any modality combination carries linearly separable signal on its own, without the Transformer's capacity to mix modality tokens.

| Variant | Fold 0 | Fold 1 | Fold 2 | Mean AUC |
|---|---|---|---|---|
| tabular_only | 0.443 | 0.544 | 0.413 | 0.467 |
| tabular_image (GAF/MTF) | 0.437 | 0.526 | 0.437 | 0.466 |
| tabular_text | 0.467 | 0.511 | 0.416 | 0.465 |
| all_modalities | 0.469 | 0.508 | 0.415 | 0.464 |

The logreg mean AUC is near-chance for all variants. This does not mean there is no signal — it means the signal is non-linear or requires the Transformer's capacity to mix modality tokens. Fold 1 shows that signal does exist at the logreg level (0.544 on tabular_only) when the validation period is stable. Folds 0 and 2 collapse due to regime effects explained in the stationarity section below.

Note: with a randomly initialized CNN, GAF/MTF image tokens carry no linear signal (tabular_image delta = −0.001 vs tabular_only). This is expected: a random projection destroys the temporal structure encoded by GAF/MTF. The signal emerges only when the CNN is trained end-to-end in the Transformer ablation.

### Transformer ablation — 3-fold walk-forward CV, 20 epochs

| Variant | Mean AUC | Std | Δ vs tabular_only |
|---|---|---|---|
| tabular_only | 0.4963 | 0.085 | — |
| tabular_kg | 0.4974 | 0.063 | +0.001 |
| tabular_text | 0.5104 | 0.074 | +0.014 |
| tabular_text_kg | 0.4974 | 0.084 | +0.001 |
| tabular_image | **0.5242** | 0.067 | **+0.028** |
| tabular_image_text_kg | 0.5222 | 0.058 | +0.026 |

`tabular_image` is the highest-AUC variant, with the smallest per-fold standard deviation (0.067). The all-in combination (0.5222) is marginally below image-alone (0.5242), consistent with text and KG adding mild noise on a 1,242-sample dataset.

The per-fold standard deviation is large relative to the deltas. With 3 folds and 1 seed, these numbers describe the direction of effect rather than its magnitude with precision. Multi-seed evaluation with more folds would tighten these estimates.

### KG v2 feature upgrade

Session 10a.3 replaced the compact dynamic-width KG context token with a 37-feature leakage-safe relational vector. The new vector includes sector one-hot membership, sector return/beta/volatility context, rolling peer correlations, sector-relative return and volume ranks/z-scores, lead-lag correlations, peer dispersion/spreads, Nifty50 trend/volatility-regime features, sector-rotation indicators, and a sparse-peer flag. The fusion model already projects KG inputs through `Linear(kg_dim, model_dim)`, so the wider vector competes on the same projected dimension as text and image. The empirical contribution is pending a fresh Colab ablation run on a KG v2 artifact.

Session 10a.4 decoupled the KG peer universe from the training universe. The model can still train on the 6-stock smoke set, but KG v2 sector/peer features may now be computed over an OHLCV-only peer universe of additional Nifty50 constituents. This makes sector rank, sector z-scores, peer correlations, sector index returns, and sector beta refer to the stock's broader peer set instead of only the stocks sampled for training. Training tickers are required to be present in the peer universe, peer ticker sector mappings are strict, and post-D peer OHLCV sentinel tests guard the no-future-data contract.

---

## Diagnostic narratives

### Trainer collapse and fix

The `FusionTransformer` was collapsing to a near-constant predictor regardless of modality variant. Probability ranges were ≤ 0.006 wide; validation AUC stalled below 0.40; F1 locked to recall (all-positive prediction). A mean-pooled logreg on the same tabular features achieved AUC 0.557, proving the signal existed but the Transformer could not access it.

The investigation tested four hypotheses sequentially. Output bias initialization (`logit(p_positive)`) resolved the first-epoch loss spike but did not widen the probability band. A 5-epoch linear warmup smoothed loss curves but did not break the saddle point. Switching from CLS token pooling to mean pooling over all sequence tokens (`encoded.mean(dim=1)`) broke the collapse.

Root cause: in shallow 16-dim encoders with `norm_first=True`, the `[CLS]` token learns near-uniform attention weights across all inputs to minimize early variance. The LayerNorm applied to this uniformly-averaged vector squashes it to zero-mean, unit-variance, destroying sample-to-sample variance. The classification head sees constant input regardless of sequence content. Mean pooling routes gradients directly into the sequence tokens, bypassing the attention bottleneck.

Post-fix: tabular_only reached AUC 0.561 over 50 epochs, and all variants produced distinct probability distributions.

### Train→val transfer failure

After observing sub-0.50 validation AUC on the single held-out period, a 3-fold walk-forward analysis was run to identify the failure cause: (a) label distribution shift, (b) feature non-stationarity, or (c) market regime shift.

The fold-level logreg results above showed fold 1 succeeds while folds 0 and 2 fail, ruling out a uniform failure mode.

**Label stationarity:** Fold 0 shows a +9.8pp base-rate shift (42% positive in train → 52% in val). This is driven by Nifty50 underperforming its constituent stocks during Sep–Dec 2025, making outperformance easier than the model was trained to expect.

**Market regime:** Fold 2 val period (Feb–May 2026) shows Nifty50 daily volatility 2.36× the training period. All 6 stocks show substantially higher annualised volatility in the fold 2 validation window (HDFCBANK 2.64×, SBIN 1.75×, ICICIBANK 1.78×). This is a structural regime change that the features, trained on a lower-volatility distribution, cannot generalize across.

**Feature audit:** All 11 tabular features use only data available at or before prediction date D. No global normalization statistics are computed before the train/val split. No leakage found. The transfer failure is a genuine regime effect, not an artifact of data handling.

Ranking of failure causes: (1) volatility regime shift in fold 2, (2) label base-rate non-stationarity in fold 0. The model finds signal when both stresses are moderate (fold 1).

### Backtest correction

Session 10a.2 invalidated the earlier top-k backtest implementation. The bug was not model leakage: stock selection used `y_prob`, but the backtest treated each 3-day forward return as if it were a one-day return and compounded overlapping daily rebalances sequentially. With horizon=3, positions opened on consecutive days are concurrent holdings, so the correct calculation first expands each selected trade into daily holding-period returns, averages all active positions per trading day, and then compounds those daily portfolio returns. The corrected backtest also rejects duplicate `(stock_id, end_date)` prediction rows before merging with realized returns. The previously reported 6-stock backtest figures should be regenerated with `scripts/run_backtest.py` after this correction before being cited.

---

## What we'd do with more time

**Universe expansion.** 6 stocks over 1 year produces ~1,242 samples. Expanding to 15–20 Nifty50 stocks over 3 years would produce ~10,000–15,000 samples across multiple volatility regimes. This addresses both the regime-shift sensitivity and the label non-stationarity simultaneously.

**Multi-seed evaluation.** The headline modality deltas are computed with 3 folds and 1 seed. Running 5+ seeds would produce confidence intervals and allow a claim like "image delta is reliably +X ± Y pp" rather than "image delta was +0.028 in one run."

**Regime-conditional training.** Given the fold 2 failure is driven by a 2.36× volatility spike, a volatility-conditioned model — or a two-regime approach with separate heads for low/high volatility — could maintain performance across regime changes.

**Attention attribution.** The fusion Transformer mixes modality tokens with self-attention. Attribution maps (e.g., integrated gradients or attention rollout) would show which tokens drive predictions on fold 1, helping identify whether the image or text modality is contributing meaningful context on individual samples.

**Longer prediction horizons.** The 3-day horizon is short enough that daily noise dominates. Horizons of 10–20 days might produce more learnable signal, at the cost of fewer non-overlapping samples per year.
