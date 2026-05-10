# Experimental Findings

This document provides the experimental detail behind the README. The canonical headline run is Run C, run ID `20260510_102537`: 6 training stocks, a 49-stock Nifty50 peer universe, 1,260 samples, 45.1% positive label rate, 20 trading-day windows, 3 trading-day horizon, 3-fold purged walk-forward CV, 3-day embargo, 20 epochs, batch size 16, seed 42, GPU.

The goal of this document is not to claim investment-grade performance. It records what changed, what survived robustness checks, and what did not.

---

## 1. Modality independence over time

Modality independence is measured with distance correlation between mean-pooled modality embeddings. A value near the shuffled baseline means a modality is close to redundant/noisy relative to another modality. A higher value means it carries information that is not trivially present in the other representation.

| Pair | Pre-news / early pipeline | Post-news / Run A-style pipeline | Run C: post-GAF/MTF + 49-peer KG |
|---|---:|---:|---:|
| `(tabular, text)` | `> 0.9` estimated | `0.143` | `0.143` |
| `(tabular, image)` | `~0.13` estimated | `0.131` | `0.131` |
| `(tabular, kg)` | `0.135` with 4-feature KG | `0.130` with 4-feature KG | `0.166` with 37-feature KG and 49 peers |
| `(text, kg)` | not measured | not used as headline | `0.173` |
| shuffled baseline | `0.041` | `0.041` | `0.041` |

The text branch improved conceptually when deterministic price-derived summaries were replaced with real news records encoded by FinBERT. That change prevents the text channel from being a simple restatement of the tabular features. In Run C, `(tabular, text)=0.143`, comfortably above the shuffled baseline of `0.041`.

The image branch improved after the project moved away from rendered candlestick/ViT-style image input and toward GAF/MTF time-series images encoded with a compact CNN. In Run C, `(tabular, image)=0.131`, again above the shuffled baseline. This supports the idea that image tokens preserve temporal-shape structure that 20-day scalar rolling statistics do not fully capture.

The KG branch shows the clearest structural change. The original compact KG had only a few sector/peer/event fields and showed `(tabular, kg)` around `0.130–0.135`. The 37-feature KG with a full 49-stock peer universe raises `(tabular, kg)` to `0.166`. That does not make KG the strongest predictive modality, but it does show that the wider relational vector carries information that is structurally distinct from tabular features.

---

## 2. The KG iteration

The KG modality went through several configurations. This sequence is important because it explains why the final KG claim is deliberately modest.

| Configuration | Δ ROC-AUC vs `tabular_only` |
|---|---:|
| 4-feature KG, 6-stock peer universe, initial run | `−0.003` |
| 4-feature KG, 6-stock peer universe, after trainer fix | `+0.001` |
| 37-feature KG, 6-stock peer universe, Run A | `+0.045` |
| 37-feature KG, 14-stock peer universe equal to training universe, Run B | `−0.028` |
| 37-feature KG, 49-stock peer universe + 6-stock training universe, Run C | `+0.014` |

The `+0.045` Run A result was attractive but did not survive the peer-universe robustness check. It was produced by the upgraded 37-feature KG while still computing relational features against a small 6-stock peer set. In such a small universe, sector ranks, peer spreads, and correlation-like features can look predictive by chance on a small validation sample.

Run C is the more conservative estimate. It keeps the supervised training universe small at 6 stocks but computes KG features against 49 Nifty50 peers. The result is a smaller but more defensible `+0.014` ROC-AUC delta. The independence result `(tabular, kg)=0.166` versus shuffled baseline `0.041` says the KG vector is structurally meaningful. The ablation delta says the predictive value is real but small at this scale.

Run B, where the 37-feature KG over a 14-stock universe produced `−0.028`, is consistent with this conclusion rather than contradictory. At the current data scale, KG is in a noise-dominated regime: richer relational features can help, hurt, or look spuriously strong depending on peer coverage and validation period. The final writeup therefore treats Run C as the honest estimate and does not claim the earlier `+0.045` as a stable effect.

---

## 3. Per-fold ablation detail: Run C

Run C evaluates five modality variants with 3-fold purged walk-forward CV. The README reports mean ± standard deviation across folds. The fold-level results are useful because the standard deviations are large relative to the deltas.

| Variant | Fold 0 ROC-AUC | Fold 1 ROC-AUC | Fold 2 ROC-AUC | Mean ± std | Δ vs `tabular_only` |
|---|---:|---:|---:|---:|---:|
| `tabular_only` | not recorded in summary | not recorded in summary | not recorded in summary | `0.478 ± 0.072` | — |
| `tabular_kg` | not recorded in summary | not recorded in summary | not recorded in summary | `0.491 ± 0.070` | `+0.014` |
| `tabular_image` | not recorded in summary | not recorded in summary | not recorded in summary | `0.518 ± 0.048` | `+0.041` |
| `tabular_text` | not recorded in summary | not recorded in summary | not recorded in summary | `0.519 ± 0.092` | `+0.041` |
| `tabular_image_text_kg` | not recorded in summary | not recorded in summary | not recorded in summary | `0.496 ± 0.094` | `+0.019` |

The summary artifact available for Run C records mean and standard deviation but not the individual fold values. The standard deviations alone are enough to constrain the interpretation: with 3 folds and 1 seed, the ranking is more meaningful than the exact delta. Text and image are tied as the strongest single auxiliary modalities in this run, KG is smaller but positive, and the all-modality model adds less than either text-alone or image-alone.

The all-modality result is a useful negative finding. Combining tabular, KG, image, and text reaches `0.496 ± 0.094`, only `+0.019` over tabular and below `tabular_image` or `tabular_text`. This suggests modality interference or insufficient fusion capacity at the current dataset size. It does not mean multimodal fusion is useless; it means the compact 1-layer, low-dimensional fusion setup is not yet strong enough to exploit all channels simultaneously.

---

## 4. Methodology corrections

### Backtest correction

The first top-K backtest implementation used the binary label as a return proxy and produced apparent total returns around `+204%` with Sharpe `6.45`. That was implausible for a model whose ROC-AUC was near chance. It was not a valid portfolio simulation.

The corrected backtest uses real forward returns from the price data, aggregates concurrent holdings as daily portfolio returns under daily rebalancing, and deduplicates predictions across folds. Under that corrected path, Run C produces:

| Metric | Value |
|---|---:|
| Model total return | `+5.9%` |
| Benchmark total return | `−4.6%` |
| Trading days | `157` |
| Rebalance dates | `155` |
| Average position count | `2.96` |
| Sharpe, rf=0 | `0.94` |
| Max drawdown | `−20.2%` |

This is a much more credible result: modestly positive, with meaningful drawdown, and consistent with a weak but nonzero predictive signal. The important outcome is not that the strategy is ready to trade. It is that the project no longer reports a performance number that depends on a label proxy or overlapping-position compounding bug.

### Trainer collapse fix

Earlier training runs produced near-constant predictions. The validation probability range was approximately `0.006`, ROC-AUC was poor, and classification metrics were dominated by a single-class prediction pattern. A linear baseline on mean-pooled tabular features showed that some signal existed, so the failure was in the training/model path rather than the dataset alone.

The investigation identified two stabilizers. First, class imbalance handling with `BCEWithLogitsLoss(pos_weight=num_neg / num_pos)` made the loss reflect the training fold's label distribution. Second, switching the fusion model's recommended pooling mode from CLS pooling to mean pooling over encoded tokens avoided routing all early gradients through one learned token in a shallow 16-dimensional encoder. The current model code still supports `pooling="cls"`, but `pooling="mean"` is the default and recommended setting.

The architectural lesson is straightforward: in a compact coursework-scale Transformer, mean pooling is less elegant than CLS pooling but more robust. It sends gradient through all modality tokens and reduced the collapse risk in the shallow encoder setting used for CPU/GPU-friendly experimentation.

---

## 5. What we would do with more time

**Multi-seed evaluation.** Run C uses seed `42`. Running 5 or more seeds would put confidence intervals around the modality deltas and show whether text/image consistently dominate KG.

**Larger training universe.** The 49-stock peer universe improves KG quality, but the supervised training universe is still only 6 stocks. Expanding to 15–20 training stocks across more sectors and multiple years would test whether the image/text deltas survive a larger sample.

**Fusion architecture changes.** The all-modality model underperforms text-alone and image-alone. Next candidates are modality dropout, learned modality weighting, deeper encoder layers, wider `model_dim`, or separate modality-specific adapter layers before fusion.

**Regime-aware training.** Short-horizon equity prediction is regime-sensitive. A volatility-conditioned head or regime tag could prevent a model trained in one volatility regime from overgeneralizing to another.

**Longer horizons.** A 3-day horizon is noisy. A 10-day or 20-day horizon might better align news, sector rotation, and price-structure signals, though it would reduce the number of independent labels per year.
