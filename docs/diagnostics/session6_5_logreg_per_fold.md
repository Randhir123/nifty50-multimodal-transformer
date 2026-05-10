# Session 6.5 Test 1 — Per-Fold Logistic Regression ROC-AUC

**Artifact**: data\processed\real_world_demo_full\real_world_multimodal_samples.npz  
**CV**: PurgedWalkForwardSplit(n_splits=3, horizon_days=3, embargo_days=0)  
**Features**: mean-pooled tabular tokens → (N, 11), StandardScaler on train set  
**Classifier**: LogisticRegression(max_iter=1000, C=1.0, random_state=42)  

## Date Ranges and Sample Counts

| Fold | Train start | Train end | Val start | Val end | N train | N val |
|---|---|---|---|---|---|---|
| 0 | 2025-07-02 | 2025-09-11 | 2025-09-15 | 2025-12-01 | 300 | 311 |
| 1 | 2025-07-02 | 2025-11-27 | 2025-12-01 | 2026-02-17 | 612 | 311 |
| 2 | 2025-07-02 | 2026-02-13 | 2026-02-17 | 2026-05-08 | 924 | 310 |

## ROC-AUC Results

| Fold | Train ROC-AUC | Val ROC-AUC | Prob range (min/mean/max) | Pred pos% | True pos% |
|---|---|---|---|---|---|
| 0 | 0.6180 | **0.4434** | 0.239 / 0.461 / 0.667 | 37.9% | 51.8% |
| 1 | 0.6017 | **0.5444** | 0.174 / 0.425 / 0.601 | 22.2% | 48.2% |
| 2 | 0.5896 | **0.4129** | 0.099 / 0.333 / 0.737 | 9.0% | 41.0% |

## Interpretation

Results are mixed across folds. See stationarity and feature-audit tests for additional evidence.