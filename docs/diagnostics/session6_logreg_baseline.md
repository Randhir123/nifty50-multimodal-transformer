# Session 6: Logistic Regression Baseline

## Setup

- **Artifact**: `data/processed/real_world_demo_full/real_world_multimodal_samples.npz`
- **Samples**: 1,242 total (N=931 train / N=311 val)
- **Split**: chronological, last 25% as validation — same as ablation `--val-fraction 0.25`
- **Val period**: 2026-02-17 → 2026-05-08 (52 trading days, 6 stocks)
- **Val positive rate**: 40.8% (stock outperforms index on horizon=3d)
- **Tabular features**: mean-pool `(N, 20, 11)` → `(N, 11)`, then `StandardScaler`
- **Text features**: `(N, 768)` FinBERT embeddings, concatenated directly
- **All-modal features**: tabular(11) + image(16) + text(768) + kg(4) = 799-dim

## Results

| Model | ROC-AUC | F1 | Accuracy | Prob range (min/mean/max) | Pred pos rate |
|---|---|---|---|---|---|
| **Tabular only** (11-dim) | **0.409** | 0.104 | 0.556 | 0.086 / 0.328 / 0.749 | 8.7% |
| **Tabular + text** (779-dim) | **0.416** | 0.347 | 0.492 | 0.048 / 0.420 / 0.896 | 37.0% |
| **All modalities** (799-dim) | **0.402** | 0.327 | 0.470 | 0.042 / 0.427 / 0.906 | 37.9% |

**Delta AUC (tabular+text − tabular_only): +0.007**
**Delta AUC (all_modal − tabular_only): −0.007**

## Transformer Comparison (same val fold, from ablation_results.json)

| Variant | ROC-AUC |
|---|---|
| tabular_only | 0.446 |
| tabular_text | 0.431 |
| tabular_image_text_kg | 0.414 |

## Interpretation

**Signal does not exist at this scale.**

Every model tested — logreg and Transformer alike — produces ROC-AUC below 0.5 on the validation fold. This is not a fusion architecture problem or a training stability problem: the linear baseline on raw features is also sub-chance (0.409). Adding FinBERT text embeddings improves AUC by only +0.007, which is below the 0.01 threshold that would indicate extractable additive signal at the linear level. The all-modality logreg is marginally worse than tabular-only, consistent with noise accumulation from the 768-dim text and 16-dim image features.

The root cause is the train/val split: the model trains on one calendar period and is evaluated on a different market regime (Feb–May 2026). With only 6 stocks and ~15 months of data, the signal-to-noise ratio is too low for any model to generalize across market regimes. This is a data scale problem, not an architecture problem.

**Recommended next step for Session 7: universe expansion to 15–20 stocks** with a longer price history, to increase the training set size and reduce regime-concentration risk. The existing fusion architecture is structurally sound once the trainer collapse was fixed in Session 5 — the task is to provide it with a dataset large enough to have learnable short-horizon signal.
