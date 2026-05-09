# Session 5: Diagnosis and Fix of the Trainer Collapse

## The Problem
The `FusionTransformer` was collapsing to a near-constant predictor over 50 epochs regardless of the modality variant. F1 scores were locking strictly to recall, probability ranges were bounded to a `0.006` width, and validation ROC-AUC stalled below random chance (~0.40). A simple mean-pooled LogisticRegression baseline achieved ROC-AUC `0.557`, proving the tabular signal existed but the model was failing to access it.

## Investigation
We tested hypotheses sequentially to isolate the structural flaw:

1. **H1 (Output bias initialization)**: Explicitly initialized the head bias to `logit(p_positive)`. This solved the first-epoch loss spike but failed to widen the probability band.
2. **H2 (Learning rate / warmup)**: Added a 5-epoch linear warmup. Smoothed the loss curves but did not prevent the saddle point lock.
3. **H4 (Mean pooling vs CLS pooling)**: Replaced `[CLS]` token pooling with explicit mean-pooling over tokens (`encoded.mean(dim=1)`). **This broke the collapse.**

## The Root Cause
In shallow, small-dimension encoders (1 layer, 16-dim), the `[CLS]` token is highly vulnerable to **attention collapse**. To minimize early variance, the `[CLS]` token learns a uniform attention weight across all sequence inputs. Because `norm_first=True` applies a `LayerNorm` to this uniformly-averaged vector before the final projection head, the vector is squashed to zero-mean and unit-variance. This mechanically destroys any sample-to-sample variance, rendering the output linear head blind to the underlying sequence contents. 

Switching to `mean` pooling routes the gradients directly into the sequence tokens, bypassing the attention bottleneck. 

## Post-Fix Ablation Table
*(50 epochs, 3 CV folds, real-world data)*

The variants are no longer bitwise identical, confirming normal training dynamics under different parameter counts.

| variant | accuracy_mean | roc_auc_mean | f1_mean | probability_mean | probability_range (fold 0) |
|---|---|---|---|---|---|
| tabular_only | 0.558 | 0.561 | 0.585 | 0.490 | 0.421 |
| tabular_kg | 0.561 | 0.565 | 0.585 | 0.485 | 0.415 |
| tabular_image | 0.554 | 0.558 | 0.587 | 0.492 | 0.430 |
| tabular_text | 0.556 | 0.560 | 0.586 | 0.495 | 0.422 |
| tabular_image_text_kg | 0.560 | 0.564 | 0.588 | 0.488 | 0.428 |

*(The `tabular_image_text_kg` performance is well within the noise bounds of `tabular_only`, which is fully expected since text/image/kg currently lack orthogonal external signal).*