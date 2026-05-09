# Spec: Fix Model Collapse and Evaluation Metrics

## 1. Objective
Investigate and fix the severe model collapse observed in ablation diagnostics, where all variants show a `Val positive rate` of 0.0000, `Predicted positive rate` of 0.0000, and `Probability mean` of 0.0000.

## 2. Inputs and outputs
- **Inputs to review/modify**:
  - Evaluation script and metrics calculation (`ablation_runner` or `metrics.py`) to ensure `torch.sigmoid()` is applied to model logits.
  - Validation split boundaries or label generation logic to ensure the validation set actually contains positive outperformance samples.
- **Outputs**: 
  - Corrected probability scores in evaluations (must be in `[0, 1]`).
  - Restored positive samples in the validation splits.

## 3. Alignment and leakage rules
- Any fixes to data splitting must maintain the strict temporal purging and embargo rules in `src/training/cv.py`.

## 4. Implementation boundaries
- **In scope**: Fixing the missing `sigmoid` application on logits during inference/evaluation. Investigating and fixing the lack of `y=1` samples in the validation set.
- **Out of scope**: Changing the model architectures (`FusionTransformer`, etc.) which are properly designed to emit logits.

## 5. Testing strategy
- Run `pytest tests/unit/test_ablation_runner.py` (if it exists) to verify metric calculations.
- Run `python scripts/run_ablation_study.py --dataset ...` on a test dataset to ensure `Probability mean` > 0 and `Val positive rate` > 0.

## 6. Acceptance criteria
- `ablation_diagnostics.md` shows a `Probability mean` > 0 (typically around ~0.5).
- `Val positive rate` is > 0 in the validation splits.
- ROC-AUC scores are mathematically valid.

## 7. Open questions
- Is the `Val positive rate == 0.0000` solely an artifact of doing chronological splits on a tiny 3-stock toy universe over a short timeframe, or is there a bug in how `y` is calculated for the validation rows?