# Agent Instructions for Nifty50 Multimodal Transformer

This repository implements a coursework-scale multimodal Transformer system for Indian equities.

The project goal is to predict whether a Nifty50 stock will outperform the Nifty50 index over a future horizon by combining:

- tabular OHLCV/time-series features;
- candlestick chart/image representations;
- company text from news, filings, guidance, or investor material;
- lightweight knowledge graph context;
- a multimodal fusion Transformer;
- visualization and workflow-ready outputs.

## Use the local skills

Before implementing any non-trivial change, inspect `.agent-skills/` and use the relevant `SKILL.md` files.

Default workflow:

1. Use `.agent-skills/spec-driven-development/SKILL.md` for new features, architecture changes, or unclear requirements.
2. Use `.agent-skills/planning-and-task-breakdown/SKILL.md` to break large work into small, ordered, verifiable tasks.
3. Use `.agent-skills/incremental-implementation/SKILL.md` while coding; prefer thin vertical slices.
4. Use `.agent-skills/test-driven-development/SKILL.md` for new behavior, bug fixes, and data-pipeline changes.
5. Use `.agent-skills/code-review-and-quality/SKILL.md` before opening or finalizing a PR.
6. Use `.agent-skills/documentation-and-adrs/SKILL.md` when updating specs, README, report docs, diagrams, or architectural decisions.
7. Use `.agent-skills/ci-cd-and-automation/SKILL.md` when changing GitHub Actions, test gates, or automation.

## Project-specific rules

- Do not jump straight to large code changes.
- Start with a spec when the task changes the model architecture, data contract, evaluation method, or final demo workflow.
- Keep each PR small and independently reviewable.
- Preserve chronological train/validation splits.
- Do not introduce future leakage: text, chart, KG, and tabular inputs for a sample must be available at or before that sample's prediction date.
- Build multimodal samples around the aligned `(stock_id, prediction_date)` contract.
- Keep default tests and demos CPU-friendly.
- Do not require GPU for CI.
- Do not commit heavy datasets, generated checkpoints, or secrets.
- Prefer deterministic toy data for unit and smoke tests.
- Report files changed, commands run, test results, generated artifacts, and remaining risks.

## Multimodal pipeline expectations

A valid multimodal sample should align all modalities to the same stock and date:

```text
(stock_id, prediction_date)
  -> tabular_tokens from the rolling OHLCV window ending at prediction_date
  -> image_tokens from the chart/window ending at prediction_date
  -> text_tokens from records with event_date <= prediction_date
  -> kg_tokens from graph context as of prediction_date
  -> y from future relative outperformance after prediction_date
```

For final-coursework readiness, prioritize evidence over novelty:

- aligned sample construction;
- leakage-safe data handling;
- ablation results;
- repeatable demo commands;
- clear architecture diagrams and report documentation;
- honest limitations.

## Verification commands

Prefer targeted commands first, then broader checks when relevant:

```bash
pytest tests/unit/test_multimodal_sample_builder.py
pytest tests/unit
pytest
python scripts/build_multimodal_samples.py --toy-output data/processed/multimodal_samples.npz
python -m src.training.train_fusion \
  --dataset data/processed/multimodal_samples.npz \
  --use-image --use-text --use-kg \
  --epochs 1 --batch-size 4 --device cpu
```

Only claim a command passed if it was actually run in the current environment or by CI.
