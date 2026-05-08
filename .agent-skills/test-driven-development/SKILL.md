---
name: test-driven-development
description: Use when implementing new behavior, data transformations, bug fixes, or model/evaluation contract changes.
---

# Test-Driven Development

Tests are the proof that the pipeline behaves as intended. For new logic, write or update tests before relying on the implementation.

## When to use

Use for:

- data-frame transformations;
- rolling-window generation;
- sample alignment;
- leakage prevention;
- model forward-pass contracts;
- metric computation;
- CLI behavior that writes artifacts;
- bug fixes.

## Testing priorities for this repo

1. **Shape contracts** — arrays have the expected rank and sample count.
2. **Temporal safety** — no future text/KG/chart information leaks into a sample.
3. **Stock boundaries** — rolling windows do not cross stock IDs.
4. **Determinism** — toy data and smoke tests are repeatable.
5. **CPU-friendly execution** — default tests do not require GPU or network.
6. **Behavior over internals** — assert outputs and artifacts, not implementation details.

## Recommended test levels

- Unit tests: pure data transforms, token builders, metric helpers.
- Smoke tests: one-epoch CPU training, small demo scripts, artifact creation.
- Integration tests: tiny end-to-end pipeline with toy or cached data.

## Common commands

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

Only say a command passed if it was actually run.

## Red flags

- a data change with no tests;
- a leakage fix without a regression test;
- a model contract change without shape tests;
- tests that depend on live yfinance/network access;
- skipped tests without a clear reason.
