---
name: ci-cd-and-automation
description: Use when changing GitHub Actions, test gates, smoke checks, packaging, or automation.
---

# CI/CD and Automation

Automation should prove that the repo stays runnable after each change.

## When to use

Use for:

- GitHub Actions changes;
- pytest configuration;
- smoke test automation;
- dependency/install changes;
- scheduled or manual experiment workflows;
- artifact-generation checks.

## CI expectations for this repo

Default CI should be:

- CPU-only;
- deterministic;
- independent of live network data;
- small enough to run quickly;
- focused on unit tests and toy-data smoke tests.

Good CI gates:

```bash
pytest tests/unit
python scripts/build_multimodal_samples.py --toy-output data/processed/multimodal_samples.npz
python -m src.training.train_fusion \
  --dataset data/processed/multimodal_samples.npz \
  --use-image --use-text --use-kg \
  --epochs 1 --batch-size 4 --device cpu
```

Live yfinance or large Nifty50 experiments should be manual, optional, or scheduled — not required for every PR.

## Secrets and artifacts

- Never commit API keys or tokens.
- Never print secrets in CI logs.
- Do not commit large generated datasets, checkpoints, or reports.
- Use `.gitignore` and GitHub artifacts for generated outputs when needed.

## Red flags

- CI requires GPU;
- CI depends on a live market-data download;
- failing tests are skipped instead of fixed;
- generated artifacts are committed to git;
- automation changes have no verification story.
