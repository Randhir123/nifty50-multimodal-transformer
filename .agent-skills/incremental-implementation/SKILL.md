---
name: incremental-implementation
description: Use while coding any multi-file change; implement in thin, testable slices.
---

# Incremental Implementation

Build one small, complete slice at a time. Each slice should compile, run tests, and be reviewable.

## Rules

- Implement the smallest useful change.
- Do not mix refactoring with new behavior unless the refactor is required.
- Do not add unrelated cleanup.
- Keep default execution CPU-friendly.
- Prefer additive changes over risky rewrites.
- Run targeted tests after relevant changes.
- Commit or PR with a clear description of what changed and why.

## ML/data-specific guidance

For this repo:

- First make toy data work.
- Then wire real data one modality at a time.
- Validate array shapes and sample counts at every boundary.
- Preserve chronological ordering.
- Never let rolling windows cross stock boundaries.
- Never let text or KG context include information after the sample date.
- Do not claim metric improvement without an ablation run.

## Slice examples

Good slices:

- Add a data container and validation tests.
- Add a CLI that writes a toy artifact.
- Wire real tabular windows into the existing artifact schema.
- Add KG numeric tokens from existing KG context.
- Add an ablation runner after the data contract is stable.

Bad slices:

- Rewrite all data loaders, models, and demo scripts at once.
- Add image, text, KG, and ablation in one PR.
- Add a new dependency when existing utilities are enough.

## Verification

Each slice should report:

- files changed;
- commands run;
- test results;
- generated artifacts;
- known limitations or follow-up tasks.
