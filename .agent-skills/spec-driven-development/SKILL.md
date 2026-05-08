---
name: spec-driven-development
description: Use before coding non-trivial features, data-contract changes, model architecture changes, evaluation changes, or ambiguous requirements.
---

# Spec-Driven Development

Write a short specification before implementation. The spec is the source of truth for what is being built, why it matters, and how it will be verified.

## When to use

Use this skill for:

- new data pipelines or sample builders;
- changes to model or fusion architecture;
- changes to label construction or evaluation;
- final demo/report workflows;
- changes touching several files;
- any task where leakage, reproducibility, or acceptance criteria are unclear.

Skip this for obvious one-line fixes or documentation typo corrections.

## Required spec sections

Create or update a file under `specs/` with:

1. **Objective** — what is being built and why.
2. **Inputs and outputs** — schemas, file paths, array shapes, command flags.
3. **Alignment and leakage rules** — especially for `(stock_id, prediction_date)` samples.
4. **Implementation boundaries** — what is in scope and out of scope.
5. **Testing strategy** — unit/smoke/integration checks.
6. **Acceptance criteria** — exact commands or artifacts proving completion.
7. **Open questions** — unresolved assumptions or trade-offs.

## Project rules

For this repo, every multimodal feature must preserve this contract:

```text
(stock_id, prediction_date)
  -> tabular inputs available at or before prediction_date
  -> image inputs available at or before prediction_date
  -> text inputs with event_date <= prediction_date
  -> KG context as of prediction_date
  -> y from the configured future horizon
```

Do not claim a feature is truly multimodal unless aligned samples and ablation evidence exist.

## Verification

Before implementing, ensure:

- the spec exists in `specs/`;
- success criteria are specific and testable;
- expected commands are listed;
- leakage and chronological split rules are explicit;
- non-goals are clearly stated.
