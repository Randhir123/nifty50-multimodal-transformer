---
name: documentation-and-adrs
description: Use when updating specs, README, final report docs, architecture diagrams, or architectural decisions.
---

# Documentation and ADRs

Document decisions and evidence, not just code.

## When to use

Use for:

- specs under `specs/`;
- README updates;
- final coursework report outline;
- architecture diagrams;
- ADRs for major design choices;
- experiment and ablation documentation;
- known limitations.

## Documentation expectations

For this repo, useful documentation should explain:

- the prediction objective;
- data sources and schemas;
- modality alignment rules;
- model/fusion architecture;
- evaluation protocol;
- commands to reproduce artifacts;
- limitations and future work.

## ADR guidance

Use `docs/decisions/ADR-NNN-short-title.md` for expensive-to-reverse decisions, such as:

- choosing the aligned NPZ schema;
- choosing toy-data-first CI strategy;
- choosing an image/text/KG encoder;
- changing label definitions or evaluation protocol.

ADR structure:

```markdown
# ADR-001: Decision title

## Status
Accepted

## Context
Why this decision is needed.

## Decision
What we decided.

## Alternatives considered
What else was considered and rejected.

## Consequences
Trade-offs, risks, and follow-up work.
```

## Red flags

- README commands are stale;
- final report overclaims performance;
- architecture docs ignore leakage risks;
- diagrams show modalities that are not actually wired into training;
- limitations are omitted.
