---
name: planning-and-task-breakdown
description: Use after a spec exists to break work into ordered, small, verifiable tasks.
---

# Planning and Task Breakdown

Break large work into small tasks that leave the repository working after each step.

## When to use

Use for:

- implementing a spec from `specs/`;
- planning multi-PR work;
- splitting model/data/evaluation work into safe slices;
- creating a final coursework readiness checklist.

## Task template

Each task should include:

```markdown
## Task N: Short title

**Goal:** What this task delivers.

**Acceptance criteria:**
- [ ] Specific, testable condition.
- [ ] Specific artifact, command, or behavior.

**Verification:**
- [ ] `pytest ...`
- [ ] `python ...`

**Likely files:**
- `src/...`
- `tests/...`

**Dependencies:** Task numbers or `None`.
```

## Project slicing guidance

Prefer this sequence for multimodal work:

1. Define or update the spec.
2. Build a deterministic toy-data path.
3. Add unit tests for shapes, alignment, and leakage.
4. Add a small CLI or smoke command.
5. Wire one real modality at a time.
6. Add ablation/evaluation only after data contracts are stable.
7. Update README/report docs.

## Red flags

- a task touches too many unrelated modules;
- no verification command exists;
- image/text/KG work is mixed into one large PR;
- tests rely on network, GPU, or large data;
- a plan ignores future-leakage risk.
