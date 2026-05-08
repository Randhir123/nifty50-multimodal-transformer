---
name: code-review-and-quality
description: Use before opening, updating, or merging PRs to review correctness, readability, architecture, security, performance, and verification evidence.
---

# Code Review and Quality

Review every meaningful change before it enters `main`.

## Review axes

1. **Correctness** — does the change satisfy the spec/task?
2. **Data integrity** — are shapes, sample counts, labels, dates, and stock IDs aligned?
3. **Leakage safety** — no future information enters features.
4. **Readability** — names and control flow are clear.
5. **Architecture** — the change fits existing module boundaries.
6. **Security** — no secrets, credentials, or unsafe external data handling.
7. **Performance** — no unnecessary heavy computation in default paths.
8. **Verification** — tests/commands/artifacts prove the claim.

## PR expectations

A good PR description includes:

- what changed;
- why it changed;
- files or modules touched;
- commands run;
- test results;
- generated artifacts, if any;
- limitations and follow-up work.

## ML-specific review checklist

- [ ] Rolling windows are chronological.
- [ ] Windows do not cross stock boundaries.
- [ ] Labels are computed from future horizon only.
- [ ] Features do not use future information.
- [ ] Optional modalities align to the same sample rows.
- [ ] Tests cover toy data and edge cases.
- [ ] CI/default tests are CPU-friendly.
- [ ] No heavy artifacts are committed.

## Red flags

- broad PR with many unrelated concerns;
- silent dependency addition;
- no tests for a behavioral change;
- claims of multimodal improvement without ablation;
- generated datasets/checkpoints committed to git;
- vague final response like "all good" without evidence.
