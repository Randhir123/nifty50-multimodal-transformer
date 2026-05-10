# Session 10a.3 KG v2 Decision

Status: implementation complete, empirical outcome pending.

KG v2 now replaces the default artifact-level KG token with a 37-feature leakage-safe sector/peer/regime vector. The old KG builder remains reachable through `scripts/run_real_world_demo.py --kg-version v1`; the default is `--kg-version v2`.

No outcome is selected yet because this checkout does not contain the completed Colab artifact or prediction CSVs needed to run the required independence, logreg, and ablation comparisons. The honest next step is to rerun the Colab experiment using the cached raw data folder, then classify the result as:

- Outcome 1 if `tabular_kg` improves by more than +0.01 ROC-AUC and logreg also improves.
- Outcome 2 if independence improves but logreg/ablation deltas remain in [0, +0.01].
- Outcome 3 if the richer vector still contributes no measurable signal.
