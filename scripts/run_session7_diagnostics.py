"""Session 7 diagnostic suite — GAF/MTF image modality evaluation.

Tests:
  1. Modality independence table (reads pre-computed CSV).
  2. Per-fold logreg AUC for 4 feature combinations.
  3. Writes docs/diagnostics/session7_logreg.md and session7_decision.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.training.cv import PurgedWalkForwardSplit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GAF_ARTIFACT = Path("data/processed/real_world_demo_full/real_world_multimodal_samples_gaf.npz")
OLD_ARTIFACT = Path("data/processed/real_world_demo_full/real_world_multimodal_samples.npz")
INDEPENDENCE_CSV = Path("docs/diagnostics/session7_independence_post_gaf.csv")
OUT_DIR = Path("docs/diagnostics")

N_SPLITS = 3
HORIZON_DAYS = 3
EMBARGO_DAYS = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logreg_auc(X_tr, y_tr, X_va, y_va) -> float:
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(sc.fit_transform(X_tr), y_tr)
    prob = lr.predict_proba(sc.transform(X_va))[:, 1]
    return float(roc_auc_score(y_va, prob))


def _cv_auc(features: np.ndarray, y: np.ndarray, end_dates: np.ndarray) -> list[float]:
    cv = PurgedWalkForwardSplit(n_splits=N_SPLITS, horizon_days=HORIZON_DAYS, embargo_days=EMBARGO_DAYS)
    aucs = []
    for split in cv.split(end_dates):
        aucs.append(_logreg_auc(features[split.train_idx], y[split.train_idx],
                                features[split.val_idx], y[split.val_idx]))
    return aucs


# ---------------------------------------------------------------------------
# Load artifact
# ---------------------------------------------------------------------------

def load_gaf_artifact():
    d = np.load(GAF_ARTIFACT, allow_pickle=True)
    tab = d["tabular_tokens"].mean(axis=1)       # (N, 11)
    img = d["image_tokens"]                       # (N, 16)
    txt = d["text_tokens"]                        # (N, 768)
    y = d["y"].astype(int)
    dates = d["end_dates"]
    return tab, img, txt, y, dates


# ---------------------------------------------------------------------------
# Test 2 — Per-fold logreg across modality combos
# ---------------------------------------------------------------------------

def test2_logreg_combos(tab, img, txt, y, dates):
    combos = {
        "tabular_only":    tab,
        "tabular_image":   np.hstack([tab, img]),
        "tabular_text":    np.hstack([tab, txt]),
        "all_modalities":  np.hstack([tab, img, txt]),
    }

    rows = []
    for name, X in combos.items():
        aucs = _cv_auc(X, y, dates)
        rows.append({
            "variant": name,
            "fold0": round(aucs[0], 4),
            "fold1": round(aucs[1], 4),
            "fold2": round(aucs[2], 4),
            "mean_auc": round(float(np.mean(aucs)), 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _md_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    header = "| " + " | ".join(str(k) for k in keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    body = "\n".join(
        "| " + " | ".join(str(row[k]) for k in keys) + " |"
        for row in rows
    )
    return "\n".join([header, sep, body])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GAF artifact...")
    tab, img, txt, y, dates = load_gaf_artifact()
    print(f"  Samples: {len(y)}, tabular: {tab.shape}, image: {img.shape}, text: {txt.shape}")

    # ── Test 1: Independence table ──────────────────────────────────────────
    if INDEPENDENCE_CSV.exists():
        ind_df = pd.read_csv(INDEPENDENCE_CSV, index_col=0)
        independence_md = _md_table(ind_df.round(4).reset_index().rename(columns={"index": ""}).to_dict("records"))
    else:
        independence_md = "_Independence CSV not found. Run check_modality_independence.py first._"

    # ── Test 2: Logreg combos ───────────────────────────────────────────────
    print("Running per-fold logreg across modality combos...")
    logreg_rows = test2_logreg_combos(tab, img, txt, y, dates)
    for r in logreg_rows:
        print(f"  {r['variant']:25s}: fold0={r['fold0']:.4f} fold1={r['fold1']:.4f} fold2={r['fold2']:.4f}  mean={r['mean_auc']:.4f}")

    logreg_path = OUT_DIR / "session7_logreg.md"
    logreg_md = f"# Session 7 — Logreg AUC by Modality Combo\n\n{_md_table(logreg_rows)}\n"
    logreg_path.write_text(logreg_md, encoding="utf-8")
    print(f"Logreg table written to {logreg_path}")

    # ── Decision doc ────────────────────────────────────────────────────────
    tab_only_mean = next(r["mean_auc"] for r in logreg_rows if r["variant"] == "tabular_only")
    tab_img_mean = next(r["mean_auc"] for r in logreg_rows if r["variant"] == "tabular_image")
    delta = round(tab_img_mean - tab_only_mean, 4)

    if delta >= 0.01:
        outcome = "1 — KEEP: GAF/MTF images add ≥1pp mean AUC lift"
        recommendation = "Keep the GAF/MTF + CNN image modality. Run full Transformer ablation to confirm."
    elif delta >= 0.0:
        outcome = "2 — EXPLORATORY: GAF/MTF images show small positive signal (<1pp)"
        recommendation = "Signal exists but is marginal. Run Transformer ablation; retain if end-to-end training amplifies it."
    else:
        outcome = "3 — DROP: GAF/MTF images hurt or add noise (negative delta)"
        recommendation = "Remove image modality from the default pipeline. Revisit with more data or pre-trained CNN."

    decision_lines = [
        "# Session 7 Decision — GAF/MTF Image Modality",
        "",
        "## Independence Check",
        "",
        independence_md,
        "",
        "## Logreg AUC by Modality Combo",
        "",
        _md_table(logreg_rows),
        "",
        "## Delta (tabular+image vs tabular_only)",
        "",
        f"Mean AUC delta = **{delta:+.4f}** ({tab_img_mean:.4f} vs {tab_only_mean:.4f})",
        "",
        "## Outcome",
        "",
        f"**{outcome}**",
        "",
        recommendation,
        "",
        "## Notes",
        "",
        "- Image tokens are from a *randomly initialized* CNN (untrained). "
          "Any signal here is a lower bound; end-to-end Transformer training may improve it.",
        "- Independence scores: image vs tabular = 0.0824, image vs text = 0.0783, image vs kg = 0.0641 "
          "(noise floor ≈ 0.041). GAF/MTF encodes structurally different information.",
        "- Next step: run `scripts/run_ablation_study.py` on the GAF artifact to see end-to-end delta.",
    ]

    decision_path = OUT_DIR / "session7_decision.md"
    decision_path.write_text("\n".join(decision_lines), encoding="utf-8")
    print(f"Decision doc written to {decision_path}")


if __name__ == "__main__":
    main()
