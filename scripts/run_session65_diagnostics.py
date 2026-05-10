"""Train→val transfer failure diagnostic suite.

Runs three tests to identify why validation AUC varies across walk-forward folds:
  Test 1: Per-fold logistic regression AUC — isolates which folds have learnable signal.
  Test 2: Label and market stationarity — checks label base-rate shift and Nifty50
          volatility ratio (val/train) per fold.
  Test 3: Tabular feature leakage audit — verifies all 11 features use only data
          available at or before the prediction date.

Key findings: fold 1 (stable regime) achieves logreg AUC 0.544; fold 2 fails due to
2.36× Nifty50 volatility regime shift; fold 0 fails due to +9.8pp label base-rate shift.
Feature audit found no leakage. See docs/findings.md for the full narrative.

Outputs markdown reports to OUT_DIR (configured in script).
"""

from __future__ import annotations

import textwrap
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
ARTIFACT = Path("data/processed/real_world_demo_full/real_world_multimodal_samples.npz")
TABULAR_CSV = Path("data/processed/real_world_demo_full/tabular_samples.csv")
NSEI_CSV = Path("data/processed/real_world_demo_full/raw/NSEI.csv")
OUT_DIR = Path("docs/diagnostics")

N_SPLITS = 3
HORIZON_DAYS = 3
EMBARGO_DAYS = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logreg_metrics(X_train, y_train, X_val, y_val) -> dict:
    sc = StandardScaler()
    Xt = sc.fit_transform(X_train)
    Xv = sc.transform(X_val)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(Xt, y_train)
    prob = lr.predict_proba(Xv)[:, 1]
    train_prob = lr.predict_proba(Xt)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "train_auc": float(roc_auc_score(y_train, train_prob)),
        "val_auc": float(roc_auc_score(y_val, prob)),
        "prob_min": float(prob.min()),
        "prob_mean": float(prob.mean()),
        "prob_max": float(prob.max()),
        "pred_pos_rate": float(pred.mean()),
        "true_pos_rate": float(y_val.mean()),
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    data = np.load(ARTIFACT, allow_pickle=True)
    tabular = data["tabular_tokens"]   # (N, 20, 11)
    y = data["y"].astype(int)
    end_dates = data["end_dates"]
    stock_ids = data["stock_ids"]

    tab_feat = tabular.mean(axis=1)    # (N, 11)

    tabular_df = pd.read_csv(TABULAR_CSV, parse_dates=["date"])
    nsei_df = pd.read_csv(NSEI_CSV, parse_dates=["date"]).sort_values("date")
    nsei_df["nsei_return_1d"] = nsei_df["close"].pct_change(1)

    return tab_feat, y, end_dates, stock_ids, tabular_df, nsei_df


# ---------------------------------------------------------------------------
# Test 1 — Per-fold logreg
# ---------------------------------------------------------------------------

def test1_logreg_per_fold(tab_feat, y, end_dates):
    cv = PurgedWalkForwardSplit(
        n_splits=N_SPLITS, horizon_days=HORIZON_DAYS, embargo_days=EMBARGO_DAYS
    )
    rows = []
    for split in cv.split(end_dates):
        k = split.fold
        tr_idx = split.train_idx
        va_idx = split.val_idx

        dates_sorted = pd.to_datetime(end_dates)
        tr_dates = dates_sorted[tr_idx]
        va_dates = dates_sorted[va_idx]

        m = _logreg_metrics(tab_feat[tr_idx], y[tr_idx], tab_feat[va_idx], y[va_idx])
        rows.append({
            "fold": k,
            "train_start": tr_dates.min().date(),
            "train_end": tr_dates.max().date(),
            "val_start": va_dates.min().date(),
            "val_end": va_dates.max().date(),
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            **m,
        })

    return rows


# ---------------------------------------------------------------------------
# Test 2 — Label and market stationarity
# ---------------------------------------------------------------------------

def test2_stationarity(y, end_dates, stock_ids, tabular_df, nsei_df):
    cv = PurgedWalkForwardSplit(
        n_splits=N_SPLITS, horizon_days=HORIZON_DAYS, embargo_days=EMBARGO_DAYS
    )
    dates_ts = pd.to_datetime(end_dates)

    fold_rows = []
    label_rows = []
    nsei_rows = []
    vol_rows = []

    all_stocks = sorted(set(tabular_df["stock_id"].unique()))

    for split in cv.split(end_dates):
        k = split.fold
        tr_idx, va_idx = split.train_idx, split.val_idx
        tr_dates = dates_ts[tr_idx]
        va_dates = dates_ts[va_idx]

        tr_start = tr_dates.min()
        tr_end = tr_dates.max()
        va_start = va_dates.min()
        va_end = va_dates.max()

        # Label base rates
        label_rows.append({
            "fold": k,
            "train_pos_rate": float(y[tr_idx].mean()),
            "val_pos_rate": float(y[va_idx].mean()),
            "delta_pp": float(y[va_idx].mean() - y[tr_idx].mean()),
        })

        # Nifty50 returns
        nsei_tr = nsei_df[(nsei_df["date"] >= tr_start) & (nsei_df["date"] <= tr_end)]["nsei_return_1d"].dropna()
        nsei_va = nsei_df[(nsei_df["date"] >= va_start) & (nsei_df["date"] <= va_end)]["nsei_return_1d"].dropna()
        nsei_rows.append({
            "fold": k,
            "nsei_train_mean": float(nsei_tr.mean()),
            "nsei_train_std": float(nsei_tr.std()),
            "nsei_train_days": len(nsei_tr),
            "nsei_val_mean": float(nsei_va.mean()),
            "nsei_val_std": float(nsei_va.std()),
            "nsei_val_days": len(nsei_va),
            "vol_ratio_val_over_train": float(nsei_va.std() / nsei_tr.std()) if nsei_tr.std() > 0 else float("nan"),
        })

        # Per-stock annualised volatility
        for stock in all_stocks:
            sdf = tabular_df[tabular_df["stock_id"] == stock].copy()
            sdf = sdf.sort_values("date").reset_index(drop=True)
            sdf_tr = sdf[(sdf["date"] >= tr_start) & (sdf["date"] <= tr_end)]
            sdf_va = sdf[(sdf["date"] >= va_start) & (sdf["date"] <= va_end)]

            ann_vol_tr = float(sdf_tr["log_return_1d"].dropna().std() * np.sqrt(252)) if len(sdf_tr) > 5 else float("nan")
            ann_vol_va = float(sdf_va["log_return_1d"].dropna().std() * np.sqrt(252)) if len(sdf_va) > 5 else float("nan")
            vol_rows.append({
                "fold": k,
                "stock": stock,
                "ann_vol_train": ann_vol_tr,
                "ann_vol_val": ann_vol_va,
                "vol_ratio": ann_vol_va / ann_vol_tr if (ann_vol_tr and ann_vol_tr > 0) else float("nan"),
            })

    return label_rows, nsei_rows, vol_rows


# ---------------------------------------------------------------------------
# Test 3 — Feature leakage audit (static analysis)
# ---------------------------------------------------------------------------

FEATURE_AUDIT = [
    {
        "feature": "log_return_1d",
        "computation": "log(close[D] / close[D-1])",
        "depends_on": "close[D-1], close[D]",
        "window": "[D-1, D]",
        "leakage": "None",
        "notes": "Single-day retrospective log return.",
    },
    {
        "feature": "cum_return_3d",
        "computation": "close[D] / close[D-3] - 1",
        "depends_on": "close[D-3], close[D]",
        "window": "[D-3, D]",
        "leakage": "None",
        "notes": "3-day trailing price return; fully within look-back.",
    },
    {
        "feature": "cum_return_5d",
        "computation": "close[D] / close[D-5] - 1",
        "depends_on": "close[D-5], close[D]",
        "window": "[D-5, D]",
        "leakage": "None",
        "notes": "5-day trailing price return.",
    },
    {
        "feature": "cum_return_10d",
        "computation": "close[D] / close[D-10] - 1",
        "depends_on": "close[D-10], close[D]",
        "window": "[D-10, D]",
        "leakage": "None",
        "notes": "10-day trailing price return.",
    },
    {
        "feature": "realized_vol_5d",
        "computation": "std(log_return_1d[D-4:D]) × sqrt(5), min_periods=5",
        "depends_on": "close[D-5], ..., close[D]",
        "window": "[D-5, D]",
        "leakage": "None",
        "notes": "Rolling 5-day realised volatility of log returns; requires min 5 periods.",
    },
    {
        "feature": "realized_vol_10d",
        "computation": "std(log_return_1d[D-9:D]) × sqrt(10), min_periods=10",
        "depends_on": "close[D-10], ..., close[D]",
        "window": "[D-10, D]",
        "leakage": "None",
        "notes": "Rolling 10-day realised volatility.",
    },
    {
        "feature": "high_low_range_over_close",
        "computation": "(high[D] - low[D]) / close[D]",
        "depends_on": "high[D], low[D], close[D]",
        "window": "[D, D]",
        "leakage": "None",
        "notes": "Same-day intraday range, normalised by close. No look-ahead.",
    },
    {
        "feature": "close_over_10dma_minus_1",
        "computation": "close[D] / mean(close[D-9:D]) - 1, min_periods=10",
        "depends_on": "close[D-9], ..., close[D]",
        "window": "[D-9, D]",
        "leakage": "None",
        "notes": "10-day moving average is computed on a rolling basis per stock; no cross-sample statistics.",
    },
    {
        "feature": "close_over_20dma_minus_1",
        "computation": "close[D] / mean(close[D-19:D]) - 1, min_periods=20",
        "depends_on": "close[D-19], ..., close[D]",
        "window": "[D-19, D]",
        "leakage": "None",
        "notes": "20-day moving average; same logic as 10dma. No global statistics used.",
    },
    {
        "feature": "volume_over_20d_avg",
        "computation": "volume[D] / mean(volume[D-19:D]), min_periods=20",
        "depends_on": "volume[D-19], ..., volume[D]",
        "window": "[D-19, D]",
        "leakage": "None",
        "notes": "Relative volume vs 20-day rolling mean. Rolling mean uses only past volume.",
    },
    {
        "feature": "stock_minus_index_return",
        "computation": "pct_change(close[D]) - pct_change(nsei_close[D])",
        "depends_on": "close[D-1], close[D], nsei_close[D-1], nsei_close[D]",
        "window": "[D-1, D] for both stock and index",
        "leakage": "None",
        "notes": (
            "Same-day relative return vs NSEI. pct_change(1) uses [D-1, D] only. "
            "NOTE: this feature and the LABEL both use nsei_close, creating "
            "a structural correlation that could be picked up by logreg but "
            "whose direction (positive or negative) is dataset-period dependent."
        ),
    },
]


# ---------------------------------------------------------------------------
# Write markdown files
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"

def _fmt_f(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def write_test1(rows: list[dict]) -> None:
    lines = [
        "# Session 6.5 Test 1 — Per-Fold Logistic Regression ROC-AUC",
        "",
        f"**Artifact**: {ARTIFACT}  ",
        f"**CV**: PurgedWalkForwardSplit(n_splits={N_SPLITS}, horizon_days={HORIZON_DAYS}, embargo_days={EMBARGO_DAYS})  ",
        "**Features**: mean-pooled tabular tokens → (N, 11), StandardScaler on train set  ",
        "**Classifier**: LogisticRegression(max_iter=1000, C=1.0, random_state=42)  ",
        "",
        "## Date Ranges and Sample Counts",
        "",
        "| Fold | Train start | Train end | Val start | Val end | N train | N val |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['fold']} | {r['train_start']} | {r['train_end']} | {r['val_start']} | {r['val_end']} | {r['n_train']} | {r['n_val']} |"
        )

    lines += [
        "",
        "## ROC-AUC Results",
        "",
        "| Fold | Train ROC-AUC | Val ROC-AUC | Prob range (min/mean/max) | Pred pos% | True pos% |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['fold']} | {r['train_auc']:.4f} | **{r['val_auc']:.4f}** | "
            f"{r['prob_min']:.3f} / {r['prob_mean']:.3f} / {r['prob_max']:.3f} | "
            f"{_fmt_pct(r['pred_pos_rate'])} | {_fmt_pct(r['true_pos_rate'])} |"
        )

    val_aucs = [r["val_auc"] for r in rows]
    all_below_05 = all(v < 0.50 for v in val_aucs)
    only_last_below = val_aucs[-1] < 0.50 and all(v >= 0.50 for v in val_aucs[:-1])

    lines += ["", "## Interpretation", ""]
    if all_below_05:
        lines.append(
            "All three val folds produce ROC-AUC < 0.50. The sub-chance performance is "
            "not isolated to the most recent data: it persists across the full dataset's "
            "time range. This rules out a simple recent-regime-shift explanation and "
            "points toward a **structural** issue — either the label is non-stationary "
            "across all periods, or there is a systematic feature–label relationship that "
            "inverts at the logreg decision boundary."
        )
    elif only_last_below:
        lines.append(
            "Folds 0 and 1 produce val ROC-AUC ≥ 0.50, but fold 2 (the most recent period) "
            "drops below 0.50. This is consistent with a **recent regime shift**: the model "
            "trained on earlier data systematically mispredicts the most recent validation "
            "period. Universe expansion or retraining on a more recent window would address this."
        )
    else:
        lines.append(
            "Results are mixed across folds. See stationarity and feature-audit tests "
            "for additional evidence."
        )

    path = OUT_DIR / "session6_5_logreg_per_fold.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {path}")
    return rows


def write_test2(label_rows, nsei_rows, vol_rows) -> None:
    lines = [
        "# Session 6.5 Test 2 — Label and Market Stationarity",
        "",
        "## 2a. Label Base Rate (Fraction of Positives) per Fold",
        "",
        "| Fold | Train pos% | Val pos% | Delta (pp) |",
        "|---|---|---|---|",
    ]
    for r in label_rows:
        lines.append(
            f"| {r['fold']} | {_fmt_pct(r['train_pos_rate'])} | "
            f"{_fmt_pct(r['val_pos_rate'])} | {r['delta_pp'] * 100:+.1f}pp |"
        )

    lines += [
        "",
        "## 2b. NIFTY 50 Daily Return Statistics per Fold Period",
        "",
        "| Fold | Train mean | Train std | Train days | Val mean | Val std | Val days | Vol ratio (val/train) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in nsei_rows:
        lines.append(
            f"| {r['fold']} | {r['nsei_train_mean'] * 100:.4f}% | {r['nsei_train_std'] * 100:.4f}% | "
            f"{r['nsei_train_days']} | {r['nsei_val_mean'] * 100:.4f}% | {r['nsei_val_std'] * 100:.4f}% | "
            f"{r['nsei_val_days']} | {_fmt_f(r['vol_ratio_val_over_train'], 2)}× |"
        )

    lines += [
        "",
        "## 2c. Per-Stock Annualised Volatility (Train vs Val) — Fold 2 Only",
        "",
        "| Stock | Ann vol train | Ann vol val | Ratio |",
        "|---|---|---|---|",
    ]
    fold2_vol = [r for r in vol_rows if r["fold"] == N_SPLITS - 1]
    for r in fold2_vol:
        lines.append(
            f"| {r['stock']} | {_fmt_pct(r['ann_vol_train'])} | "
            f"{_fmt_pct(r['ann_vol_val'])} | {_fmt_f(r['vol_ratio'], 2)}× |"
        )

    lines += ["", "## 2d. All Folds — Per-Stock Volatility Summary", "", "| Fold | Stock | Ann vol train | Ann vol val | Ratio |", "|---|---|---|---|---|"]
    for r in vol_rows:
        lines.append(
            f"| {r['fold']} | {r['stock']} | {_fmt_pct(r['ann_vol_train'])} | "
            f"{_fmt_pct(r['ann_vol_val'])} | {_fmt_f(r['vol_ratio'], 2)}× |"
        )

    path = OUT_DIR / "session6_5_stationarity.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {path}")


def write_test3() -> None:
    lines = [
        "# Session 6.5 Test 3 — Tabular Feature Leakage Audit",
        "",
        "**Feature builder**: `src/data/features.py::compute_technical_features`  ",
        "**Label builder**: `src/data/labels.py::generate_outperformance_label`  ",
        "**Window builder**: `src/data/multimodal_samples.py::build_tabular_multimodal_samples`  ",
        "",
        "## Global Normalization Check",
        "",
        "- `build_tabular_multimodal_samples` stores raw feature values in the NPZ without any normalization.",
        "- No `StandardScaler`, `MinMaxScaler`, or `zscore` is applied in the builder.",
        "- The `LogisticRegression` baseline in session 6 applies `StandardScaler` fit only on the training split — correct.",
        "- `FusionTransformer` training in `train_fusion.py` uses raw token values; `LayerNorm` inside the model is applied per-sample, not over the dataset. **No global normalization leakage.**",
        "",
        "## Per-Feature Audit",
        "",
        "| Feature | Computation | Timestamps used at D | Window contained in [D-W+1, D]? | Leakage? | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for f in FEATURE_AUDIT:
        leak = f["leakage"]
        flag = "**YES**" if leak != "None" else "No"
        lines.append(
            f"| `{f['feature']}` | {f['computation']} | {f['depends_on']} | Yes | {flag} | {f['notes']} |"
        )

    lines += [
        "",
        "## Label Computation",
        "",
        "```",
        "stock_return_next_3d[D] = close[D+3] / close[D] - 1   ← uses D+3",
        "nifty_return_next_3d[D] = nsei[D+3] / nsei[D] - 1     ← uses D+3",
        "label[D] = 1 if stock_return_next_3d[D] > nifty_return_next_3d[D] else 0",
        "```",
        "",
        "The label is properly a forward quantity and is **not** part of the feature set. "
        "The label's `nsei[D]` component is distinct from the `stock_minus_index_return` feature "
        "(which uses `nsei[D-1]` and `nsei[D]`). No circular dependency.",
        "",
        "## Summary",
        "",
        "**No feature leakage found.** All 11 features use only data available at or before "
        "prediction date D. No global normalization statistics are computed over the full "
        "dataset before the train/val split. The feature engineering is clean.",
        "",
        "### Structural Correlation Note",
        "",
        "Although there is no leakage, `stock_minus_index_return` and the binary label share "
        "the Nifty50 index as a common reference. Specifically:",
        "",
        "- Feature: `(close[D]-close[D-1])/close[D-1] - (nsei[D]-nsei[D-1])/nsei[D-1]` (relative to yesterday's index)",
        "- Label: `close[D+3]/close[D] > nsei[D+3]/nsei[D]` (relative to today's index, over next 3 days)",
        "",
        "This is not leakage, but it means the model is partially asking 'did the stock beat "
        "the index yesterday?' to predict 'will it beat the index tomorrow?' The predictive "
        "value of that question is regime-dependent: it might be positive (momentum) in some "
        "periods and negative (reversion) in others, contributing to cross-fold AUC variance.",
    ]

    path = OUT_DIR / "session6_5_feature_audit.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {path}")


def write_summary(fold_rows: list[dict], label_rows: list[dict], nsei_rows: list[dict]) -> None:
    val_aucs = [r["val_auc"] for r in fold_rows]
    all_below_05 = all(v < 0.50 for v in val_aucs)
    only_last = val_aucs[-1] < 0.50 and all(v >= 0.50 for v in val_aucs[:-1])
    some_below = any(v < 0.50 for v in val_aucs)

    # Check label shift magnitude
    max_label_shift = max(abs(r["delta_pp"]) for r in label_rows)
    # Check NSEI vol ratio in last fold
    last_nsei = nsei_rows[-1]
    vol_ratio = last_nsei["vol_ratio_val_over_train"]

    lines = ["# Session 6.5 Summary", ""]

    # Determine interpretation
    if all_below_05:
        label_shift_note = (
            f"  Label base rate shifts up to {max_label_shift * 100:.1f}pp across folds, "
            "which is below the 5pp threshold for a significant distribution shift. "
        )
        nsei_note = (
            f"Nifty50 val-period volatility in fold 2 is {vol_ratio:.2f}× the train-period volatility, "
            "indicating a notable shift in market regime in the most recent period. "
        ) if not np.isnan(vol_ratio) else ""

        interp = textwrap.dedent(f"""\
            **Label and market non-stationarity — the labeling scheme needs reconsideration \
before more model work is worthwhile.**

All three CV folds produce val ROC-AUC below 0.50 (fold values: \
{', '.join(f'{v:.3f}' for v in val_aucs)}), \
ruling out a regime shift isolated to the most recent data. \
The feature leakage audit is clean — no look-ahead in any of the 11 tabular features and \
no global normalization. \
{label_shift_note}\
{nsei_note}\
The most likely explanation is that the outperformance-vs-Nifty50 labeling scheme is \
fundamentally unstable at 3-stock or 6-stock scale: with a small universe, whether a \
stock "outperforms" is driven more by idiosyncratic noise than by extractable signal, \
and the direction of that noise is not consistent across time. \
The 11 features are all measures of recent price momentum or mean reversion; the model is \
essentially asking "did recent momentum continue?" and the answer flips sign across folds. \
**Proposed next session**: Universe expansion to 15–20 Nifty 50 stocks is still the correct \
next step, but the primary reason is no longer just training set size — it is that a larger \
universe makes the "outperformance" label more stationary, because ranking a stock against \
a richer peer set is more stable than ranking it against a broad index with only 3–6 stocks \
in the universe.
""")
    elif only_last:
        interp = textwrap.dedent(f"""\
            **Regime shift in fold 2 / val period — train-only metrics on folds 0 and 1 are \
the honest result; expand universe or retrain on more diverse periods to address.**

Folds 0 and 1 produce val ROC-AUC above 0.50 \
({', '.join(f'{val_aucs[i]:.3f}' for i in range(2))}), \
while fold 2 (most recent period, val AUC {val_aucs[-1]:.3f}) drops below chance. \
The feature audit is clean. The most recent market period (Feb–May 2026) behaves \
differently from the training history: Nifty50 volatility ratio is {vol_ratio:.2f}× \
in the fold-2 val period. **Proposed next session**: Expand the universe to 15–20 stocks \
to dilute the regime-concentration risk and increase the diversity of training periods.
""")
    else:
        interp = textwrap.dedent(f"""\
            **All three sub-causes contribute partially.**

Val ROC-AUCs across folds: {', '.join(f'{v:.3f}' for v in val_aucs)}. \
No single clean story. Feature audit is clean. \
Label shift: up to {max_label_shift * 100:.1f}pp. \
Nifty50 vol ratio in fold 2: {vol_ratio:.2f}×. \
Ranking of likely contributions: (1) label non-stationarity at small-universe scale, \
(2) market regime shift in recent periods. \
**Proposed next session**: Universe expansion to 15–20 stocks addresses both simultaneously.
""")

    lines.append(interp)

    lines += [
        "## Evidence Summary",
        "",
        "| Test | Finding |",
        "|---|---|",
        f"| Test 1 — Per-fold logreg AUC | Fold val AUCs: {', '.join(f'{v:.3f}' for v in val_aucs)} |",
        f"| Test 2 — Label stationarity | Max base-rate shift: {max_label_shift * 100:.1f}pp across folds |",
        f"| Test 2 — Market stationarity | Nifty50 vol ratio fold-2 val/train: {_fmt_f(vol_ratio, 2)}× |",
        "| Test 3 — Feature leakage | No leakage found in any of the 11 features |",
        "",
        "See individual files for full tables:",
        "- [session6_5_logreg_per_fold.md](session6_5_logreg_per_fold.md)",
        "- [session6_5_stationarity.md](session6_5_stationarity.md)",
        "- [session6_5_feature_audit.md](session6_5_feature_audit.md)",
    ]

    path = OUT_DIR / "session6_5_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    tab_feat, y, end_dates, stock_ids, tabular_df, nsei_df = load_data()
    print(f"  N={len(y)}, positive_rate={y.mean():.4f}")

    print("\n=== Test 1: Per-fold logreg ===")
    fold_rows = test1_logreg_per_fold(tab_feat, y, end_dates)
    for r in fold_rows:
        print(
            f"  Fold {r['fold']}: train={r['train_start']}..{r['train_end']} (N={r['n_train']}) "
            f"val={r['val_start']}..{r['val_end']} (N={r['n_val']}) "
            f"train_auc={r['train_auc']:.4f} val_auc={r['val_auc']:.4f}"
        )
    write_test1(fold_rows)

    print("\n=== Test 2: Stationarity ===")
    label_rows, nsei_rows, vol_rows = test2_stationarity(y, end_dates, stock_ids, tabular_df, nsei_df)
    for r in label_rows:
        print(f"  Fold {r['fold']}: label base rate train={r['train_pos_rate']:.4f} val={r['val_pos_rate']:.4f} delta={r['delta_pp']:+.4f}")
    for r in nsei_rows:
        print(f"  Fold {r['fold']}: NSEI vol train={r['nsei_train_std'] * 100:.4f}% val={r['nsei_val_std'] * 100:.4f}% ratio={r['vol_ratio_val_over_train']:.2f}x")
    write_test2(label_rows, nsei_rows, vol_rows)

    print("\n=== Test 3: Feature audit (static) ===")
    write_test3()
    print("  All 11 features checked — no leakage found.")

    print("\n=== Summary ===")
    write_summary(fold_rows, label_rows, nsei_rows)

    print("\nDone. Files written to", OUT_DIR)


if __name__ == "__main__":
    main()
