"""Compute pairwise modality independence from a multimodal NPZ artifact.

For each pair of modalities, computes the mean absolute Pearson correlation across all
cross-modal feature pairs after PCA reduction to --max-dim (default 50) per modality.

Score interpretation:
    ~0.04        noise floor (shuffled random pairs); modality carries no independent signal
    0.08–0.17    independent signal confirmed; modality contributes information not present in others
    >0.9         redundant; modality is a near-re-encoding of another (e.g., price-derived text)

Reference values from the project artifact:
    (tabular, text) after real news: 0.170   — well above noise floor
    (tabular, image) with candlestick ViT: 0.047   — at noise floor (no useful signal)
    (tabular, image) after GAF/MTF + CNN: 0.082    — independent temporal-shape signal confirmed

Usage:
    python scripts/check_modality_independence.py \
        --artifact data/processed/real_world_demo/real_world_multimodal_samples_gaf.npz \
        --output-csv data/processed/real_world_demo/modality_independence.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _prepare(arr: np.ndarray, max_dim: int) -> np.ndarray:
    """Mean-pool time dimension if 3-D, standardise, optionally PCA-reduce."""
    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    arr = StandardScaler().fit_transform(arr.astype(np.float64))
    if arr.shape[1] > max_dim:
        arr = PCA(n_components=max_dim, random_state=42).fit_transform(arr)
    return arr


def _mean_abs_corr(A: np.ndarray, B: np.ndarray) -> float:
    """Mean absolute Pearson r across all (Da x Db) cross-modal feature pairs.

    Both A and B must already be zero-mean and unit-std per column.
    """
    n = A.shape[0]
    corr_matrix = (A.T @ B) / (n - 1)  # (Da, Db)
    return float(np.abs(corr_matrix).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Modality independence diagnostics")
    parser.add_argument("--artifact", required=True, help="Path to .npz file")
    parser.add_argument(
        "--output-csv",
        default="data/processed/modality_independence.csv",
        help="Where to write the 4×4 dependence table",
    )
    parser.add_argument("--max-dim", type=int, default=50, help="PCA cap per modality")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.artifact, allow_pickle=True)

    raw: dict[str, np.ndarray] = {}
    for key, label in [
        ("tabular_tokens", "tabular"),
        ("text_tokens", "text"),
        ("image_tokens", "image"),
        ("kg_tokens", "kg"),
    ]:
        if key in data:
            raw[label] = _prepare(data[key], args.max_dim)

    names = list(raw.keys())

    table: dict[str, dict[str, float]] = {}
    for name_i in names:
        table[name_i] = {}
        for name_j in names:
            if name_i == name_j:
                table[name_i][name_j] = 1.0
            else:
                table[name_i][name_j] = _mean_abs_corr(raw[name_i], raw[name_j])

    df = pd.DataFrame(table, index=names)
    print(df.to_string(float_format="{:.4f}".format))

    rng = np.random.RandomState(args.shuffle_seed)
    if "text" in raw and "tabular" in raw:
        shuffled = raw["text"].copy()
        rng.shuffle(shuffled)
        baseline = _mean_abs_corr(raw["tabular"], shuffled)
        print(f"\nShuffled baseline (tabular vs shuffled_text): {baseline:.4f}")
        print("(Expected: ~0.00 for a purely random pairing)")

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
