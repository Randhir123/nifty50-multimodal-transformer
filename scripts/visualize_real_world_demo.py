"""Generate visualization artifacts from a completed real-world demo run.

Reads the NPZ artifact and ablation CSV produced by run_real_world_demo.py
and writes publication-ready PNGs to the demo output directory.

Usage::

    python scripts/visualize_real_world_demo.py \\
        --demo-dir data/processed/real_world_demo

Outputs:
    <demo-dir>/modality_embedding_projection.png
    <demo-dir>/modality_embedding_projection.csv
    <demo-dir>/ablations/accuracy_vs_majority_baseline.png
    <demo-dir>/ablations/roc_auc_by_variant.png
    <demo-dir>/ablations/f1_by_variant.png
    <demo-dir>/ablations/positive_rate_diagnostics.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def plot_ablation_metrics(results: pd.DataFrame, output_dir: Path) -> None:
    """Write one bar chart per metric group to output_dir."""
    plot_df = results.set_index("variant")

    configs = [
        (
            ["accuracy", "majority_class_accuracy"],
            "Accuracy vs majority-class baseline",
            "accuracy_vs_majority_baseline.png",
        ),
        (
            ["roc_auc"],
            "ROC-AUC by modality combination",
            "roc_auc_by_variant.png",
        ),
        (
            ["f1"],
            "F1 by modality combination",
            "f1_by_variant.png",
        ),
        (
            ["val_positive_rate", "predicted_positive_rate"],
            "Actual vs predicted positive rate",
            "positive_rate_diagnostics.png",
        ),
    ]

    for cols, title, filename in configs:
        available = [c for c in cols if c in plot_df.columns]
        if not available:
            print(f"  Skipping {filename}: columns {cols} not found in results")
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_df[available].plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out = output_dir / filename
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"  Wrote: {out}")


def plot_embedding_projections(data: dict, output_dir: Path) -> None:
    """Project each modality's mean vector to 2D with PCA and save an overlay scatter."""
    modalities: dict[str, np.ndarray] = {}
    for name, reduce in [
        ("tabular", True),
        ("image", False),
        ("text", False),
        ("kg", False),
    ]:
        key = f"{name}_tokens"
        if key not in data:
            continue
        arr = data[key]
        modalities[name] = arr.mean(axis=1) if reduce and arr.ndim == 3 else arr

    if not modalities:
        print("  No modality arrays found; skipping embedding projection.")
        return

    rows = []
    for modality, values in modalities.items():
        n_components = min(2, values.shape[1])
        coords = PCA(n_components=n_components, random_state=42).fit_transform(values)
        if n_components == 1:
            coords = np.c_[coords[:, 0], np.zeros(values.shape[0])]
        for i, (x, y) in enumerate(coords):
            rows.append(
                {
                    "sample_id": i,
                    "stock_id": str(data["stock_ids"][i]) if "stock_ids" in data else str(i),
                    "end_date": str(data["end_dates"][i]) if "end_dates" in data else "",
                    "label": int(data["y"][i]) if "y" in data else -1,
                    "modality": modality,
                    "x": float(x),
                    "y": float(y),
                }
            )

    proj_df = pd.DataFrame(rows)
    csv_path = output_dir / "modality_embedding_projection.csv"
    proj_df.to_csv(csv_path, index=False)
    print(f"  Wrote: {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 7))
    for modality, frame in proj_df.groupby("modality"):
        ax.scatter(frame["x"], frame["y"], label=modality, alpha=0.65, s=35)
    ax.set_title("PCA embedding projections by modality (actual artifact vectors)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(title="Modality")
    plt.tight_layout()
    png_path = output_dir / "modality_embedding_projection.png"
    plt.savefig(png_path, dpi=160)
    plt.close()
    print(f"  Wrote: {png_path}")

    # Second view: colour by label
    if "y" in data:
        fig, ax = plt.subplots(figsize=(10, 7))
        label_palette = {0: "tab:blue", 1: "tab:orange"}
        label_names = {0: "underperform (y=0)", 1: "outperform (y=1)"}
        for modality, frame in proj_df.groupby("modality"):
            for lv, group in frame.groupby("label"):
                ax.scatter(
                    group["x"],
                    group["y"],
                    label=f"{modality} / {label_names.get(lv, lv)}",
                    alpha=0.6,
                    s=30,
                    color=label_palette.get(lv, "gray"),
                    marker={"tabular": "o", "image": "s", "text": "^", "kg": "D"}.get(modality, "o"),
                )
        ax.set_title("PCA embeddings coloured by outperformance label")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(title="modality / label", fontsize=7, ncol=2)
        plt.tight_layout()
        label_png = output_dir / "modality_embedding_by_label.png"
        plt.savefig(label_png, dpi=160)
        plt.close()
        print(f"  Wrote: {label_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualization artifacts from a real-world demo run."
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default="data/processed/real_world_demo",
        help="Path to the real-world demo output directory",
    )
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    ablation_dir = demo_dir / "ablations"

    print(f"Demo directory: {demo_dir}")

    # Ablation charts
    ablation_csv = ablation_dir / "ablation_results.csv"
    if ablation_csv.exists():
        print("\nPlotting ablation metrics ...")
        results = pd.read_csv(ablation_csv)
        ablation_dir.mkdir(parents=True, exist_ok=True)
        plot_ablation_metrics(results, ablation_dir)
    else:
        print(
            f"\nNo ablation_results.csv at {ablation_csv}. "
            "Re-run with --run-ablations to generate ablation outputs."
        )

    # Embedding projections
    npz_path = demo_dir / "real_world_multimodal_samples.npz"
    if npz_path.exists():
        print("\nProjecting modality embeddings ...")
        data = dict(np.load(npz_path, allow_pickle=False))
        plot_embedding_projections(data, demo_dir)
    else:
        print(
            f"\nNo NPZ artifact at {npz_path}. "
            "Run scripts/run_real_world_demo.py first."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
