"""Run fusion-model ablations across modality combinations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.training.cv import PurgedWalkForwardSplit
from src.training.train_fusion import (
    FusionArrays,
    load_fusion_arrays,
    slice_fusion_arrays,
    train_on_arrays,
)


@dataclass(frozen=True)
class AblationVariant:
    name: str
    use_image: bool = False
    use_text: bool = False
    use_kg: bool = False

    @property
    def required_keys(self) -> set[str]:
        keys = {"tabular_tokens", "y", "end_dates"}
        if self.use_image:
            keys.add("image_tokens")
        if self.use_text:
            keys.add("text_tokens")
        if self.use_kg:
            keys.add("kg_tokens")
        return keys

    @property
    def modalities(self) -> list[str]:
        values = ["tabular"]
        if self.use_image:
            values.append("image")
        if self.use_text:
            values.append("text")
        if self.use_kg:
            values.append("kg")
        return values


DEFAULT_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("tabular_only"),
    AblationVariant("tabular_kg", use_kg=True),
    AblationVariant("tabular_image", use_image=True),
    AblationVariant("tabular_text", use_text=True),
    AblationVariant("tabular_image_text_kg", use_image=True, use_text=True, use_kg=True),
)


def available_dataset_keys(dataset_path: str | Path) -> set[str]:
    data = np.load(dataset_path, allow_pickle=False)
    return set(data.files)


def select_variants(
    dataset_keys: set[str],
    *,
    variants: tuple[AblationVariant, ...] = DEFAULT_VARIANTS,
    strict: bool = False,
) -> list[AblationVariant]:
    selected: list[AblationVariant] = []
    missing_messages: list[str] = []
    for variant in variants:
        missing = sorted(variant.required_keys - dataset_keys)
        if missing:
            message = f"Skipping {variant.name}; missing keys: {missing}"
            if strict:
                missing_messages.append(message)
            else:
                print(message)
            continue
        selected.append(variant)
    if strict and missing_messages:
        raise ValueError("; ".join(missing_messages))
    if not selected:
        raise ValueError("No ablation variants can run for the provided dataset")
    return selected


def build_train_command(
    *,
    dataset: str | Path,
    checkpoint_path: str | Path,
    variant: AblationVariant,
    epochs: int,
    batch_size: int,
    device: str,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    ff_dim: int,
    val_fraction: float,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.training.train_fusion",
        "--dataset",
        str(dataset),
        "--checkpoint-path",
        str(checkpoint_path),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--model-dim",
        str(model_dim),
        "--num-heads",
        str(num_heads),
        "--num-layers",
        str(num_layers),
        "--ff-dim",
        str(ff_dim),
        "--val-fraction",
        str(val_fraction),
    ]
    if variant.use_image:
        command.append("--use-image")
    if variant.use_text:
        command.append("--use-text")
    if variant.use_kg:
        command.append("--use-kg")
    return command


def _load_trusted_local_checkpoint(checkpoint_path: str | Path) -> dict[str, object]:
    """Load a checkpoint written by this local ablation run.

    PyTorch 2.6+ defaults ``torch.load`` to ``weights_only=True``. These
    checkpoints intentionally store small NumPy validation arrays for diagnostics,
    so we explicitly load the trusted local file with ``weights_only=False``.
    """
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_checkpoint_metrics(checkpoint_path: str | Path) -> dict[str, float]:
    checkpoint = _load_trusted_local_checkpoint(checkpoint_path)
    metrics = checkpoint.get("val_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Checkpoint does not include val_metrics: {checkpoint_path}")
    return {key: float(value) for key, value in metrics.items()}


def load_checkpoint_predictions(checkpoint_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load validation labels, probabilities, dates, and stock_ids from a checkpoint."""
    checkpoint = _load_trusted_local_checkpoint(checkpoint_path)
    if "val_y_true" not in checkpoint or "val_y_prob" not in checkpoint:
        raise ValueError(
            "Checkpoint does not include validation predictions. "
            "Re-run ablations with the updated training script."
        )
    y_true = np.asarray(checkpoint["val_y_true"]).astype(np.int64)
    y_prob = np.asarray(checkpoint["val_y_prob"]).astype(np.float32)
    end_dates = checkpoint.get("val_end_dates")
    if end_dates is not None:
        end_dates = np.asarray(end_dates)
    stock_ids = checkpoint.get("val_stock_ids")
    if stock_ids is not None:
        stock_ids = np.asarray(stock_ids)
    return y_true, y_prob, end_dates, stock_ids


def summarize_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Return threshold and class-distribution diagnostics."""
    y_pred = (y_prob >= 0.5).astype(np.int64)
    total = int(y_true.shape[0])
    positives = int(y_true.sum())
    negatives = total - positives
    positive_rate = positives / total if total else 0.0
    majority_class_accuracy = max(positive_rate, 1.0 - positive_rate) if total else 0.0
    predicted_positive_rate = float(y_pred.mean()) if total else 0.0
    return {
        "val_count": float(total),
        "val_positive_count": float(positives),
        "val_negative_count": float(negatives),
        "val_positive_rate": float(positive_rate),
        "majority_class_accuracy": float(majority_class_accuracy),
        "predicted_positive_rate": predicted_positive_rate,
        "probability_min": float(np.min(y_prob)) if total else 0.0,
        "probability_mean": float(np.mean(y_prob)) if total else 0.0,
        "probability_max": float(np.max(y_prob)) if total else 0.0,
    }


def write_prediction_scores(
    *,
    variant: AblationVariant,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    end_dates: np.ndarray | None,
    stock_ids: np.ndarray | None,
    output_dir: str | Path,
) -> Path:
    """Write validation probabilities for one ablation variant."""
    output_path = Path(output_dir)
    scores_path = output_path / f"prediction_scores_{variant.name}.csv"
    y_pred = (y_prob >= 0.5).astype(np.int64)
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["row_id", "end_date", "stock_id", "y_true", "y_prob", "y_pred"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, (label, prob, pred) in enumerate(zip(y_true, y_prob, y_pred)):
            writer.writerow(
                {
                    "row_id": i,
                    "end_date": "" if end_dates is None else str(end_dates[i]),
                    "stock_id": "" if stock_ids is None else str(stock_ids[i]),
                    "y_true": int(label),
                    "y_prob": float(prob),
                    "y_pred": int(pred),
                }
            )
    return scores_path


_BASE_FIELDNAMES = [
    "variant",
    "modalities",
    "n_folds",
    "checkpoint_path",
    "prediction_scores_path",
    "command",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "val_count",
    "val_positive_count",
    "val_negative_count",
    "val_positive_rate",
    "majority_class_accuracy",
    "predicted_positive_rate",
    "probability_min",
    "probability_mean",
    "probability_max",
]


def write_results(results: list[dict[str, object]], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "ablation_results.csv"
    json_path = output_path / "ablation_results.json"

    # Collect extra keys introduced by CV aggregation (e.g. accuracy_mean, f1_std).
    extra_keys: list[str] = []
    seen = set(_BASE_FIELDNAMES)
    for row in results:
        for key in row:
            if key not in seen:
                extra_keys.append(key)
                seen.add(key)
    fieldnames = _BASE_FIELDNAMES + extra_keys

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return csv_path, json_path


def write_diagnostics(results: list[dict[str, object]], output_dir: str | Path) -> Path:
    """Write a human-readable diagnostics summary for demo interpretation."""
    output_path = Path(output_dir)
    diagnostics_path = output_path / "ablation_diagnostics.md"
    lines = [
        "# Ablation diagnostics",
        "",
        "This file summarizes validation diagnostics for the ablation run.",
        "",
        "These metrics are useful for checking whether the pipeline is working and whether a run has collapsed to a majority-class prediction. They are not a portfolio backtest and should not be presented as investment performance.",
        "",
        "## Summary table",
        "",
        "| Variant | Accuracy | ROC-AUC | F1 | Val positive rate | Majority baseline | Predicted positive rate | Probability mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| {variant} | {accuracy:.4f} | {roc_auc:.4f} | {f1:.4f} | {val_positive_rate:.4f} | {majority_class_accuracy:.4f} | {predicted_positive_rate:.4f} | {probability_mean:.4f} |".format(
                variant=row["variant"],
                accuracy=float(row.get("accuracy", 0.0)),
                roc_auc=float(row.get("roc_auc", 0.0)),
                f1=float(row.get("f1", 0.0)),
                val_positive_rate=float(row.get("val_positive_rate", 0.0)),
                majority_class_accuracy=float(row.get("majority_class_accuracy", 0.0)),
                predicted_positive_rate=float(row.get("predicted_positive_rate", 0.0)),
                probability_mean=float(row.get("probability_mean", 0.0)),
            )
        )

    collapsed = [
        str(row["variant"])
        for row in results
        if float(row.get("predicted_positive_rate", 0.0)) in (0.0, 1.0)
    ]
    lines.extend(["", "## Interpretation notes", ""])
    if collapsed:
        lines.extend(
            [
                "The following variants predicted only one class at the default 0.5 threshold:",
                "",
                "```text",
                *collapsed,
                "```",
                "",
                "When this happens, accuracy can match the majority-class baseline while precision, recall, or F1 may be uninformative. Use ROC-AUC and the exported probability score files to inspect whether the ranking signal varies before making any claim about model quality.",
            ]
        )
    else:
        lines.append(
            "No variant predicted only one class at the default 0.5 threshold. Still compare accuracy to the majority baseline and inspect probability distributions before interpreting results."
        )
    lines.extend(
        [
            "",
            "## Recommended demo language",
            "",
            "> The ablation runner is working and produces comparable metrics across modality combinations. However, these compact runs are pipeline evidence, not investment-grade performance claims. We inspect majority-class baseline, positive prediction rate, and ROC-AUC to avoid over-interpreting accuracy.",
            "",
        ]
    )
    diagnostics_path.write_text("\n".join(lines), encoding="utf-8")
    return diagnostics_path


def _make_variant_arrays(full_arrays: FusionArrays, variant: AblationVariant) -> FusionArrays:
    """Return a FusionArrays view that exposes only the modalities used by *variant*."""
    return FusionArrays(
        tabular_tokens=full_arrays.tabular_tokens,
        y=full_arrays.y,
        end_dates=full_arrays.end_dates,
        image_tokens=full_arrays.image_tokens if variant.use_image else None,
        text_tokens=full_arrays.text_tokens if variant.use_text else None,
        kg_tokens=full_arrays.kg_tokens if variant.use_kg else None,
    )


def aggregate_fold_metrics(
    fold_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Return mean and std of each metric key across folds."""
    if not fold_metrics:
        return {}
    keys = list(fold_metrics[0].keys())
    aggregated: dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=np.float64)
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values, ddof=0))
    return aggregated


def write_fold_results(
    fold_rows: list[dict[str, object]], output_dir: str | Path
) -> Path:
    """Write per-fold detail rows to ``ablation_results_folds.csv``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    folds_path = output_path / "ablation_results_folds.csv"
    if not fold_rows:
        return folds_path
    fieldnames = list(fold_rows[0].keys())
    with folds_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in fold_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return folds_path


def _run_cv_variant(
    *,
    variant: AblationVariant,
    full_arrays: FusionArrays,
    splitter: PurgedWalkForwardSplit,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run walk-forward CV for one ablation variant.

    Returns (aggregated_result_row, per_fold_rows).
    """
    variant_arrays = _make_variant_arrays(full_arrays, variant)
    fold_metrics_list: list[dict[str, float]] = []
    fold_rows: list[dict[str, object]] = []

    for cv_split in splitter.split(full_arrays.end_dates):
        fold_k = cv_split.fold
        train_arrays = slice_fusion_arrays(variant_arrays, cv_split.train_idx)
        val_arrays = slice_fusion_arrays(variant_arrays, cv_split.val_idx)

        checkpoint_path = checkpoint_dir / f"{variant.name}_fold{fold_k}.pt"
        print(f"  CV fold {fold_k}: train={len(cv_split.train_idx)} val={len(cv_split.val_idx)}")

        fold_val_metrics = train_on_arrays(
            train_arrays,
            val_arrays,
            args=args,
            checkpoint_path=checkpoint_path,
        )
        fold_metrics_list.append(fold_val_metrics)

        y_true, y_prob, end_dates, stock_ids = load_checkpoint_predictions(checkpoint_path)
        diagnostics = summarize_predictions(y_true, y_prob)

        fold_rows.append(
            {
                "variant": variant.name,
                "fold": fold_k,
                "modalities": "+".join(variant.modalities),
                "checkpoint_path": str(checkpoint_path),
                **fold_val_metrics,
                **diagnostics,
            }
        )

    aggregated = aggregate_fold_metrics(fold_metrics_list)
    n_folds = len(fold_metrics_list)

    result_row: dict[str, object] = {
        "variant": variant.name,
        "modalities": "+".join(variant.modalities),
        "n_folds": n_folds,
        "checkpoint_path": str(checkpoint_dir / f"{variant.name}_fold*.pt"),
        "prediction_scores_path": "",
        "command": f"cv:{n_folds} folds",
        "accuracy": aggregated.get("accuracy_mean", ""),
        "precision": aggregated.get("precision_mean", ""),
        "recall": aggregated.get("recall_mean", ""),
        "f1": aggregated.get("f1_mean", ""),
        "roc_auc": aggregated.get("roc_auc_mean", ""),
        **aggregated,
    }
    return result_row, fold_rows


def run_ablation_study(args: argparse.Namespace) -> list[dict[str, object]]:
    use_cv = (not getattr(args, "single_split", False)) and getattr(args, "cv_splits", 1) > 1

    dataset_keys = available_dataset_keys(args.dataset)
    variants = select_variants(dataset_keys, strict=args.strict)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []

    if use_cv:
        full_arrays = load_fusion_arrays(
            args.dataset,
            use_image="image_tokens" in dataset_keys,
            use_text="text_tokens" in dataset_keys,
            use_kg="kg_tokens" in dataset_keys,
        )
        splitter = PurgedWalkForwardSplit(
            n_splits=args.cv_splits,
            horizon_days=args.horizon_days,
            embargo_days=args.embargo_days,
        )
        all_fold_rows: list[dict[str, object]] = []
        for variant in variants:
            print(f"Running CV ablation variant: {variant.name} ({args.cv_splits} folds)")
            result_row, fold_rows = _run_cv_variant(
                variant=variant,
                full_arrays=full_arrays,
                splitter=splitter,
                args=args,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
            )
            results.append(result_row)
            all_fold_rows.extend(fold_rows)

        folds_path = write_fold_results(all_fold_rows, output_dir)
        print(f"Wrote per-fold results: {folds_path}")
    else:
        for variant in variants:
            checkpoint_path = checkpoint_dir / f"{variant.name}.pt"
            command = build_train_command(
                dataset=args.dataset,
                checkpoint_path=checkpoint_path,
                variant=variant,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                model_dim=args.model_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                ff_dim=args.ff_dim,
                val_fraction=args.val_fraction,
            )
            print(f"Running ablation variant: {variant.name}")
            subprocess.run(command, check=True)
            metrics = load_checkpoint_metrics(checkpoint_path)
            y_true, y_prob, end_dates, stock_ids = load_checkpoint_predictions(checkpoint_path)
            diagnostics = summarize_predictions(y_true, y_prob)
            scores_path = write_prediction_scores(
                variant=variant,
                y_true=y_true,
                y_prob=y_prob,
                end_dates=end_dates,
                stock_ids=stock_ids,
                output_dir=output_dir,
            )
            results.append(
                {
                    "variant": variant.name,
                    "modalities": "+".join(variant.modalities),
                    "checkpoint_path": str(checkpoint_path),
                    "prediction_scores_path": str(scores_path),
                    "command": " ".join(command),
                    "accuracy": metrics.get("accuracy", ""),
                    "precision": metrics.get("precision", ""),
                    "recall": metrics.get("recall", ""),
                    "f1": metrics.get("f1", ""),
                    "roc_auc": metrics.get("roc_auc", ""),
                    **diagnostics,
                }
            )

    csv_path, json_path = write_results(results, output_dir)
    diagnostics_path = write_diagnostics(results, output_dir)
    print(f"Wrote ablation CSV: {csv_path}")
    print(f"Wrote ablation JSON: {json_path}")
    print(f"Wrote ablation diagnostics: {diagnostics_path}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fusion modality ablation study")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="data/processed/ablations")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--ff-dim", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--pooling", type=str, default="mean", choices=["cls", "mean"]
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--strict", action="store_true")
    # Walk-forward CV arguments
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=1,
        help="Number of walk-forward CV folds (>1 activates CV mode).",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=3,
        help="Label horizon in calendar days; used to purge overlapping train samples.",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=0,
        help="Additional calendar-day buffer before each CV test fold.",
    )
    parser.add_argument(
        "--single-split",
        action="store_true",
        help="Force single-split mode (original subprocess behaviour); overrides --cv-splits.",
    )
    return parser


def main() -> None:
    run_ablation_study(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
