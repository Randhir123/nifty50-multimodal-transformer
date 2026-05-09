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


def load_checkpoint_metrics(checkpoint_path: str | Path) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metrics = checkpoint.get("val_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Checkpoint does not include val_metrics: {checkpoint_path}")
    return {key: float(value) for key, value in metrics.items()}


def load_checkpoint_predictions(checkpoint_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load validation labels, probabilities, and dates from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
    return y_true, y_prob, end_dates


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
    output_dir: str | Path,
) -> Path:
    """Write validation probabilities for one ablation variant."""
    output_path = Path(output_dir)
    scores_path = output_path / f"prediction_scores_{variant.name}.csv"
    y_pred = (y_prob >= 0.5).astype(np.int64)
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["row_id", "end_date", "y_true", "y_prob", "y_pred"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, (label, prob, pred) in enumerate(zip(y_true, y_prob, y_pred)):
            writer.writerow(
                {
                    "row_id": i,
                    "end_date": "" if end_dates is None else str(end_dates[i]),
                    "y_true": int(label),
                    "y_prob": float(prob),
                    "y_pred": int(pred),
                }
            )
    return scores_path


def write_results(results: list[dict[str, object]], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "ablation_results.csv"
    json_path = output_path / "ablation_results.json"

    fieldnames = [
        "variant",
        "modalities",
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


def run_ablation_study(args: argparse.Namespace) -> list[dict[str, object]]:
    dataset_keys = available_dataset_keys(args.dataset)
    variants = select_variants(dataset_keys, strict=args.strict)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
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
        y_true, y_prob, end_dates = load_checkpoint_predictions(checkpoint_path)
        diagnostics = summarize_predictions(y_true, y_prob)
        scores_path = write_prediction_scores(
            variant=variant,
            y_true=y_true,
            y_prob=y_prob,
            end_dates=end_dates,
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
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    run_ablation_study(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
