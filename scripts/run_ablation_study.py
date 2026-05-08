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


def write_results(results: list[dict[str, object]], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "ablation_results.csv"
    json_path = output_path / "ablation_results.json"

    fieldnames = [
        "variant",
        "modalities",
        "checkpoint_path",
        "command",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return csv_path, json_path


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
        results.append(
            {
                "variant": variant.name,
                "modalities": "+".join(variant.modalities),
                "checkpoint_path": str(checkpoint_path),
                "command": " ".join(command),
                "accuracy": metrics.get("accuracy", ""),
                "precision": metrics.get("precision", ""),
                "recall": metrics.get("recall", ""),
                "f1": metrics.get("f1", ""),
                "roc_auc": metrics.get("roc_auc", ""),
            }
        )
    csv_path, json_path = write_results(results, output_dir)
    print(f"Wrote ablation CSV: {csv_path}")
    print(f"Wrote ablation JSON: {json_path}")
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
