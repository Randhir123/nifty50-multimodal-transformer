from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_ablation_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_ablation_study.py"
    spec = importlib.util.spec_from_file_location("run_ablation_study", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load ablation script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ablation = _load_ablation_module()
AblationVariant = ablation.AblationVariant
build_train_command = ablation.build_train_command
select_variants = ablation.select_variants
summarize_predictions = ablation.summarize_predictions
write_diagnostics = ablation.write_diagnostics
write_prediction_scores = ablation.write_prediction_scores
write_results = ablation.write_results


def test_select_variants_skips_missing_optional_modalities() -> None:
    selected = select_variants({"tabular_tokens", "y", "end_dates"})

    assert [variant.name for variant in selected] == ["tabular_only"]


def test_select_variants_includes_all_when_keys_exist() -> None:
    selected = select_variants(
        {"tabular_tokens", "y", "end_dates", "kg_tokens", "image_tokens", "text_tokens"}
    )

    assert [variant.name for variant in selected] == [
        "tabular_only",
        "tabular_kg",
        "tabular_image",
        "tabular_text",
        "tabular_image_text_kg",
    ]


def test_select_variants_strict_raises_for_missing_modality() -> None:
    with pytest.raises(ValueError, match="Skipping tabular_kg"):
        select_variants({"tabular_tokens", "y", "end_dates"}, strict=True)


def test_build_train_command_adds_modality_flags() -> None:
    command = build_train_command(
        dataset="dataset.npz",
        checkpoint_path="checkpoint.pt",
        variant=AblationVariant("all", use_image=True, use_text=True, use_kg=True),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_dim=16,
        num_heads=4,
        num_layers=1,
        ff_dim=32,
        val_fraction=0.25,
    )

    assert command[:3] == [sys.executable, "-m", "src.training.train_fusion"]
    assert "--use-image" in command
    assert "--use-text" in command
    assert "--use-kg" in command
    assert "dataset.npz" in command
    assert "checkpoint.pt" in command


def test_summarize_predictions_detects_majority_class_collapse() -> None:
    summary = summarize_predictions(
        np.array([0, 0, 0, 1], dtype=np.int64),
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    )

    assert summary["val_count"] == 4.0
    assert summary["val_positive_count"] == 1.0
    assert summary["val_negative_count"] == 3.0
    assert summary["val_positive_rate"] == 0.25
    assert summary["majority_class_accuracy"] == 0.75
    assert summary["predicted_positive_rate"] == 0.0
    assert summary["probability_min"] == pytest.approx(0.1)
    assert summary["probability_max"] == pytest.approx(0.4)


def test_write_prediction_scores_creates_per_variant_csv(tmp_path) -> None:
    path = write_prediction_scores(
        variant=AblationVariant("tabular_only"),
        y_true=np.array([0, 1], dtype=np.int64),
        y_prob=np.array([0.4, 0.7], dtype=np.float32),
        end_dates=np.array(["2025-01-01", "2025-01-02"]),
        output_dir=tmp_path,
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert path.name == "prediction_scores_tabular_only.csv"
    assert rows[0]["y_true"] == "0"
    assert rows[0]["y_pred"] == "0"
    assert rows[1]["y_true"] == "1"
    assert rows[1]["y_pred"] == "1"


def test_write_results_creates_csv_and_json(tmp_path) -> None:
    results = [
        {
            "variant": "tabular_only",
            "modalities": "tabular",
            "checkpoint_path": "ckpt.pt",
            "prediction_scores_path": "scores.csv",
            "command": "python -m src.training.train_fusion",
            "accuracy": 0.5,
            "precision": 0.6,
            "recall": 0.7,
            "f1": 0.65,
            "roc_auc": 0.75,
            "val_count": 10.0,
            "val_positive_count": 3.0,
            "val_negative_count": 7.0,
            "val_positive_rate": 0.3,
            "majority_class_accuracy": 0.7,
            "predicted_positive_rate": 0.2,
            "probability_min": 0.1,
            "probability_mean": 0.4,
            "probability_max": 0.9,
        }
    ]

    csv_path, json_path = write_results(results, tmp_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["variant"] == "tabular_only"
    assert rows[0]["modalities"] == "tabular"
    assert rows[0]["prediction_scores_path"] == "scores.csv"
    assert rows[0]["majority_class_accuracy"] == "0.7"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload[0]["f1"] == 0.65


def test_write_diagnostics_flags_single_class_predictions(tmp_path) -> None:
    path = write_diagnostics(
        [
            {
                "variant": "tabular_only",
                "accuracy": 0.7,
                "roc_auc": 0.6,
                "f1": 0.0,
                "val_positive_rate": 0.3,
                "majority_class_accuracy": 0.7,
                "predicted_positive_rate": 0.0,
                "probability_mean": 0.4,
            }
        ],
        tmp_path,
    )

    content = path.read_text(encoding="utf-8")
    assert "tabular_only" in content
    assert "predicted only one class" in content
    assert "not a portfolio backtest" in content
