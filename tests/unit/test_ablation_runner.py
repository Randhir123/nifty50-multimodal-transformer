from __future__ import annotations

import csv
import json
import sys

import pytest

from scripts.run_ablation_study import (
    AblationVariant,
    build_train_command,
    select_variants,
    write_results,
)


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


def test_write_results_creates_csv_and_json(tmp_path) -> None:
    results = [
        {
            "variant": "tabular_only",
            "modalities": "tabular",
            "checkpoint_path": "ckpt.pt",
            "command": "python -m src.training.train_fusion",
            "accuracy": 0.5,
            "precision": 0.6,
            "recall": 0.7,
            "f1": 0.65,
            "roc_auc": 0.75,
        }
    ]

    csv_path, json_path = write_results(results, tmp_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["variant"] == "tabular_only"
    assert rows[0]["modalities"] == "tabular"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload[0]["f1"] == 0.65
