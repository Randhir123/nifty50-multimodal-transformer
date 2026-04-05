"""Training entry point for the stock-news text branch."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import TextSampleDataset, create_text_sample_dataset
from src.models.text_encoder import TextEncoder, TextEncoderConfig
from src.training.evaluate import compute_binary_classification_metrics


class StockNewsTextDataset(Dataset[tuple[str, Tensor]]):
    """Dataset wrapper over sample-level text and labels."""

    def __init__(self, texts: np.ndarray, labels: np.ndarray) -> None:
        if texts.ndim != 1:
            raise ValueError("texts must be 1D")
        if labels.ndim != 1:
            raise ValueError("labels must be 1D")
        if texts.shape[0] != labels.shape[0]:
            raise ValueError("texts and labels must have same number of samples")

        self._texts = texts.astype(object, copy=False)
        self._labels = torch.from_numpy(labels.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> tuple[str, Tensor]:
        return str(self._texts[index]), self._labels[index]


@dataclass(frozen=True)
class TextBranchArrays:
    """Text training arrays sorted by sample date."""

    texts: np.ndarray
    y: np.ndarray
    sample_dates: np.ndarray


def load_text_samples_table(
    path: str | Path,
    *,
    text_col: str,
    label_col: str,
    date_col: str,
) -> TextBranchArrays:
    """Load and normalize text samples from CSV or Parquet."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Samples file not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(data_path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("samples file must be .csv or .parquet")

    dataset: TextSampleDataset = create_text_sample_dataset(
        df,
        text_col=text_col,
        label_col=label_col,
        date_col=date_col,
    )
    return TextBranchArrays(
        texts=np.asarray(dataset.texts, dtype=object),
        y=np.asarray(dataset.y, dtype=np.int64),
        sample_dates=np.asarray(dataset.sample_dates),
    )


def time_based_split(
    arrays: TextBranchArrays,
    *,
    val_fraction: float,
) -> tuple[TextBranchArrays, TextBranchArrays]:
    """Split text samples chronologically into train and validation."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    order = np.argsort(arrays.sample_dates)
    texts = arrays.texts[order]
    y = arrays.y[order]
    sample_dates = arrays.sample_dates[order]

    split_idx = int((1.0 - val_fraction) * len(texts))
    if split_idx <= 0 or split_idx >= len(texts):
        raise ValueError("Split produced empty train or validation set")

    train = TextBranchArrays(texts=texts[:split_idx], y=y[:split_idx], sample_dates=sample_dates[:split_idx])
    val = TextBranchArrays(texts=texts[split_idx:], y=y[split_idx:], sample_dates=sample_dates[split_idx:])
    return train, val


def _text_collate_fn(batch: list[tuple[str, Tensor]]) -> tuple[list[str], Tensor]:
    texts = [text for text, _ in batch]
    labels = torch.stack([label for _, label in batch], dim=0)
    return texts, labels


def run_epoch(
    model: TextEncoder,
    loader: DataLoader[tuple[str, Tensor]],
    *,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one train/eval epoch and return loss, labels, probabilities."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for texts, labels in loader:
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(texts)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    mean_loss = total_loss / max(total_count, 1)
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return mean_loss, y_true.astype(np.int64), y_prob.astype(np.float32)


def train_text_branch(args: argparse.Namespace) -> None:
    """Main training routine for the text branch."""
    arrays = load_text_samples_table(
        args.samples,
        text_col=args.text_col,
        label_col=args.label_col,
        date_col=args.date_col,
    )
    train_arrays, val_arrays = time_based_split(arrays, val_fraction=args.val_fraction)

    train_dataset = StockNewsTextDataset(train_arrays.texts, train_arrays.y)
    val_dataset = StockNewsTextDataset(val_arrays.texts, val_arrays.y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_text_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_text_collate_fn,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    config = TextEncoderConfig(
        pretrained_model_name=args.pretrained_model_name,
        max_length=args.max_length,
        dropout=args.dropout,
        use_mean_pooling=(args.pooling == "mean"),
    )
    model = TextEncoder(config).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_f1 = -float("inf")
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_y, train_prob = run_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_y, val_prob = run_epoch(
            model,
            val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        train_metrics = compute_binary_classification_metrics(train_y, train_prob)
        val_metrics = compute_binary_classification_metrics(val_y, val_prob)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

    print(f"Saved best checkpoint to: {checkpoint_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI args for text branch training."""
    parser = argparse.ArgumentParser(description="Train stock-news text branch")
    parser.add_argument("--samples", type=str, required=True, help="CSV/Parquet with date, text, label")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--date-col", type=str, default="date")

    parser.add_argument("--checkpoint-path", type=str, default="data/processed/checkpoints/text_encoder.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--pretrained-model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_text_branch(args)


if __name__ == "__main__":
    main()
