"""Training entry point for the candlestick image Transformer branch."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from src.data.dataset import ImagePathDataset, create_image_path_dataset
from src.models.image_transformer import ImageTransformer, ImageTransformerConfig
from src.training.evaluate import compute_binary_classification_metrics


class CandlestickImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """Torch dataset for candlestick chart image paths and binary labels."""

    def __init__(self, image_paths: np.ndarray, labels: np.ndarray, *, image_size: int) -> None:
        if image_paths.ndim != 1:
            raise ValueError("image_paths must be 1D")
        if labels.ndim != 1:
            raise ValueError("labels must be 1D")
        if image_paths.shape[0] != labels.shape[0]:
            raise ValueError("image_paths and labels must have same number of samples")

        self._image_paths = image_paths
        self._labels = torch.from_numpy(labels.astype(np.float32, copy=False))
        self._transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = Path(str(self._image_paths[index]))
        if not image_path.exists():
            raise FileNotFoundError(f"Missing chart image: {image_path}")

        image = read_image(str(image_path), mode=ImageReadMode.RGB)
        image = self._transform(image)
        label = self._labels[index]
        return image, label


@dataclass(frozen=True)
class ImageBranchArrays:
    """Image training arrays sorted by sample date."""

    image_paths: np.ndarray
    y: np.ndarray
    sample_dates: np.ndarray


def load_image_samples_table(
    path: str | Path,
    *,
    image_path_col: str,
    label_col: str,
    date_col: str,
    require_existing_files: bool,
) -> ImageBranchArrays:
    """Load and normalize image samples from CSV or Parquet."""
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

    dataset: ImagePathDataset = create_image_path_dataset(
        df,
        image_path_col=image_path_col,
        label_col=label_col,
        date_col=date_col,
        require_existing_files=require_existing_files,
    )
    return ImageBranchArrays(
        image_paths=np.asarray(dataset.image_paths, dtype=object),
        y=np.asarray(dataset.y, dtype=np.int64),
        sample_dates=np.asarray(dataset.sample_dates),
    )


def time_based_split(
    arrays: ImageBranchArrays,
    *,
    val_fraction: float,
) -> tuple[ImageBranchArrays, ImageBranchArrays]:
    """Split image samples chronologically into train and validation."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    order = np.argsort(arrays.sample_dates)
    image_paths = arrays.image_paths[order]
    y = arrays.y[order]
    sample_dates = arrays.sample_dates[order]

    split_idx = int((1.0 - val_fraction) * len(image_paths))
    if split_idx <= 0 or split_idx >= len(image_paths):
        raise ValueError("Split produced empty train or validation set")

    train = ImageBranchArrays(
        image_paths=image_paths[:split_idx],
        y=y[:split_idx],
        sample_dates=sample_dates[:split_idx],
    )
    val = ImageBranchArrays(
        image_paths=image_paths[split_idx:],
        y=y[split_idx:],
        sample_dates=sample_dates[split_idx:],
    )
    return train, val


def run_epoch(
    model: ImageTransformer,
    loader: DataLoader[tuple[Tensor, Tensor]],
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

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    mean_loss = total_loss / max(total_count, 1)
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return mean_loss, y_true.astype(np.int64), y_prob.astype(np.float32)


def train_image_branch(args: argparse.Namespace) -> None:
    """Main training routine for the candlestick image branch."""
    arrays = load_image_samples_table(
        args.samples,
        image_path_col=args.image_path_col,
        label_col=args.label_col,
        date_col=args.date_col,
        require_existing_files=args.require_existing_files,
    )
    train_arrays, val_arrays = time_based_split(arrays, val_fraction=args.val_fraction)

    train_dataset = CandlestickImageDataset(train_arrays.image_paths, train_arrays.y, image_size=args.image_size)
    val_dataset = CandlestickImageDataset(val_arrays.image_paths, val_arrays.y, image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    config = ImageTransformerConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = ImageTransformer(config).to(device)

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
    """Build CLI args for image branch training."""
    parser = argparse.ArgumentParser(description="Train candlestick image Transformer branch")
    parser.add_argument("--samples", type=str, required=True, help="CSV/Parquet with date, chart_path, label")
    parser.add_argument("--image-path-col", type=str, default="chart_path")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--require-existing-files", action="store_true")

    parser.add_argument("--checkpoint-path", type=str, default="data/processed/checkpoints/image_transformer.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_image_branch(args)


if __name__ == "__main__":
    main()
