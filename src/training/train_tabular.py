"""Training entry point for the tabular Transformer baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.models.tabular_transformer import TabularTransformer, TabularTransformerConfig
from src.training.evaluate import compute_binary_classification_metrics


@dataclass(frozen=True)
class RollingWindowArrays:
    """Loaded rolling-window arrays from a saved dataset artifact."""

    X: np.ndarray
    y: np.ndarray
    end_dates: np.ndarray


class NumpyRollingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch dataset wrapper over rolling-window numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D [num_samples, window_len, feature_dim], got {X.shape}"
            )
        if y.ndim != 1:
            raise ValueError(f"y must be 1D [num_samples], got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have matching number of samples")

        self._X = torch.from_numpy(X.astype(np.float32, copy=False))
        self._y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return self._X.size(0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._X[index], self._y[index]


def load_rolling_window_arrays(path: str | Path) -> RollingWindowArrays:
    """Load arrays from a `.npz` artifact containing ``X``, ``y``, ``end_dates``."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=False)
    required = ("X", "y", "end_dates")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing keys in dataset artifact: {missing}")

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"]).astype(np.int64)
    end_dates = np.asarray(data["end_dates"])

    return RollingWindowArrays(X=X, y=y, end_dates=end_dates)


def time_based_split(
    arrays: RollingWindowArrays,
    *,
    val_fraction: float = 0.2,
) -> tuple[RollingWindowArrays, RollingWindowArrays]:
    """Split arrays by chronological order into train and validation sets."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    order = np.argsort(arrays.end_dates)
    X = arrays.X[order]
    y = arrays.y[order]
    end_dates = arrays.end_dates[order]

    split_idx = int((1.0 - val_fraction) * len(X))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Split produced empty train or validation set")

    train = RollingWindowArrays(
        X=X[:split_idx], y=y[:split_idx], end_dates=end_dates[:split_idx]
    )
    val = RollingWindowArrays(
        X=X[split_idx:], y=y[split_idx:], end_dates=end_dates[split_idx:]
    )
    return train, val


def run_epoch(
    model: TabularTransformer,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one train/eval epoch and return loss and predictions."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(batch_y.detach().cpu().numpy())

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    mean_loss = total_loss / max(total_count, 1)
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return mean_loss, y_true.astype(np.int64), y_prob.astype(np.float32)


def train_tabular_transformer(args: argparse.Namespace) -> None:
    """Main training routine for tabular baseline."""
    arrays = load_rolling_window_arrays(args.dataset)
    train_arrays, val_arrays = time_based_split(arrays, val_fraction=args.val_fraction)

    train_dataset = NumpyRollingWindowDataset(train_arrays.X, train_arrays.y)
    val_dataset = NumpyRollingWindowDataset(val_arrays.X, val_arrays.y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    config = TabularTransformerConfig(
        feature_dim=train_arrays.X.shape[-1],
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_len=max(train_arrays.X.shape[1], val_arrays.X.shape[1]) + 1,
        pooling=args.pooling,
    )
    model = TabularTransformer(config).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

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
    """Build CLI args for tabular baseline training."""
    parser = argparse.ArgumentParser(description="Train tabular Transformer baseline")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to .npz with X, y, end_dates"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="data/processed/checkpoints/tabular_transformer.pt",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_tabular_transformer(args)


if __name__ == "__main__":
    main()
