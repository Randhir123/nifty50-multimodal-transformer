"""Training entry point for multimodal fusion Transformer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.models.fusion import FusionTransformer, FusionTransformerConfig
from src.training.evaluate import compute_binary_classification_metrics


@dataclass(frozen=True)
class FusionArrays:
    """Fusion arrays loaded from a `.npz` artifact."""

    tabular_tokens: np.ndarray
    y: np.ndarray
    end_dates: np.ndarray
    image_tokens: np.ndarray | None = None
    text_tokens: np.ndarray | None = None
    kg_tokens: np.ndarray | None = None
    stock_ids: np.ndarray | None = None


class FusionDataset(Dataset[tuple[dict[str, Tensor], Tensor]]):
    """Dataset wrapper for fusion modality tensors."""

    def __init__(self, arrays: FusionArrays) -> None:
        self._inputs: dict[str, Tensor] = {
            "tabular_tokens": torch.from_numpy(
                arrays.tabular_tokens.astype(np.float32, copy=False)
            )
        }
        if arrays.image_tokens is not None:
            self._inputs["image_tokens"] = torch.from_numpy(
                arrays.image_tokens.astype(np.float32, copy=False)
            )
        if arrays.text_tokens is not None:
            self._inputs["text_tokens"] = torch.from_numpy(
                arrays.text_tokens.astype(np.float32, copy=False)
            )
        if arrays.kg_tokens is not None:
            self._inputs["kg_tokens"] = torch.from_numpy(
                arrays.kg_tokens.astype(np.float32, copy=False)
            )

        self._y = torch.from_numpy(arrays.y.astype(np.float32, copy=False))

        sample_count = self._y.shape[0]
        for name, values in self._inputs.items():
            if values.shape[0] != sample_count:
                raise ValueError(f"Sample count mismatch for {name}")

    def __len__(self) -> int:
        return int(self._y.shape[0])

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Tensor]:
        item = {name: values[index] for name, values in self._inputs.items()}
        return item, self._y[index]


def load_fusion_arrays(
    path: str | Path, *, use_image: bool, use_text: bool, use_kg: bool
) -> FusionArrays:
    """Load multimodal arrays from an `.npz` dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=False)

    required = ("tabular_tokens", "y", "end_dates")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required keys in dataset artifact: {missing}")

    def _optional(name: str, enabled: bool) -> np.ndarray | None:
        if not enabled:
            return None
        if name not in data:
            raise ValueError(
                f"Requested modality '{name}' but key is missing in dataset"
            )
        return np.asarray(data[name], dtype=np.float32)

    return FusionArrays(
        tabular_tokens=np.asarray(data["tabular_tokens"], dtype=np.float32),
        y=np.asarray(data["y"]).astype(np.int64),
        end_dates=np.asarray(data["end_dates"]),
        image_tokens=_optional("image_tokens", use_image),
        text_tokens=_optional("text_tokens", use_text),
        kg_tokens=_optional("kg_tokens", use_kg),
        stock_ids=np.asarray(data["stock_ids"]) if "stock_ids" in data else None,
    )


def time_based_split(
    arrays: FusionArrays,
    *,
    val_fraction: float,
) -> tuple[FusionArrays, FusionArrays]:
    """Split arrays chronologically into train/validation."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    order = np.argsort(arrays.end_dates)

    def _take(a: np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
        if a is None:
            return None
        return a[idx]

    split_idx = int((1.0 - val_fraction) * len(order))
    if split_idx <= 0 or split_idx >= len(order):
        raise ValueError("Split produced empty train or validation set")

    train_idx = order[:split_idx]
    val_idx = order[split_idx:]

    train_arrays = FusionArrays(
        tabular_tokens=arrays.tabular_tokens[train_idx],
        y=arrays.y[train_idx],
        end_dates=arrays.end_dates[train_idx],
        image_tokens=_take(arrays.image_tokens, train_idx),
        text_tokens=_take(arrays.text_tokens, train_idx),
        kg_tokens=_take(arrays.kg_tokens, train_idx),
        stock_ids=_take(arrays.stock_ids, train_idx),
    )
    val_arrays = FusionArrays(
        tabular_tokens=arrays.tabular_tokens[val_idx],
        y=arrays.y[val_idx],
        end_dates=arrays.end_dates[val_idx],
        image_tokens=_take(arrays.image_tokens, val_idx),
        text_tokens=_take(arrays.text_tokens, val_idx),
        kg_tokens=_take(arrays.kg_tokens, val_idx),
        stock_ids=_take(arrays.stock_ids, val_idx),
    )
    return train_arrays, val_arrays


def run_epoch(
    model: FusionTransformer,
    loader: DataLoader[tuple[dict[str, Tensor], Tensor]],
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

    for batch_inputs, labels in loader:
        labels = labels.to(device)
        model_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(**model_inputs)
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


def slice_fusion_arrays(arrays: FusionArrays, idx: np.ndarray) -> FusionArrays:
    """Return a new :class:`FusionArrays` containing only the rows at *idx*."""

    def _take(a: np.ndarray | None) -> np.ndarray | None:
        return None if a is None else a[idx]

    return FusionArrays(
        tabular_tokens=arrays.tabular_tokens[idx],
        y=arrays.y[idx],
        end_dates=arrays.end_dates[idx],
        image_tokens=_take(arrays.image_tokens),
        text_tokens=_take(arrays.text_tokens),
        kg_tokens=_take(arrays.kg_tokens),
        stock_ids=_take(arrays.stock_ids),
    )


def train_on_arrays(
    train_arrays: FusionArrays,
    val_arrays: FusionArrays,
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> dict[str, float]:
    """Train on pre-split arrays and return the best-epoch validation metrics.

    This is the core training loop, extracted so that both the single-split CLI
    path and the walk-forward CV path can reuse it without going through
    subprocess.
    """
    train_dataset = FusionDataset(train_arrays)
    val_dataset = FusionDataset(val_arrays)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    config = FusionTransformerConfig(
        tabular_dim=train_arrays.tabular_tokens.shape[-1],
        image_dim=(
            train_arrays.image_tokens.shape[-1]
            if train_arrays.image_tokens is not None
            else None
        ),
        text_dim=(
            train_arrays.text_tokens.shape[-1]
            if train_arrays.text_tokens is not None
            else None
        ),
        kg_dim=(
            train_arrays.kg_tokens.shape[-1]
            if train_arrays.kg_tokens is not None
            else None
        ),
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        pooling=args.pooling,
        max_tokens=args.max_tokens,
    )
    model = FusionTransformer(config).to(device)

    # Calculate positive weight to prevent model collapse on imbalanced real-world splits
    num_pos = int(train_arrays.y.sum())
    num_neg = len(train_arrays.y) - num_pos
    pos_weight_val = num_neg / max(num_pos, 1)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_val_metrics: dict[str, float] = {}
    best_val_y: np.ndarray = np.empty(0, dtype=np.int64)
    best_val_prob: np.ndarray = np.empty(0, dtype=np.float32)
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
            f"train_f1={train_metrics.get('f1', 0.0):.4f} val_f1={val_metrics.get('f1', 0.0):.4f}"
        )

        # Use val_loss for reliable checkpointing, especially when val sets lack positive classes
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            best_val_y = val_y
            best_val_prob = val_prob
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "val_loss": float(val_loss),
                    "val_y_true": val_y.astype(np.int64),
                    "val_y_prob": val_prob.astype(np.float32),
                    "val_end_dates": val_arrays.end_dates,
                    "val_stock_ids": val_arrays.stock_ids,
                },
                checkpoint_path,
            )

    print(f"Saved best checkpoint to: {checkpoint_path}")
    _ = best_val_y, best_val_prob  # available if callers need them later
    return best_val_metrics


def train_fusion_transformer(args: argparse.Namespace) -> None:
    """Main training routine for multimodal fusion."""
    arrays = load_fusion_arrays(
        args.dataset,
        use_image=args.use_image,
        use_text=args.use_text,
        use_kg=args.use_kg,
    )
    train_arrays, val_arrays = time_based_split(arrays, val_fraction=args.val_fraction)
    train_on_arrays(
        train_arrays,
        val_arrays,
        args=args,
        checkpoint_path=Path(args.checkpoint_path),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI args for multimodal fusion training."""
    parser = argparse.ArgumentParser(description="Train multimodal fusion Transformer")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to .npz with fusion arrays"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="data/processed/checkpoints/fusion_transformer.pt",
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean"])
    parser.add_argument("--max-tokens", type=int, default=4096)

    parser.add_argument(
        "--use-image", action="store_true", help="Enable image modality"
    )
    parser.add_argument("--use-text", action="store_true", help="Enable text modality")
    parser.add_argument("--use-kg", action="store_true", help="Enable KG modality")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_fusion_transformer(args)


if __name__ == "__main__":
    main()
