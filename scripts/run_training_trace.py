"""Run N training epochs on one ablation variant + fold and emit a per-epoch trace CSV.

Useful for diagnosing training dynamics such as probability collapse (output range < 0.01),
saddle-point lock, and loss curve anomalies. The root cause of the original collapse was
CLS token pooling in shallow 16-dim encoders; mean pooling (now the default) resolved it.

Outputs one row per epoch to --output-csv with columns: epoch, tr_loss, vl_loss, tr_f1,
vl_f1, vl_roc_auc, vl_prob_min, vl_prob_mean, vl_prob_max, vl_prob_range.

Usage:
    python scripts/run_training_trace.py \
        --artifact data/processed/real_world_demo/real_world_multimodal_samples_gaf.npz \
        --variant tabular_only --fold 1 --epochs 50 \
        --output-csv data/processed/trace.csv

Key flags match run_ablation_study.py: --model-dim, --num-heads, --num-layers, --ff-dim.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.fusion import FusionTransformer, FusionTransformerConfig
from src.training.cv import PurgedWalkForwardSplit
from src.training.evaluate import compute_binary_classification_metrics
from src.training.train_fusion import (
    FusionArrays,
    FusionDataset,
    load_fusion_arrays,
    slice_fusion_arrays,
)

_VARIANT_FLAGS: dict[str, dict[str, bool]] = {
    "tabular_only":          dict(use_image=False, use_text=False, use_kg=False),
    "tabular_kg":            dict(use_image=False, use_text=False, use_kg=True),
    "tabular_image":         dict(use_image=True,  use_text=False, use_kg=False),
    "tabular_text":          dict(use_image=False, use_text=True,  use_kg=False),
    "tabular_image_text_kg": dict(use_image=True,  use_text=True,  use_kg=True),
}


def _build_fold_arrays(
    artifact: str,
    variant: str,
    fold_idx: int,
    *,
    cv_splits: int,
    horizon_days: int,
    embargo_days: int,
) -> tuple[FusionArrays, FusionArrays]:
    flags = _VARIANT_FLAGS[variant]
    arrays = load_fusion_arrays(artifact, **flags)
    splitter = PurgedWalkForwardSplit(
        n_splits=cv_splits, horizon_days=horizon_days, embargo_days=embargo_days
    )
    folds = list(splitter.split(arrays.end_dates))
    cv = folds[fold_idx]
    return slice_fusion_arrays(arrays, cv.train_idx), slice_fusion_arrays(arrays, cv.val_idx)


def run_trace(args: argparse.Namespace) -> list[dict]:
    train_arrays, val_arrays = _build_fold_arrays(
        args.artifact,
        args.variant,
        args.fold,
        cv_splits=args.cv_splits,
        horizon_days=args.horizon_days,
        embargo_days=args.embargo_days,
    )
    print(
        f"Variant={args.variant}  fold={args.fold}  "
        f"train={len(train_arrays.y)}  val={len(val_arrays.y)}"
    )
    pos_train = float(train_arrays.y.mean())
    pos_val = float(val_arrays.y.mean())
    print(f"  train_pos={pos_train:.4f}  val_pos={pos_val:.4f}")
    print(f"  pooling={args.pooling}  lr={args.lr:.2e}  warmup_epochs={args.warmup_epochs}"
          f"  grad_clip={args.grad_clip}  bias_init_to_prior={args.bias_init_to_prior}")
    print()

    device = torch.device("cpu")
    config = FusionTransformerConfig(
        tabular_dim=train_arrays.tabular_tokens.shape[-1],
        image_dim=(
            train_arrays.image_tokens.shape[-1]
            if train_arrays.image_tokens is not None else None
        ),
        text_dim=(
            train_arrays.text_tokens.shape[-1]
            if train_arrays.text_tokens is not None else None
        ),
        kg_dim=(
            train_arrays.kg_tokens.shape[-1]
            if train_arrays.kg_tokens is not None else None
        ),
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        pooling=args.pooling,
        max_tokens=4096,
    )
    model = FusionTransformer(config).to(device)

    # H1: initialise classifier bias to log-odds of training positive rate
    if args.bias_init_to_prior:
        eps = 1e-6
        log_odds = float(np.log((pos_train + eps) / (1.0 - pos_train + eps)))
        with torch.no_grad():
            model.classifier.bias.fill_(log_odds)
        print(f"  [H1] classifier bias initialised to {log_odds:.4f}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # H2: linear LR warmup
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if args.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.05,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )

    train_loader = DataLoader(
        FusionDataset(train_arrays), batch_size=args.batch_size, shuffle=False
    )
    val_loader = DataLoader(
        FusionDataset(val_arrays), batch_size=args.batch_size, shuffle=False
    )

    header = (
        f"{'ep':>3} | {'tr_loss':>8} {'vl_loss':>8} | "
        f"{'tr_f1':>6} {'vl_f1':>6} {'vl_roc':>7} | "
        f"{'vl_pmin':>8} {'vl_pmean':>9} {'vl_pmax':>8} {'vl_range':>9} | "
        f"{'vl_pred+':>9} {'vl_true+':>9} | {'lr':>9}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        tr_probs_list: list[np.ndarray] = []
        tr_labels_list: list[np.ndarray] = []

        for batch_inputs, labels in train_loader:
            labels = labels.to(device)
            model_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**model_inputs)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            with torch.no_grad():
                tr_probs_list.append(torch.sigmoid(logits).cpu().numpy())
                tr_labels_list.append(labels.cpu().numpy())
            tr_loss_sum += loss.item() * len(labels)
            tr_n += len(labels)

        if scheduler is not None:
            scheduler.step()

        tr_loss = tr_loss_sum / tr_n
        tr_m = compute_binary_classification_metrics(
            np.concatenate(tr_labels_list).astype(np.int64),
            np.concatenate(tr_probs_list).astype(np.float32),
        )

        # ── val ────────────────────────────────────────────────────────────
        model.eval()
        vl_loss_sum, vl_n = 0.0, 0
        vl_probs_list: list[np.ndarray] = []
        vl_labels_list: list[np.ndarray] = []

        with torch.no_grad():
            for batch_inputs, labels in val_loader:
                labels = labels.to(device)
                model_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                logits = model(**model_inputs)
                loss = criterion(logits, labels)
                vl_probs_list.append(torch.sigmoid(logits).cpu().numpy())
                vl_labels_list.append(labels.cpu().numpy())
                vl_loss_sum += loss.item() * len(labels)
                vl_n += len(labels)

        vl_loss = vl_loss_sum / vl_n
        vl_prob = np.concatenate(vl_probs_list).astype(np.float32)
        vl_y = np.concatenate(vl_labels_list).astype(np.int64)
        vl_m = compute_binary_classification_metrics(vl_y, vl_prob)
        vl_pred_pos = float((vl_prob >= 0.5).mean())
        vl_true_pos = float(vl_y.mean())
        vl_range = float(vl_prob.max() - vl_prob.min())
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "tr_loss": round(tr_loss, 6),
            "vl_loss": round(vl_loss, 6),
            "tr_f1": round(float(tr_m["f1"]), 6),
            "vl_f1": round(float(vl_m["f1"]), 6),
            "vl_roc_auc": round(float(vl_m["roc_auc"]), 6),
            "vl_prob_min": round(float(vl_prob.min()), 6),
            "vl_prob_mean": round(float(vl_prob.mean()), 6),
            "vl_prob_max": round(float(vl_prob.max()), 6),
            "vl_prob_range": round(vl_range, 6),
            "vl_pred_pos_rate": round(vl_pred_pos, 6),
            "vl_true_pos_rate": round(vl_true_pos, 6),
            "lr": current_lr,
        }
        rows.append(row)

        print(
            f"{epoch:>3} | {tr_loss:>8.4f} {vl_loss:>8.4f} | "
            f"{tr_m['f1']:>6.4f} {vl_m['f1']:>6.4f} {vl_m['roc_auc']:>7.4f} | "
            f"{vl_prob.min():>8.4f} {vl_prob.mean():>9.4f} {vl_prob.max():>8.4f} {vl_range:>9.4f} | "
            f"{vl_pred_pos:>9.4f} {vl_true_pos:>9.4f} | {current_lr:.2e}"
        )

    return rows


def _save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTrace saved to: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-epoch training trace for collapse diagnostics")
    p.add_argument("--artifact", required=True, help="Path to .npz multimodal artifact")
    p.add_argument(
        "--variant", default="tabular_only", choices=list(_VARIANT_FLAGS),
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default=None, help="CSV output path (optional)")
    # CV split parameters
    p.add_argument("--cv-splits", type=int, default=3)
    p.add_argument("--horizon-days", type=int, default=3)
    p.add_argument("--embargo-days", type=int, default=3)
    # Model parameters
    p.add_argument("--model-dim", type=int, default=16)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--ff-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training parameters
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # Hypothesis flags
    p.add_argument("--pooling", default="cls", choices=["cls", "mean"],
                   help="H4: 'mean' replaces CLS pooling with mean-over-tokens")
    p.add_argument("--bias-init-to-prior", action="store_true",
                   help="H1: initialise classifier bias to logit(train_pos_rate)")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="H2: linear LR warmup over N epochs (0 = disabled)")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="H5: gradient clip max-norm (0 = disabled)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = run_trace(args)
    if args.output:
        _save_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()
