"""Build aligned multimodal fusion sample artifacts.

The first CLI mode writes a deterministic toy dataset that exercises the full
fusion contract: tabular, image, text, KG, labels, stock IDs, and end dates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.multimodal_samples import build_toy_multimodal_samples, save_multimodal_samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build aligned multimodal samples")
    parser.add_argument(
        "--toy-output",
        type=str,
        default="data/processed/multimodal_samples.npz",
        help="Path to write a deterministic toy multimodal NPZ artifact.",
    )
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--tabular-dim", type=int, default=4)
    parser.add_argument("--image-dim", type=int, default=12)
    parser.add_argument("--text-dim", type=int, default=16)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    arrays = build_toy_multimodal_samples(
        num_samples=args.num_samples,
        window_size=args.window_size,
        tabular_dim=args.tabular_dim,
        image_dim=args.image_dim,
        text_dim=args.text_dim,
    )
    output_path = save_multimodal_samples(arrays, Path(args.toy_output))
    print(f"Saved toy multimodal sample artifact to: {output_path}")
    print(
        "Shapes: "
        f"tabular={arrays.tabular_tokens.shape}, "
        f"image={arrays.image_tokens.shape if arrays.image_tokens is not None else None}, "
        f"text={arrays.text_tokens.shape if arrays.text_tokens is not None else None}, "
        f"kg={arrays.kg_tokens.shape if arrays.kg_tokens is not None else None}, "
        f"y={arrays.y.shape}"
    )


if __name__ == "__main__":
    main()
