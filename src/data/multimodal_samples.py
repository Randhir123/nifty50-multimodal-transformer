"""Aligned multimodal sample artifact helpers.

The functions in this module define the project-level bridge between modality
branches and the fusion Transformer.  The first implementation slice is small
and deterministic: it can build toy aligned arrays, validate the artifact
schema, and convert leakage-safe text/KG context into numeric tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import hashlib

import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.data.kg_features_v2 import build_kg_v2
from src.data.text import build_company_text_input, normalize_company_text_records
from src.kg.query_graph import retrieve_kg_context
from src.models.image_transformer import ImageTransformer, ImageTransformerConfig
from src.viz.charts import resolve_chart_path

# Lazy imports for GAF/MTF path to avoid hard dep on scipy/pyts in toy tests
def _import_gaf_mtf():
    from src.data.timeseries_images import compute_gaf, compute_mtf
    return compute_gaf, compute_mtf


def _import_image_cnn():
    from src.models.image_cnn import ImageCNN, ImageCNNConfig
    return ImageCNN, ImageCNNConfig


@dataclass(frozen=True)
class MultimodalSampleArrays:
    """Aligned arrays for one multimodal fusion dataset artifact.

    Required arrays use the same first dimension. Optional modality arrays are
    included only when that modality is available for the same sample rows.
    """

    tabular_tokens: np.ndarray
    y: np.ndarray
    end_dates: np.ndarray
    stock_ids: np.ndarray
    image_tokens: np.ndarray | None = None
    text_tokens: np.ndarray | None = None
    kg_tokens: np.ndarray | None = None

    def validate(self) -> None:
        """Validate basic shape and sample-count invariants."""
        if self.tabular_tokens.ndim != 3:
            raise ValueError("tabular_tokens must be 3D [samples, window, features]")
        sample_count = self.tabular_tokens.shape[0]
        if sample_count == 0:
            raise ValueError("multimodal sample artifact must not be empty")

        required_lengths = {
            "y": self.y.shape[0],
            "end_dates": self.end_dates.shape[0],
            "stock_ids": self.stock_ids.shape[0],
        }
        for name, length in required_lengths.items():
            if length != sample_count:
                raise ValueError(
                    f"Sample count mismatch for {name}: expected {sample_count}, got {length}"
                )

        for name, values in (
            ("image_tokens", self.image_tokens),
            ("text_tokens", self.text_tokens),
            ("kg_tokens", self.kg_tokens),
        ):
            if values is None:
                continue
            if values.ndim not in (2, 3):
                raise ValueError(f"{name} must be 2D or 3D, got {values.ndim}D")
            if values.shape[0] != sample_count:
                raise ValueError(
                    f"Sample count mismatch for {name}: expected {sample_count}, got {values.shape[0]}"
                )

    def to_npz_kwargs(self) -> dict[str, np.ndarray]:
        """Return keyword arguments suitable for ``numpy.savez_compressed``."""
        self.validate()
        payload: dict[str, np.ndarray] = {
            "tabular_tokens": self.tabular_tokens.astype(np.float32, copy=False),
            "y": self.y.astype(np.int64, copy=False),
            "end_dates": self.end_dates,
            "stock_ids": self.stock_ids.astype(str),
        }
        if self.image_tokens is not None:
            payload["image_tokens"] = self.image_tokens.astype(np.float32, copy=False)
        if self.text_tokens is not None:
            payload["text_tokens"] = self.text_tokens.astype(np.float32, copy=False)
        if self.kg_tokens is not None:
            payload["kg_tokens"] = self.kg_tokens.astype(np.float32, copy=False)
        return payload


def save_multimodal_samples(arrays: MultimodalSampleArrays, path: str | Path) -> Path:
    """Save aligned multimodal arrays to a compressed `.npz` artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays.to_npz_kwargs())
    return output_path


def _validate_tabular_inputs(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    stock_col: str,
    date_col: str,
    label_col: str,
    window_size: int,
) -> None:
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")

    required = [stock_col, date_col, label_col, *feature_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_tabular_multimodal_samples(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    stock_col: str = "stock_id",
    date_col: str = "date",
    label_col: str = "label",
    window_size: int = 60,
    dropna: bool = True,
) -> MultimodalSampleArrays:
    """Build aligned multimodal samples from real tabular rolling windows."""
    _validate_tabular_inputs(
        df,
        feature_cols=feature_cols,
        stock_col=stock_col,
        date_col=date_col,
        label_col=label_col,
        window_size=window_size,
    )

    work_df = df.copy()
    work_df[stock_col] = work_df[stock_col].astype(str).str.strip()
    work_df[date_col] = pd.to_datetime(work_df[date_col])
    if dropna:
        work_df = work_df.dropna(
            subset=[stock_col, date_col, label_col, *feature_cols]
        ).reset_index(drop=True)

    work_df = work_df.sort_values([stock_col, date_col]).reset_index(drop=True)

    windows: list[np.ndarray] = []
    labels: list[int] = []
    end_dates: list[pd.Timestamp] = []
    stock_ids: list[str] = []

    for stock_id, stock_df in work_df.groupby(stock_col, sort=True):
        stock_frame = stock_df.sort_values(date_col).reset_index(drop=True)
        if len(stock_frame) < window_size:
            continue

        features = stock_frame.loc[:, list(feature_cols)].to_numpy(dtype=np.float32)
        stock_labels = stock_frame.loc[:, label_col].to_numpy(dtype=np.int64)
        stock_dates = stock_frame.loc[:, date_col].to_numpy()

        sample_count = len(stock_frame) - window_size + 1
        for start_idx in range(sample_count):
            end_idx = start_idx + window_size - 1
            windows.append(features[start_idx : end_idx + 1])
            labels.append(int(stock_labels[end_idx]))
            end_dates.append(pd.to_datetime(stock_dates[end_idx]))
            stock_ids.append(str(stock_id))

    if not windows:
        raise ValueError(
            "No tabular samples could be built; check window_size and per-stock row counts"
        )

    arrays = MultimodalSampleArrays(
        tabular_tokens=np.stack(windows).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        end_dates=np.asarray(end_dates, dtype="datetime64[ns]"),
        stock_ids=np.asarray(stock_ids, dtype=str),
    )
    arrays.validate()
    return arrays


def infer_numeric_feature_columns(
    df: pd.DataFrame,
    *,
    stock_col: str = "stock_id",
    date_col: str = "date",
    label_col: str = "label",
) -> list[str]:
    """Infer numeric tabular feature columns from a dataframe."""
    excluded = {stock_col, date_col, label_col}
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def stable_text_token(text: str, *, dim: int = 16) -> np.ndarray:
    """Create a deterministic lightweight token vector for text.

    This is not a semantic embedding.  It is a CPU-only placeholder that makes
    the aligned artifact and fusion path testable until a real text encoder is
    wired into the same schema.
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    values = np.resize(raw, dim) / 255.0
    return values.astype(np.float32)


def stable_image_token(identifier: str, *, dim: int = 12) -> np.ndarray:
    """Create a deterministic placeholder token for a chart/image identifier."""
    return stable_text_token(f"image::{identifier}", dim=dim)


def resolve_image_paths_for_samples(
    *,
    stock_ids: Sequence[str],
    end_dates: Sequence[Any],
    chart_dir: str | Path,
) -> list[Path]:
    """Resolve deterministic chart paths aligned to sample rows."""
    if len(stock_ids) != len(end_dates):
        raise ValueError("stock_ids and end_dates must have identical lengths")
    return [
        resolve_chart_path(str(stock_id), pd.Timestamp(end_date), output_dir=chart_dir)
        for stock_id, end_date in zip(stock_ids, end_dates, strict=True)
    ]


def load_chart_image_tensors(
    image_paths: Sequence[str | Path],
    *,
    image_size: int,
) -> torch.Tensor:
    """Load chart images into a float tensor of shape [batch, 3, H, W]."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency installed via torchvision/Pillow
        raise ImportError("Pillow is required to load chart images") from exc

    tensors: list[np.ndarray] = []
    for raw_path in image_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Chart image not found: {path}")
        image = Image.open(path).convert("RGB").resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensors.append(np.transpose(array, (2, 0, 1)))
    if not tensors:
        raise ValueError("image_paths must not be empty")
    return torch.from_numpy(np.stack(tensors).astype(np.float32))


def build_image_tokens_for_samples(
    *,
    stock_ids: Sequence[str],
    end_dates: Sequence[Any],
    chart_dir: str | Path,
    config: ImageTransformerConfig | None = None,
    batch_size: int = 8,
    device: str = "cpu",
) -> np.ndarray:
    """Encode aligned chart images into image tokens using ``ImageTransformer``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_config = config or ImageTransformerConfig()
    paths = resolve_image_paths_for_samples(
        stock_ids=stock_ids, end_dates=end_dates, chart_dir=chart_dir
    )
    images = load_chart_image_tensors(paths, image_size=model_config.image_size)
    torch_device = torch.device(
        device if torch.cuda.is_available() or device == "cpu" else "cpu"
    )
    model = ImageTransformer(model_config).to(torch_device)
    model.eval()

    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, images.size(0), batch_size):
            batch = images[start : start + batch_size].to(torch_device)
            embeddings = model.encode_images(batch)
            batches.append(embeddings.cpu().numpy())
    return np.concatenate(batches, axis=0).astype(np.float32)


def attach_image_tokens(
    arrays: MultimodalSampleArrays,
    *,
    chart_dir: str | Path,
    config: ImageTransformerConfig | None = None,
    batch_size: int = 8,
    device: str = "cpu",
) -> MultimodalSampleArrays:
    """Return a copy of ``arrays`` with chart-image tokens aligned by row."""
    image_tokens = build_image_tokens_for_samples(
        stock_ids=arrays.stock_ids,
        end_dates=arrays.end_dates,
        chart_dir=chart_dir,
        config=config,
        batch_size=batch_size,
        device=device,
    )
    enriched = replace(arrays, image_tokens=image_tokens)
    enriched.validate()
    return enriched


def build_text_tokens_for_samples(
    samples: pd.DataFrame,
    text_records: pd.DataFrame,
    *,
    stock_col: str = "stock_id",
    date_col: str = "date",
    top_k: int = 5,
    dim: int = 16,
) -> np.ndarray:
    """Build deterministic text tokens with an as-of-date cutoff.

    Only records with ``event_date <= sample[date_col]`` are visible to a sample.
    """
    if stock_col not in samples.columns or date_col not in samples.columns:
        raise ValueError(f"samples must include '{stock_col}' and '{date_col}'")
    normalized_records = normalize_company_text_records(text_records)
    tokens: list[np.ndarray] = []
    for row in samples.itertuples(index=False):
        stock_id = getattr(row, stock_col)
        as_of_date = getattr(row, date_col)
        text = build_company_text_input(
            normalized_records,
            stock_id=stock_id,
            as_of_date=as_of_date,
            top_k=top_k,
        )
        tokens.append(stable_text_token(text, dim=dim))
    return np.stack(tokens).astype(np.float32)


def build_text_tokens_for_sample_arrays(
    arrays: MultimodalSampleArrays,
    text_records: pd.DataFrame,
    *,
    top_k: int = 5,
    dim: int = 16,
) -> np.ndarray:
    """Build text tokens aligned to ``arrays.stock_ids`` and ``arrays.end_dates``."""
    samples = pd.DataFrame({"stock_id": arrays.stock_ids, "date": arrays.end_dates})
    return build_text_tokens_for_samples(samples, text_records, top_k=top_k, dim=dim)


def attach_text_tokens(
    arrays: MultimodalSampleArrays,
    text_records: pd.DataFrame,
    *,
    top_k: int = 5,
    dim: int = 16,
) -> MultimodalSampleArrays:
    """Return a copy of ``arrays`` with leakage-safe text tokens aligned by row."""
    text_tokens = build_text_tokens_for_sample_arrays(
        arrays,
        text_records,
        top_k=top_k,
        dim=dim,
    )
    enriched = replace(arrays, text_tokens=text_tokens)
    enriched.validate()
    return enriched


def kg_context_to_token(context: dict[str, Any], *, event_types: list[str] | None = None) -> np.ndarray:
    """Flatten normalized KG context to a stable numeric token."""
    flags = context.get("event_flags", {}) or {}
    ordered_event_types = event_types or sorted(str(k) for k in flags)
    values = [
        float(context.get("peer_count") or 0.0),
        float(context.get("peer_avg_recent_return") or 0.0),
        float(context.get("sector_avg_recent_return") or 0.0),
    ]
    values.extend(float(flags.get(event_type, 0)) for event_type in ordered_event_types)
    return np.asarray(values, dtype=np.float32)


def build_kg_tokens_from_contexts(contexts: list[dict[str, Any]]) -> np.ndarray:
    """Convert a list of normalized KG context dictionaries into a token matrix."""
    event_types = sorted(
        {
            str(event_type)
            for context in contexts
            for event_type in (context.get("event_flags", {}) or {})
        }
    )
    return np.stack(
        [kg_context_to_token(context, event_types=event_types) for context in contexts]
    ).astype(np.float32)


def build_kg_tokens_for_samples(
    graph: nx.Graph,
    *,
    stock_ids: Sequence[str],
    end_dates: Sequence[Any],
    returns: pd.DataFrame | None = None,
    lookback_periods: int = 5,
    event_lookback_days: int = 7,
    index_id: str = "NIFTY50",
) -> np.ndarray:
    """Build KG tokens aligned to ``stock_ids`` and ``end_dates`` sample rows."""
    if len(stock_ids) != len(end_dates):
        raise ValueError("stock_ids and end_dates must have identical lengths")
    contexts = [
        retrieve_kg_context(
            graph,
            stock_id=str(stock_id),
            as_of_date=end_date,
            returns=returns,
            lookback_periods=lookback_periods,
            event_lookback_days=event_lookback_days,
            index_id=index_id,
        )
        for stock_id, end_date in zip(stock_ids, end_dates, strict=True)
    ]
    return build_kg_tokens_from_contexts(contexts)


def attach_kg_tokens(
    arrays: MultimodalSampleArrays,
    graph: nx.Graph,
    *,
    returns: pd.DataFrame | None = None,
    lookback_periods: int = 5,
    event_lookback_days: int = 7,
    index_id: str = "NIFTY50",
) -> MultimodalSampleArrays:
    """Return a copy of ``arrays`` with KG tokens aligned by row."""
    kg_tokens = build_kg_tokens_for_samples(
        graph,
        stock_ids=arrays.stock_ids,
        end_dates=arrays.end_dates,
        returns=returns,
        lookback_periods=lookback_periods,
        event_lookback_days=event_lookback_days,
        index_id=index_id,
    )
    enriched = replace(arrays, kg_tokens=kg_tokens)
    enriched.validate()
    return enriched


def attach_kg_v2_tokens(
    arrays: MultimodalSampleArrays,
    *,
    universe_ohlcv: dict[str, pd.DataFrame],
    peer_ohlcv: dict[str, pd.DataFrame] | None = None,
    benchmark_ohlcv: pd.DataFrame,
    sector_mapping: dict[str, str] | None = None,
) -> MultimodalSampleArrays:
    """Return a copy of ``arrays`` with KG v2 relational features aligned by row."""
    result = build_kg_v2(
        training_ohlcv=universe_ohlcv,
        peer_ohlcv=peer_ohlcv or universe_ohlcv,
        benchmark_ohlcv=benchmark_ohlcv,
        sector_mapping=sector_mapping,
        stock_ids=arrays.stock_ids,
        end_dates=arrays.end_dates,
    )
    enriched = replace(arrays, kg_tokens=result.values)
    enriched.validate()
    return enriched


def build_gaf_mtf_image_tokens(
    *,
    stock_ids: Sequence[str],
    end_dates: Sequence[Any],
    raw_dir: str | Path,
    image_size: int = 32,
    window_size: int = 20,
    output_dim: int = 16,
    device: str = "cpu",
) -> np.ndarray:
    """Build CNN-encoded GAF/MTF image tokens from raw per-stock CSVs.

    For each (stock_id, end_date) pair, loads close prices from
    ``raw_dir/{STOCK_UPPER_UNDERSCORE}.csv``, extracts a trailing window of
    length ``window_size`` ending on or before ``end_date``, computes a
    2-channel (GAF + MTF) image, and encodes via ``ImageCNN``.

    Samples missing a raw CSV or with insufficient history receive a
    zero-filled token vector.

    Returns:
        np.ndarray: shape [N, output_dim], dtype float32
    """
    compute_gaf, compute_mtf = _import_gaf_mtf()
    ImageCNN, ImageCNNConfig = _import_image_cnn()

    raw_dir = Path(raw_dir)

    stock_close: dict[str, pd.Series] = {}
    for sid in {str(s) for s in stock_ids}:
        fname = sid.upper().replace(".", "_") + ".csv"
        csv_path = raw_dir / fname
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["date"])
            df = df.sort_values("date").set_index("date")
            stock_close[sid] = df["close"]

    images: list[np.ndarray] = []
    for stock_id, end_date in zip(stock_ids, end_dates, strict=True):
        sid = str(stock_id)
        ts = pd.Timestamp(end_date).normalize()

        if sid in stock_close:
            close = stock_close[sid]
            prices = close[close.index <= ts].iloc[-window_size:].values
        else:
            prices = np.array([], dtype=np.float32)

        if len(prices) < 2:
            images.append(np.zeros((2, image_size, image_size), dtype=np.float32))
            continue

        gaf = compute_gaf(prices, image_size=image_size)
        mtf = compute_mtf(prices, image_size=image_size)
        images.append(np.stack([gaf, mtf], axis=0).astype(np.float32))

    config = ImageCNNConfig(image_size=image_size, in_channels=2, output_dim=output_dim)
    model = ImageCNN(config).to(device)
    model.eval()

    batch = torch.from_numpy(np.stack(images).astype(np.float32)).to(device)
    with torch.no_grad():
        embeddings = model.encode_images(batch)
    return embeddings.cpu().numpy().astype(np.float32)


def attach_gaf_mtf_image_tokens(
    arrays: MultimodalSampleArrays,
    *,
    raw_dir: str | Path,
    image_size: int = 32,
    output_dim: int = 16,
    device: str = "cpu",
) -> MultimodalSampleArrays:
    """Return a copy of ``arrays`` with GAF/MTF image tokens aligned by row."""
    window_size = arrays.tabular_tokens.shape[1]
    image_tokens = build_gaf_mtf_image_tokens(
        stock_ids=arrays.stock_ids,
        end_dates=arrays.end_dates,
        raw_dir=raw_dir,
        image_size=image_size,
        window_size=window_size,
        output_dim=output_dim,
        device=device,
    )
    enriched = replace(arrays, image_tokens=image_tokens)
    enriched.validate()
    return enriched


def build_toy_multimodal_samples(
    *,
    num_samples: int = 12,
    window_size: int = 5,
    tabular_dim: int = 4,
    image_dim: int = 12,
    text_dim: int = 16,
) -> MultimodalSampleArrays:
    """Build a deterministic toy multimodal artifact for tests and smoke runs."""
    if num_samples < 4:
        raise ValueError("num_samples must be at least 4 for chronological splits")
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=num_samples, freq="D")
    stock_ids = np.array(["RELIANCE.NS" if i % 2 == 0 else "TCS.NS" for i in range(num_samples)])

    tabular_tokens = rng.normal(
        loc=0.0, scale=1.0, size=(num_samples, window_size, tabular_dim)
    ).astype(np.float32)
    y = (tabular_tokens[:, -1, 0] > 0.0).astype(np.int64)

    sample_frame = pd.DataFrame(
        {"stock_id": stock_ids, "date": dates, "chart_id": [f"chart-{i}" for i in range(num_samples)]}
    )
    image_tokens = np.stack(
        [stable_image_token(chart_id, dim=image_dim) for chart_id in sample_frame["chart_id"]]
    ).astype(np.float32)

    text_records = pd.DataFrame(
        [
            {
                "stock_id": stock_id,
                "event_date": date - pd.Timedelta(days=1),
                "source_type": "news",
                "title": f"{stock_id} update {i}",
                "body_text": f"Market update available before sample {i}.",
            }
            for i, (stock_id, date) in enumerate(zip(stock_ids, dates, strict=True))
        ]
    )
    text_tokens = build_text_tokens_for_samples(
        sample_frame, text_records, dim=text_dim
    )

    contexts = [
        {
            "peer_count": 1,
            "peer_avg_recent_return": float(i) / 100.0,
            "sector_avg_recent_return": float(i + 1) / 100.0,
            "event_flags": {"earnings": int(i % 2 == 0), "guidance": int(i % 3 == 0)},
        }
        for i in range(num_samples)
    ]
    kg_tokens = build_kg_tokens_from_contexts(contexts)

    arrays = MultimodalSampleArrays(
        tabular_tokens=tabular_tokens,
        y=y,
        end_dates=dates.to_numpy(),
        stock_ids=stock_ids,
        image_tokens=image_tokens,
        text_tokens=text_tokens,
        kg_tokens=kg_tokens,
    )
    arrays.validate()
    return arrays
