"""Data pipeline package: features, labels, and dataset construction."""

from .dataset import RollingWindowDataset, create_rolling_transformer_dataset, load_ohlcv_csv
from .features import compute_technical_features
from .labels import generate_outperformance_label
from .text import (
    CompanyTextRecord,
    build_company_text_input,
    build_company_text_inputs_for_samples,
    extract_pdf_text_record,
    normalize_company_text_records,
    records_to_dataframe,
)

__all__ = [
    "RollingWindowDataset",
    "compute_technical_features",
    "create_rolling_transformer_dataset",
    "generate_outperformance_label",
    "load_ohlcv_csv",
    "CompanyTextRecord",
    "normalize_company_text_records",
    "build_company_text_input",
    "build_company_text_inputs_for_samples",
    "extract_pdf_text_record",
    "records_to_dataframe",
]
