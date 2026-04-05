from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


TOY_DIR = Path("data/toy")


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    """Keep tests deterministic across local and CI runs."""
    np.random.seed(7)
    torch.manual_seed(7)


@pytest.fixture
def toy_ohlcv() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load toy stock/index OHLCV datasets with parsed dates."""
    stock_df = pd.read_csv(TOY_DIR / "stock_ohlcv.csv")
    index_df = pd.read_csv(TOY_DIR / "index_ohlcv.csv")
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    index_df["date"] = pd.to_datetime(index_df["date"])
    return stock_df, index_df


@pytest.fixture
def toy_event_records() -> pd.DataFrame:
    """Load toy event records used by KG tests."""
    return pd.read_csv(TOY_DIR / "event_records.csv")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign markers based on test directory."""
    for item in items:
        path = str(item.fspath)
        if "/tests/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/tests/smoke/" in path:
            item.add_marker(pytest.mark.smoke)
        elif "/tests/integration/" in path:
            item.add_marker(pytest.mark.integration)
