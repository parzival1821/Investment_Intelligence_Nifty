from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("archive-2")
PRICE_FILE = "NIFTY50_all.csv"
METADATA_FILE = "stock_metadata.csv"

PRICE_COLUMNS = [
    "Date",
    "Symbol",
    "Open",
    "High",
    "Low",
    "Close",
    "VWAP",
    "Volume",
    "Turnover",
    "Trades",
]

METADATA_COLUMNS = ["Company Name", "Industry", "Symbol"]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Place the Kaggle NIFTY-50 files under archive-2/."
        )


def load_price_data(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load and normalize the aggregate NIFTY-50 price history."""
    data_dir = Path(data_dir)
    path = data_dir / PRICE_FILE
    _require_file(path)

    df = pd.read_csv(path, usecols=lambda col: col in PRICE_COLUMNS)
    missing = sorted(set(PRICE_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Price data is missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = [col for col in PRICE_COLUMNS if col not in {"Date", "Symbol"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Symbol", "Close"])
        .sort_values(["Symbol", "Date"])
        .reset_index(drop=True)
    )
    return df


def load_metadata(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load company metadata and keep one row per symbol."""
    data_dir = Path(data_dir)
    path = data_dir / METADATA_FILE
    _require_file(path)

    metadata = pd.read_csv(path)
    missing = sorted(set(METADATA_COLUMNS) - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    metadata = metadata[METADATA_COLUMNS].drop_duplicates("Symbol")
    metadata = metadata.rename(columns={"Company Name": "Company"})
    return metadata


def add_company_metadata(prices: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach company and industry labels, falling back to symbol names."""
    enriched = prices.merge(metadata, on="Symbol", how="left")
    enriched["Company"] = enriched["Company"].fillna(enriched["Symbol"])
    enriched["Industry"] = enriched["Industry"].fillna("Unclassified")
    return enriched


def available_symbols(prices: pd.DataFrame) -> list[str]:
    return sorted(prices["Symbol"].dropna().unique().tolist())


def filter_symbol(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    stock = prices.loc[prices["Symbol"] == symbol].copy()
    if stock.empty:
        raise ValueError(f"No rows found for symbol {symbol}")
    return stock.sort_values("Date").reset_index(drop=True)


def latest_market_snapshot(prices: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Return the latest available row per symbol with metadata attached."""
    latest_idx = prices.groupby("Symbol")["Date"].idxmax()
    latest = prices.loc[latest_idx].copy().reset_index(drop=True)
    return add_company_metadata(latest, metadata)

