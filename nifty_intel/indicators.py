from __future__ import annotations

import numpy as np
import pandas as pd


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window, min_periods=window).mean()
    avg_loss = losses.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_technical_indicators(stock: pd.DataFrame) -> pd.DataFrame:
    """Add technical features for a single symbol ordered by date."""
    df = stock.sort_values("Date").copy()
    close = df["Close"]

    df["Daily Return"] = close.pct_change()
    df["Next Return"] = df["Daily Return"].shift(-1)
    df["MA20"] = close.rolling(20, min_periods=5).mean()
    df["MA50"] = close.rolling(50, min_periods=10).mean()
    df["MA200"] = close.rolling(200, min_periods=40).mean()
    df["Volatility20"] = df["Daily Return"].rolling(20, min_periods=10).std()
    df["RSI14"] = relative_strength_index(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    rolling20 = close.rolling(20, min_periods=10)
    mid = rolling20.mean()
    band_std = rolling20.std()
    df["Bollinger Upper"] = mid + (2 * band_std)
    df["Bollinger Lower"] = mid - (2 * band_std)
    df["Momentum20"] = close.pct_change(20)
    df["Volume Change"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    wealth = (1 + df["Daily Return"].fillna(0)).cumprod()
    peak = wealth.cummax()
    df["Drawdown"] = (wealth / peak) - 1

    return df


def add_indicators_by_symbol(prices: pd.DataFrame) -> pd.DataFrame:
    frames = [
        add_technical_indicators(group)
        for _, group in prices.groupby("Symbol", sort=False)
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
