from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_RISK_FREE_RATE = 0.05


def max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    wealth = (1 + clean).cumprod()
    peak = wealth.cummax()
    return float(((wealth / peak) - 1).min())


def annualized_return(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    total_return = float((1 + clean).prod() - 1)
    years = max(len(clean) / TRADING_DAYS, 1 / TRADING_DAYS)
    return (1 + total_return) ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> float:
    vol = annualized_volatility(returns)
    if vol == 0:
        return 0.0
    return float((annualized_return(returns) - risk_free_rate) / vol)


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> float:
    clean = returns.dropna()
    downside = clean[clean < 0]
    if len(downside) < 2:
        return 0.0
    downside_vol = float(downside.std() * np.sqrt(TRADING_DAYS))
    if downside_vol == 0:
        return 0.0
    return float((annualized_return(clean) - risk_free_rate) / downside_vol)


def risk_summary(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    return {
        "annualized_return": annualized_return(clean),
        "annualized_volatility": annualized_volatility(clean),
        "sharpe_ratio": sharpe_ratio(clean),
        "sortino_ratio": sortino_ratio(clean),
        "max_drawdown": max_drawdown(clean),
    }


def summarize_symbol_risk(features: pd.DataFrame, trailing_days: int = TRADING_DAYS) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for symbol, group in features.groupby("Symbol"):
        recent = group.sort_values("Date").tail(trailing_days)
        metrics = risk_summary(recent["Daily Return"])
        metrics["symbol"] = symbol
        metrics["latest_close"] = float(recent["Close"].iloc[-1]) if not recent.empty else 0.0
        metrics["momentum_20d"] = float(recent["Momentum20"].iloc[-1]) if not recent.empty else 0.0
        rows.append(metrics)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.rename(columns={"symbol": "Symbol"})

