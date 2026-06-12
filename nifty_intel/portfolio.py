from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_intel.risk import TRADING_DAYS, summarize_symbol_risk

PROFILE_CONFIG = {
    "Conservative": {
        "max_weight": 0.22,
        "description": "Prioritizes lower volatility, smaller drawdowns, and stable risk-adjusted returns.",
    },
    "Balanced": {
        "max_weight": 0.28,
        "description": "Balances return potential with volatility and risk-adjusted performance.",
    },
    "Aggressive": {
        "max_weight": 0.35,
        "description": "Accepts higher volatility for stronger return and momentum potential.",
    },
}


def _normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan).fillna(series.median())
    if clean.nunique(dropna=True) <= 1:
        normalized = pd.Series(0.5, index=series.index)
    else:
        normalized = (clean - clean.min()) / (clean.max() - clean.min())
    return normalized if higher_is_better else 1 - normalized


def _cap_and_normalize(weights: pd.Series, cap: float) -> pd.Series:
    weights = weights.clip(lower=0).astype(float)
    if weights.sum() <= 0:
        weights = pd.Series(1 / len(weights), index=weights.index)
    else:
        weights = weights / weights.sum()

    for _ in range(10):
        over = weights > cap
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights.loc[over] = cap
        under = ~over
        if not under.any() or weights.loc[under].sum() <= 0:
            break
        weights.loc[under] += (weights.loc[under] / weights.loc[under].sum()) * excess

    return weights / weights.sum()


def _score_candidates(candidates: pd.DataFrame, profile: str) -> pd.Series:
    return_score = _normalize(candidates["annualized_return"])
    sharpe_score = _normalize(candidates["sharpe_ratio"])
    vol_score = _normalize(candidates["annualized_volatility"], higher_is_better=False)
    drawdown_score = _normalize(candidates["max_drawdown"])
    momentum_score = _normalize(candidates["momentum_20d"])

    if profile == "Conservative":
        score = (0.42 * sharpe_score) + (0.38 * vol_score) + (0.20 * drawdown_score)
    elif profile == "Aggressive":
        score = (
            (0.40 * return_score)
            + (0.28 * momentum_score)
            + (0.20 * sharpe_score)
            + (0.12 * drawdown_score)
        )
    else:
        score = (
            (0.32 * sharpe_score)
            + (0.28 * return_score)
            + (0.22 * vol_score)
            + (0.18 * momentum_score)
        )

    return score.clip(lower=0)


def _explain(row: pd.Series, profile: str) -> str:
    sharpe = row["sharpe_ratio"]
    volatility = row["annualized_volatility"]
    annual_return = row["annualized_return"]
    drawdown = row["max_drawdown"]
    momentum = row["momentum_20d"]

    if profile == "Conservative":
        return (
            f"Selected for comparatively controlled volatility ({volatility:.1%}) "
            f"and risk-adjusted performance (Sharpe {sharpe:.2f})."
        )
    if profile == "Aggressive":
        return (
            f"Selected for stronger return potential ({annual_return:.1%}) "
            f"and recent momentum ({momentum:.1%}), with drawdown risk of {drawdown:.1%}."
        )
    return (
        f"Selected for a balanced mix of annualized return ({annual_return:.1%}), "
        f"Sharpe {sharpe:.2f}, and volatility ({volatility:.1%})."
    )


def build_portfolio(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    profile: str = "Balanced",
    top_n: int = 8,
    trailing_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    if profile not in PROFILE_CONFIG:
        raise ValueError(f"Unknown investor profile: {profile}")

    summary = summarize_symbol_risk(features, trailing_days=trailing_days)
    if summary.empty:
        raise ValueError("No risk summary available for portfolio construction")

    counts = features.groupby("Symbol")["Date"].count().rename("observations")
    candidates = summary.merge(counts, on="Symbol", how="left")
    candidates = candidates.loc[candidates["observations"] >= min(126, trailing_days)].copy()
    if candidates.empty:
        raise ValueError("Not enough observations to build a portfolio")

    candidates["score"] = _score_candidates(candidates, profile)
    candidates = candidates.sort_values("score", ascending=False).head(top_n).copy()
    if profile == "Conservative":
        raw_weight = candidates["score"] / candidates["annualized_volatility"].replace(0, np.nan)
    elif profile == "Aggressive":
        raw_weight = candidates["score"] * (1 + candidates["momentum_20d"].clip(lower=-0.5))
    else:
        raw_weight = candidates["score"]

    candidates["weight"] = _cap_and_normalize(
        raw_weight.fillna(candidates["score"]), PROFILE_CONFIG[profile]["max_weight"]
    )

    portfolio = candidates.merge(metadata, on="Symbol", how="left")
    portfolio["Company"] = portfolio["Company"].fillna(portfolio["Symbol"])
    portfolio["Industry"] = portfolio["Industry"].fillna("Unclassified")
    portfolio["Explanation"] = portfolio.apply(lambda row: _explain(row, profile), axis=1)

    columns = [
        "Symbol",
        "Company",
        "Industry",
        "weight",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "momentum_20d",
        "score",
        "Explanation",
    ]
    return portfolio[columns].sort_values("weight", ascending=False).reset_index(drop=True)


def profile_description(profile: str) -> str:
    return PROFILE_CONFIG.get(profile, PROFILE_CONFIG["Balanced"])["description"]
