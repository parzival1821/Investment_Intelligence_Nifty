from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifty_intel.data import (
    available_symbols,
    filter_symbol,
    latest_market_snapshot,
    load_metadata,
    load_price_data,
)
from nifty_intel.indicators import add_technical_indicators
from nifty_intel.indicators import add_indicators_by_symbol
from nifty_intel.portfolio import build_portfolio
from nifty_intel.prediction import train_predictor
from nifty_intel.risk import risk_summary


def main() -> None:
    prices = load_price_data()
    metadata = load_metadata()
    symbols = available_symbols(prices)
    if not symbols:
        raise AssertionError("No symbols loaded from price data")

    sample_symbol = "RELIANCE" if "RELIANCE" in symbols else symbols[0]
    stock = add_technical_indicators(filter_symbol(prices, sample_symbol))
    if stock.empty or "RSI14" not in stock.columns:
        raise AssertionError("Technical indicators were not generated")

    metrics = risk_summary(stock["Daily Return"])
    required_metrics = {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
    }
    if required_metrics - set(metrics):
        raise AssertionError("Risk summary is missing required metrics")

    snapshot = latest_market_snapshot(prices, metadata)
    if snapshot.empty or "Company" not in snapshot.columns:
        raise AssertionError("Latest market snapshot did not join metadata")

    prediction = train_predictor(stock)
    if prediction.predictions.empty or "rmse" not in prediction.metrics:
        raise AssertionError("Prediction engine did not return expected output")

    portfolio_features = add_indicators_by_symbol(
        prices.loc[prices["Symbol"].isin(symbols[:12])].copy()
    )
    portfolio = build_portfolio(portfolio_features, metadata, profile="Balanced", top_n=5)
    if portfolio.empty or abs(float(portfolio["weight"].sum()) - 1) > 0.001:
        raise AssertionError("Portfolio builder did not return normalized weights")

    print(
        f"Smoke test passed: {len(prices):,} rows, {len(symbols)} symbols, "
        f"sample={sample_symbol}, sharpe={metrics['sharpe_ratio']:.2f}, "
        f"model={prediction.model_name}, portfolio={len(portfolio)} holdings"
    )


if __name__ == "__main__":
    main()
