# NIFTY-50 Investment Intelligence

AI-powered investment intelligence dashboard for historical NIFTY-50 market data. The project turns daily OHLCV data into technical indicators, stock-level risk analytics, next-day return prediction, and explainable portfolio suggestions for conservative, balanced, and aggressive investor profiles.

> Educational project only. This is not financial advice and does not use live market data.

## Features

- Market overview across NIFTY-50 constituents
- Stock analytics with moving averages, RSI, MACD, Bollinger Bands, volatility, and drawdown
- Random Forest next-day return prediction with chronological train/test evaluation
- Baseline comparison using previous-day return
- Risk metrics: annualized return, volatility, Sharpe ratio, Sortino ratio, and maximum drawdown
- Portfolio construction for conservative, balanced, and aggressive profiles
- Plain-English explanations for portfolio recommendations

## Dataset

Use the organizer-provided NIFTY-50 Kaggle dataset locally. Place these files at:

```text
archive-2/NIFTY50_all.csv
archive-2/stock_metadata.csv
```

The dataset folders are intentionally ignored by Git to keep the repository lightweight and reproducible.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Smoke Test

```bash
python scripts/smoke_test.py
```

## Methodology

The prototype focuses on decision support rather than live trading. It computes technical indicators from historical prices, evaluates risk-adjusted performance, trains per-stock next-day return models using chronological validation, and ranks stocks for profile-specific portfolio construction.

## Limitations

- Uses historical data ending in April 2021.
- Does not include news, sentiment, macroeconomic data, or live prices.
- Predictions are short-horizon statistical estimates and should not be treated as trading signals.
