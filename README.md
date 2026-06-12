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

## Project Structure

```text
app.py                  # Streamlit dashboard
nifty_intel/data.py     # CSV loading, validation, metadata joins
nifty_intel/indicators.py
nifty_intel/risk.py
nifty_intel/prediction.py
nifty_intel/portfolio.py
scripts/smoke_test.py   # End-to-end sanity check
REPORT.md               # Technical report draft
PRESENTATION_OUTLINE.md # 8-slide submission outline
```

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

Then open:

```text
http://localhost:8501
```

## Smoke Test

```bash
python scripts/smoke_test.py
```

Expected output includes the row count, number of symbols, selected model, and portfolio holding count.

## Methodology

The prototype focuses on decision support rather than live trading. It computes technical indicators from historical prices, evaluates risk-adjusted performance, trains per-stock next-day return models using chronological validation, and ranks stocks for profile-specific portfolio construction.

## Dashboard Sections

- **Market Overview:** risk-return map, industry coverage, and top Sharpe performers.
- **Stock Analytics:** price trend, moving averages, volume, RSI, MACD, and risk KPIs.
- **Predictor:** next-day return model metrics, baseline comparison, actual vs predicted returns, and feature importance.
- **Portfolio Builder:** conservative, balanced, and aggressive allocations with explanations.
- **Methodology:** concise summary of assumptions, metrics, and limitations.

## Reproducibility

- No live market data or financial APIs are used.
- All computations come from the local CSV files listed above.
- The train/test split is chronological to avoid future leakage.
- The smoke test verifies data loading, feature engineering, prediction, and portfolio construction.

## Screenshots

Run the app locally and capture the main dashboard tabs for final submission assets:

```bash
streamlit run app.py
```

## Limitations

- Uses historical data ending in April 2021.
- Does not include news, sentiment, macroeconomic data, or live prices.
- Predictions are short-horizon statistical estimates and should not be treated as trading signals.
