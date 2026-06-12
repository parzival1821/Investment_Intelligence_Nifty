# Presentation Outline: NIFTY-50 Investment Intelligence

## Slide 1: Title

NIFTY-50 Investment Intelligence Platform  
Akshat - 23115007  
Electrical Engineering, completed 3rd year, entering 4th year  
Solo Project

## Slide 2: Problem Overview

- Investors need more than raw price charts.
- Goal: convert historical NIFTY-50 data into actionable decision support.
- Focus: analytics, prediction, risk, portfolio construction, and explainability.

## Slide 3: Dataset

- Organizer-provided NIFTY-50 historical market data.
- Local aggregate: 235,192 rows, 65 historical symbols.
- Date range: January 2000 to April 2021.
- Fields: OHLC, VWAP, volume, turnover, trades, deliverable volume.

## Slide 4: Feature Engineering

- Daily and next-day returns.
- Moving averages, RSI, MACD, Bollinger Bands.
- Rolling volatility, momentum, volume change, drawdown.
- Metadata enrichment with company and industry labels.

## Slide 5: Prediction Engine

- Target: next-trading-day return.
- Model: Random Forest Regressor.
- Validation: chronological 80/20 split.
- Metrics: MAE, RMSE, R2, directional accuracy.
- Baseline: previous-day return.

## Slide 6: Risk and Portfolio Analytics

- Risk metrics: annualized return, volatility, Sharpe, Sortino, max drawdown.
- Profiles: conservative, balanced, aggressive.
- Weighted scoring and capped allocations.
- Plain-English explanation for every holding.

## Slide 7: Dashboard Demo

- Market overview risk-return map.
- Stock analytics tab.
- Predictor tab with feature importance.
- Portfolio builder with profile-specific allocations.

## Slide 8: Impact and Future Work

- Practical decision-support prototype.
- Transparent assumptions and no live market dependency.
- Future work: walk-forward validation, backtesting, sector constraints, downloadable reports.
