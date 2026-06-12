# Technical Report: NIFTY-50 Investment Intelligence

**Participant:** Akshat  
**Problem Statement:** Data-Driven Investment Intelligence Using NIFTY-50 Market Data  
**Submission Type:** AI-powered investment intelligence dashboard using organizer-provided datasets

## 1. Problem Understanding

The objective is to transform historical NIFTY-50 stock market data into a practical investment intelligence platform. Instead of only predicting prices, the system supports decisions through stock analytics, risk assessment, machine learning forecasts, portfolio construction, and transparent explanations.

## 2. Dataset and EDA

The implementation uses the organizer-provided NIFTY-50 aggregate file `archive-2/NIFTY50_all.csv` and metadata file `archive-2/stock_metadata.csv`. The aggregate dataset contains daily stock-level records with open, high, low, close, VWAP, volume, turnover, trades, and deliverable volume fields. The local copy spans January 2000 to April 2021 and includes 235,192 rows across 65 historical symbols.

Important observations:

- The dataset includes long time coverage, enabling trend, volatility, and drawdown analysis.
- Some historical symbols are not present in the latest metadata; the app falls back to the symbol name and an "Unclassified" industry label.
- The `Trades` column has many missing values, so it is not used as a core prediction feature.
- Since the data is historical and not live, all outputs are decision-support signals rather than current investment recommendations.

## 3. Feature Engineering

For each symbol, the pipeline computes:

- Daily return and next-day return target
- 20, 50, and 200 day moving averages
- 20 day rolling volatility
- RSI-14
- MACD and MACD signal
- Bollinger Bands
- 20 day momentum
- Volume change
- Drawdown from cumulative return peaks

These features balance interpretability, financial relevance, and dashboard performance.

## 4. Methodology and Model Architecture

The prediction target is next-trading-day return. The model uses a chronological 80/20 train-test split, avoiding shuffled validation because financial time series must preserve temporal order. The primary model is a `RandomForestRegressor`, chosen because it handles nonlinear relationships, feature interactions, and mixed-scale technical indicators without heavy preprocessing.

Reported metrics:

- MAE
- RMSE
- R2
- Directional accuracy

The app also compares the model to a naive baseline that uses the previous day's return as the next-day prediction. If scikit-learn is unavailable, a NumPy linear regression fallback keeps the project reproducible.

## 5. Risk Assessment Methodology

Risk analytics are computed from daily returns:

- Annualized return
- Annualized volatility
- Sharpe ratio using a 5% annual risk-free rate
- Sortino ratio
- Maximum drawdown

These metrics are shown at stock level and reused in portfolio construction.

## 6. Portfolio Construction Logic

The portfolio module creates three investor profiles:

- Conservative: prioritizes lower volatility, lower drawdowns, and stronger Sharpe ratio.
- Balanced: blends annualized return, volatility, Sharpe ratio, and momentum.
- Aggressive: emphasizes return and momentum while still accounting for drawdown.

Each profile ranks stocks using a weighted score, selects the top holdings, normalizes allocation weights to 100%, and applies a single-stock concentration cap. Every holding includes a plain-English explanation based on its risk-return profile.

## 7. Explainability Techniques

The dashboard exposes:

- Feature importance for the prediction model
- Baseline vs model comparison
- Risk-return scatter plots
- Per-holding portfolio explanations
- A methodology tab describing assumptions and limitations

This makes the system easier to evaluate as an investment intelligence prototype rather than a black-box predictor.

## 8. Key Insights and Results

The smoke test validates the complete path from data loading to portfolio output. It loads 235,192 rows, computes indicators, trains a prediction model, evaluates metrics, and builds a normalized portfolio. The dashboard provides a usable interface for comparing stocks, inspecting model behavior, and generating profile-based portfolios.

Key insights:

- Risk-adjusted metrics are more useful than raw returns for comparing stocks across sectors.
- The portfolio module gives different allocations for conservative, balanced, and aggressive investor profiles, making the output more decision-oriented.
- The model is evaluated against a simple baseline so the prediction output is not presented without context.
- Explanations make recommendations easier to audit and more suitable for an investment intelligence interface.

## 9. Limitations and Future Work

Limitations:

- Dataset ends in April 2021.
- No live prices, financial APIs, news, sentiment, or macroeconomic signals are used.
- Short-horizon return prediction is noisy and should not be interpreted as financial advice.

Future improvements:

- Add walk-forward validation.
- Add sector-level portfolio constraints.
- Include more robust backtesting.
- Add downloadable reports.
- Compare Random Forest with gradient boosting and time-series models.
