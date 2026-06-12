from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nifty_intel.data import (
    add_company_metadata,
    available_symbols,
    filter_symbol,
    latest_market_snapshot,
    load_metadata,
    load_price_data,
)
from nifty_intel.indicators import add_indicators_by_symbol, add_technical_indicators
from nifty_intel.portfolio import PROFILE_CONFIG, build_portfolio, profile_description
from nifty_intel.prediction import train_predictor
from nifty_intel.risk import risk_summary, summarize_symbol_risk


st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon=":material/monitoring:",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            border: 1px solid #d9dee8;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #fbfcfe;
        }
        div[data-testid="stMetricLabel"] {font-size: 0.82rem;}
        h1, h2, h3 {letter-spacing: 0;}
        .caption-box {
            border-left: 4px solid #2a6f97;
            padding: 0.6rem 0.8rem;
            background: #f5f8fb;
            border-radius: 4px;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_base_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_price_data()
    metadata = load_metadata()
    return prices, metadata


@st.cache_data(show_spinner="Calculating technical indicators...")
def load_feature_data() -> pd.DataFrame:
    prices, _ = load_base_data()
    return add_indicators_by_symbol(prices)


@st.cache_data(show_spinner="Training prediction model...")
def cached_prediction(symbol: str) -> object:
    prices, _ = load_base_data()
    stock = add_technical_indicators(filter_symbol(prices, symbol))
    return train_predictor(stock)


def pct(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.{decimals}%}"


def num(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def metric_row(metrics: dict[str, float]) -> None:
    cols = st.columns(5)
    cols[0].metric("Annual Return", pct(metrics["annualized_return"]))
    cols[1].metric("Volatility", pct(metrics["annualized_volatility"]))
    cols[2].metric("Sharpe", num(metrics["sharpe_ratio"]))
    cols[3].metric("Sortino", num(metrics["sortino_ratio"]))
    cols[4].metric("Max Drawdown", pct(metrics["max_drawdown"]))


def price_chart(stock: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock["Date"], y=stock["Close"], name="Close", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=stock["Date"], y=stock["MA20"], name="MA20", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=stock["Date"], y=stock["MA50"], name="MA50", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=stock["Date"], y=stock["MA200"], name="MA200", line=dict(width=1)))
    fig.update_layout(
        title=f"{symbol} price trend",
        height=390,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="Close price",
    )
    return fig


def indicator_chart(stock: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock["Date"], y=stock["RSI14"], name="RSI14", line=dict(color="#2a6f97")))
    fig.add_hline(y=70, line_dash="dot", line_color="#b23a48")
    fig.add_hline(y=30, line_dash="dot", line_color="#2d6a4f")
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="RSI")
    return fig


def macd_chart(stock: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock["Date"], y=stock["MACD"], name="MACD", line=dict(color="#3a506b")))
    fig.add_trace(
        go.Scatter(
            x=stock["Date"],
            y=stock["MACD Signal"],
            name="Signal",
            line=dict(color="#e09f3e"),
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="MACD")
    return fig


prices, metadata = load_base_data()
features = load_feature_data()
symbols = available_symbols(prices)
metadata_map = metadata.set_index("Symbol")["Company"].to_dict()

st.title("NIFTY-50 Investment Intelligence")
st.caption("Historical market analytics, risk-aware prediction, and explainable portfolio construction.")

with st.sidebar:
    st.header("Controls")
    default_symbol = "RELIANCE" if "RELIANCE" in symbols else symbols[0]
    selected_symbol = st.selectbox(
        "Stock",
        symbols,
        index=symbols.index(default_symbol),
        format_func=lambda symbol: f"{symbol} - {metadata_map.get(symbol, symbol)}",
    )
    profile = st.selectbox("Investor profile", list(PROFILE_CONFIG), index=1)
    min_date = prices["Date"].min().date()
    max_date = prices["Date"].max().date()
    start_date, end_date = st.date_input(
        "Analysis window",
        value=(pd.Timestamp("2016-01-01").date(), max_date),
        min_value=min_date,
        max_value=max_date,
    )
    top_n = st.slider("Portfolio holdings", 5, 12, 8)
    st.divider()
    st.caption("Data source: organizer-provided Kaggle NIFTY-50 files. No live market data.")

stock_all = features.loc[features["Symbol"] == selected_symbol].copy()
window_mask = (stock_all["Date"].dt.date >= start_date) & (stock_all["Date"].dt.date <= end_date)
stock = stock_all.loc[window_mask].copy()
if stock.empty:
    st.error("No rows are available for the selected stock and date range.")
    st.stop()

latest = stock.iloc[-1]
stock_meta = add_company_metadata(stock.tail(1), metadata).iloc[0]
period_return = (stock["Close"].iloc[-1] / stock["Close"].iloc[0]) - 1
stock_metrics = risk_summary(stock["Daily Return"])

overview_tab, stock_tab, predictor_tab, portfolio_tab, methodology_tab = st.tabs(
    ["Market Overview", "Stock Analytics", "Predictor", "Portfolio Builder", "Methodology"]
)

with overview_tab:
    st.subheader("Market Overview")
    market_snapshot = latest_market_snapshot(prices, metadata)
    risk_table = summarize_symbol_risk(features).merge(metadata, on="Symbol", how="left")
    risk_table["Company"] = risk_table["Company"].fillna(risk_table["Symbol"])
    risk_table["Industry"] = risk_table["Industry"].fillna("Unclassified")
    risk_table["Positive Sharpe"] = risk_table["sharpe_ratio"].clip(lower=0.01)

    cols = st.columns(4)
    cols[0].metric("Symbols", f"{prices['Symbol'].nunique()}")
    cols[1].metric("Rows", f"{len(prices):,}")
    cols[2].metric("Date Range", f"{prices['Date'].min().year}-{prices['Date'].max().year}")
    cols[3].metric("Latest Close Median", num(market_snapshot["Close"].median()))

    left, right = st.columns([1.2, 1])
    with left:
        scatter = px.scatter(
            risk_table,
            x="annualized_volatility",
            y="annualized_return",
            color="Industry",
            size="Positive Sharpe",
            hover_name="Company",
            hover_data={"Symbol": True, "sharpe_ratio": ":.2f", "max_drawdown": ":.1%"},
            labels={
                "annualized_volatility": "Annualized volatility",
                "annualized_return": "Annualized return",
            },
            title="Risk-return map",
        )
        scatter.update_layout(height=430, margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(scatter, width="stretch")
    with right:
        industry_counts = metadata["Industry"].value_counts().reset_index()
        industry_counts.columns = ["Industry", "Stocks"]
        bar = px.bar(
            industry_counts,
            x="Stocks",
            y="Industry",
            orientation="h",
            title="Metadata coverage by industry",
            color="Stocks",
            color_continuous_scale="Teal",
        )
        bar.update_layout(height=430, margin=dict(l=20, r=20, t=45, b=20), showlegend=False)
        st.plotly_chart(bar, width="stretch")

    st.dataframe(
        risk_table.sort_values("sharpe_ratio", ascending=False)
        .head(12)[
            [
                "Symbol",
                "Company",
                "Industry",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "max_drawdown",
            ]
        ],
        width="stretch",
        column_config={
            "annualized_return": st.column_config.NumberColumn("Annual Return", format="%.2f"),
            "annualized_volatility": st.column_config.NumberColumn("Volatility", format="%.2f"),
            "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn("Max Drawdown", format="%.2f"),
        },
        hide_index=True,
    )

with stock_tab:
    st.subheader(f"{selected_symbol} - {stock_meta['Company']}")
    st.markdown(
        f"<div class='caption-box'>{stock_meta['Industry']} - latest available close "
        f"{latest['Close']:,.2f} on {latest['Date'].date()}</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    cols[0].metric("Latest Close", num(latest["Close"]))
    cols[1].metric("Window Return", pct(period_return))
    cols[2].metric("RSI14", num(latest["RSI14"]))
    cols[3].metric("20D Momentum", pct(latest["Momentum20"]))
    cols[4].metric("20D Volatility", pct(latest["Volatility20"] * (252**0.5)))
    metric_row(stock_metrics)

    st.plotly_chart(price_chart(stock, selected_symbol), width="stretch")
    volume = px.bar(stock, x="Date", y="Volume", title="Volume")
    volume.update_layout(height=240, margin=dict(l=20, r=20, t=45, b=20))
    st.plotly_chart(volume, width="stretch")

    left, right = st.columns(2)
    left.plotly_chart(indicator_chart(stock), width="stretch")
    right.plotly_chart(macd_chart(stock), width="stretch")

with predictor_tab:
    st.subheader("Next-Day Return Predictor")
    prediction = cached_prediction(selected_symbol)
    pred_metrics = prediction.metrics
    baseline_metrics = prediction.baseline_metrics

    cols = st.columns(4)
    cols[0].metric("Model", prediction.model_name)
    cols[1].metric("RMSE", f"{pred_metrics['rmse']:.4f}", delta=f"baseline {baseline_metrics['rmse']:.4f}")
    cols[2].metric("MAE", f"{pred_metrics['mae']:.4f}", delta=f"baseline {baseline_metrics['mae']:.4f}")
    cols[3].metric("Directional Accuracy", pct(pred_metrics["directional_accuracy"]))

    predictions = prediction.predictions.tail(180)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Target Next Return"],
            name="Actual next return",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Predicted Next Return"],
            name="Predicted next return",
        )
    )
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=35, b=20), yaxis_tickformat=".1%")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([0.95, 1.05])
    with left:
        importance = prediction.feature_importance.head(10)
        imp_fig = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top feature importance",
        )
        imp_fig.update_layout(height=350, margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(imp_fig, width="stretch")
    with right:
        st.dataframe(
            prediction.predictions.tail(12).sort_values("Date", ascending=False),
            width="stretch",
            hide_index=True,
            column_config={
                "Target Next Return": st.column_config.NumberColumn("Actual Return", format="%.4f"),
                "Predicted Next Return": st.column_config.NumberColumn("Predicted Return", format="%.4f"),
                "Baseline Prediction": st.column_config.NumberColumn("Baseline", format="%.4f"),
            },
        )

with portfolio_tab:
    st.subheader(f"{profile} Portfolio")
    st.markdown(
        f"<div class='caption-box'>{profile_description(profile)}</div>",
        unsafe_allow_html=True,
    )
    portfolio = build_portfolio(features, metadata, profile=profile, top_n=top_n)

    cols = st.columns(4)
    cols[0].metric("Holdings", len(portfolio))
    cols[1].metric("Largest Weight", pct(portfolio["weight"].max()))
    cols[2].metric(
        "Weighted Return",
        pct(float((portfolio["weight"] * portfolio["annualized_return"]).sum())),
    )
    cols[3].metric(
        "Weighted Volatility",
        pct(float((portfolio["weight"] * portfolio["annualized_volatility"]).sum())),
    )

    weights = px.bar(
        portfolio.sort_values("weight"),
        x="weight",
        y="Symbol",
        color="Industry",
        orientation="h",
        hover_name="Company",
        title="Portfolio allocation",
        labels={"weight": "Weight"},
    )
    weights.update_layout(height=390, margin=dict(l=20, r=20, t=45, b=20), xaxis_tickformat=".0%")
    st.plotly_chart(weights, width="stretch")

    portfolio_display = portfolio.copy()
    portfolio_display["weight"] = portfolio_display["weight"] * 100
    st.dataframe(
        portfolio_display,
        width="stretch",
        hide_index=True,
        column_config={
            "weight": st.column_config.ProgressColumn("Weight", format="%.1f%%", min_value=0, max_value=100),
            "annualized_return": st.column_config.NumberColumn("Annual Return", format="%.2f"),
            "annualized_volatility": st.column_config.NumberColumn("Volatility", format="%.2f"),
            "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "sortino_ratio": st.column_config.NumberColumn("Sortino", format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn("Max Drawdown", format="%.2f"),
            "momentum_20d": st.column_config.NumberColumn("20D Momentum", format="%.2f"),
            "score": st.column_config.NumberColumn("Score", format="%.2f"),
        },
    )

with methodology_tab:
    st.subheader("Methodology")
    st.markdown(
        """
        **Data pipeline.** The app loads the organizer-provided NIFTY-50 aggregate price file and company metadata, validates the required columns, parses dates, and enriches symbols with company and industry labels.

        **Feature engineering.** Each stock receives daily returns, moving averages, RSI, MACD, Bollinger Bands, rolling volatility, momentum, volume change, and drawdown features.

        **Prediction.** The model predicts next-trading-day return using a chronological 80/20 split. Random Forest is used when scikit-learn is available; a NumPy linear regression fallback keeps the interface reproducible.

        **Evaluation.** The predictor reports MAE, RMSE, R2, and directional accuracy, and compares against a previous-day-return baseline.

        **Portfolio logic.** Conservative, balanced, and aggressive profiles use different weighted scores across return, volatility, Sharpe ratio, drawdown, and momentum. Weights are normalized and capped to reduce concentration.

        **Limitations.** The dataset ends in April 2021 and this prototype does not use live prices, news, sentiment, or proprietary data. Outputs are educational decision-support signals, not investment advice.
        """
    )
