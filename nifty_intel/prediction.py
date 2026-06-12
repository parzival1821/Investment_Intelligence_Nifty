from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "Daily Return",
    "Return5",
    "Return20",
    "MA20 Distance",
    "MA50 Distance",
    "Volatility20",
    "RSI14",
    "MACD",
    "MACD Signal",
    "Momentum20",
    "Volume Change",
    "Drawdown",
]


@dataclass
class PredictionResult:
    model_name: str
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


class NumpyLinearRegressor:
    """Small fallback model used when scikit-learn is unavailable."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLinearRegressor":
        design = np.column_stack([np.ones(len(x)), x])
        self.coef_, *_ = np.linalg.lstsq(design, y, rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        design = np.column_stack([np.ones(len(x)), x])
        return design @ self.coef_


def build_model_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Create a clean supervised learning frame from indicator features."""
    df = features.sort_values("Date").copy()
    df["Return5"] = df["Close"].pct_change(5)
    df["Return20"] = df["Close"].pct_change(20)
    df["MA20 Distance"] = (df["Close"] / df["MA20"]) - 1
    df["MA50 Distance"] = (df["Close"] / df["MA50"]) - 1
    df["Target Next Return"] = df["Next Return"]

    model_frame = df[["Date", "Close", "Target Next Return", *FEATURE_COLUMNS]].replace(
        [np.inf, -np.inf], np.nan
    )
    return model_frame.dropna().reset_index(drop=True)


def chronological_split(
    model_frame: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(model_frame) < 80:
        raise ValueError("At least 80 usable rows are required for prediction")
    split_idx = int(len(model_frame) * train_fraction)
    split_idx = min(max(split_idx, 1), len(model_frame) - 1)
    return model_frame.iloc[:split_idx].copy(), model_frame.iloc[split_idx:].copy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if denominator == 0 else float(1 - (np.sum(errors**2) / denominator))
    directional_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
    }


def _train_model(x_train: np.ndarray, y_train: np.ndarray):
    try:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return "Random Forest Regressor", model
    except Exception:
        model = NumpyLinearRegressor().fit(x_train, y_train)
        return "NumPy Linear Regression Fallback", model


def train_predictor(features: pd.DataFrame) -> PredictionResult:
    model_frame = build_model_frame(features)
    train, test = chronological_split(model_frame)

    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["Target Next Return"].to_numpy(dtype=float)
    x_test = test[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test["Target Next Return"].to_numpy(dtype=float)

    model_name, model = _train_model(x_train, y_train)
    y_pred = model.predict(x_test)
    baseline_pred = test["Daily Return"].to_numpy(dtype=float)

    prediction_rows = test[["Date", "Close", "Target Next Return"]].copy()
    prediction_rows["Predicted Next Return"] = y_pred
    prediction_rows["Baseline Prediction"] = baseline_pred
    prediction_rows["Actual Direction"] = np.where(
        prediction_rows["Target Next Return"] >= 0, "Up", "Down"
    )
    prediction_rows["Predicted Direction"] = np.where(
        prediction_rows["Predicted Next Return"] >= 0, "Up", "Down"
    )

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif getattr(model, "coef_", None) is not None:
        coef = np.abs(model.coef_[1:])
        values = coef / coef.sum() if coef.sum() else coef
    else:
        values = np.zeros(len(FEATURE_COLUMNS))

    importance = (
        pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return PredictionResult(
        model_name=model_name,
        metrics=_metrics(y_test, y_pred),
        baseline_metrics=_metrics(y_test, baseline_pred),
        predictions=prediction_rows,
        feature_importance=importance,
    )
