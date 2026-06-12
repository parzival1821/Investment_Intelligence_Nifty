# Model Artifacts

This project trains prediction models at runtime from the organizer-provided local CSV files.

No serialized model binary is committed because:

- the dataset is intentionally kept local and uncommitted;
- the selected stock and date window can change inside the dashboard;
- runtime training keeps the model reproducible from the documented pipeline instead of relying on a stale artifact.

The model implementation lives in `nifty_intel/prediction.py`. The main model is a `RandomForestRegressor` with a chronological train/test split. If scikit-learn is unavailable, the same interface falls back to a NumPy linear regression model.
