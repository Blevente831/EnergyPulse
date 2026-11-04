"""Baselines, Random Forest training, evaluation and next-24-hour forecasting."""

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from energypulse.config import DB_PATH, HORIZON_HOURS, METRICS_PATH, MODEL_PATH, TEST_DAYS, FORECAST_HISTORY_HOURS
from energypulse.features import BASELINES, FEATURE_COLUMNS, TARGET, make_features


# --------------------------------------------------------------------------- data
def load_hourly(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Read the hourly table written by the pipeline."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM hourly_readings", conn, parse_dates=["timestamp"])
    # asfreq gives the index a fixed hourly frequency, which make_features requires.
    return df.set_index("timestamp").sort_index().asfreq("h")


def chronological_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    """Train on the past, test on the final `test_days` days. Never shuffle time series."""
    cutoff = df.index.max() - pd.Timedelta(days=test_days)
    return df[df.index <= cutoff], df[df.index > cutoff]


# ------------------------------------------------------------------------ metrics
def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        # MAPE blows up near zero, so guard the denominator.
        "mape_pct": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-3)) * 100),
        "smape_pct": float(
            np.mean(2 * np.abs(err) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-3)) * 100
        ),
    }


# ------------------------------------------------------------------------- model
def build_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,  # each leaf must average >= 5 hours, which damps noise
        n_jobs=-1,
        random_state=42,  # makes results reproducible
    )


def train_and_evaluate(hourly: pd.DataFrame, test_days: int = TEST_DAYS):
    """Build features, split by time, fit the forest, score it against the baselines."""
    table = make_features(hourly[TARGET]) ## makes the feature inputs for the model
    # Rows with a missing target or feature are dropped. This is where the
    # long outages are excluded from both training and testing.
    table = table.dropna(subset=[TARGET] + FEATURE_COLUMNS)
    train, test = chronological_split(table, test_days) 

    model = build_model() ## builds the model
    model.fit(train[FEATURE_COLUMNS], train[TARGET]) ## fits the data on the model to learn

    results = {
        name: evaluate(test[TARGET], test[col]) for name, col in BASELINES.items() ## evaluates the baselines and calculates the errors
    }
    results["random_forest"] = evaluate(test[TARGET], model.predict(test[FEATURE_COLUMNS])) ## predicts AI values for the test period then compares them with the actual values


    best_baseline = min(BASELINES, key=lambda n: results[n]["mae"]) ## best mae between naive yesterday and naive last week
    gain = 1 - results["random_forest"]["mae"] / results[best_baseline]["mae"] ## how much the model was better(worked with less error) than the naive baselines for prediction



    report = {
        "split": {
            "train_start": str(train.index.min()),
            "train_end": str(train.index.max()),
            "test_start": str(test.index.min()),
            "test_end": str(test.index.max()),
            "n_train": len(train),
            "n_test": len(test),
        },
        "metrics": results,  ## contains the errors for both naive and predicted
        "best_baseline": best_baseline, 
        "mae_improvement_vs_best_baseline_pct": round(gain * 100, 2), ## improvement in procent
        "feature_importance": dict(
            sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()),
                key=lambda kv: -kv[1],
            )
        ),
    }
    return model, report


def forecast_next_24h(model, hourly: pd.DataFrame) -> pd.DataFrame:
    """Forecast the 24 hours after the last observed hour."""
    series = hourly[TARGET]
    origin = series.last_valid_index()  # the last hour we actually know
    series = series.loc[:origin] ## cuts off at the last observed hour

    future = pd.date_range(origin + pd.Timedelta(hours=1), periods=HORIZON_HOURS, freq="h") 
    full_index = pd.date_range(series.index[0], future[-1], freq="h") ## adds to the series the next 24 hours 
    extended = series.reindex(full_index)  # future hours are NaN, which is fine:
    # every feature looks back >= 24h, so none of them needs the unknown future.

    X = make_features(extended).loc[future, FEATURE_COLUMNS]
    if X.isna().any().any():
        raise ValueError("Not enough recent history to build features (data gap near the end).")
    return pd.DataFrame({"forecast_kwh": model.predict(X)}, index=future).rename_axis("timestamp")

 
def backtest_origin(model, hourly: pd.DataFrame, origin) -> pd.DataFrame:
    """Forecast the 24 hours after `origin` using only data up to `origin`.
 
    Returns the forecast next to what actually happened (NaN if that is still in the
    future) and the "same hour yesterday" baseline, so the three can be compared.
    """
    history = hourly.loc[:origin].tail(FORECAST_HISTORY_HOURS)
    if history.empty or history[TARGET].last_valid_index() is None:
        raise ValueError("No data at or before the requested time.")
    fc = forecast_next_24h(model, history.asfreq("h"))
 
    series = hourly[TARGET]
    actual = series.reindex(fc.index).to_numpy()
    yesterday = series.reindex(fc.index - pd.Timedelta(hours=HORIZON_HOURS)).to_numpy()
    return fc.assign(actual_kwh=actual, naive_yesterday=yesterday)
 
 
def score_forecast(result: pd.DataFrame) -> dict:
    """MAE of the model and of the yesterday baseline over hours where both can be judged."""
    valid = result.dropna(subset=["actual_kwh", "naive_yesterday"])
    if valid.empty:
        return {"hours": 0, "mae_model": None, "mae_yesterday": None}
    return {
        "hours": len(valid),
        "mae_model": float((valid["actual_kwh"] - valid["forecast_kwh"]).abs().mean()),
        "mae_yesterday": float((valid["actual_kwh"] - valid["naive_yesterday"]).abs().mean()),
    }
def save_artifacts(model, report: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(report, indent=2))


def load_model(path: str | Path = MODEL_PATH):
    return joblib.load(path)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    if not Path(DB_PATH).exists():
        raise SystemExit("Database not found. Run `python -m energypulse.pipeline` first.")

    hourly = load_hourly()
    model, report = train_and_evaluate(hourly)
    save_artifacts(model, report)

    s = report["split"]
    print(f"Train: {s['train_start']} -> {s['train_end']}  ({s['n_train']:,} hours)")
    print(f"Test:  {s['test_start']} -> {s['test_end']}  ({s['n_test']:,} hours)\n")
    print(f"{'model':<18}{'MAE':>8}{'RMSE':>8}{'MAPE%':>8}{'sMAPE%':>9}")
    for name, m in report["metrics"].items():
        print(f"{name:<18}{m['mae']:>8.3f}{m['rmse']:>8.3f}{m['mape_pct']:>8.1f}{m['smape_pct']:>9.1f}")
    print(f"\nRandom Forest vs best baseline ({report['best_baseline']}): "
          f"{report['mae_improvement_vs_best_baseline_pct']:+.1f}% MAE")
    print("\nFeature importance:")
    for name, imp in report["feature_importance"].items():
        print(f"  {name:<15}{imp:.3f}")

    print("\nNext 24 hours:")
    print(forecast_next_24h(model, hourly).round(3).to_string())
    print(f"\nSaved: {MODEL_PATH}\n       {METRICS_PATH}")


if __name__ == "__main__":
    main()