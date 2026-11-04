"""Feature engineering for the next-24-hour forecast.

Core rule: to predict the value at time T, we may only use data from
T - 24h or earlier. That way one model can forecast every hour of the next day
from data we already have at forecast time, with no recursion and no leakage.
"""

import pandas as pd

from energypulse.config import HORIZON_HOURS

TARGET = "energy_kwh"

# Calendar features describe the time we are predicting, so they are always known.
CALENDAR_FEATURES = ["hour", "dayofweek", "month", "is_weekend"]

# History features: every one of them looks back at least HORIZON_HOURS.
HISTORY_FEATURES = [
    "lag_24",  # same hour yesterday
    "lag_48",  # same hour two days ago
    "lag_168",  # same hour last week
    "roll_mean_24",  # average level of the day ending at lag_24
    "roll_mean_168",  # average level of the week ending at lag_24
]

FEATURE_COLUMNS = HISTORY_FEATURES + CALENDAR_FEATURES

# The two baselines are simply two of the features.
BASELINES = {"naive_yesterday": "lag_24", "naive_last_week": "lag_168"}


def make_features(series: pd.Series) -> pd.DataFrame:
    """Build the feature table from an hourly series (index = every hour, no holes).

    Missing hours must be NaN in the series, not absent, so that shift(24)
    means exactly 24 hours and not "24 rows".
    """
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.freq is None:
        raise ValueError("series needs a DatetimeIndex with a fixed hourly frequency")

    h = HORIZON_HOURS
    # shifted[T] = series[T - 24h]. Every history feature is built from this.
    shifted = series.shift(h)

    df = pd.DataFrame({TARGET: series}, index=series.index)
    df["lag_24"] = shifted
    df["lag_48"] = series.shift(2 * h)
    df["lag_168"] = series.shift(168)
    # Rolling windows tolerate some missing hours (need at least half present).
    df["roll_mean_24"] = shifted.rolling(24, min_periods=12).mean()
    df["roll_mean_168"] = shifted.rolling(168, min_periods=84).mean()

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df