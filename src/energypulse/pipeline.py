"""Load, validate, clean and aggregate the UCI household power dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from energypulse.config import DB_PATH, HOURLY_CSV, RAW_PATH

NUMERIC_COLS = [
    "Global_active_power",  # kW, minute average
    "Global_reactive_power",  # kVAR
    "Voltage",  # V
    "Global_intensity",  # A
    "Sub_metering_1",  # Wh (kitchen)
    "Sub_metering_2",  # Wh (laundry)
    "Sub_metering_3",  # Wh (water heater, A/C)
]

# Plausible physical ranges. Values outside become NaN.
VALID_RANGES = {
    "Global_active_power": (0, 12),
    "Global_reactive_power": (0, 5),
    "Voltage": (200, 260),
    "Global_intensity": (0, 60),
    "Sub_metering_1": (0, 100),
    "Sub_metering_2": (0, 100),
    "Sub_metering_3": (0, 100),
}

MIN_MINUTES_PER_HOUR = 45  # require 75% of minutes to trust an hourly value
MAX_INTERP_GAP_HOURS = 3  # fill gaps up to this length, flag the rest


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw file: semicolon separated, '?' marks missing values."""
    df = pd.read_csv(path, sep=";", na_values=["?", ""], low_memory=False)
    ts = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df = df.drop(columns=["Date", "Time"])
    df.index = pd.DatetimeIndex(ts, name="timestamp")
    return df


def clean_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, drop bad timestamps/duplicates, null out impossible values."""
    df = df[df.index.notna()].copy()  # unparseable timestamps/bad date formats are removed and copies the new one without errors
    df = df[~df.index.duplicated(keep="first")].sort_index() ## duplicated minutes are kept only the first appearance

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce") ## convert to numeric if any column contains not numeric values
        low, high = VALID_RANGES[col]
        df[col] = df[col].where(df[col].between(low, high))
    return df


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute data to hourly, requiring enough valid minutes."""
    grouped = df.resample("h") ## remake to hourly blocks 
    hourly = pd.DataFrame(
        {
            # Mean kW over one hour equals energy in kWh for that hour.
            "energy_kwh": grouped["Global_active_power"].mean(),
            "reactive_kvar": grouped["Global_reactive_power"].mean(),
            "voltage": grouped["Voltage"].mean(),
            "intensity": grouped["Global_intensity"].mean(),
            "sub1_wh": grouped["Sub_metering_1"].sum(min_count=1), ## forcing to make those missing for one hour to be NaN by setting min count to 1
            "sub2_wh": grouped["Sub_metering_2"].sum(min_count=1),
            "sub3_wh": grouped["Sub_metering_3"].sum(min_count=1),
            "minutes_observed": grouped["Global_active_power"].count(), ## counting the minutes over an hour  
        }
    )
    too_sparse = hourly["minutes_observed"] < MIN_MINUTES_PER_HOUR ## if there is not enough minutes in an hour at least 45 minutes 
    value_cols = hourly.columns.drop("minutes_observed")
    hourly.loc[too_sparse, value_cols] = np.nan  ## too sparse hours(< 45 minutes) are set to NaN
    return hourly.asfreq("h")  # make every hour explicit, even fully missing


def fill_gaps(hourly: pd.DataFrame, max_gap: int = MAX_INTERP_GAP_HOURS) -> pd.DataFrame:
    """Interpolate short gaps only; flag everything filled or left missing.

    Note: pandas' interpolate(limit=n) would partially fill the start of a long
    gap, so we measure each gap's full length and only fill the short ones.
    """
    out = hourly.copy()
    was_missing = out["energy_kwh"].isna()

    run_id = (~was_missing).cumsum()  # constant within each run of NaNs, not missing add up and missing habe the same id
    gap_len = was_missing.groupby(run_id).transform("sum") ## each row gets the sum of the gap it is part of by run_id
    fillable = was_missing & (gap_len <= max_gap) ## is fillable if it is missing and gap is less than 3 hours

    value_cols = out.columns.drop("minutes_observed") ## new dataframe without the minutes_observed column
    interpolated = out[value_cols].interpolate(method="time", limit_area="inside") ## interpolate/fill based on previous and next data points, only inside the given data 
    out.loc[fillable, value_cols] = interpolated.loc[fillable] ## locate the interpolated values into the out table without minute_obs 

    out["is_imputed"] = fillable & out["energy_kwh"].notna() ## column which shows the imputed/interpolated values, good for anomaly because it is not real data just imputed
    out["is_long_gap"] = out["energy_kwh"].isna()  # exclude from training, more than 3 hours gap
    out["minutes_observed"] = out["minutes_observed"].fillna(0).astype(int)
    return out

def run_pipeline(raw_path: str | Path) -> pd.DataFrame:
    return fill_gaps(to_hourly(clean_minutes(load_raw(raw_path))))
 
 
def save_to_sqlite(hourly: pd.DataFrame, db_path: str | Path = DB_PATH) -> None:
    import sqlite3
 
    db_path = Path(db_path)            ## creating the .db file 
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        hourly.reset_index().to_sql("hourly_readings", conn, if_exists="replace", index=False)
 
 
def main() -> None:
    """Run the full pipeline: raw file -> clean hourly data -> SQLite + CSV."""
    if not RAW_PATH.exists():
        raise SystemExit(f"Raw file not found: {RAW_PATH}\nDownload it from the UCI page first.")
 
    hourly = run_pipeline(RAW_PATH)
    save_to_sqlite(hourly)
    hourly.to_csv(HOURLY_CSV)     ## creating the .csv file
 
    usable = ~hourly["is_long_gap"]
    print(f"Hours: {len(hourly):,} | usable: {usable.sum():,} | "
          f"imputed: {hourly['is_imputed'].sum()} | long gaps: {hourly['is_long_gap'].sum()}")
    print(f"Saved: {DB_PATH}\n       {HOURLY_CSV}")
 
 
if __name__ == "__main__":
    main()