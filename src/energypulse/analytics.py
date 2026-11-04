"""Calculations behind the dashboard: KPIs, trends, daily/weekly patterns, outages.

Plain pandas, no Streamlit or Plotly, so everything here is unit-tested.

Reminder: `energy_kwh` is the energy used in one hour. Because the hour is one hour
long, it is numerically the same as the average power in kW during that hour.
Missing hours are NaN and are ignored, never treated as zero.
"""

import pandas as pd

TARGET = "energy_kwh"
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

UNIT_LABELS = {
    "Hourly": "kWh per hour",
    "Daily": "kWh per day",
    "Weekly": "kWh per week",
    "Monthly": "kWh per month",
}
_RESAMPLE_RULES = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}  # weeks start on Monday
MIN_COVERAGE = 0.75  # a day/week/month needs at least 75% of its hours to be shown


def slice_range(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows between two timestamps, inclusive."""
    return df.loc[start:end]


def kpis(view: pd.DataFrame) -> dict:
    """Headline numbers for the selected period."""
    s = view[TARGET]
    observed = int(s.notna().sum())
    result = {
        "hours": len(view),
        "observed_hours": observed,
        "completeness_pct": observed / len(view) * 100 if len(view) else 0.0,
        "imputed_hours": int(view["is_imputed"].sum()),
        "outage_hours": int(view["is_long_gap"].sum()),
    }
    if observed == 0:
        return {**result, "observed_total_kwh": 0.0, "avg_daily_kwh": None, "avg_power_kw": None,
                "peak_kwh": None, "peak_time": None, "busiest_hour": None}

    result.update(
        observed_total_kwh=float(s.sum()),
        # mean x 24 stays honest when some hours are missing, unlike a plain sum
        avg_daily_kwh=float(s.mean() * 24),
        avg_power_kw=float(s.mean()),
        peak_kwh=float(s.max()),
        peak_time=s.idxmax(),
        busiest_hour=int(s.groupby(view.index.hour).mean().idxmax()),
    )
    return result


def aggregate(series: pd.Series, freq: str) -> pd.Series:
    """Hourly, or kWh per day/week/month, adjusted for missing hours.

    Each period's value is (average hourly kWh) x (hours in a full period), so a day with
    a few missing hours is not understated. Periods with under 75% of their hours are
    left empty rather than shown misleadingly.
    """
    if freq == "Hourly":
        return series.rename("value").rename_axis("timestamp")

    groups = series.resample(_RESAMPLE_RULES[freq], label="left", closed="left")
    mean, present, total = groups.mean(), groups.count(), groups.size()

    if freq == "Daily":
        full_hours = 24
    elif freq == "Weekly":
        full_hours = 168
    else:
        full_hours = mean.index.days_in_month * 24

    out = mean * full_hours
    out[(present / total) < MIN_COVERAGE] = float("nan")
    return out.rename("value").rename_axis("timestamp")


def hourly_profile(view: pd.DataFrame) -> pd.DataFrame:
    """Average kWh for each hour of the day, weekdays versus weekends."""
    s = view[TARGET]
    prof = s.groupby([view.index.hour, view.index.dayofweek >= 5]).mean().unstack()
    prof = prof.reindex(index=range(24), columns=[False, True])
    prof.columns = ["Weekday", "Weekend"]
    prof.index.name = "hour"
    return prof


def weekday_profile(view: pd.DataFrame) -> pd.DataFrame:
    """Average kWh per day for each day of the week."""
    daily = view[TARGET].groupby(view.index.dayofweek).mean().reindex(range(7)) * 24
    return pd.DataFrame({"weekday": WEEKDAY_NAMES, "avg_daily_kwh": daily.to_numpy()})


def heatmap_table(view: pd.DataFrame) -> pd.DataFrame:
    """Average kWh for every weekday x hour combination (7 rows, 24 columns)."""
    table = view[TARGET].groupby([view.index.dayofweek, view.index.hour]).mean().unstack()
    table = table.reindex(index=range(7), columns=range(24))
    table.index = WEEKDAY_NAMES
    table.columns.name = "hour"
    return table


def weekend_effect(view: pd.DataFrame):
    """How much higher (+) or lower (-) weekend consumption is than weekday, in percent."""
    s = view[TARGET]
    weekend = s[view.index.dayofweek >= 5].mean()
    weekday = s[view.index.dayofweek < 5].mean()
    if pd.isna(weekend) or pd.isna(weekday) or weekday == 0:
        return None
    return float((weekend / weekday - 1) * 100)


def outages(view: pd.DataFrame) -> pd.DataFrame:
    """Runs of consecutive hours inside long data outages."""
    flag = view["is_long_gap"].astype(bool)
    run_id = (flag != flag.shift()).cumsum()
    frame = pd.DataFrame({"flag": flag.to_numpy(), "run": run_id.to_numpy(), "ts": view.index})
    out = (
        frame[frame["flag"]]
        .groupby("run")
        .agg(start=("ts", "min"), end=("ts", "max"), hours=("ts", "size"))
        .reset_index(drop=True)
    )
    out["days"] = (out["hours"] / 24).round(1)
    return out


def completeness_by_month(view: pd.DataFrame) -> pd.DataFrame:
    """Share of hours with real data, per calendar month."""
    share = view[TARGET].notna().groupby(view.index.to_period("M")).mean() * 100
    return pd.DataFrame({"month": share.index.astype(str), "completeness_pct": share.to_numpy()})
