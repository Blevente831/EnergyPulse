"""Anomaly detection with a robust rolling baseline.

Idea: an hour is unusual if it differs a lot from what that same hour looked like
on comparable recent days (judged in relative terms, since consumption is skewed). "Comparable" means the same hour of day and the same
day type (weekday or weekend), so a normal 8 pm peak is never flagged just for
being higher than 3 am.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from energypulse.config import (
    ANOMALY_LOG_SCALE,
    ANOMALY_MIN_DEVIATION_KWH,
    ANOMALY_MIN_SCALE,
    ANOMALY_THRESHOLD,
    ANOMALY_WINDOW_WEEKS,
    DB_PATH,
)

TARGET = "energy_kwh"
WEEKDAYS_PER_WEEK, WEEKEND_DAYS_PER_WEEK = 5, 2
MAD_TO_SIGMA = 1.4826  # makes MAD comparable to a standard deviation for normal data


def _mad(window: np.ndarray) -> float:
    """Median absolute deviation, ignoring missing values."""
    w = window[~np.isnan(window)]
    return float(np.median(np.abs(w - np.median(w))))


def _rolling_median_mad(series: pd.Series, window: int):
    """Rolling median and MAD of the PREVIOUS `window` observations of a series."""
    prior = series.shift(1)  # the current point must not be part of its own baseline
    min_periods = max(5, window // 2)
    roll = prior.rolling(window, min_periods=min_periods)
    return roll.median(), roll.apply(_mad, raw=True)


def detect_anomalies(
    hourly: pd.DataFrame,
    window_weeks: int = ANOMALY_WINDOW_WEEKS,
    threshold: float = ANOMALY_THRESHOLD, ## it is 3.5x more than the usual spread
    min_scale: float = ANOMALY_MIN_SCALE,  ## 15% minimum deviation to avoid sensibility for quiet periods
    log_scale: bool = ANOMALY_LOG_SCALE,   ## if True measures in procent not in kwH
    min_deviation: float = ANOMALY_MIN_DEVIATION_KWH, ## at least 0.25 KwH difference
) -> pd.DataFrame:
    """Score every hour. Returns one row per hour with expected value, z-score and flag.

    With log_scale=True the z-score measures relative change ("40% above typical"),
    which suits skewed, multiplicative data like household consumption. `min_scale`
    is then in log units (0.15 is roughly 15%); otherwise it is in kWh.
    An hour is flagged only if it is both statistically unusual (|z| > threshold)
    and at least `min_deviation` kWh away from normal.
    """
    x = hourly[TARGET].astype(float)
    if "is_imputed" in hourly:
        # Interpolated hours are not real measurements, so never judge or learn from them.
        x = x.where(~hourly["is_imputed"].astype(bool))

    values = np.log(x.clip(lower=0.01)) if log_scale else x  # limit guards against log(0), calculates the natural log of the values
    expected = pd.Series(np.nan, index=x.index)
    spread = pd.Series(np.nan, index=x.index)

    keys = pd.DataFrame({"hour": x.index.hour, "weekend": x.index.dayofweek >= 5}, index=x.index)
    for (_, weekend), idx in keys.groupby(["hour", "weekend"]).groups.items():         ## 48 separate series 
        window = window_weeks * (WEEKEND_DAYS_PER_WEEK if weekend else WEEKDAYS_PER_WEEK) ## the hours of the last 5 weeks grouped by weekdays and weekends
        ## 5*5 for weekdays and 5*2 for weekends
        med, mad = _rolling_median_mad(values.loc[idx], window) ## median of the values at a certain hour and their spread in mad
        expected.loc[idx] = med ## expected value at a certain hour
        spread.loc[idx] = mad * MAD_TO_SIGMA ## the spread at that time 

    scale = spread.clip(lower=min_scale) ## if spread is under 0.15 
    z = (values - expected) / scale 

    expected_kwh = np.exp(expected) if log_scale else expected  # back to kWh for reporting from log to exp(kwH)
    out = pd.DataFrame({TARGET: x, "expected_kwh": expected_kwh, "scale": scale, "z_score": z})
    big_enough = (x - expected_kwh).abs() >= min_deviation   ## the difference is at least 0.25 KwH
    out["is_anomaly"] = (z.abs() > threshold) & big_enough  # NaN comparisons are False, True only if z > 3.5 and is at least 0.25 kwH
    out["direction"] = np.where(out["is_anomaly"], np.where(z > 0, "spike", "dip"), "") ## spike - too high, dip - too low
    return out


def group_events(flagged: pd.DataFrame, max_gap_hours: int = 3) -> pd.DataFrame:
    """Merge flagged hours of the same direction into one event (gaps up to max_gap_hours allowed)."""
    cols = ["start", "end", "hours", "direction", "peak_abs_z", "excess_kwh"]
    if flagged.empty:
        return pd.DataFrame(columns=cols)

    f = flagged.sort_index().rename_axis("timestamp").reset_index()
    f["abs_z"] = f["z_score"].abs()
    f["deviation"] = f[TARGET] - f["expected_kwh"]

    gap = f["timestamp"].diff() > pd.Timedelta(hours=max_gap_hours)
    changed = f["direction"] != f["direction"].shift()
    f["event_id"] = (gap | changed).cumsum()

    events = f.groupby("event_id").agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        hours=("timestamp", "size"),
        direction=("direction", "first"),
        peak_abs_z=("abs_z", "max"),
        excess_kwh=("deviation", "sum"),  # total kWh above (+) or below (-) expectation
    )
    return events.reset_index(drop=True)[cols]


def sensitivity_table(hourly, windows=(3, 5, 8), thresholds=(3.0, 3.5, 4.0, 5.0)) -> pd.DataFrame:
    """Share of hours flagged for each window/threshold combination (in percent)."""
    rows = {}
    for w in windows:
        scored = detect_anomalies(hourly, window_weeks=w, threshold=np.inf)
        valid = scored["z_score"].notna()
        big = (scored[TARGET] - scored["expected_kwh"]).abs() >= ANOMALY_MIN_DEVIATION_KWH
        rows[f"{w} weeks"] = {
            f"|z| > {t}": float(((scored["z_score"].abs() > t) & big)[valid].mean() * 100)
            for t in thresholds
        }
    return pd.DataFrame(rows).T


def injection_benchmark(
    hourly: pd.DataFrame, factors=(1.5, 2.0, 3.0, 5.0), n: int = 150, seed: int = 0, **kwargs
) -> pd.DataFrame:
    """Self-test: multiply random hours by a factor (spikes) or divide (dips), count catches."""
    rng = np.random.default_rng(seed)
    base = detect_anomalies(hourly, **kwargs)
    # Only inject where a baseline exists (the first weeks cannot be judged) and the hour is normal.
    candidates = base.index[base["z_score"].notna() & ~base["is_anomaly"]]

    rows = []
    for f in factors:
        chosen = candidates[rng.choice(len(candidates), size=min(2 * n, len(candidates)), replace=False)]
        spikes, dips = chosen[: len(chosen) // 2], chosen[len(chosen) // 2 :]
        tweaked = hourly.copy()
        tweaked.loc[spikes, TARGET] = tweaked.loc[spikes, TARGET] * f
        tweaked.loc[dips, TARGET] = tweaked.loc[dips, TARGET] / f
        flags = detect_anomalies(tweaked, **kwargs)["is_anomaly"]
        rows.append(
            {
                "factor": f,
                "spike_recall_pct": float(flags.loc[spikes].mean() * 100),
                "dip_recall_pct": float(flags.loc[dips].mean() * 100),
            }
        )
    return pd.DataFrame(rows).set_index("factor")


def save_anomalies(scored: pd.DataFrame, events: pd.DataFrame, db_path: str | Path = DB_PATH) -> None:
    flagged = scored.loc[scored["is_anomaly"], [TARGET, "expected_kwh", "z_score", "direction"]]
    with sqlite3.connect(db_path) as conn:
        flagged.reset_index().to_sql("anomalies", conn, if_exists="replace", index=False)
        events.to_sql("anomaly_events", conn, if_exists="replace", index=False)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    from energypulse.forecast import load_hourly

    if not Path(DB_PATH).exists():
        raise SystemExit("Database not found. Run `python -m energypulse.pipeline` first.")

    hourly = load_hourly(DB_PATH)
    scored = detect_anomalies(hourly)
    flagged = scored[scored["is_anomaly"]]
    events = group_events(flagged)
    save_anomalies(scored, events, DB_PATH)

    judged = scored["z_score"].notna().sum()
    print(f"Judged {judged:,} hours | flagged {len(flagged):,} ({len(flagged) / judged:.2%}) "
          f"| {len(events):,} events")
    print(f"Spikes: {(flagged['direction'] == 'spike').sum():,} | dips: {(flagged['direction'] == 'dip').sum():,}")

    print("\nFlagged hours by hour of day:")
    print(flagged.groupby(flagged.index.hour).size().reindex(range(24), fill_value=0).to_string())

    print("\nFlagged hours by month:")
    print(flagged.groupby(flagged.index.to_period("M")).size().to_string())

    print("\nTop 10 events by strength:")
    top = events.nlargest(10, "peak_abs_z").round({"peak_abs_z": 2, "excess_kwh": 2})
    print(top.to_string(index=False))

    print("\nLongest events:")
    print(events.nlargest(5, "hours").round({"peak_abs_z": 2, "excess_kwh": 2}).to_string(index=False))

    print("\nSensitivity: % of hours flagged")
    print(sensitivity_table(hourly).round(2).to_string())

    print("\nSelf-test: injected spikes (x factor) and dips (/ factor) caught (%)")
    print(injection_benchmark(hourly).round(1).to_string())
    print(f"\nSaved tables 'anomalies' and 'anomaly_events' to {DB_PATH}")


if __name__ == "__main__":
    main()
