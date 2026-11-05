"""EnergyPulse dashboard.

Run from the project root:  python -m streamlit run dashboard/app.py
"""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from energypulse import analytics, db
from energypulse.config import (
    ANOMALY_MIN_DEVIATION_KWH,
    ANOMALY_THRESHOLD,
    ANOMALY_WINDOW_WEEKS,
    DB_PATH,
    METRICS_PATH,
    MODEL_PATH,
)
from energypulse.forecast import backtest_origin, load_model, score_forecast

TARGET = analytics.TARGET

st.set_page_config(page_title="EnergyPulse", page_icon="⚡", layout="wide")


# ------------------------------------------------------------------ helpers
def show_chart(fig) -> None:
    """Plotly charts fill the page width by default in every Streamlit version."""
    st.plotly_chart(fig)


def show_table(df: pd.DataFrame) -> None:
    st.dataframe(df, hide_index=True)


def csv_button(container, label: str, df: pd.DataFrame, filename: str, index: bool = False) -> None:
    container.download_button(label, df.to_csv(index=index), file_name=filename, mime="text/csv")


@st.cache_data(show_spinner="Loading data...")
def load_hourly() -> pd.DataFrame:
    return db.read_hourly(DB_PATH)


@st.cache_data
def load_anomalies() -> pd.DataFrame:
    return db.read_anomalies(DB_PATH)


@st.cache_data
def load_events() -> pd.DataFrame:
    return db.read_events(DB_PATH)


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH) if MODEL_PATH.exists() else None


def load_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


# --------------------------------------------------------------------- data
if not DB_PATH.exists():
    st.error("Database not found. Run `python -m energypulse.pipeline` first.")
    st.stop()

hourly = load_hourly()
data_start, data_end = hourly.index.min(), hourly.index.max()
last_day = data_end.normalize()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚡ EnergyPulse")
preset = st.sidebar.selectbox(
    "Period", ["Last 30 days", "Last 90 days", "Last 12 months", "All data", "Custom"], index=2
)
days_back = {"Last 30 days": 30, "Last 90 days": 90, "Last 12 months": 365}

if preset in days_back:
    start = max(data_start, last_day - pd.Timedelta(days=days_back[preset] - 1))
    end = data_end
elif preset == "All data":
    start, end = data_start, data_end
else:
    default_start = max(data_start, last_day - pd.Timedelta(days=89))
    picked = st.sidebar.date_input(
        "Date range",
        value=(default_start.date(), last_day.date()),
        min_value=data_start.date(),
        max_value=last_day.date(),
    )
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start = pd.Timestamp(picked[0])
        end = pd.Timestamp(picked[1]) + pd.Timedelta(hours=23)
    else:
        st.sidebar.info("Pick an end date to continue.")
        st.stop()

granularity = st.sidebar.radio("Trend resolution", list(analytics.UNIT_LABELS), index=1)

if st.sidebar.button("Reload data from database"):
    st.cache_data.clear()

view = analytics.slice_range(hourly, start, end)

st.title("EnergyPulse: household electricity")
st.caption(f"{start:%d %b %Y} to {end:%d %b %Y}  |  {len(view):,} hours selected")

if view[TARGET].notna().sum() == 0:
    st.warning("There is no measured data in this period. Pick a different range.")
    st.stop()

tab_overview, tab_patterns, tab_forecast, tab_anomalies, tab_quality = st.tabs(
    ["Overview", "Patterns", "Forecast", "Anomalies", "Data quality"]
)

# ----------------------------------------------------------------- overview
with tab_overview:
    k = analytics.kpis(view)
    effect = analytics.weekend_effect(view)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average per day", f"{k['avg_daily_kwh']:.1f} kWh")
    c2.metric("Average power", f"{k['avg_power_kw']:.2f} kW")
    c3.metric("Peak hour", f"{k['peak_kwh']:.2f} kWh", delta=f"{k['peak_time']:%d %b %Y, %H:%M}", delta_color="off")
    c4.metric("Data completeness", f"{k['completeness_pct']:.1f}%")

    c5, c6, c7, _ = st.columns(4)
    c5.metric("Measured total", f"{k['observed_total_kwh']:,.0f} kWh", help="Sum of the hours that were actually measured.")
    c6.metric("Typical busiest hour", f"{k['busiest_hour']:02d}:00")
    c7.metric("Weekend vs weekday", "n/a" if effect is None else f"{effect:+.0f}%")

    trend = analytics.aggregate(view[TARGET], granularity)
    unit = analytics.UNIT_LABELS[granularity]
    fig = px.line(trend.reset_index(), x="timestamp", y="value",
                  labels={"timestamp": "", "value": unit}, title=f"Consumption ({unit})")
    show_chart(fig)
    if granularity != "Hourly":
        st.caption("Totals are adjusted for missing hours; periods with under 75% of their data are left blank.")

    stamp = f"{start:%Y%m%d}_{end:%Y%m%d}"
    d1, d2 = st.columns(2)
    csv_button(d1, "Download hourly data (CSV)", view.reset_index(), f"energypulse_hourly_{stamp}.csv")
    csv_button(d2, f"Download {granularity.lower()} trend (CSV)",
               trend.rename(unit).reset_index(), f"energypulse_{granularity.lower()}_{stamp}.csv")

# ----------------------------------------------------------------- patterns
with tab_patterns:
    st.subheader("Average day, weekdays versus weekends")
    prof = analytics.hourly_profile(view).reset_index().melt(id_vars="hour", var_name="day type", value_name="kWh")
    show_chart(px.line(prof, x="hour", y="kWh", color="day type", markers=True,
                       labels={"hour": "Hour of day", "kWh": "Average kWh per hour"}))

    left, right = st.columns(2)
    with left:
        st.subheader("Average by weekday")
        show_chart(px.bar(analytics.weekday_profile(view), x="weekday", y="avg_daily_kwh",
                          labels={"weekday": "", "avg_daily_kwh": "Average kWh per day"}))
    with right:
        st.subheader("Weekday x hour")
        show_chart(px.imshow(analytics.heatmap_table(view), aspect="auto", color_continuous_scale="YlOrRd",
                             labels={"x": "Hour of day", "y": "", "color": "kWh"}))

# ----------------------------------------------------------------- forecast
with tab_forecast:
    model, metrics = get_model(), load_metrics()
    if model is None:
        st.info("No trained model yet. Run `python -m energypulse.forecast` first.")
    else:
        st.subheader("The next 24 hours")
        origin = hourly[TARGET].last_valid_index()
        try:
            latest = backtest_origin(model, hourly, origin)
        except ValueError as exc:
            st.warning(f"Could not forecast from the latest data: {exc}")
        else:
            recent = hourly[TARGET].loc[origin - pd.Timedelta(hours=47): origin]
            both = pd.concat([
                recent.rename("kWh").to_frame().assign(series="Actual"),
                latest["forecast_kwh"].rename("kWh").to_frame().assign(series="Forecast"),
            ]).rename_axis("timestamp").reset_index()
            show_chart(px.line(both, x="timestamp", y="kWh", color="series",
                               labels={"kWh": "kWh per hour", "timestamp": ""},
                               title=f"Forecast from {origin:%d %b %Y, %H:%M}"))
            csv_button(st, "Download forecast (CSV)", latest.reset_index(), "energypulse_forecast.csv")

        st.divider()
        st.subheader("Back-test a day")
        st.caption("The model forecasts a whole day using only data from before midnight. "
                   "Compare with what really happened, and with simply repeating yesterday.")
        min_day = (data_start + pd.Timedelta(days=14)).date()
        default_day = max(min_day, (last_day - pd.Timedelta(days=30)).date())
        day = st.date_input("Day to forecast", value=default_day, min_value=min_day, max_value=last_day.date())

        if metrics and pd.Timestamp(day) < pd.Timestamp(metrics["split"]["test_start"]).normalize():
            st.warning("This day is inside the model's training period, so it has already seen it and the "
                       f"result is flattering. Pick a day after {metrics['split']['test_start'][:10]} "
                       "for an honest test.")

        try:
            bt = backtest_origin(model, hourly, pd.Timestamp(day) - pd.Timedelta(hours=1))
        except ValueError as exc:
            st.warning(f"Cannot forecast this day: {exc}")
        else:
            scores = score_forecast(bt)
            if scores["hours"]:
                m1, m2 = st.columns(2)
                m1.metric("Model error (MAE)", f"{scores['mae_model']:.2f} kWh/h")
                m2.metric("Repeat-yesterday error (MAE)", f"{scores['mae_yesterday']:.2f} kWh/h",
                          delta=f"{scores['mae_model'] - scores['mae_yesterday']:+.2f} for the model",
                          delta_color="inverse")
            names = {"actual_kwh": "Actual", "forecast_kwh": "Random Forest", "naive_yesterday": "Yesterday (baseline)"}
            long = bt.rename(columns=names).reset_index().melt(
                id_vars="timestamp", value_vars=list(names.values()), var_name="series", value_name="kWh")
            show_chart(px.line(long, x="timestamp", y="kWh", color="series",
                               labels={"kWh": "kWh per hour", "timestamp": ""}))

        if metrics:
            st.divider()
            st.subheader("Model accuracy on the held-out year")
            table = pd.DataFrame(metrics["metrics"]).T.reset_index().rename(columns={"index": "model"})
            show_table(table.round(3))
            split = metrics["split"]
            st.caption(
                f"Test period {split['test_start'][:10]} to {split['test_end'][:10]} ({split['n_test']:,} hours). "
                f"Random Forest vs best baseline ({metrics['best_baseline']}): "
                f"{metrics['mae_improvement_vs_best_baseline_pct']:+.1f}% MAE. "
                "MAPE is inflated by quiet night hours; MAE and RMSE are the fairer guides."
            )
            imp = (pd.Series(metrics["feature_importance"]).sort_values()
                   .rename("importance").rename_axis("feature").reset_index())
            show_chart(px.bar(imp, x="importance", y="feature", orientation="h", title="Feature importance"))

# ---------------------------------------------------------------- anomalies
with tab_anomalies:
    anomalies, events = load_anomalies(), load_events()
    if anomalies.empty and events.empty:
        st.info("No anomalies yet. Run `python -m energypulse.anomaly` first.")
    else:
        kinds = st.multiselect("Show", ["spike", "dip"], default=["spike", "dip"])
        hits = anomalies.loc[start:end]
        hits = hits[hits["direction"].isin(kinds)]
        evs = events[(events["end"] >= start) & (events["start"] <= end) & events["direction"].isin(kinds)]

        a1, a2, a3 = st.columns(3)
        a1.metric("Flagged hours", f"{len(hits):,}")
        a2.metric("Events", f"{len(evs):,}")
        a3.metric("Share of hours flagged", f"{len(hits) / max(view[TARGET].notna().sum(), 1):.2%}")

        fig = px.line(view.reset_index(), x="timestamp", y=TARGET,
                      labels={"timestamp": "", TARGET: "kWh per hour"}, title="Consumption with flagged hours")
        if not hits.empty:
            dots = px.scatter(hits.reset_index(), x="timestamp", y="energy_kwh", color="direction",
                              color_discrete_map={"spike": "crimson", "dip": "royalblue"})
            fig.add_traces(dots.data)
        show_chart(fig)

        st.subheader("Events, strongest first")
        if evs.empty:
            st.write("No events in this period.")
        else:
            table = evs.sort_values("peak_abs_z", ascending=False).round({"peak_abs_z": 1, "excess_kwh": 1})
            show_table(table)
            e1, e2 = st.columns(2)
            csv_button(e1, "Download events (CSV)", table, "energypulse_anomaly_events.csv")
            csv_button(e2, "Download flagged hours (CSV)", hits.reset_index(), "energypulse_anomalies.csv")

        with st.expander("How are anomalies decided?"):
            st.markdown(
                f"Each hour is compared with the same hour on the previous **{ANOMALY_WINDOW_WEEKS} weeks** "
                "of the same day type (weekday or weekend), so a normal evening peak is never flagged. "
                "The comparison uses relative change on a log scale with the median and median absolute "
                "deviation, which resist being distorted by past anomalies. An hour is flagged when its "
                f"robust z-score exceeds **{ANOMALY_THRESHOLD}** and it also differs by at least "
                f"**{ANOMALY_MIN_DEVIATION_KWH} kWh**. Consecutive flagged hours are merged into one event."
            )

# ------------------------------------------------------------ data quality
with tab_quality:
    q1, q2, q3 = st.columns(3)
    q1.metric("Hours with real data", f"{k['observed_hours']:,} of {k['hours']:,}")
    q2.metric("Interpolated hours", f"{k['imputed_hours']:,}", help="Short gaps filled by interpolation.")
    q3.metric("Hours in long outages", f"{k['outage_hours']:,}", help="Left empty, never filled in.")

    st.subheader("Long outages in this period")
    gaps = analytics.outages(view)
    if gaps.empty:
        st.write("None.")
    else:
        show_table(gaps)

    show_chart(px.bar(analytics.completeness_by_month(view), x="month", y="completeness_pct",
                      labels={"month": "", "completeness_pct": "% of hours with data"},
                      title="Data completeness by month"))
