"""EnergyPulse REST API.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request

from energypulse import db
from energypulse.config import (
    API_DEFAULT_LIMIT,
    API_MAX_LIMIT,
    DB_PATH,
    FORECAST_HISTORY_HOURS,
    METRICS_PATH,
    MODEL_PATH,
)
from energypulse.features import TARGET
from energypulse.forecast import forecast_next_24h, load_model
from energypulse.schemas import (
    AnomaliesPage,
    EventsPage,
    ForecastResponse,
    Health,
    MetricsResponse,
    ReadingsPage,
)

Limit = Annotated[int, Query(ge=1, le=API_MAX_LIMIT, description="Rows per page")]
Offset = Annotated[int, Query(ge=0, description="Rows to skip")]


# ------------------------------------------------------------------ helpers
def _check_range(start: datetime | None, end: datetime | None) -> None:
    if start and end and start > end:
        raise HTTPException(status_code=422, detail="'start' must not be later than 'end'.")


def _run(request: Request, func, *args, **kwargs):
    """Call a db function and turn 'database/table missing' into a clear 503."""
    try:
        return func(request.app.state.db_path, *args, **kwargs)
    except db.DatabaseMissing:
        raise HTTPException(503, "Database not found. Run `python -m energypulse.pipeline` first.")
    except db.TableMissing:
        raise HTTPException(503, "Data not available yet. Run the pipeline (and `python -m energypulse.anomaly` for anomalies).")


def _get_model(request: Request):
    state = request.app.state
    if state.model is None:
        if not Path(state.model_path).exists():
            raise HTTPException(503, "Model not trained yet. Run `python -m energypulse.forecast` first.")
        state.model = load_model(state.model_path)  # loaded once, then reused
    return state.model


# ------------------------------------------------------------------ the app
def create_app(db_path=DB_PATH, model_path=MODEL_PATH, metrics_path=METRICS_PATH) -> FastAPI:
    """App factory: tests can pass temporary paths instead of the real data."""
    app = FastAPI(
        title="EnergyPulse API",
        description="Household electricity readings, 24-hour forecasts, anomalies and model metrics.",
        version="0.1.0",
    )
    app.state.db_path = db_path
    app.state.model_path = model_path
    app.state.metrics_path = metrics_path
    app.state.model = None

    @app.get("/health", response_model=Health, tags=["meta"])
    def health(request: Request):
        """Is the service up, and what does it have available?"""
        state = request.app.state
        info = {
            "status": "ok",
            "database_found": Path(state.db_path).exists(),
            "model_found": Path(state.model_path).exists(),
        }
        if info["database_found"]:
            try:
                rng = db.data_range(state.db_path)
                info.update(data_start=rng["start"], data_end=rng["end"], hours=rng["hours"])
            except db.TableMissing:
                pass
        return Health(**info)

    @app.get("/readings", response_model=ReadingsPage, tags=["data"])
    def readings(
        request: Request,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: Limit = API_DEFAULT_LIMIT,
        offset: Offset = 0,
        usable_only: Annotated[bool, Query(description="Skip hours inside long data outages")] = False,
    ):
        """Hourly consumption, oldest first, paginated. Dates look like 2010-03-01T00:00:00."""
        _check_range(start, end)
        total, items = _run(request, db.fetch_readings, start, end, limit, offset, usable_only)
        return ReadingsPage(total=total, limit=limit, offset=offset, items=items)

    @app.get("/forecast", response_model=ForecastResponse, tags=["forecast"])
    def forecast(
        request: Request,
        as_of: Annotated[
            datetime | None,
            Query(description="Forecast from this past moment instead of from now. Real values are added for comparison."),
        ] = None,
    ):
        """Forecast the 24 hours after the last known hour (or after `as_of`)."""
        model = _get_model(request)
        history = _run(request, db.fetch_history, as_of, FORECAST_HISTORY_HOURS)
        if history.empty or history[TARGET].last_valid_index() is None:
            raise HTTPException(404, "No readings at or before the requested time.")
        try:
            fc = forecast_next_24h(model, history)
        except ValueError as exc:  # e.g. an outage right before the forecast origin
            raise HTTPException(422, str(exc))

        # When forecasting from the past, the real values are known: include them.
        _, actual_rows = _run(request, db.fetch_readings, fc.index[0].to_pydatetime(),
                              fc.index[-1].to_pydatetime(), len(fc), 0, False)
        actual = {pd.Timestamp(r["timestamp"]): r["energy_kwh"] for r in actual_rows}

        items = [
            {"timestamp": ts, "forecast_kwh": float(row.forecast_kwh), "actual_kwh": actual.get(ts)}
            for ts, row in fc.iterrows()
        ]
        return ForecastResponse(
            as_of=history[TARGET].last_valid_index().to_pydatetime(),
            horizon_hours=len(fc),
            items=items,
        )

    @app.get("/anomalies", response_model=AnomaliesPage, tags=["anomalies"])
    def anomalies(
        request: Request,
        start: datetime | None = None,
        end: datetime | None = None,
        direction: Literal["spike", "dip"] | None = None,
        limit: Limit = API_DEFAULT_LIMIT,
        offset: Offset = 0,
    ):
        """Individual hours flagged as unusual."""
        _check_range(start, end)
        total, items = _run(request, db.fetch_anomalies, start, end, direction, limit, offset)
        return AnomaliesPage(total=total, limit=limit, offset=offset, items=items)

    @app.get("/anomalies/events", response_model=EventsPage, tags=["anomalies"])
    def anomaly_events(
        request: Request,
        start: datetime | None = None,
        end: datetime | None = None,
        direction: Literal["spike", "dip"] | None = None,
        limit: Limit = API_DEFAULT_LIMIT,
        offset: Offset = 0,
    ):
        """Flagged hours merged into episodes (a three-day absence is one event, not 72 rows)."""
        _check_range(start, end)
        total, items = _run(request, db.fetch_events, start, end, direction, limit, offset)
        return EventsPage(total=total, limit=limit, offset=offset, items=items)

    @app.get("/metrics", response_model=MetricsResponse, tags=["forecast"])
    def metrics(request: Request):
        """How accurate the forecaster was on the held-out test year, against the baselines."""
        path = Path(request.app.state.metrics_path)
        if not path.exists():
            raise HTTPException(503, "No metrics yet. Run `python -m energypulse.forecast` first.")
        return json.loads(path.read_text())

    return app


app = create_app()