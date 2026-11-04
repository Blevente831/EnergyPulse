"""Response shapes for the API. FastAPI validates every response against these
and uses them to generate the interactive documentation at /docs."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Health(BaseModel):
    status: str
    database_found: bool
    model_found: bool
    data_start: datetime | None = None
    data_end: datetime | None = None
    hours: int | None = None


class Reading(BaseModel):
    timestamp: datetime
    energy_kwh: float | None = None  # None where data is missing
    reactive_kvar: float | None = None
    voltage: float | None = None
    intensity: float | None = None
    sub1_wh: float | None = None
    sub2_wh: float | None = None
    sub3_wh: float | None = None
    minutes_observed: int | None = None
    is_imputed: bool
    is_long_gap: bool


class ReadingsPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[Reading]


class ForecastPoint(BaseModel):
    timestamp: datetime
    forecast_kwh: float
    actual_kwh: float | None = None  # filled only when forecasting from a past point


class ForecastResponse(BaseModel):
    as_of: datetime  # the last hour of real data the forecast was based on
    horizon_hours: int
    items: list[ForecastPoint]


class AnomalyPoint(BaseModel):
    timestamp: datetime
    energy_kwh: float
    expected_kwh: float
    z_score: float
    direction: Literal["spike", "dip"]


class AnomaliesPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AnomalyPoint]


class AnomalyEvent(BaseModel):
    start: datetime
    end: datetime
    hours: int
    direction: Literal["spike", "dip"]
    peak_abs_z: float
    excess_kwh: float


class EventsPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AnomalyEvent]


class ErrorMetrics(BaseModel):
    mae: float
    rmse: float
    mape_pct: float
    smape_pct: float


class SplitInfo(BaseModel):
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int


class MetricsResponse(BaseModel):
    split: SplitInfo
    metrics: dict[str, ErrorMetrics]
    best_baseline: str
    mae_improvement_vs_best_baseline_pct: float
    feature_importance: dict[str, float]
