"""Read-only queries over the SQLite database. Plain Python, no web framework.

Every value that comes from a caller goes through a `?` placeholder, never into
the SQL text, so a malicious query parameter cannot change the query.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

TS_FORMAT = "%Y-%m-%d %H:%M:%S"  # how timestamps are stored (sortable text)

READING_COLUMNS = [
    "timestamp", "energy_kwh", "reactive_kvar", "voltage", "intensity",
    "sub1_wh", "sub2_wh", "sub3_wh", "minutes_observed", "is_imputed", "is_long_gap",
]


class DatabaseMissing(Exception):
    """The database file does not exist yet."""


class TableMissing(Exception):
    """A table has not been created yet (for example, anomalies were never computed)."""


@contextmanager
def connect(db_path: str | Path):
    """Open the database read-only, so the API can never modify the data."""
    path = Path(db_path)
    if not path.exists():
        raise DatabaseMissing(str(path))
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _fmt(dt: datetime) -> str:
    return dt.strftime(TS_FORMAT)


def _where(conditions: list[str]) -> str:
    return f"WHERE {' AND '.join(conditions)}" if conditions else ""


def _query(conn, sql: str, params: list):
    try:
        return conn.execute(sql, params)
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            raise TableMissing(str(exc)) from exc
        raise


def _page(conn, table: str, columns: str, conditions, params, order_by, limit, offset):
    """Return (total matching rows, one page of rows as dicts)."""
    where = _where(conditions)
    total = _query(conn, f"SELECT COUNT(*) FROM {table} {where}", params).fetchone()[0]
    rows = _query(
        conn,
        f"SELECT {columns} FROM {table} {where} ORDER BY {order_by} LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    return total, [dict(r) for r in rows]


# ---------------------------------------------------------------- readings
def fetch_readings(db_path, start=None, end=None, limit=168, offset=0, usable_only=False):
    conditions, params = [], []
    if start:
        conditions.append("timestamp >= ?"); params.append(_fmt(start))
    if end:
        conditions.append("timestamp <= ?"); params.append(_fmt(end))
    if usable_only:
        conditions.append("is_long_gap = 0")
    with connect(db_path) as conn:
        return _page(conn, "hourly_readings", ", ".join(READING_COLUMNS),
                     conditions, params, "timestamp", limit, offset)


def data_range(db_path) -> dict:
    with connect(db_path) as conn:
        row = _query(conn, "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM hourly_readings", []).fetchone()
    return {"start": row[0], "end": row[1], "hours": row[2]}


def fetch_history(db_path, end=None, hours=336) -> pd.DataFrame:
    """The most recent `hours` rows up to `end`, as a DataFrame with a regular hourly index."""
    conditions, params = [], []
    if end:
        conditions.append("timestamp <= ?"); params.append(_fmt(end))
    sql = f"SELECT * FROM hourly_readings {_where(conditions)} ORDER BY timestamp DESC LIMIT ?"
    with connect(db_path) as conn:
        _query(conn, "SELECT 1 FROM hourly_readings LIMIT 1", [])  # clear error if table is missing
        df = pd.read_sql_query(sql, conn, params=params + [hours], parse_dates=["timestamp"])
    if df.empty:
        return df
    return df.set_index("timestamp").sort_index().asfreq("h")


# --------------------------------------------------------------- anomalies
def fetch_anomalies(db_path, start=None, end=None, direction=None, limit=168, offset=0):
    conditions, params = [], []
    if start:
        conditions.append("timestamp >= ?"); params.append(_fmt(start))
    if end:
        conditions.append("timestamp <= ?"); params.append(_fmt(end))
    if direction:
        conditions.append("direction = ?"); params.append(direction)
    with connect(db_path) as conn:
        return _page(conn, "anomalies", "timestamp, energy_kwh, expected_kwh, z_score, direction",
                     conditions, params, "timestamp", limit, offset)


def fetch_events(db_path, start=None, end=None, direction=None, limit=168, offset=0):
    """Events that overlap the requested window."""
    conditions, params = [], []
    if start:
        conditions.append('"end" >= ?'); params.append(_fmt(start))
    if end:
        conditions.append('"start" <= ?'); params.append(_fmt(end))
    if direction:
        conditions.append("direction = ?"); params.append(direction)
    with connect(db_path) as conn:
        return _page(conn, "anomaly_events", '"start", "end", hours, direction, peak_abs_z, excess_kwh',
                     conditions, params, '"start"', limit, offset)


# ------------------------------------------------- whole tables (dashboard)
def read_hourly(db_path) -> pd.DataFrame:
    """All hourly rows as a DataFrame with a regular hourly index and boolean flags."""
    with connect(db_path) as conn:
        _query(conn, "SELECT 1 FROM hourly_readings LIMIT 1", [])  # clear error if table is missing
        df = pd.read_sql_query("SELECT * FROM hourly_readings", conn, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index().asfreq("h")
    df[["is_imputed", "is_long_gap"]] = df[["is_imputed", "is_long_gap"]].fillna(0).astype(bool)
    return df


def read_anomalies(db_path) -> pd.DataFrame:
    """Flagged hours indexed by timestamp. Empty (not an error) if anomalies were never computed."""
    cols = ["energy_kwh", "expected_kwh", "z_score", "direction"]
    try:
        with connect(db_path) as conn:
            _query(conn, "SELECT 1 FROM anomalies LIMIT 1", [])
            df = pd.read_sql_query("SELECT * FROM anomalies", conn, parse_dates=["timestamp"])
    except TableMissing:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="timestamp"))
    return df.set_index("timestamp").sort_index()


def read_events(db_path) -> pd.DataFrame:
    """Anomaly events. Empty (not an error) if anomalies were never computed."""
    cols = ["start", "end", "hours", "direction", "peak_abs_z", "excess_kwh"]
    try:
        with connect(db_path) as conn:
            _query(conn, "SELECT 1 FROM anomaly_events LIMIT 1", [])
            return pd.read_sql_query("SELECT * FROM anomaly_events ORDER BY \"start\"", conn, parse_dates=["start", "end"])
    except TableMissing:
        return pd.DataFrame(columns=cols)