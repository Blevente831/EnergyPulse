"""Project paths and settings. Everything is resolved relative to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Data
RAW_PATH = ROOT / "data" / "raw" / "household_power_consumption.txt"
DB_PATH = ROOT / "data" / "processed" / "energypulse.db"
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "household_power_consumption.txt"
PROCESSED_DIR = DATA_DIR / "processed"
HOURLY_CSV = PROCESSED_DIR / "hourly.csv"
# Model artifacts
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

# Forecasting
HORIZON_HOURS = 24  # how far ahead we forecast
TEST_DAYS = 365  # the final year is held out for testing

# Anomaly detection
ANOMALY_WINDOW_WEEKS = 5  # how much recent history forms the baseline
ANOMALY_THRESHOLD = 3.5  # flag when |robust z-score| exceeds this
ANOMALY_LOG_SCALE = True  # judge relative (%) changes instead of absolute kWh changes
ANOMALY_MIN_SCALE = 0.15  # floor on the spread (15% in log space), so tiny deviations are never flagged
ANOMALY_MIN_DEVIATION_KWH = 0.25  # an alert must also differ from normal by at least this much
 
# API
API_DEFAULT_LIMIT = 168  # one week of hourly rows
API_MAX_LIMIT = 5000
FORECAST_HISTORY_HOURS = 336  # two weeks of history is plenty for the 168-hour lag features
 