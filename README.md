EnergyPulse — Energy Analytics & Forecasting

EnergyPulse is a Python-based energy analytics platform that combines data engineering, machine learning, and interactive visualization to explore electricity consumption, forecast future usage, and identify unusual measurements.

Features

Energy analytics: Hourly and daily consumption trends, peak usage, and weekday comparisons.

Forecasting: Next-24-hour electricity consumption forecasts using a Random Forest model and baseline comparisons.

Anomaly detection: Identify unusual measurements using a rolling statistical baseline.

Data pipeline: Data validation, missing-value handling, hourly aggregation, and SQLite storage.

Interactive dashboard: Streamlit charts, date filters, KPI summaries, and CSV exports.

REST API: FastAPI endpoints for readings, forecasts, anomalies, and model metrics.

Testing: Automated tests for data processing, forecasting, database operations, and API behavior.

Technology Stack

Python

Pandas and NumPy

Scikit-learn

SQLite and SQLAlchemy

Streamlit and Plotly

FastAPI and Pydantic

Pytest

Git and GitHub Actions

Dataset

This project uses the Individual Household Electric Power Consumption dataset from the UCI Machine Learning Repository.

Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The dataset contains minute-level household electricity measurements collected between December 2006 and November 2010. Raw readings are validated and aggregated into hourly observations before analysis.

Architecture

Ingest raw electricity measurements.

Validate timestamps and numeric values.

Aggregate measurements into hourly readings.

Store processed data in SQLite.

Generate analytics and train forecasting models.

Save forecasts, anomaly results, and evaluation metrics.

Serve insights through Streamlit and predictions through FastAPI.

Machine Learning

The forecasting pipeline compares a persistence baseline, a seasonal baseline, and a Random Forest Regressor.

Features include calendar variables, historical lags, and rolling statistics. Models are evaluated using chronological data splits to reduce the risk of future-data leakage.

Evaluation metrics include Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

Anomaly detection uses a rolling statistical baseline to identify unusual consumption patterns.

<img width="1886" height="830" alt="overview" src="https://github.com/user-attachments/assets/34ce4074-2a1a-4fa5-9b1b-fa40affad29d" />

<img width="1877" height="840" alt="Patterns" src="https://github.com/user-attachments/assets/1032d0fa-4944-4520-bf89-b09328db3a7e" />

<img width="1912" height="842" alt="Forecast1" src="https://github.com/user-attachments/assets/fed17baf-f5f8-4bba-83f4-396dc87c29b3" />

<img width="1902" height="850" alt="Forecast2" src="https://github.com/user-attachments/assets/c8068511-1c8e-4deb-84a4-4b0e2dedec24" />

<img width="1916" height="872" alt="Forecast3" src="https://github.com/user-attachments/assets/5b9aacea-345a-414b-bc10-7ddec095db52" />

<img width="1917" height="862" alt="Anomalies1" src="https://github.com/user-attachments/assets/57a4de05-50d5-41b3-8a10-066a5b0005bd" />

<img width="1847" height="650" alt="Dataquality3" src="https://github.com/user-attachments/assets/79062c65-7563-4374-aea9-cfc0b219747b" />

<img width="1907" height="732" alt="Dataquality1" src="https://github.com/user-attachments/assets/6dd6b866-bbcc-4574-97c8-2d8ce6260fcf" />

<img width="1895" height="827" alt="Anomalies2" src="https://github.com/user-attachments/assets/661e0777-f493-4dcc-9fa9-fb1438c3b3b0" />

<img width="1847" height="650" alt="Dataquality3" src="https://github.com/user-attachments/assets/a4e67c79-f70a-48f8-aca1-258a7a36dee9" />





