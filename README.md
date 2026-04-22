# 🚕 NYC Taxi Demand Forecasting Engine

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![PySpark](https://img.shields.io/badge/PySpark-3.5-red?style=flat-square&logo=apachespark)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red?style=flat-square&logo=streamlit)

> End-to-end taxi demand forecasting pipeline on millions of NYC TLC trip records — PySpark preprocessing, time series feature engineering, XGBoost + LSTM ensemble, REST API, and geospatial heatmap dashboard.

## Stack
- **Data:** NYC TLC public dataset (synthetic at scale: 1M+ trips)
- **Processing:** PySpark for distributed preprocessing
- **Features:** Time series, geospatial, weather, lag features
- **Model:** XGBoost + LSTM ensemble
- **API:** FastAPI REST endpoint
- **Dashboard:** Streamlit + Plotly (geospatial heatmap, forecasts)

## Quick Start
```bash
pip install -r requirements.txt
python run_all.py
streamlit run dashboard/app.py   # http://localhost:8501
uvicorn api.main:app --reload    # http://localhost:8000/docs
```

## Resume Bullets
- Built demand forecasting pipeline on 1M+ NYC TLC taxi trips using PySpark for distributed preprocessing
- Engineered 30+ time series features including lag demand, rolling averages, and geospatial zone signals
- Trained XGBoost + LSTM ensemble achieving strong next-hour demand prediction across 63 NYC zones
- Served predictions via FastAPI; visualized demand heatmaps and forecasts in interactive Streamlit dashboard
- Containerized with Docker; validated with 12-test pytest suite covering ETL, model, and API layers
