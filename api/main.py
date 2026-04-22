"""api/main.py — FastAPI REST endpoint for taxi demand prediction."""
import os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).parent.parent / "models"

app = FastAPI(
    title="NYC Taxi Demand Forecasting API",
    description="Predicts next-hour taxi demand per zone using XGBoost + LSTM ensemble",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = scaler = feature_names = None


@app.on_event("startup")
def load_model():
    global model, scaler, feature_names
    if (MODEL_DIR / "xgb_model.pkl").exists():
        model         = joblib.load(MODEL_DIR / "xgb_model.pkl")
        scaler        = joblib.load(MODEL_DIR / "scaler.pkl")
        feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
        print("Model loaded.")
    else:
        print("WARNING: Model not found. Run models/train.py first.")


class DemandInput(BaseModel):
    zone: int             = Field(..., example=42, description="NYC taxi zone ID (1-63)")
    hour: int             = Field(..., example=8,  description="Hour of day (0-23)")
    dayofweek: int        = Field(..., example=1,  description="Day of week (0=Mon)")
    month: int            = Field(..., example=3,  description="Month (1-12)")
    demand_lag_1h: float  = Field(..., example=45.0)
    demand_lag_24h: float = Field(..., example=38.0)
    demand_lag_168h: float= Field(..., example=41.0)
    avg_fare: float       = Field(..., example=18.5)
    avg_distance: float   = Field(..., example=2.8)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/metrics")
def get_metrics():
    p = MODEL_DIR / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "Metrics not found. Train model first.")
    return json.load(open(p))


@app.post("/predict")
def predict(inp: DemandInput):
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    row = {f: 0.0 for f in feature_names}
    row.update({
        "zone": inp.zone,
        "hour": inp.hour,
        "dayofweek": inp.dayofweek,
        "month": inp.month,
        "dayofmonth": 15,
        "is_weekend": int(inp.dayofweek >= 5),
        "is_rush_hour": int(inp.hour in [7,8,9,16,17,18,19]),
        "is_night": int(inp.hour in [22,23,0,1,2,3]),
        "demand_lag_1h": inp.demand_lag_1h,
        "demand_lag_2h": inp.demand_lag_1h * 0.95,
        "demand_lag_3h": inp.demand_lag_1h * 0.9,
        "demand_lag_6h": inp.demand_lag_1h * 0.85,
        "demand_lag_12h": inp.demand_lag_1h * 0.8,
        "demand_lag_24h": inp.demand_lag_24h,
        "demand_lag_48h": inp.demand_lag_24h * 0.98,
        "demand_lag_168h": inp.demand_lag_168h,
        "demand_rolling_mean_3h": inp.demand_lag_1h,
        "demand_rolling_mean_6h": inp.demand_lag_1h * 0.95,
        "demand_rolling_mean_12h": inp.demand_lag_1h * 0.9,
        "demand_rolling_mean_24h": inp.demand_lag_24h,
        "demand_rolling_std_3h": inp.demand_lag_1h * 0.1,
        "demand_rolling_std_6h": inp.demand_lag_1h * 0.12,
        "zone_avg_demand": inp.demand_lag_168h,
        "zone_demand_ratio": inp.demand_lag_1h / (inp.demand_lag_168h + 1e-6),
        "is_high_demand_zone": int(inp.zone in [4,12,42,47,49,1,7,32,40,44]),
        "avg_fare": inp.avg_fare,
        "avg_distance": inp.avg_distance,
        "avg_duration": inp.avg_distance * 4,
        "total_passengers": inp.demand_lag_1h * 1.3,
    })

    X    = pd.DataFrame([row])[feature_names]
    X_sc = pd.DataFrame(scaler.transform(X), columns=feature_names)
    pred = float(max(0, model.predict(X_sc)[0]))

    trend = "INCREASING" if pred > inp.demand_lag_1h * 1.05 else             "DECREASING" if pred < inp.demand_lag_1h * 0.95 else "STABLE"

    return {
        "zone": inp.zone,
        "predicted_demand": round(pred, 1),
        "current_demand": inp.demand_lag_1h,
        "demand_change_pct": round((pred - inp.demand_lag_1h) / (inp.demand_lag_1h + 1e-6) * 100, 1),
        "trend": trend,
        "is_rush_hour": bool(inp.hour in [7,8,9,16,17,18,19]),
        "is_high_demand_zone": bool(inp.zone in [4,12,42,47,49,1,7,32,40,44]),
    }


@app.post("/predict/batch")
def predict_batch(inputs: List[DemandInput]):
    return [predict(inp) for inp in inputs]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
