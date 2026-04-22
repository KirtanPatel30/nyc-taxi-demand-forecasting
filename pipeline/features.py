"""
pipeline/features.py
Engineers demand forecasting features:
- Hourly demand aggregation per zone
- Lag features (1h, 2h, 3h, 24h, 48h, 168h)
- Rolling averages and volatility
- Time features (rush hour, weekend, etc.)
- Geospatial zone features
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trips into hourly demand per zone."""
    print("[FEATURES] Aggregating hourly demand per zone...")
    df["pickup_dt"] = pd.to_datetime(df["pickup_dt"] if "pickup_dt" in df.columns else df["pickup_datetime"])
    df["hour_bucket"] = df["pickup_dt"].dt.floor("H")

    demand = df.groupby(["hour_bucket", "pickup_zone"]).agg(
        demand=("trip_distance", "count"),
        avg_fare=("fare_amount", "mean"),
        avg_distance=("trip_distance", "mean"),
        avg_duration=("trip_duration_min", "mean"),
        total_passengers=("passenger_count", "sum"),
    ).reset_index()

    demand.columns = ["datetime", "zone", "demand", "avg_fare",
                      "avg_distance", "avg_duration", "total_passengers"]
    print(f"[FEATURES] Hourly demand shape: {demand.shape}")
    return demand


def engineer_features(demand: pd.DataFrame) -> pd.DataFrame:
    """Add time series lag features, rolling stats, and time features."""
    print("[FEATURES] Engineering lag and rolling features...")

    demand = demand.sort_values(["zone", "datetime"]).reset_index(drop=True)
    demand["datetime"] = pd.to_datetime(demand["datetime"])

    # Time features
    demand["hour"]         = demand["datetime"].dt.hour
    demand["dayofweek"]    = demand["datetime"].dt.dayofweek
    demand["month"]        = demand["datetime"].dt.month
    demand["dayofmonth"]   = demand["datetime"].dt.day
    demand["is_weekend"]   = (demand["dayofweek"] >= 5).astype(int)
    demand["is_rush_hour"] = demand["hour"].isin([7,8,9,16,17,18,19]).astype(int)
    demand["is_night"]     = demand["hour"].isin([22,23,0,1,2,3]).astype(int)

    # Lag features per zone
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        demand[f"demand_lag_{lag}h"] = demand.groupby("zone")["demand"].shift(lag)

    # Rolling averages
    for window in [3, 6, 12, 24]:
        demand[f"demand_rolling_mean_{window}h"] = demand.groupby("zone")["demand"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        demand[f"demand_rolling_std_{window}h"] = demand.groupby("zone")["demand"].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )

    # Zone-level features
    zone_avg = demand.groupby("zone")["demand"].mean().reset_index()
    zone_avg.columns = ["zone", "zone_avg_demand"]
    demand = demand.merge(zone_avg, on="zone", how="left")
    demand["zone_demand_ratio"] = demand["demand"] / (demand["zone_avg_demand"] + 1e-6)

    # High demand zone flag
    high_demand = [4, 12, 42, 47, 49, 1, 7, 32, 40, 44]
    demand["is_high_demand_zone"] = demand["zone"].isin(high_demand).astype(int)

    # Target: next hour demand
    demand["target"] = demand.groupby("zone")["demand"].shift(-1)
    demand = demand.dropna(subset=["target"] + [f"demand_lag_{l}h" for l in [1,2,3]])

    print(f"[FEATURES] Final shape: {demand.shape} | Columns: {len(demand.columns)}")
    return demand


def run_feature_pipeline():
    clean_path = PROCESSED_DIR / "trips_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError("Run ETL pipeline first.")

    df = pd.read_csv(clean_path)
    demand = aggregate_hourly_demand(df)
    features = engineer_features(demand)

    out = PROCESSED_DIR / "features.csv"
    features.to_csv(out, index=False)
    print(f"[FEATURES] Saved to {out}")
    return features


if __name__ == "__main__":
    run_feature_pipeline()
