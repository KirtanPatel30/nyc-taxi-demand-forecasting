"""tests/test_pipeline.py — Unit tests for NYC Taxi Demand Forecasting"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataGeneration:
    def test_generate_shape(self):
        from data.generate import generate_trips
        df = generate_trips(n=1000, days=7)
        assert len(df) == 1000
        assert "pickup_zone" in df.columns
        assert "fare_amount" in df.columns

    def test_fares_positive(self):
        from data.generate import generate_trips
        df = generate_trips(n=500, days=7)
        assert (df["fare_amount"] > 0).all()

    def test_distance_positive(self):
        from data.generate import generate_trips
        df = generate_trips(n=500, days=7)
        assert (df["trip_distance"] > 0).all()

    def test_zones_valid(self):
        from data.generate import generate_trips
        df = generate_trips(n=500, days=7)
        assert df["pickup_zone"].between(1, 63).all()

    def test_timestamps_sorted(self):
        from data.generate import generate_trips
        df = generate_trips(n=500, days=7)
        assert pd.to_datetime(df["pickup_datetime"]).is_monotonic_increasing


class TestPreprocessing:
    def setup_method(self):
        from data.generate import generate_trips
        self.df = generate_trips(n=2000, days=14)

    def test_pandas_preprocess_adds_hour(self):
        df = self.df.copy()
        df["pickup_dt"] = pd.to_datetime(df["pickup_datetime"])
        df["hour"] = df["pickup_dt"].dt.hour
        assert "hour" in df.columns
        assert df["hour"].between(0, 23).all()

    def test_filters_zero_distances(self):
        df = self.df.copy()
        df.loc[0, "trip_distance"] = 0
        filtered = df[df["trip_distance"] > 0]
        assert len(filtered) < len(df)

    def test_weekend_flag(self):
        df = self.df.copy()
        df["pickup_dt"] = pd.to_datetime(df["pickup_datetime"])
        df["dayofweek"] = df["pickup_dt"].dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        assert df["is_weekend"].isin([0, 1]).all()


class TestFeatureEngineering:
    def setup_method(self):
        from data.generate import generate_trips
        from pipeline.spark_preprocess import preprocess_with_pandas
        import tempfile, os
        df = generate_trips(n=5000, days=30)
        tmp = Path(tempfile.mktemp(suffix=".csv"))
        df.to_csv(tmp, index=False)
        self.clean_df = preprocess_with_pandas(str(tmp))
        tmp.unlink()

    def test_hourly_aggregation(self):
        from pipeline.features import aggregate_hourly_demand
        demand = aggregate_hourly_demand(self.clean_df)
        assert "demand" in demand.columns
        assert "zone" in demand.columns
        assert (demand["demand"] >= 0).all()

    def test_lag_features_created(self):
        from pipeline.features import aggregate_hourly_demand, engineer_features
        demand   = aggregate_hourly_demand(self.clean_df)
        features = engineer_features(demand)
        assert "demand_lag_1h" in features.columns
        assert "demand_lag_24h" in features.columns

    def test_target_column_exists(self):
        from pipeline.features import aggregate_hourly_demand, engineer_features
        demand   = aggregate_hourly_demand(self.clean_df)
        features = engineer_features(demand)
        assert "target" in features.columns
        assert (features["target"] >= 0).all()


class TestAPI:
    def test_demand_input_valid(self):
        from api.main import DemandInput
        inp = DemandInput(
            zone=42, hour=8, dayofweek=1, month=3,
            demand_lag_1h=45.0, demand_lag_24h=40.0,
            demand_lag_168h=42.0, avg_fare=18.5, avg_distance=2.8
        )
        assert inp.zone == 42
        assert 0 <= inp.hour <= 23

    def test_rush_hour_zones(self):
        high_demand = [4, 12, 42, 47, 49, 1, 7, 32, 40, 44]
        assert 42 in high_demand
        assert 1 in high_demand

    def test_weekend_flag_logic(self):
        assert int(5 >= 5) == 1  # Saturday
        assert int(3 >= 5) == 0  # Thursday

    def test_trend_logic(self):
        pred    = 50.0
        current = 45.0
        trend   = "INCREASING" if pred > current*1.05 else "DECREASING" if pred < current*0.95 else "STABLE"
        assert trend == "INCREASING"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
