"""
pipeline/spark_preprocess.py
PySpark preprocessing pipeline for NYC taxi data.
Falls back to Pandas if PySpark is unavailable.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_with_spark(input_path: str) -> pd.DataFrame:
    """PySpark preprocessing pipeline."""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType, IntegerType

        print("[SPARK] Starting Spark session...")
        spark = SparkSession.builder             .appName("NYCTaxiPreprocessing")             .config("spark.driver.memory", "2g")             .config("spark.sql.shuffle.partitions", "8")             .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        print("[SPARK] Reading data...")
        df = spark.read.csv(input_path, header=True, inferSchema=True)
        print(f"[SPARK] Loaded {df.count():,} rows")

        # Parse timestamps
        df = df.withColumn("pickup_dt",  F.to_timestamp("pickup_datetime"))
        df = df.withColumn("dropoff_dt", F.to_timestamp("dropoff_datetime"))

        # Extract time features
        df = df.withColumn("hour",       F.hour("pickup_dt"))
        df = df.withColumn("dayofweek",  F.dayofweek("pickup_dt"))
        df = df.withColumn("month",      F.month("pickup_dt"))
        df = df.withColumn("dayofmonth", F.dayofmonth("pickup_dt"))
        df = df.withColumn("weekofyear", F.weekofyear("pickup_dt"))
        df = df.withColumn("is_weekend", (F.col("dayofweek") >= 6).cast(IntegerType()))
        df = df.withColumn("is_rush_hour",
            ((F.col("hour").between(7,9)) | (F.col("hour").between(16,19))).cast(IntegerType()))

        # Data quality filters
        df = df.filter(F.col("trip_distance") > 0)
        df = df.filter(F.col("fare_amount") > 0)
        df = df.filter(F.col("passenger_count") > 0)
        df = df.filter(F.col("trip_duration_min") > 0)

        # Speed sanity check
        df = df.withColumn("speed_mph",
            F.col("trip_distance") / (F.col("trip_duration_min") / 60 + 0.001))
        df = df.filter(F.col("speed_mph") < 80)

        result = df.toPandas()
        spark.stop()
        print(f"[SPARK] Preprocessing complete. Shape: {result.shape}")
        return result

    except Exception as e:
        print(f"[SPARK] PySpark unavailable ({e}). Falling back to Pandas.")
        return preprocess_with_pandas(input_path)


def preprocess_with_pandas(input_path: str) -> pd.DataFrame:
    """Pandas fallback preprocessing."""
    print("[PANDAS] Running Pandas preprocessing...")
    df = pd.read_csv(input_path)
    print(f"[PANDAS] Loaded {len(df):,} rows")

    df["pickup_dt"]  = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_dt"] = pd.to_datetime(df["dropoff_datetime"])

    df["hour"]        = df["pickup_dt"].dt.hour
    df["dayofweek"]   = df["pickup_dt"].dt.dayofweek
    df["month"]       = df["pickup_dt"].dt.month
    df["dayofmonth"]  = df["pickup_dt"].dt.day
    df["weekofyear"]  = df["pickup_dt"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["dayofweek"] >= 5).astype(int)
    df["is_rush_hour"]= df["hour"].isin([7,8,9,16,17,18,19]).astype(int)

    # Data quality
    df = df[df["trip_distance"] > 0]
    df = df[df["fare_amount"]   > 0]
    df = df[df["passenger_count"] > 0]
    df = df[df["trip_duration_min"] > 0]
    df["speed_mph"] = df["trip_distance"] / (df["trip_duration_min"] / 60 + 0.001)
    df = df[df["speed_mph"] < 80]

    print(f"[PANDAS] Preprocessing complete. Shape: {df.shape}")
    return df


def run_preprocessing():
    input_path = str(RAW_DIR / "trips.csv")
    df = preprocess_with_spark(input_path)
    out = PROCESSED_DIR / "trips_clean.csv"
    # Save sample for faster downstream processing
    df.sample(min(200_000, len(df)), random_state=42).to_csv(out, index=False)
    print(f"[PREPROCESS] Saved {min(200_000, len(df)):,} cleaned rows to {out}")
    return df


if __name__ == "__main__":
    run_preprocessing()
