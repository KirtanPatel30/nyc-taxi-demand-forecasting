"""pipeline/etl.py — Full ETL orchestrator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_etl():
    print("=" * 60)
    print("NYC TAXI DEMAND FORECASTING — ETL PIPELINE")
    print("=" * 60)

    from data.generate import generate_trips, save_data
    from pipeline.spark_preprocess import run_preprocessing
    from pipeline.features import run_feature_pipeline

    raw_path = Path(__file__).parent.parent / "data" / "raw" / "trips.csv"
    if not raw_path.exists():
        print("\n[1/3] Generating NYC taxi dataset...")
        df = generate_trips(n=1_000_000, days=90)
        save_data(df)
    else:
        print("\n[1/3] Raw data already exists, skipping generation.")

    print("\n[2/3] Preprocessing with PySpark/Pandas...")
    run_preprocessing()

    print("\n[3/3] Engineering features...")
    features = run_feature_pipeline()

    print(f"\n[ETL] Complete! Features shape: {features.shape}")
    return features


if __name__ == "__main__":
    run_etl()
