"""run_all.py — Run the entire NYC Taxi Demand Forecasting pipeline."""
import subprocess, sys
from pathlib import Path

def run(cmd, desc):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    r = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
    if r.returncode != 0:
        print(f"ERROR: failed with code {r.returncode}")
        sys.exit(r.returncode)

if __name__ == "__main__":
    print("\n🚕 NYC TAXI DEMAND FORECASTING — FULL PIPELINE")
    print("=" * 60)
    run("python pipeline/etl.py",     "STEP 1/3: ETL — generate + clean + engineer features")
    run("python models/train.py",     "STEP 2/3: Train XGBoost + LSTM ensemble model")
    run("python -m pytest tests/ -v", "STEP 3/3: Run unit tests")
    print("\n" + "="*60)
    print("  ✅ ALL STEPS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  Dashboard:  streamlit run dashboard/app.py  → http://localhost:8501")
    print("  API:        uvicorn api.main:app --reload   → http://localhost:8000/docs")
    print("=" * 60)
