"""
models/train.py
Trains XGBoost + LSTM ensemble for taxi demand forecasting.
Uses TimeSeriesSplit for temporal cross-validation.
"""

import sys, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR     = Path(__file__).parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR     = MODEL_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "hour", "dayofweek", "month", "dayofmonth", "is_weekend",
    "is_rush_hour", "is_night", "zone",
    "demand_lag_1h", "demand_lag_2h", "demand_lag_3h",
    "demand_lag_6h", "demand_lag_12h", "demand_lag_24h",
    "demand_lag_48h", "demand_lag_168h",
    "demand_rolling_mean_3h", "demand_rolling_mean_6h",
    "demand_rolling_mean_12h", "demand_rolling_mean_24h",
    "demand_rolling_std_3h", "demand_rolling_std_6h",
    "zone_avg_demand", "zone_demand_ratio", "is_high_demand_zone",
    "avg_fare", "avg_distance", "avg_duration", "total_passengers",
]


def load_data():
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        print("Features not found. Running ETL...")
        from pipeline.etl import run_etl
        run_etl()
    df = pd.read_csv(path)
    print(f"[DATA] Loaded {len(df):,} rows")
    return df


def build_lstm_model(input_dim: int):
    """Build a simple LSTM for sequence prediction."""
    try:
        import torch
        import torch.nn as nn

        class LSTMDemand(nn.Module):
            def __init__(self, input_size, hidden=64, layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, layers,
                                    batch_first=True, dropout=0.2)
                self.fc   = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze()

        return LSTMDemand(input_size=input_dim)
    except Exception:
        return None


def train_lstm(X_train, y_train, feature_names):
    """Train LSTM if PyTorch is available."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        print("[LSTM] Training LSTM model...")
        model  = build_lstm_model(len(feature_names))
        if model is None:
            return None

        X_t = torch.FloatTensor(X_train.values).unsqueeze(1)
        y_t = torch.FloatTensor(y_train.values)
        ds  = TensorDataset(X_t, y_t)
        dl  = DataLoader(ds, batch_size=512, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(5):
            total_loss = 0
            for xb, yb in dl:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[LSTM] Epoch {epoch+1}/5 — Loss: {total_loss/len(dl):.4f}")

        torch.save(model.state_dict(), MODEL_DIR / "lstm_model.pt")
        print("[LSTM] LSTM saved.")
        return model

    except Exception as e:
        print(f"[LSTM] Skipping LSTM ({e})")
        return None


def train_xgboost(X_train, y_train):
    print("[XGB] Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    X_tr, X_val = X_train.iloc[:int(len(X_train)*0.9)], X_train.iloc[int(len(X_train)*0.9):]
    y_tr, y_val = y_train.iloc[:int(len(y_train)*0.9)], y_train.iloc[int(len(y_train)*0.9):]
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
    return model


def evaluate(xgb_model, lstm_model, X_test, y_test, feature_names, scaler):
    print("\n[EVAL] Evaluating ensemble...")
    X_sc   = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    xgb_pred = xgb_model.predict(X_sc)

    if lstm_model is not None:
        try:
            import torch
            lstm_model.eval()
            with torch.no_grad():
                X_t    = torch.FloatTensor(X_sc.values).unsqueeze(1)
                lstm_p = lstm_model(X_t).numpy()
            final_pred = (xgb_pred * 0.6 + lstm_p * 0.4)
            print("[EVAL] Using XGBoost(60%) + LSTM(40%) ensemble")
        except Exception:
            final_pred = xgb_pred
    else:
        final_pred = xgb_pred
        print("[EVAL] Using XGBoost only (LSTM unavailable)")

    final_pred = np.clip(final_pred, 0, None)
    mae  = mean_absolute_error(y_test, final_pred)
    rmse = np.sqrt(mean_squared_error(y_test, final_pred))
    r2   = r2_score(y_test, final_pred)

    print(f"\n{'='*50}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  R²:    {r2:.4f}")
    print(f"{'='*50}")

    metrics = {"mae": round(mae,4), "rmse": round(rmse,4), "r2": round(r2,4)}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False)
    imp.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    return metrics, final_pred


def run_training():
    print("=" * 60)
    print("NYC TAXI — MODEL TRAINING")
    print("=" * 60)

    df = load_data()
    available  = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["target"]
    print(f"[PREP] Features: {len(available)} | Samples: {len(X):,}")

    split      = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"[SPLIT] Train: {len(X_train):,} | Test: {len(X_test):,}")

    scaler     = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=available)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test), columns=available)

    # TimeSeriesSplit CV
    print("\n[CV] TimeSeriesSplit cross-validation...")
    tscv    = TimeSeriesSplit(n_splits=5)
    cv_mdl  = XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(cv_mdl, X_train_sc, y_train, cv=tscv, scoring="r2")
    print(f"[CV] R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    xgb_model  = train_xgboost(X_train_sc, y_train)
    lstm_model = train_lstm(X_train_sc, y_train, available)
    metrics, _ = evaluate(xgb_model, lstm_model, X_test, y_test, available, scaler)

    joblib.dump(xgb_model, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(scaler,    MODEL_DIR / "scaler.pkl")
    joblib.dump(available, MODEL_DIR / "feature_names.pkl")
    print("\n[SAVE] Model artifacts saved.")
    print("[TRAINING] Complete!")
    return xgb_model, metrics


if __name__ == "__main__":
    run_training()
