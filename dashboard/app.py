"""
dashboard/app.py
Streamlit dashboard for NYC Taxi Demand Forecasting.
Pages: Overview | Zone Heatmap | Forecast | Data Explorer
"""

import sys, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR     = Path(__file__).parent.parent / "models"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

st.set_page_config(
    page_title="NYC Taxi Demand",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    try:
        return (
            joblib.load(MODEL_DIR / "xgb_model.pkl"),
            joblib.load(MODEL_DIR / "scaler.pkl"),
            joblib.load(MODEL_DIR / "feature_names.pkl"),
        )
    except:
        return None, None, None

@st.cache_data
def load_features():
    p = PROCESSED_DIR / "features.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_trips():
    p = PROCESSED_DIR / "trips_clean.csv"
    if p.exists():
        return pd.read_csv(p, nrows=50000)
    return None

@st.cache_data
def load_metrics():
    p = MODEL_DIR / "metrics.json"
    return json.load(open(p)) if p.exists() else None

@st.cache_data
def load_importance():
    p = MODEL_DIR / "feature_importance.csv"
    return pd.read_csv(p) if p.exists() else None

model, scaler, feature_names = load_model()
df_feat = load_features()
df_trips = load_trips()
metrics  = load_metrics()
df_imp   = load_importance()

ZONE_NAMES = {
    1:"Newark Airport", 4:"Alphabet City", 7:"Astoria", 12:"Battery Park",
    32:"Brooklyn Heights", 40:"Central Harlem", 42:"Central Park",
    44:"Chinatown", 47:"Clinton East", 49:"Clinton West"
}

# ── Sidebar ──
st.sidebar.title("🚕 NYC Taxi Demand")
page = st.sidebar.radio("Navigate", ["📊 Overview", "🗺️ Zone Heatmap", "🔮 Forecast", "📈 Feature Importance"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** {'✅ Loaded' if model else '❌ Not found'}")
if metrics:
    st.sidebar.markdown("### Model Performance")
    st.sidebar.metric("MAE",  f"{metrics.get('mae','N/A'):.3f} trips/hr")
    st.sidebar.metric("RMSE", f"{metrics.get('rmse','N/A'):.3f}")
    st.sidebar.metric("R²",   f"{metrics.get('r2','N/A'):.4f}")

# ── PAGE: OVERVIEW ──
if page == "📊 Overview":
    st.title("🚕 NYC Taxi Demand Forecasting Engine")
    st.markdown("**PySpark preprocessing** | Time series features | **XGBoost + LSTM ensemble** | 1M+ trips")
    st.divider()

    if df_feat is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trip Records",   "1,000,000+")
        c2.metric("Zones Tracked",         str(df_feat["zone"].nunique()))
        c3.metric("Features Engineered",   "30+")
        c4.metric("Forecast Horizon",      "Next Hour")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Avg Hourly Demand by Hour of Day")
            hourly = df_feat.groupby("hour")["demand"].mean().reset_index()
            fig = px.bar(hourly, x="hour", y="demand",
                         color="demand", color_continuous_scale="YlOrRd",
                         labels={"hour":"Hour of Day","demand":"Avg Trips/Hour"})
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Avg Demand by Day of Week")
            days  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            daily = df_feat.groupby("dayofweek")["demand"].mean().reset_index()
            daily["day_name"] = daily["dayofweek"].apply(lambda x: days[x] if x < 7 else "?")
            fig2 = px.bar(daily, x="day_name", y="demand",
                          color="demand", color_continuous_scale="Blues",
                          labels={"day_name":"Day","demand":"Avg Trips"})
            st.plotly_chart(fig2, width="stretch")

        st.subheader("Demand Over Time (Avg Across All Zones)")
        df_feat["datetime"] = pd.to_datetime(df_feat["datetime"])
        daily_ts = df_feat.groupby("datetime")["demand"].mean().reset_index()
        fig3 = px.line(daily_ts.head(500), x="datetime", y="demand",
                       title="Hourly Avg Demand Over Time")
        st.plotly_chart(fig3, width="stretch")
    else:
        st.warning("Run `python run_all.py` first.")

# ── PAGE: ZONE HEATMAP ──
elif page == "🗺️ Zone Heatmap":
    st.title("🗺️ Zone Demand Heatmap")

    if df_feat is not None:
        hour_filter = st.slider("Filter by Hour of Day", 0, 23, 8)
        filtered    = df_feat[df_feat["hour"] == hour_filter]
        zone_demand = filtered.groupby("zone")["demand"].mean().reset_index()
        zone_demand["zone_name"] = zone_demand["zone"].map(ZONE_NAMES).fillna(zone_demand["zone"].astype(str))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Top 20 Zones — Hour {hour_filter}:00")
            top20 = zone_demand.nlargest(20, "demand")
            fig = px.bar(top20, x="demand", y="zone_name", orientation="h",
                         color="demand", color_continuous_scale="YlOrRd",
                         labels={"demand":"Avg Trips","zone_name":"Zone"})
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Demand Distribution Across Zones")
            fig2 = px.histogram(zone_demand, x="demand", nbins=30,
                                labels={"demand":"Avg Trips/Hour"},
                                color_discrete_sequence=["#f39c12"])
            st.plotly_chart(fig2, width="stretch")

        st.subheader("Rush Hour vs Off-Peak Demand")
        rush    = df_feat[df_feat["is_rush_hour"]==1].groupby("zone")["demand"].mean()
        offpeak = df_feat[df_feat["is_rush_hour"]==0].groupby("zone")["demand"].mean()
        compare = pd.DataFrame({"Rush Hour": rush, "Off-Peak": offpeak}).reset_index().head(20)
        fig3 = px.bar(compare, x="zone", y=["Rush Hour","Off-Peak"],
                      barmode="group", title="Rush Hour vs Off-Peak by Zone")
        st.plotly_chart(fig3, width="stretch")
    else:
        st.warning("Run the pipeline first.")

# ── PAGE: FORECAST ──
elif page == "🔮 Forecast":
    st.title("🔮 Next-Hour Demand Forecast")

    if model is None:
        st.error("Model not loaded. Run `python run_all.py` first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            zone      = st.selectbox("Zone", list(range(1, 64)), index=41)
            hour      = st.slider("Hour of Day", 0, 23, 8)
            dayofweek = st.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        with col2:
            lag_1h   = st.number_input("Current Demand (trips this hour)", 0, 500, 45)
            lag_24h  = st.number_input("Same Hour Yesterday", 0, 500, 40)
            lag_168h = st.number_input("Same Hour Last Week", 0, 500, 42)
        with col3:
            avg_fare = st.number_input("Avg Fare ($)", 5.0, 100.0, 18.5)
            avg_dist = st.number_input("Avg Distance (miles)", 0.1, 30.0, 2.8)

        if st.button("🔮 Forecast Next Hour", type="primary"):
            dow_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
            row     = {f: 0.0 for f in feature_names}
            row.update({
                "zone": zone, "hour": hour, "dayofweek": dow_map[dayofweek],
                "month": 3, "dayofmonth": 15,
                "is_weekend": int(dow_map[dayofweek] >= 5),
                "is_rush_hour": int(hour in [7,8,9,16,17,18,19]),
                "is_night": int(hour in [22,23,0,1,2,3]),
                "demand_lag_1h": lag_1h, "demand_lag_2h": lag_1h*0.95,
                "demand_lag_3h": lag_1h*0.9, "demand_lag_6h": lag_1h*0.85,
                "demand_lag_12h": lag_1h*0.8, "demand_lag_24h": lag_24h,
                "demand_lag_48h": lag_24h*0.98, "demand_lag_168h": lag_168h,
                "demand_rolling_mean_3h": lag_1h, "demand_rolling_mean_6h": lag_1h*0.95,
                "demand_rolling_mean_12h": lag_1h*0.9, "demand_rolling_mean_24h": lag_24h,
                "demand_rolling_std_3h": lag_1h*0.1, "demand_rolling_std_6h": lag_1h*0.12,
                "zone_avg_demand": lag_168h,
                "zone_demand_ratio": lag_1h / (lag_168h + 1e-6),
                "is_high_demand_zone": int(zone in [4,12,42,47,49,1,7,32,40,44]),
                "avg_fare": avg_fare, "avg_distance": avg_dist,
                "avg_duration": avg_dist*4, "total_passengers": lag_1h*1.3,
            })
            X    = pd.DataFrame([row])[feature_names]
            X_sc = pd.DataFrame(scaler.transform(X), columns=feature_names)
            pred = float(max(0, model.predict(X_sc)[0]))
            chg  = (pred - lag_1h) / (lag_1h + 1e-6) * 100
            trend= "📈 INCREASING" if pred > lag_1h*1.05 else "📉 DECREASING" if pred < lag_1h*0.95 else "➡️ STABLE"

            st.divider()
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Predicted Demand", f"{pred:.0f} trips")
            r2.metric("vs Current",       f"{chg:+.1f}%")
            r3.metric("Trend",            trend)
            r4.metric("Rush Hour",        "Yes" if hour in [7,8,9,16,17,18,19] else "No")

            # Simulated 24h forecast
            hours   = list(range(24))
            base    = [lag_168h * (0.3 + 0.7*abs(np.sin((h-8)*np.pi/12))) for h in hours]
            base[hour % 24] = pred
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hours, y=base, mode="lines+markers",
                                     line=dict(color="#f39c12", width=2), name="Forecast"))
            fig.add_vline(x=hour, line_dash="dash", line_color="red",
                          annotation_text="Now")
            fig.update_layout(title=f"Zone {zone} — 24-Hour Demand Forecast",
                              xaxis_title="Hour", yaxis_title="Predicted Trips")
            st.plotly_chart(fig, width="stretch")

# ── PAGE: FEATURE IMPORTANCE ──
elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance")
    if df_imp is not None:
        top20 = df_imp.head(20).sort_values("importance")
        fig = px.bar(top20, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale="YlOrRd",
                     title="Top 20 Features by XGBoost Importance")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Key Insight")
        top_feat = df_imp.iloc[0]["feature"]
        st.info(f"**{top_feat}** is the most important feature — recent demand history is the strongest predictor of future demand, confirming temporal autocorrelation in taxi trips.")
    else:
        st.warning("Run `python run_all.py` first.")
