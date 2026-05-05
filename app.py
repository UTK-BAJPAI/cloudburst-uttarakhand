"""
Cloudburst Risk Prediction System - Streamlit GUI
Real-time enabled version (with Live Monitor at bottom)
"""
from __future__ import annotations
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import model_utils  # noqa: F401  -- ensures pickle classes resolve

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

# ---------- Page setup + theme ----------
st.set_page_config(
    page_title="Cloudburst Risk Prediction - Uttarakhand",
    page_icon=":cloud_with_lightning_and_rain:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0b1424 0%, #0f1b30 100%); color: #e6eefb; }
    h1, h2, h3 { color: #e6eefb; letter-spacing: .2px; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,.04);
        padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,.08); }
    .stButton>button { background: linear-gradient(90deg, #2563eb, #06b6d4);
        color: white; border: 0; padding: 0.6rem 1.2rem; border-radius: 10px;
        font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artefacts():
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    return model, scaler, metadata


@st.cache_data(show_spinner=False)
def load_dataset():
    p = DATA_DIR / "powerbi_dataset.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["Date"])


def predict_cloudburst(rain, temp, humidity, wind, month=7, prev_temp=None,
                       model=None, scaler=None, metadata=None):
    if model is None:
        model, scaler, metadata = load_artefacts()
    if rain <= 2.5: intensity = 0
    elif rain <= 7.5: intensity = 1
    elif rain <= 35.5: intensity = 2
    elif rain <= 64.5: intensity = 3
    elif rain <= 100: intensity = 4
    else: intensity = 5
    feats = {
        "PRECTOT": rain, "T2M": temp, "RH2M": humidity, "WS2M": wind,
        "Rainfall_Intensity": intensity,
        "Humidity_Rainfall": rain * humidity,
        "Temperature_Change": 0.0 if prev_temp is None else (temp - prev_temp),
        "Heat_Index": temp + 0.05 * humidity - 0.10 * wind,
        "Month_sin": float(np.sin(2 * np.pi * month / 12)),
        "Month_cos": float(np.cos(2 * np.pi * month / 12)),
    }
    X = np.array([[feats[c] for c in metadata["feature_cols"]]], dtype=float)
    X = scaler.transform(X)
    proba = float(model.predict_proba(X)[0, 1])
    thr = metadata["tuned_threshold"]
    if proba >= max(thr, 0.6): level = "High"
    elif proba >= 0.30: level = "Medium"
    else: level = "Low"
    return {"risk_level": level, "probability": proba, "threshold": thr}


# ---------- UI ----------
st.title(":cloud_with_lightning_and_rain: Cloudburst Risk Prediction - Uttarakhand")
st.caption("Real-time cloudburst risk monitoring for Himalayan hotspots, "
           "powered by a Random Forest trained on NASA POWER data and live-fed by Open-Meteo.")

try:
    model, scaler, metadata = load_artefacts()
except FileNotFoundError:
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.stop()

dataset = load_dataset()

with st.sidebar:
    st.header("Context")
    locations = sorted(dataset["Location"].unique()) if not dataset.empty else \
        ["Kedarnath (Rudraprayag)"]
    location = st.selectbox("Location", locations, index=0)
    month = st.slider("Month of year", 1, 12, datetime.now().month)
    st.divider()
    st.markdown(f"**Operational threshold:** `{metadata['tuned_threshold']:.3f}`")
    st.markdown(f"**Test ROC-AUC:** `{metadata['metrics']['roc_auc']:.3f}`")
    st.markdown(f"**Test accuracy:** `{metadata['metrics']['accuracy_tuned']:.3f}`")

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("Meteorological Inputs")
    rain = st.slider("Rainfall (mm/day)", 0.0, 350.0, 25.0, 1.0)
    temp = st.slider("Temperature (degC)", -10.0, 45.0, 22.0, 0.5)
    humidity = st.slider("Relative Humidity (%)", 0.0, 100.0, 75.0, 1.0)
    wind = st.slider("Wind Speed (m/s)", 0.0, 25.0, 3.0, 0.1)
    predict_btn = st.button("Predict Cloudburst Risk", use_container_width=True)

with right:
    st.subheader("Risk Assessment")
    if predict_btn:
        result = predict_cloudburst(rain, temp, humidity, wind, month=month,
                                    model=model, scaler=scaler, metadata=metadata)
        prob_pct = result["probability"] * 100
        level = result["risk_level"]
        colour = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}[level]
        st.markdown(
            f"""
            <div style="padding:24px;border-radius:18px;
                background:linear-gradient(135deg,{colour}33,{colour}11);
                border:1px solid {colour}66;">
              <h2 style="margin:0;color:{colour};">{level} Risk</h2>
              <p style="font-size:48px;margin:6px 0;font-weight:700;">{prob_pct:.1f}%</p>
              <p style="opacity:.8;margin:0;">Probability of cloudburst given the inputs above.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(result["probability"], 0.0), 1.0))
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.insert(0, {
            "Time": datetime.utcnow().strftime("%H:%M:%S UTC"),
            "Location": location, "Rain (mm)": rain, "Temp (C)": temp,
            "Humidity (%)": humidity, "Wind (m/s)": wind,
            "Probability": f"{prob_pct:.1f}%", "Risk": level,
        })
    else:
        st.info("Move the sliders and click **Predict Cloudburst Risk** to score the snapshot.")

if st.session_state.get("history"):
    st.subheader("Recent predictions (this session)")
    st.dataframe(pd.DataFrame(st.session_state.history),
                 use_container_width=True, hide_index=True)

st.divider()
fcol, hcol = st.columns([1, 1], gap="large")
with fcol:
    st.subheader("Feature Importance")
    fi = pd.DataFrame(metadata["feature_importance"].items(),
                      columns=["Feature", "Importance"]).sort_values("Importance")
    st.bar_chart(fi.set_index("Feature"))
with hcol:
    st.subheader(f"Historical observations - {location}")
    if not dataset.empty:
        sub = dataset[dataset["Location"] == location].copy()
        sub["Year"] = sub["Date"].dt.year
        per_year = sub.groupby("Year")["Cloudburst"].sum().rename("Events")
        st.bar_chart(per_year)
    else:
        st.write("No historical data available.")


# ===== LIVE MONITOR SECTION (cloud-friendly inline fetch) =====
import requests as _rq

LIVE_SITES = [
    ("Arakot (Uttarkashi)",            "Uttarkashi",  30.88,  78.20),
    ("Badrinath (Chamoli)",            "Chamoli",     30.74,  79.49),
    ("Dharali (Uttarkashi)",           "Uttarkashi",  31.04,  78.73),
    ("Kedarnath (Rudraprayag)",        "Rudraprayag", 30.735, 79.066),
    ("Malpa (Pithoragarh)",            "Pithoragarh", 30.23,  80.72),
    ("Mandakini Valley (Rudraprayag)", "Rudraprayag", 30.45,  79.20),
]


@st.cache_data(ttl=3600, show_spinner="Fetching live weather...")
def _fetch_live(model_id):
    url = "https://api.open-meteo.com/v1/forecast"
    rows = []
    now = datetime.now()
    for name, district, lat, lon in LIVE_SITES:
        try:
            r = _rq.get(url, params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "daily": "precipitation_sum",
                "wind_speed_unit": "ms",
                "timezone": "Asia/Kolkata",
                "forecast_days": 1,
            }, timeout=20).json()
            cur = r["current"]
            rain = float((r.get("daily", {}).get("precipitation_sum") or [0])[0] or 0)
            res = predict_cloudburst(
                rain, cur["temperature_2m"], cur["relative_humidity_2m"],
                cur["wind_speed_10m"], month=now.month,
                model=model, scaler=scaler, metadata=metadata)
            rows.append({
                "Location": name, "District": district,
                "Rain (mm)": rain, "Temp (C)": cur["temperature_2m"],
                "Humidity (%)": cur["relative_humidity_2m"],
                "Probability": res["probability"], "Risk_Level": res["risk_level"],
            })
        except Exception:
            continue
    return pd.DataFrame(rows), now


st.divider()
st.subheader(":satellite: Live Real-Time Monitor")
latest, fetched_at = _fetch_live(id(model))

if latest.empty:
    st.warning("Could not fetch live data. Try refresh.")
else:
    high = latest[latest["Risk_Level"] == "High"]
    medium = latest[latest["Risk_Level"] == "Medium"]
    c1, c2, c3 = st.columns(3)
    c1.metric("HIGH risk sites", len(high))
    c2.metric("MEDIUM risk sites", len(medium))
    c3.metric("Last fetched", fetched_at.strftime("%H:%M, %d %b"))

    if not high.empty:
        st.error(f":rotating_light: {len(high)} HIGH-risk site(s) right now!")
    elif not medium.empty:
        st.warning(f":warning: {len(medium)} MEDIUM-risk site(s)")
    else:
        st.success(":white_check_mark: All sites Low risk - normal conditions")

    st.dataframe(latest, use_container_width=True, hide_index=True)

    if st.button(":arrows_counterclockwise: Refresh now"):
        st.cache_data.clear()
        st.rerun()