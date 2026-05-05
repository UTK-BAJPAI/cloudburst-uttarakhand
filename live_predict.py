"""Real-time cloudburst risk monitor using Open-Meteo (no API key needed)."""
from __future__ import annotations
import json, os, pickle, smtplib, sys
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import model_utils  # noqa

ROOT = Path(__file__).resolve().parent
LIVE_CSV = ROOT / "data" / "live_predictions.csv"

# Optional email alerts
ALERT_EMAIL_FROM = os.getenv("ALERT_FROM", "")
ALERT_EMAIL_PASS = os.getenv("ALERT_PASS", "")
ALERT_EMAIL_TO   = os.getenv("ALERT_TO",   "")

SITES = [
    ("Arakot (Uttarkashi)",            "Uttarkashi",  30.88,  78.20),
    ("Badrinath (Chamoli)",            "Chamoli",     30.74,  79.49),
    ("Dharali (Uttarkashi)",           "Uttarkashi",  31.04,  78.73),
    ("Kedarnath (Rudraprayag)",        "Rudraprayag", 30.735, 79.066),
    ("Malpa (Pithoragarh)",            "Pithoragarh", 30.23,  80.72),
    ("Mandakini Valley (Rudraprayag)", "Rudraprayag", 30.45,  79.20),
]

with open(ROOT / "models" / "model.pkl",  "rb") as f: MODEL = pickle.load(f)
with open(ROOT / "models" / "scaler.pkl", "rb") as f: SCALER = pickle.load(f)
META = json.loads((ROOT / "models" / "metadata.json").read_text())


def fetch_weather(lat, lon):
    """Live observation + 24h precipitation total from Open-Meteo (free, no key)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "daily": "precipitation_sum",
        "wind_speed_unit": "ms",
        "timezone": "Asia/Kolkata",
        "forecast_days": 1,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    cur = j["current"]
    daily_rain = j.get("daily", {}).get("precipitation_sum", [0.0])[0] or 0.0
    return {
        "PRECTOT": float(daily_rain),
        "T2M":     float(cur["temperature_2m"]),
        "RH2M":    float(cur["relative_humidity_2m"]),
        "WS2M":    float(cur["wind_speed_10m"]),
    }


def featurise(rain, temp, hum, wind, month):
    if rain <= 2.5: i = 0
    elif rain <= 7.5: i = 1
    elif rain <= 35.5: i = 2
    elif rain <= 64.5: i = 3
    elif rain <= 100: i = 4
    else: i = 5
    return [rain, temp, hum, wind, i, rain*hum, 0.0,
            temp + 0.05*hum - 0.10*wind,
            np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)]


def send_alert(rows):
    if not (ALERT_EMAIL_FROM and ALERT_EMAIL_PASS and ALERT_EMAIL_TO):
        return
    body = "\n".join(f"{r['Location']}: {r['Probability']:.0%} risk - rain {r['PRECTOT']:.1f} mm"
                     for r in rows)
    msg = MIMEText(f"HIGH cloudburst risk:\n\n{body}\n\nGenerated: {datetime.now()}")
    msg["Subject"] = "[CLOUDBURST ALERT] High risk in Uttarakhand"
    msg["From"] = ALERT_EMAIL_FROM; msg["To"] = ALERT_EMAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(ALERT_EMAIL_FROM, ALERT_EMAIL_PASS)
        s.send_message(msg)


def main():
    now = datetime.now()
    out_rows = []
    for name, district, lat, lon in SITES:
        try:
            w = fetch_weather(lat, lon)
        except Exception as e:
            print(f"[WARN] {name}: {e}"); continue
        feats = featurise(w["PRECTOT"], w["T2M"], w["RH2M"], w["WS2M"], now.month)
        x = SCALER.transform(np.array([feats]))
        proba = float(MODEL.predict_proba(x)[0, 1])
        risk = "High" if proba >= max(META["tuned_threshold"], 0.6) else \
               "Medium" if proba >= 0.30 else "Low"
        out_rows.append({
            "Timestamp": now.isoformat(timespec="minutes"),
            "Location": name, "District": district,
            "Latitude": lat, "Longitude": lon,
            **w, "Probability": proba, "Risk_Level": risk,
        })
        print(f"{name:35s}  rain={w['PRECTOT']:6.1f} mm  T={w['T2M']:5.1f}C  "
              f"RH={w['RH2M']:5.1f}%  prob={proba:5.1%}  -> {risk}")

    if not out_rows:
        sys.exit("No locations fetched.")
    df_new = pd.DataFrame(out_rows)
    LIVE_CSV.parent.mkdir(exist_ok=True)
    if LIVE_CSV.exists():
        df_new.to_csv(LIVE_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LIVE_CSV, index=False)
    print(f"Appended {len(df_new)} rows to {LIVE_CSV}")

    high = [r for r in out_rows if r["Risk_Level"] == "High"]
    if high:
        try: send_alert(high); print(f"Alert sent for {len(high)} sites.")
        except Exception as e: print(f"[ALERT FAILED] {e}")


if __name__ == "__main__":
    main()