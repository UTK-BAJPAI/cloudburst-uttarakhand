# Cloudburst Risk Prediction System — Uttarakhand (India)

End-to-end pipeline that ingests historical Uttarakhand cloudburst location
metadata, fetches (or synthesises) daily NASA POWER style meteorological
features, trains a Random Forest classifier, ships an interactive Streamlit
GUI, and exports a Power-BI-ready dataset plus a full dashboard build guide.

---

## 1. What's in the box

```
outputs/
├── data_pipeline.py            Step-1: parse CSV → fetch/synth weather → label & balance
├── train_model.py              Step-2: feature engineering → Random Forest → metrics → pickle
├── app.py                      Step-3: Streamlit GUI (sliders, predict button, gauges)
├── visualizations.py           Step-4: feature importance, scatter, monthly trend, heatmap
├── _build_pbi_guide.py         Helper that regenerates the .docx guide
├── Power_BI_Dashboard_Guide.docx   Step-5: dashboard build manual (8 sections, DAX + visuals)
├── powerbi_theme.json          Step-5: dark theme matching the Streamlit GUI
├── requirements.txt
├── data/
│   ├── cloudburst_dataset.csv  513 rows, 13 columns (raw + label)
│   └── powerbi_dataset.csv     513 rows, 23 columns (raw + features + predictions)
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── metadata.json           feature list, tuned threshold, metrics, importances
└── figures/
    ├── feature_importance.png
    ├── rainfall_vs_cloudburst.png
    ├── risk_distribution.png
    ├── monthly_trend.png
    └── district_heatmap.png
```

---

## 2. Prerequisites

* Python 3.10+
* Windows / macOS / Linux
* Power BI Desktop (free) for the dashboard — Microsoft Store or
  https://powerbi.microsoft.com/desktop/

```bash
pip install -r requirements.txt
```

> If `pip install scikit-learn` is blocked in your environment, the training
> script falls back to a NumPy-only baseline classifier so the pipeline still
> produces every artefact end-to-end. **Install scikit-learn for the real
> Random Forest model.**

---

## 3. Quick start

```bash
# 1. Build the dataset (synthetic mode is the default - takes seconds)
python data_pipeline.py

# 2. Train the model + export the Power BI dataset
python train_model.py

# 3. Generate the static figures
python visualizations.py

# 4. Launch the interactive GUI
streamlit run app.py
```

To pull live data from NASA POWER instead of synthesising:

```bash
python data_pipeline.py --use-api
```

---

## 4. The Streamlit GUI

* Sliders for rainfall (mm/day), temperature (°C), humidity (%) and wind
  speed (m/s).
* "Predict" button runs the pickled model and shows risk level (Low /
  Medium / High) plus probability percentage.
* Sidebar lets you pick a hotspot (Kedarnath, Badrinath, …) and a month;
  the month feeds the cyclical feature so seasonality is preserved.
* In-session prediction history table.
* Feature-importance bar chart and a per-location historical chart.

---

## 5. The Power BI dashboard

The dashboard is **not** a binary `.pbix` (those cannot be reliably
hand-authored without Power BI Desktop), but **everything you need to
build it in 15 minutes is shipped here**:

1. `data/powerbi_dataset.csv` — the data model (already enriched with
   predictions, risk levels, calendar attributes, lat/lon).
2. `Power_BI_Dashboard_Guide.docx` — eight sections covering import,
   relationships, DAX measures, visuals per page (Map, KPIs, charts,
   heatmaps, decomposition tree), slicers and operational refresh.
3. `powerbi_theme.json` — the dark theme that matches the Streamlit GUI.

Open the .docx and follow it top to bottom; the result is a four-page
report with a map of high-risk zones, KPI cards, monthly trends and a
district-level risk matrix.

---

## 6. Model details (cheat sheet)

* **Algorithm:** RandomForest (400 trees, balanced class weights).
* **Train/test split:** 80 / 20, stratified.
* **Operational threshold:** tuned to keep recall ≥ 90 % (we'd rather
  false-alarm than miss a real cloudburst event).
* **Features (10):** PRECTOT, T2M, RH2M, WS2M, Rainfall_Intensity,
  Humidity_Rainfall, Temperature_Change, Heat_Index, Month_sin, Month_cos.
* **Outputs:** confusion matrix + classification report + `metadata.json`.

---

## 7. Error handling baked in

* NASA POWER fetch wrapped in `try/except` → silent fallback to synthetic data.
* Synthetic generator is physics-aware (monsoon-modulated Bernoulli/gamma
  rainfall, convective tail, latitude-dependent temperature).
* Both classes guaranteed in the training set via 1:2 down-sampling.
* Feature engineering is shared between training and the GUI (same
  `FEATURE_COLS`) so inference cannot drift from training.
* Threshold tuning never returns a value that would collapse recall below 0.9.

---

## 8. Bonus features

* `models/model.pkl` + `models/scaler.pkl` (pickled).
* Structured logging via `logging` in every module.
* Threshold tuning configurable via `tune_threshold(min_recall=...)`.
* `visualizations.py` doubles as documentation for the Power BI visuals.

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` (the script will still run on the NumPy fallback, but the real model needs sklearn). |
| `streamlit: command not found` | `pip install streamlit` then re-run `streamlit run app.py`. |
| NASA POWER timeouts | Use the default synthetic mode: just run `python data_pipeline.py` without `--use-api`. |
| Power BI map shows numbers | Set Latitude / Longitude → Data Category in Power Query. |
| Power BI MonthName slicer is alphabetical | Right-click MonthName → Sort By Column → Month. |
