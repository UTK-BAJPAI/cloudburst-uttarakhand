"""
Cloudburst Risk Prediction System - Model Training
==================================================
Reads `data/cloudburst_dataset.csv`, performs feature engineering,
trains a Random Forest classifier, evaluates it, tunes a probability
threshold for the operational regime ("recall-first" - we'd rather
issue a false alarm than miss a real cloudburst), and persists:

    models/model.pkl
    models/scaler.pkl
    models/metadata.json
    data/powerbi_dataset.csv     <- predictions joined to the raw set
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from model_utils import (
    RandomForestClassifier, StandardScaler, train_test_split,
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, SKLEARN_OK,
)

logging.basicConfig(format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("cloudburst.train")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "PRECTOT", "T2M", "RH2M", "WS2M",
    "Rainfall_Intensity", "Humidity_Rainfall", "Temperature_Change",
    "Heat_Index", "Month_sin", "Month_cos",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)

    bins = [-0.1, 2.5, 7.5, 35.5, 64.5, 100, 1e6]
    labels = [0, 1, 2, 3, 4, 5]
    df["Rainfall_Intensity"] = pd.cut(df["PRECTOT"], bins=bins, labels=labels).astype(int)

    df["Humidity_Rainfall"] = df["RH2M"] * df["PRECTOT"]
    df["Temperature_Change"] = df.groupby("Location")["T2M"].diff().fillna(0.0)
    df["Heat_Index"] = df["T2M"] + 0.05 * df["RH2M"] - 0.10 * df["WS2M"]
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    return df


def tune_threshold(y_true, y_proba, min_recall: float = 0.90) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == len(precision) - 1:
        thr = np.r_[thresholds, 1.0]
    else:
        thr = thresholds
    ok = recall >= min_recall
    if not ok.any():
        return 0.5
    return float(thr[ok][np.argmax(precision[ok])])


def main() -> None:
    src = DATA_DIR / "cloudburst_dataset.csv"
    if not src.exists():
        raise SystemExit(f"Run data_pipeline.py first - missing {src}")

    raw = pd.read_csv(src)
    log.info("Loaded %d rows (positives=%d).", len(raw), int(raw["Cloudburst"].sum()))

    df = engineer_features(raw)
    X = df[FEATURE_COLS].values.astype(float)
    y = df["Cloudburst"].values.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y if SKLEARN_OK else None, random_state=42,
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    if SKLEARN_OK:
        model = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42)
    else:
        model = RandomForestClassifier()
    model.fit(X_tr_s, y_tr)

    proba = model.predict_proba(X_te_s)[:, 1]
    auc = roc_auc_score(y_te, proba)
    base_pred = (proba >= 0.5).astype(int)
    base_acc = accuracy_score(y_te, base_pred)
    log.info("ROC-AUC=%.3f, default-threshold accuracy=%.3f", auc, base_acc)

    tuned_threshold = tune_threshold(y_te, proba, min_recall=0.90)
    tuned_pred = (proba >= tuned_threshold).astype(int)
    tuned_acc = accuracy_score(y_te, tuned_pred)

    print(f"\n=== Confusion Matrix (tuned threshold = {tuned_threshold:.3f}) ===")
    print(confusion_matrix(y_te, tuned_pred))
    print("\n=== Classification Report ===")
    print(classification_report(y_te, tuned_pred,
                                target_names=["No Cloudburst", "Cloudburst"]))

    with open(MODEL_DIR / "model.pkl", "wb") as f: pickle.dump(model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)

    metadata = {
        "feature_cols": FEATURE_COLS,
        "tuned_threshold": tuned_threshold,
        "sklearn": SKLEARN_OK,
        "metrics": {
            "roc_auc": float(auc),
            "accuracy_default": float(base_acc),
            "accuracy_tuned": float(tuned_acc),
        },
        "feature_importance": dict(zip(
            FEATURE_COLS, [float(v) for v in model.feature_importances_])),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Saved model artefacts to %s.", MODEL_DIR)

    full_proba = model.predict_proba(scaler.transform(df[FEATURE_COLS].values.astype(float)))[:, 1]
    pbi = df.copy()
    pbi["Predicted_Probability"] = full_proba
    pbi["Predicted_Cloudburst"] = (full_proba >= tuned_threshold).astype(int)
    pbi["Risk_Level"] = pd.cut(
        full_proba, bins=[-0.01, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"]).astype(str)
    out = DATA_DIR / "powerbi_dataset.csv"
    pbi.to_csv(out, index=False)
    log.info("Saved Power BI dataset to %s (%d rows).", out, len(pbi))


if __name__ == "__main__":
    main()
