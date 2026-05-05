"""
Cloudburst Risk Prediction System - Visualisations
==================================================
Reads `data/powerbi_dataset.csv` and `models/metadata.json` and writes:

    figures/feature_importance.png
    figures/rainfall_vs_cloudburst.png
    figures/risk_distribution.png
    figures/monthly_trend.png
    figures/district_heatmap.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
DATA = ROOT / "data" / "powerbi_dataset.csv"
META = ROOT / "models" / "metadata.json"

sns.set_theme(style="whitegrid", context="talk")
PALETTE = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}


def fig_feature_importance() -> None:
    meta = json.loads(META.read_text())
    fi = pd.Series(meta["feature_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    fi.plot.barh(ax=ax, color="#2563eb")
    ax.set_title("Feature Importance - Random Forest")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=150)
    plt.close(fig)


def fig_rainfall_vs_cloudburst(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df, x="PRECTOT", y="RH2M",
        hue="Cloudburst", style="Cloudburst", s=60, alpha=0.75,
        palette={0: "#60a5fa", 1: "#ef4444"}, ax=ax,
    )
    ax.axvline(100, color="#374151", lw=1.2, ls="--", label="100 mm/day threshold")
    ax.set_title("Rainfall vs Humidity, coloured by Cloudburst label")
    ax.set_xlabel("Daily rainfall (mm)")
    ax.set_ylabel("Relative humidity (%)")
    ax.legend(title="Cloudburst")
    fig.tight_layout()
    fig.savefig(FIG / "rainfall_vs_cloudburst.png", dpi=150)
    plt.close(fig)


def fig_risk_distribution(df: pd.DataFrame) -> None:
    counts = df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 6))
    counts.plot.bar(ax=ax, color=[PALETTE[k] for k in counts.index])
    ax.set_title("Predicted Risk Distribution")
    ax.set_ylabel("Number of days")
    ax.set_xlabel("")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, f"{int(v)}",
                ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "risk_distribution.png", dpi=150)
    plt.close(fig)


def fig_monthly_trend(df: pd.DataFrame) -> None:
    monthly = (df.assign(M=df["Month"])
                 .groupby("M")
                 .agg(events=("Cloudburst", "sum"),
                      avg_rain=("PRECTOT", "mean"))
                 .reindex(range(1, 13)).fillna(0))
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.bar(monthly.index, monthly["events"], color="#ef4444", alpha=0.7,
            label="Cloudburst events")
    ax1.set_ylabel("Cloudburst events", color="#ef4444")
    ax1.set_xlabel("Month")
    ax1.set_xticks(range(1, 13))
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["avg_rain"], color="#2563eb",
             marker="o", lw=2.5, label="Mean daily rain")
    ax2.set_ylabel("Mean daily rain (mm)", color="#2563eb")
    ax1.set_title("Monthly cloudburst trend vs mean rainfall")
    fig.tight_layout()
    fig.savefig(FIG / "monthly_trend.png", dpi=150)
    plt.close(fig)


def fig_district_heatmap(df: pd.DataFrame) -> None:
    pivot = (df.pivot_table(index="District", columns="Month",
                            values="Cloudburst", aggfunc="sum", fill_value=0)
               .reindex(columns=range(1, 13), fill_value=0))
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, cmap="rocket_r", annot=True, fmt="d", linewidths=.4, ax=ax)
    ax.set_title("Cloudburst events by District and Month")
    ax.set_xlabel("Month"); ax.set_ylabel("District")
    fig.tight_layout()
    fig.savefig(FIG / "district_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if not DATA.exists() or not META.exists():
        raise SystemExit("Run data_pipeline.py and train_model.py first.")
    df = pd.read_csv(DATA, parse_dates=["Date"])
    fig_feature_importance()
    fig_rainfall_vs_cloudburst(df)
    fig_risk_distribution(df)
    fig_monthly_trend(df)
    fig_district_heatmap(df)
    print(f"Saved 5 figures to {FIG}")


if __name__ == "__main__":
    main()
