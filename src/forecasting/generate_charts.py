"""
Graficos de comparacion de modelos de forecasting (track de referencia).
Proyecto: Sistema inteligente de gestion de inventarios - Comercial La Feria

Entrada:  data/model_comparison_results.csv, data/prediction_details.json
          (salida de train_forecasting.py)
Salida:   data/charts/grafico_comparacion_mape.png
          data/charts/grafico_distribucion_modelos.png
          data/charts/grafico_forecast_vs_real.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CHARTS_DIR = DATA_DIR / "charts"

RESULTS_PATH = DATA_DIR / "model_comparison_results.csv"
PREDICTIONS_PATH = DATA_DIR / "prediction_details.json"

MODEL_COLORS = {"ARIMA": "#4C72B0", "Prophet": "#DD8452", "LSTM": "#55A868"}


def chart_mape_by_model(df_res, out_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(df_res))
    w = 0.25
    ax.bar(x - w, df_res["ARIMA_MAPE"], width=w, label="ARIMA", color=MODEL_COLORS["ARIMA"])
    ax.bar(x, df_res["Prophet_MAPE"], width=w, label="Prophet", color=MODEL_COLORS["Prophet"])
    ax.bar(x + w, df_res["LSTM_MAPE"], width=w, label="LSTM", color=MODEL_COLORS["LSTM"])
    ax.axhline(15, color="red", linestyle="--", linewidth=1, label="Umbral proyecto (15%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_res["product_code"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Comparacion de error (MAPE) por modelo y producto - Comercial La Feria")
    ax.set_ylim(0, 60)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def chart_selected_model(df_res, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = df_res["selected_model"].value_counts()
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
           colors=[MODEL_COLORS.get(k, "#8C8C8C") for k in counts.index], startangle=90)
    ax.set_title(f"Modelo seleccionado automaticamente\n({len(df_res)} productos)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def chart_forecast_vs_real(df_res, detail, out_path):
    # Primeros 4 productos disponibles en el detalle (evita depender de codigos fijos)
    selection = [c for c in df_res["product_code"] if c in detail][:4]
    if not selection:
        print("  (sin productos con detalle de predicciones, se omite forecast_vs_real)")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, code in zip(axes.flat, selection):
        p = detail[code]
        dates = pd.to_datetime(p["test_dates"])
        ax.plot(dates, p["actual"], label="Real", color="black", marker="o", linewidth=2)
        for model_name in ("ARIMA", "Prophet", "LSTM"):
            if model_name in p:
                ax.plot(dates, p[model_name], label=model_name, linestyle="--")
        row = df_res[df_res.product_code == code].iloc[0]
        ax.set_title(f"{code}\n{row['product_name'][:35]}\n"
                     f"Modelo: {row['selected_model']} (MAPE {row['best_mape']:.1f}%)", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    df_res = pd.read_csv(RESULTS_PATH)
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        detail = json.load(f)

    chart_mape_by_model(df_res, CHARTS_DIR / "grafico_comparacion_mape.png")
    chart_selected_model(df_res, CHARTS_DIR / "grafico_distribucion_modelos.png")
    chart_forecast_vs_real(df_res, detail, CHARTS_DIR / "grafico_forecast_vs_real.png")

    print(f"Graficos generados en {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
