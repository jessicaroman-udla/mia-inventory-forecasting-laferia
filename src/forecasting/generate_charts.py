import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]

with open(ROOT_DIR / "prediction_details.json", "r", encoding="utf-8") as f:
    detail = json.load(f)

df_res = pd.read_csv(ROOT_DIR / "model_comparison_results.csv")

# ---------- 1. Grafico de barras: MAPE por modelo y producto ----------
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(df_res))
w = 0.25
ax.bar(x - w, df_res["ARIMA_MAPE"], width=w, label="ARIMA", color="#4C72B0")
ax.bar(x, df_res["Prophet_MAPE"], width=w, label="Prophet", color="#DD8452")
ax.bar(x + w, df_res["LSTM_MAPE"], width=w, label="LSTM", color="#55A868")
ax.axhline(15, color="red", linestyle="--", linewidth=1, label="Project threshold (15%)")
ax.set_xticks(x)
ax.set_xticklabels(df_res["product_code"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("MAPE (%)")
ax.set_title("Error comparison (MAPE) by model and product — Comercial La Feria")
ax.set_ylim(0, 60)  # recortamos para visualizar bien la mayoria (el atipico se anota aparte)
ax.legend()
plt.tight_layout()
plt.savefig(ROOT_DIR / "grafico_comparacion_mape.png", dpi=150)
plt.close()

# ---------- 2. Distribucion de modelo seleccionado ----------
fig, ax = plt.subplots(figsize=(5, 5))
counts = df_res["selected_model"].value_counts()
colors = {"ARIMA": "#4C72B0", "Prophet": "#DD8452", "LSTM": "#55A868"}
ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
       colors=[colors[k] for k in counts.index], startangle=90)
ax.set_title("Automatically selected model\n(15 category-A products)")
plt.tight_layout()
plt.savefig(ROOT_DIR / "grafico_distribucion_modelos.png", dpi=150)
plt.close()

# ---------- 3. Forecast vs Real para 4 productos representativos ----------
selection = ["A111010-0022", "A111013-0001", "A102012-0070", "C114053-0005"]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, code in zip(axes.flat, selection):
    p = detail[code]
    dates = pd.to_datetime(p["test_dates"])
    ax.plot(dates, p["actual"], label="Actual", color="black", marker="o", linewidth=2)
    if "ARIMA" in p:
        ax.plot(dates, p["ARIMA"], label="ARIMA", linestyle="--")
    if "Prophet" in p:
        ax.plot(dates, p["Prophet"], label="Prophet", linestyle="--")
    if "LSTM" in p:
        ax.plot(dates, p["LSTM"], label="LSTM", linestyle="--")
    row = df_res[df_res.product_code == code].iloc[0]
    ax.set_title(f"{code}\n{row['product_name'][:35]}\nSelected model: {row['selected_model']} (MAPE {row['best_mape']:.1f}%)", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(ROOT_DIR / "grafico_forecast_vs_real.png", dpi=150)
plt.close()

print("Charts generated: grafico_comparacion_mape.png, grafico_distribucion_modelos.png, grafico_forecast_vs_real.png")
