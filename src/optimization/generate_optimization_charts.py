"""
Graficos del MILP (track de referencia) y del algoritmo genetico de transferencias.
Proyecto: Sistema inteligente de gestion de inventarios - Comercial La Feria

Entrada:  data/resultado_milp_reorden.json      (milp_reorder.py)
          data/resultado_ga_transferencias.json (ga_transfers.py)
Salida:   data/charts/grafico_plan_compra_milp.png
          data/charts/grafico_dias_inventario_ga.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CHARTS_DIR = DATA_DIR / "charts"

MILP_PATH = DATA_DIR / "resultado_milp_reorden.json"
GA_PATH = DATA_DIR / "resultado_ga_transferencias.json"


def chart_milp_plan(milp, out_path):
    res_sorted = sorted(milp["results"], key=lambda r: -r["order_quantity"])
    products = [r["product"] for r in res_sorted]
    quantities = [r["order_quantity"] for r in res_sorted]
    colors = ["#C44E52" if r["order_placed"] else "#8C8C8C" for r in res_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(products, quantities, color=colors)
    ax.set_xlabel("Unidades a ordenar")
    ax.set_title("Plan optimo de reabastecimiento (MILP)\n"
                 f"Capital usado: ${milp['capital_used_usd']:,.0f} de "
                 f"${milp['parameters']['capital_budget']:,.0f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def chart_ga_inventory_days(ga, out_path):
    products_ga = [r["product"] for r in ga]
    days_before, days_after = [], []
    for r in ga:
        before = [v for k, v in r["inventory_days_before"].items() if k != "ESMAL01" and v < 999]
        after = [v for k, v in r["inventory_days_after"].items() if k != "ESMAL01"]
        days_before.append(np.mean(before) if before else 0)
        days_after.append(np.mean(after) if after else 0)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(products_ga))
    w = 0.35
    ax.bar(x - w / 2, days_before, width=w, label="Antes de transferencias", color="#C44E52")
    ax.bar(x + w / 2, days_after, width=w, label="Despues (algoritmo genetico)", color="#55A868")
    ax.axhspan(7, 30, alpha=0.1, color="green", label="Rango objetivo (7-30 dias)")
    ax.set_xticks(x)
    ax.set_xticklabels(products_ga, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Dias de inventario (promedio entre sucursales)")
    ax.set_title("Efecto del algoritmo genetico de transferencias sobre los dias de cobertura")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MILP_PATH, encoding="utf-8") as f:
        milp = json.load(f)
    with open(GA_PATH, encoding="utf-8") as f:
        ga = json.load(f)

    chart_milp_plan(milp, CHARTS_DIR / "grafico_plan_compra_milp.png")
    chart_ga_inventory_days(ga, CHARTS_DIR / "grafico_dias_inventario_ga.png")
    print(f"Graficos generados en {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
