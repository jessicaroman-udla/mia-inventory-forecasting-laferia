import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]

with open(ROOT_DIR / "resultado_milp_reorden.json", encoding="utf-8") as f:
    milp = json.load(f)
with open(ROOT_DIR / "resultado_ga_transferencias.json", encoding="utf-8") as f:
    ga = json.load(f)

# ---------- Grafico 1: Plan de compra (MILP) ----------
fig, ax = plt.subplots(figsize=(10, 5))
res = [r for r in milp["results"]]
res_sorted = sorted(res, key=lambda r: -r["order_quantity"])
products = [r["product"] for r in res_sorted]
quantities = [r["order_quantity"] for r in res_sorted]
colors = ["#C44E52" if r["order_placed"] else "#8C8C8C" for r in res_sorted]
ax.barh(products, quantities, color=colors)
ax.set_xlabel("Units to order")
ax.set_title(f"Optimal replenishment plan (MILP)\nCapital used: ${milp['capital_used_usd']:,.0f} of ${milp['parameters']['capital_budget']:,.0f}")
plt.tight_layout()
plt.savefig(ROOT_DIR / "grafico_plan_compra_milp.png", dpi=150)
plt.close()

# ---------- Grafico 2: Dias de inventario antes/despues por producto (promedio entre sucursales activas) ----------
products_ga = [r["product"] for r in ga]
days_before = []
days_after = []
for r in ga:
    before = [v for k, v in r["inventory_days_before"].items() if k != "ESMAL01" and v < 999]
    after = [v for k, v in r["inventory_days_after"].items() if k != "ESMAL01"]
    days_before.append(np.mean(before) if before else 0)
    days_after.append(np.mean(after) if after else 0)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(products_ga))
w = 0.35
ax.bar(x - w/2, days_before, width=w, label="Before transfers", color="#C44E52")
ax.bar(x + w/2, days_after, width=w, label="After transfers (GA)", color="#55A868")
ax.axhspan(7, 30, alpha=0.1, color="green", label="Target range (7-30 days)")
ax.set_xticks(x)
ax.set_xticklabels(products_ga, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Days of inventory (average across warehouses)")
ax.set_title("Effect of the transfer genetic algorithm on coverage days")
ax.legend()
plt.tight_layout()
plt.savefig(ROOT_DIR / "grafico_dias_inventario_ga.png", dpi=150)
plt.close()

print("Charts generated: grafico_plan_compra_milp.png, grafico_dias_inventario_ga.png")
