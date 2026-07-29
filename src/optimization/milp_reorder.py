"""
Optimizador MILP (Programacion Lineal Entera Mixta) con PuLP
Proyecto: Sistema inteligente de gestion de inventarios - Comercial La Feria
Seccion 7.2: "Optimizador MILP (PuLP) para el calculo de cantidades optimas de
pedido y punto de reorden, sujeto a restricciones de capital, lead time y
capacidad de almacenamiento"

Datos: 15 productos categoria A, stock real (inventario.stock_global),
precio real (inventario.ventas), demanda semanal estimada a partir de las
series historicas usadas en el modulo de forecasting (src/forecasting/train_forecasting.py).

SUPUESTOS DOCUMENTADOS (deben verificarse/ajustarse con datos reales de la empresa):
  - Lead time: el campo lead_time de inventario.productos esta en 0 para
    todos los productos de la muestra (no poblado). Se asume el valor de
    referencia de PRODUCTOS NACIONALES declarado en la seccion 1.1 del
    documento capstone: 10 dias (rango 7-15 dias).
  - Costo unitario: la columna "costo" de inventario.ventas presenta valores
    inconsistentes con "precio" (costo promedio mayor que precio promedio en
    la mayoria de productos), lo cual es en si mismo un hallazgo relevante
    para la seccion de calidad de datos del proyecto. Se aproxima el costo
    unitario como el 60% del precio de venta promedio (margen tipico de
    abarrotes), y debe reemplazarse por el costo real de compra (BP1.Price
    o equivalente en SAP) antes de usar este modelo en produccion.
  - Nivel de servicio objetivo: 95% (z = 1.65), consistente con el enfoque
    de stock de seguridad estadistico de la seccion 7.2.
  - Presupuesto de capital y capacidad de almacenamiento: se definen como
    parametros configurables (ver PARAMETROS) a falta de un valor declarado
    explicitamente por la empresa; deben ajustarse con el area financiera.
"""
import json
import math
from pathlib import Path

import pulp

ROOT_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------
# 0. Parametros generales (ajustables)
# ---------------------------------------------------------------
LEAD_TIME_DAYS = 10          # supuesto: producto nacional (seccion 1.1)
SERVICE_LEVEL_Z = 1.65       # 95% de nivel de servicio
DAYS_PER_WEEK = 7
FIXED_ORDER_COST = 15.0      # USD por orden de compra colocada (estimado)
ANNUAL_HOLDING_RATE = 0.20   # 20% anual del valor del inventario (estimado estandar retail)
CAPITAL_BUDGET = 40000.0     # USD disponibles para reposicion en el ciclo actual
WAREHOUSE_CAPACITY_UNITS = 400000  # unidades totales que pueden recibirse en el ciclo

# ---------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------
with open(ROOT_DIR / "stock_data.json", encoding="utf-8") as f:
    stock = json.load(f)
with open(ROOT_DIR / "prices.json", encoding="utf-8") as f:
    prices = json.load(f)
with open(ROOT_DIR / "demand_statistics.json", encoding="utf-8") as f:
    demand = json.load(f)

products = list(stock.keys())
total_stock = {p: sum(stock[p].values()) for p in products}

daily_demand = {p: demand[p]["weekly_mean"] / DAYS_PER_WEEK for p in products}
daily_std = {p: demand[p]["weekly_std"] / math.sqrt(DAYS_PER_WEEK) for p in products}
unit_cost = {p: prices[p] * 0.60 for p in products}

# ---------------------------------------------------------------
# 2. Punto de reorden y stock de seguridad (formula estadistica estandar)
#    ROP = demanda_diaria * lead_time + stock_seguridad
#    stock_seguridad = z * std_diaria * sqrt(lead_time)
# ---------------------------------------------------------------
safety_stock = {
    p: SERVICE_LEVEL_Z * daily_std[p] * math.sqrt(LEAD_TIME_DAYS) for p in products
}
reorder_point = {
    p: daily_demand[p] * LEAD_TIME_DAYS + safety_stock[p] for p in products
}

print("=" * 100)
print(f"{'Product':16s} {'Current stock':>14s} {'Reorder point':>14s} {'Safety stock':>13s} {'Needs order?':>14s}")
print("=" * 100)
for p in products:
    needs_order = "YES" if total_stock[p] < reorder_point[p] else "no"
    print(f"{p:16s} {total_stock[p]:14,.0f} {reorder_point[p]:14,.0f} {safety_stock[p]:13,.0f} {needs_order:>14s}")

# ---------------------------------------------------------------
# 3. Modelo MILP: cantidad optima de pedido por producto
#    Variables:
#      q[p]   >= 0            cantidad a pedir (continua)
#      y[p]   in {0,1}        1 si se coloca orden para el producto p
#    Objetivo: minimizar costo total = costo fijo de ordenar + costo de
#              mantener inventario promedio (stock_actual + q/2)
#    Restricciones:
#      - Si y[p] = 0  => q[p] = 0           (enlace big-M)
#      - Si stock_actual[p] < ROP[p] => se debe ordenar al menos hasta cubrir ROP
#      - Presupuesto de capital: sum(q[p] * costo_unit[p]) <= PRESUPUESTO_CAPITAL
#      - Capacidad de almacenamiento: sum(q[p]) <= CAPACIDAD_ALMACEN_UNIDADES
# ---------------------------------------------------------------
prob = pulp.LpProblem("Reabastecimiento_ComercialLaFeria", pulp.LpMinimize)

q = {p: pulp.LpVariable(f"q_{p}", lowBound=0) for p in products}
y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in products}

BIG_M = {p: max(reorder_point[p] * 4, 10000) for p in products}
daily_holding_cost = {p: unit_cost[p] * ANNUAL_HOLDING_RATE / 365 for p in products}

# Objetivo
prob += pulp.lpSum(
    FIXED_ORDER_COST * y[p] + daily_holding_cost[p] * (total_stock[p] + q[p] / 2) * DAYS_PER_WEEK
    for p in products
), "Total_cost"

for p in products:
    # Enlace: no se puede pedir sin activar la orden
    prob += q[p] <= BIG_M[p] * y[p], f"link_{p}"
    # Si hay deficit frente al ROP, la orden debe cubrir al menos la brecha
    deficit = max(reorder_point[p] - total_stock[p], 0)
    if deficit > 0:
        prob += q[p] >= deficit * y[p] - BIG_M[p] * (1 - y[p]), f"cover_deficit_{p}"
        prob += y[p] == 1, f"force_order_{p}"  # si hay deficit, es obligatorio ordenar

# Restriccion de capital
prob += pulp.lpSum(q[p] * unit_cost[p] for p in products) <= CAPITAL_BUDGET, "capital"
# Restriccion de capacidad de almacenamiento
prob += pulp.lpSum(q[p] for p in products) <= WAREHOUSE_CAPACITY_UNITS, "capacity"

solver = pulp.PULP_CBC_CMD(msg=0)
status = prob.solve(solver)

print("\n" + "=" * 100)
print(f"Solution status: {pulp.LpStatus[status]}")
print(f"Optimized total cost: ${pulp.value(prob.objective):,.2f}")
print("=" * 100)

results = []
for p in products:
    qty = q[p].value() or 0.0
    order_placed = int(round(y[p].value() or 0))
    order_cost = qty * unit_cost[p]
    results.append({
        "product": p,
        "current_stock": round(total_stock[p], 1),
        "reorder_point": round(reorder_point[p], 1),
        "safety_stock": round(safety_stock[p], 1),
        "order_placed": order_placed,
        "order_quantity": round(qty, 1),
        "order_cost_usd": round(order_cost, 2),
    })
    if order_placed:
        print(f"  -> ORDER {p:16s}: {qty:10,.0f} units  (cost ${order_cost:,.2f})")

capital_used = sum(r["order_cost_usd"] for r in results)
units_ordered = sum(r["order_quantity"] for r in results)
print(f"\nTotal capital used: ${capital_used:,.2f} of ${CAPITAL_BUDGET:,.2f} available ({capital_used/CAPITAL_BUDGET*100:.1f}%)")
print(f"Total units to receive: {units_ordered:,.0f} of {WAREHOUSE_CAPACITY_UNITS:,.0f} capacity ({units_ordered/WAREHOUSE_CAPACITY_UNITS*100:.1f}%)")

with open(ROOT_DIR / "resultado_milp_reorden.json", "w", encoding="utf-8") as f:
    json.dump({
        "parameters": {
            "lead_time_days": LEAD_TIME_DAYS,
            "service_level_z": SERVICE_LEVEL_Z,
            "capital_budget": CAPITAL_BUDGET,
            "warehouse_capacity": WAREHOUSE_CAPACITY_UNITS,
        },
        "solution_status": pulp.LpStatus[status],
        "total_cost_usd": pulp.value(prob.objective),
        "capital_used_usd": capital_used,
        "units_ordered": units_ordered,
        "results": results,
    }, f, ensure_ascii=False, indent=2)

print(f"\nResult saved to {ROOT_DIR / 'resultado_milp_reorden.json'}")
