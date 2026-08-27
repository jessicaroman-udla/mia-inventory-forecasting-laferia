"""
Optimizador MILP (Programacion Lineal Entera Mixta) con PuLP - track de referencia.
Proyecto: Sistema inteligente de gestion de inventarios - Comercial La Feria
Seccion 7.2: calculo de cantidades optimas de pedido y punto de reorden, sujeto
a restricciones de capital, lead time y capacidad de almacenamiento.

Version simple (sin BD activa): trabaja sobre los 15 productos categoria A de
mayor venta a nivel nacional, usando los JSON que genera extract_data.py. Para
la version por producto-sucursal ver optimizacion_milp_piloto.py.

Entrada:  data/stock_data.json, data/prices.json, data/demand_statistics.json
Salida:   data/resultado_milp_reorden.json

SUPUESTOS DOCUMENTADOS (verificar/ajustar con datos reales de la empresa):
  - Lead time: el campo lead_time de inventario.productos esta en 0 para todos
    los productos de la muestra (no poblado). Se asume el valor de referencia de
    PRODUCTOS NACIONALES de la seccion 1.1 del capstone: 10 dias (rango 7-15).
  - Costo unitario: la columna "costo" de inventario.ventas es inconsistente con
    "precio" (hallazgo de calidad de datos). Se aproxima como el 60% del precio
    de venta promedio (margen tipico de abarrotes); reemplazar por el costo real
    de compra (BP1.Price en SAP) antes de produccion.
  - Nivel de servicio objetivo: 95% (z = 1.65).
  - Presupuesto de capital y capacidad de almacenamiento: parametros
    configurables abajo, a ajustar con el area financiera.
"""
import json
import math
from pathlib import Path

import pulp

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

STOCK_PATH = DATA_DIR / "stock_data.json"
PRICES_PATH = DATA_DIR / "prices.json"
DEMAND_PATH = DATA_DIR / "demand_statistics.json"
OUTPUT_PATH = DATA_DIR / "resultado_milp_reorden.json"

# ---------------------------------------------------------------
# Parametros generales (ajustables)
# ---------------------------------------------------------------
LEAD_TIME_DAYS = 10            # supuesto: producto nacional (seccion 1.1)
SERVICE_LEVEL_Z = 1.65        # 95% de nivel de servicio
DAYS_PER_WEEK = 7
FIXED_ORDER_COST = 15.0       # USD por orden de compra colocada (estimado)
ANNUAL_HOLDING_RATE = 0.20    # 20% anual del valor del inventario (estandar retail)
CAPITAL_BUDGET = 40000.0      # USD disponibles para reposicion en el ciclo actual
WAREHOUSE_CAPACITY_UNITS = 400000  # unidades totales que pueden recibirse en el ciclo


def load_inputs():
    with open(STOCK_PATH, encoding="utf-8") as f:
        stock = json.load(f)
    with open(PRICES_PATH, encoding="utf-8") as f:
        prices = json.load(f)
    with open(DEMAND_PATH, encoding="utf-8") as f:
        demand = json.load(f)
    return stock, prices, demand


def compute_reorder_points(products, demand, prices):
    """ROP = demanda_diaria * lead_time + z * std_diaria * sqrt(lead_time)."""
    daily_demand = {p: demand[p]["weekly_mean"] / DAYS_PER_WEEK for p in products}
    daily_std = {p: demand[p]["weekly_std"] / math.sqrt(DAYS_PER_WEEK) for p in products}
    unit_cost = {p: prices[p] * 0.60 for p in products}

    safety_stock = {
        p: SERVICE_LEVEL_Z * daily_std[p] * math.sqrt(LEAD_TIME_DAYS) for p in products
    }
    reorder_point = {
        p: daily_demand[p] * LEAD_TIME_DAYS + safety_stock[p] for p in products
    }
    return unit_cost, safety_stock, reorder_point


def solve_milp(products, total_stock, unit_cost, reorder_point):
    """
    Variables:  q[p] >= 0 (cantidad a pedir), y[p] in {0,1} (1 si se ordena p).
    Objetivo:   minimizar costo fijo de ordenar + costo de mantener inventario.
    Restricciones: enlace big-M q-y, cobertura del deficit frente al ROP,
                   presupuesto de capital, capacidad de almacenamiento.
    """
    prob = pulp.LpProblem("Reabastecimiento_ComercialLaFeria", pulp.LpMinimize)

    q = {p: pulp.LpVariable(f"q_{p}", lowBound=0) for p in products}
    y = {p: pulp.LpVariable(f"y_{p}", cat="Binary") for p in products}

    big_m = {p: max(reorder_point[p] * 4, 10000) for p in products}
    daily_holding_cost = {p: unit_cost[p] * ANNUAL_HOLDING_RATE / 365 for p in products}

    prob += pulp.lpSum(
        FIXED_ORDER_COST * y[p]
        + daily_holding_cost[p] * (total_stock[p] + q[p] / 2) * DAYS_PER_WEEK
        for p in products
    ), "Total_cost"

    for p in products:
        prob += q[p] <= big_m[p] * y[p], f"link_{p}"
        deficit = max(reorder_point[p] - total_stock[p], 0)
        if deficit > 0:
            prob += q[p] >= deficit * y[p] - big_m[p] * (1 - y[p]), f"cover_deficit_{p}"
            prob += y[p] == 1, f"force_order_{p}"

    prob += pulp.lpSum(q[p] * unit_cost[p] for p in products) <= CAPITAL_BUDGET, "capital"
    prob += pulp.lpSum(q[p] for p in products) <= WAREHOUSE_CAPACITY_UNITS, "capacity"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return prob, status, q, y


def main():
    DATA_DIR.mkdir(exist_ok=True)
    stock, prices, demand = load_inputs()

    products = list(stock.keys())
    total_stock = {p: sum(stock[p].values()) for p in products}
    unit_cost, safety_stock, reorder_point = compute_reorder_points(products, demand, prices)

    print("=" * 100)
    print(f"{'Producto':16s} {'Stock actual':>14s} {'Punto reorden':>14s} "
          f"{'Stock seguridad':>15s} {'Ordenar?':>10s}")
    print("=" * 100)
    for p in products:
        needs_order = "SI" if total_stock[p] < reorder_point[p] else "no"
        print(f"{p:16s} {total_stock[p]:14,.0f} {reorder_point[p]:14,.0f} "
              f"{safety_stock[p]:15,.0f} {needs_order:>10s}")

    prob, status, q, y = solve_milp(products, total_stock, unit_cost, reorder_point)

    print("\n" + "=" * 100)
    print(f"Estado de la solucion: {pulp.LpStatus[status]}")
    print(f"Costo total optimizado: ${pulp.value(prob.objective):,.2f}")
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
            print(f"  -> ORDEN {p:16s}: {qty:10,.0f} unidades  (costo ${order_cost:,.2f})")

    capital_used = sum(r["order_cost_usd"] for r in results)
    units_ordered = sum(r["order_quantity"] for r in results)
    print(f"\nCapital usado: ${capital_used:,.2f} de ${CAPITAL_BUDGET:,.2f} "
          f"({capital_used / CAPITAL_BUDGET * 100:.1f}%)")
    print(f"Unidades a recibir: {units_ordered:,.0f} de {WAREHOUSE_CAPACITY_UNITS:,.0f} "
          f"({units_ordered / WAREHOUSE_CAPACITY_UNITS * 100:.1f}%)")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
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

    print(f"\nResultado guardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
