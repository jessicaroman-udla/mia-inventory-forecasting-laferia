"""
Algoritmo Genetico (DEAP) para balanceo de transferencias inter-sucursales
Proyecto: Sistema inteligente de gestion de inventarios - Comercial La Feria
Seccion 7.2: "Algoritmo genetico desarrollado con DEAP para el balanceo de
inventarios entre sucursales"

Logica: para cada producto se calculan dias de inventario por sucursal
(stock_actual / demanda_diaria_sucursal). Las sucursales con exceso (muchos
dias de cobertura) pueden transferir hacia sucursales con deficit (pocos
dias de cobertura), respetando las rutas de transferencia priorizadas ya
definidas en inventario.almacenes (prioridad_transferencia) y sin superar
el stock disponible de la sucursal origen.

El algoritmo genetico busca el plan de transferencias (cuanto mover en cada
ruta origen->destino) que minimiza una funcion de costo compuesta por:
  - Penalizacion por dias de inventario fuera del rango objetivo [7, 30] dias
    (parametros dias_seguridad y dias_exceso de inventario.almacenes)
  - Penalizacion por cada transferencia realizada (costo logistico fijo)
  - Penalizacion fuerte si se transfiere mas stock del disponible

Datos reales usados: inventario.stock_global (stock por producto/almacen),
inventario.almacenes (rutas de prioridad y umbrales de dias) y ventas
recientes por almacen (para estimar demanda diaria por sucursal).
"""
import json
import random
from pathlib import Path

import numpy as np
from deap import base, creator, tools, algorithms

random.seed(42)
np.random.seed(42)

ROOT_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------
with open(ROOT_DIR / "stock_data.json", encoding="utf-8") as f:
    stock = json.load(f)
with open(ROOT_DIR / "warehouse_sale.json", encoding="utf-8") as f:
    sales_by_warehouse = json.load(f)
with open(ROOT_DIR / "warehouses.json", encoding="utf-8") as f:
    warehouses_info = json.load(f)

WAREHOUSES = [w["code"] for w in warehouses_info]
SAFETY_DAYS = 7      # dias_seguridad en inventario.almacenes
EXCESS_DAYS = 30      # dias_exceso en inventario.almacenes
WINDOW_DAYS = 180     # ventana usada para estimar venta_almacen (6 meses)

# Rutas de transferencia permitidas: (origen, destino) segun prioridad_transferencia
# NOTA: se excluyen las rutas hacia/desde ESMAL01 para esta corrida, ya que
# el analisis de warehouse_sale.json muestra 0 unidades vendidas en los
# ultimos 6 meses para los 15 productos categoria A en esa sucursal. Enviar
# inventario de productos de alta rotacion a un punto sin demanda historica
# constituiria inventario muerto; esto deberia validarse con el area
# comercial (podria tratarse de una sucursal nueva, en remodelacion, o
# especializada en otras categorias) antes de habilitar la ruta.
routes = []
for w in warehouses_info:
    origin = w["code"]
    for destination in w["transfer_priority"]:
        if "ESMAL01" in (origin, destination):
            continue
        routes.append((origin, destination))

print(f"Enabled transfer routes ({len(routes)}):")
for o, d in routes:
    print(f"  {o} -> {d}")

# ---------------------------------------------------------------
# 2. Calculo de dias de inventario por producto y sucursal
# ---------------------------------------------------------------
def daily_demand_warehouse(product, warehouse):
    return sales_by_warehouse[product].get(warehouse, 0.0) / WINDOW_DAYS

def inventory_days(product, warehouse):
    dd = daily_demand_warehouse(product, warehouse)
    st = stock[product].get(warehouse, 0.0)
    if dd <= 0.01:
        return 999.0 if st > 0 else 0.0
    return st / dd

# ---------------------------------------------------------------
# 3. Definicion del problema de optimizacion (por producto)
# ---------------------------------------------------------------
FIXED_TRANSFER_COST = 5.0    # penalizacion por cada transferencia activada
OUT_OF_RANGE_WEIGHT = 1.0
INFEASIBILITY_WEIGHT = 1000.0  # penalizacion por transferir mas de lo disponible


def evaluate_plan(individual, product, product_routes):
    """individual: lista de cantidades a transferir, una por ruta habilitada."""
    simulated_stock = dict(stock[product])
    penalty = 0.0
    n_transfers = 0

    # Aplicar transferencias en el orden de las rutas (con chequeo de disponibilidad)
    for quantity, (origin, destination) in zip(individual, product_routes):
        quantity = max(0.0, quantity)
        available = simulated_stock.get(origin, 0.0)
        if quantity > available:
            penalty += INFEASIBILITY_WEIGHT * (quantity - available)
            quantity = available
        if quantity > 1e-6:
            n_transfers += 1
            simulated_stock[origin] = simulated_stock.get(origin, 0.0) - quantity
            simulated_stock[destination] = simulated_stock.get(destination, 0.0) + quantity

    # Penalizar dias de inventario fuera del rango objetivo [SAFETY_DAYS, EXCESS_DAYS]
    for warehouse in WAREHOUSES:
        dd = daily_demand_warehouse(product, warehouse)
        st = simulated_stock.get(warehouse, 0.0)
        if dd > 0.01:
            days = st / dd
            if days < SAFETY_DAYS:
                penalty += OUT_OF_RANGE_WEIGHT * (SAFETY_DAYS - days) ** 2
            elif days > EXCESS_DAYS:
                penalty += OUT_OF_RANGE_WEIGHT * (days - EXCESS_DAYS) ** 2
        else:
            # Sucursal sin demanda historica (ej. almacen dormido/nuevo):
            # cualquier stock alli es inventario muerto -> penalizar directamente
            # en proporcion al stock (no se puede expresar en "dias" porque dd=0)
            penalty += OUT_OF_RANGE_WEIGHT * (st / 50.0) ** 2

    penalty += FIXED_TRANSFER_COST * n_transfers
    return (penalty,)


# ---------------------------------------------------------------
# 4. Configuracion DEAP (algoritmo genetico)
# ---------------------------------------------------------------
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)


def optimize_product_transfers(product):
    product_routes = routes  # mismas rutas para todos los productos
    n_genes = len(product_routes)
    max_stock = max(stock[product].values()) if stock[product] else 1000.0

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, max_stock * 0.3)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_plan, product=product, product_routes=product_routes)
    toolbox.register("mate", tools.cxBlend, alpha=0.4)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=max_stock * 0.05, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=60)
    # Un individuo "sin transferencias" como referencia (baseline)
    pop[0] = creator.Individual([0.0] * n_genes)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(
        pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=40,
        stats=stats, halloffame=hof, verbose=False,
    )

    best = hof[0]
    # Recalcular resultado final (stock resultante y transferencias efectivas)
    simulated_stock = dict(stock[product])
    plan = []
    for quantity, (origin, destination) in zip(best, product_routes):
        quantity = max(0.0, min(quantity, simulated_stock.get(origin, 0.0)))
        if quantity > 1.0:  # ignorar transferencias insignificantes
            simulated_stock[origin] -= quantity
            simulated_stock[destination] = simulated_stock.get(destination, 0.0) + quantity
            plan.append({"origin": origin, "destination": destination, "quantity": round(quantity, 1)})
    return {
        "product": product,
        "final_fitness": best.fitness.values[0],
        "transfer_plan": plan,
        "stock_before": {k: round(v, 1) for k, v in stock[product].items()},
        "stock_after": {k: round(v, 1) for k, v in simulated_stock.items()},
        "inventory_days_before": {w: round(inventory_days(product, w), 1) for w in WAREHOUSES},
        "inventory_days_after": {
            w: round((simulated_stock.get(w, 0.0) / daily_demand_warehouse(product, w))
                      if daily_demand_warehouse(product, w) > 0.01 else 0.0, 1)
            for w in WAREHOUSES
        },
    }


# ---------------------------------------------------------------
# 5. Ejecutar para todos los productos
# ---------------------------------------------------------------
results = []
for product in stock.keys():
    print(f"\nOptimizing transfers for {product} ...")
    r = optimize_product_transfers(product)
    results.append(r)
    if r["transfer_plan"]:
        print(f"  Suggested plan ({len(r['transfer_plan'])} transfers):")
        for t in r["transfer_plan"]:
            print(f"    {t['origin']} -> {t['destination']}: {t['quantity']:,.0f} units")
    else:
        print("  No transfers needed (inventory already balanced)")

with open(ROOT_DIR / "resultado_ga_transferencias.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------
# 6. Resumen
# ---------------------------------------------------------------
total_transfers = sum(len(r["transfer_plan"]) for r in results)
products_with_transfer = sum(1 for r in results if r["transfer_plan"])
print("\n" + "=" * 80)
print(f"SUMMARY: {products_with_transfer} of {len(results)} products require transfers")
print(f"Total suggested movements: {total_transfers}")
print(f"Result saved to {ROOT_DIR / 'resultado_ga_transferencias.json'}")
