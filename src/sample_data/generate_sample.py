"""
generate_sample.py

Genera un dataset de ejemplo SINTETICO (datos ficticios, sin relacion con
Comercial La Feria) para poder ejecutar el pipeline completo SIN base de datos
ni credenciales -- pensado para que el tutor / evaluador reproduzca los
resultados en su maquina.

Reemplaza el paso `python src/extraction/extract_data.py` (que necesita
PostgreSQL) escribiendo los mismos archivos de entrada en data/:

    data/parsed.json              (series de venta semanal por producto)
    data/abc_classification.json  (clasificacion ABC)
    data/stock_data.json          (stock por producto y sucursal)
    data/prices.json              (precio de venta promedio)
    data/demand_statistics.json   (media y desviacion de demanda semanal)
    data/warehouse_sale.json      (ventas por sucursal)
    data/warehouses.json          (sucursales y rutas de transferencia)

Determinista (semilla fija): dos corridas producen exactamente los mismos
archivos.

Uso:
    python src/sample_data/generate_sample.py
"""
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

SEED = 42
N_WEEKS = 90
FIRST_MONDAY = date(2024, 4, 1)

WAREHOUSES = [
    {"code": "MAYVE01", "name": "MAYORISTA",  "safety_days": 7, "transfer_priority": ["MATVE01", "CHOVE01"]},
    {"code": "MATVE01", "name": "MATRIZ",     "safety_days": 7, "transfer_priority": ["MAYVE01", "ESMVE01"]},
    {"code": "CHOVE01", "name": "SUCURSAL-3", "safety_days": 7, "transfer_priority": ["MAYVE01", "MATVE01"]},
    {"code": "ESMVE01", "name": "SUCURSAL-4", "safety_days": 7, "transfer_priority": ["MATVE01", "MAYVE01"]},
]

# (codigo, nombre, categoria, nivel_base_semanal, amplitud_estacional, prob_semana_cero)
PRODUCTS = [
    ("SKU-A001", "ACEITE GIRASOL 1L (ejemplo)",       "A", 480, 0.35, 0.00),
    ("SKU-A002", "ARROZ BLANCO 5KG (ejemplo)",         "A", 610, 0.25, 0.00),
    ("SKU-A003", "AZUCAR MORENA 2KG (ejemplo)",        "A", 350, 0.20, 0.00),
    ("SKU-A004", "DETERGENTE POLVO 900G (ejemplo)",    "A", 275, 0.30, 0.00),
    ("SKU-A005", "LECHE ENTERA UHT 1L (ejemplo)",      "A", 520, 0.15, 0.00),
    ("SKU-B001", "ATUN LOMITO LATA 170G (ejemplo)",    "B",  95, 0.40, 0.05),
    ("SKU-B002", "SHAMPOO ANTICASPA 375ML (ejemplo)",  "B",  70, 0.35, 0.08),
    ("SKU-B003", "GALLETA VAINILLA 380G (ejemplo)",    "B",  60, 0.45, 0.10),
    ("SKU-B004", "MAYONESA POTE 500G (ejemplo)",       "B",  48, 0.30, 0.12),
    ("SKU-C001", "VELA AROMATICA #20 (ejemplo)",       "C",  14, 0.50, 0.30),
    ("SKU-C002", "ESPONJA MIXTA X3 (ejemplo)",         "C",   9, 0.40, 0.40),
    ("SKU-C003", "PILAS AA BLISTER X4 (ejemplo)",      "C",   6, 0.35, 0.45),
]

CATEGORIES_TO_FORECAST = ("A", "B")


def weeks():
    return [FIRST_MONDAY + timedelta(weeks=i) for i in range(N_WEEKS)]


def make_series(rng, base, seasonal_amp, p_zero):
    """Serie semanal = nivel * (1 + tendencia) * estacionalidad + ruido, con ceros ocasionales."""
    out = []
    for i in range(N_WEEKS):
        trend = 1 + 0.0025 * i                      # leve crecimiento
        seasonal = 1 + seasonal_amp * math.sin(2 * math.pi * (i % 52) / 52)
        noise = rng.gauss(1.0, 0.18)
        units = base * trend * seasonal * max(noise, 0.05)
        if rng.random() < p_zero:
            units = 0.0
        out.append(round(max(units, 0.0), 2))
    return out


def build():
    rng = random.Random(SEED)
    all_weeks = weeks()
    week_str = [d.strftime("%Y-%m-%d") for d in all_weeks]

    parsed_rows = []
    prices = {}
    demand_stats = {}
    stock_data = {}
    warehouse_sale = {}
    abc_products = []

    # Reparto fijo de cada producto entre sucursales (participacion)
    for code, name, cat, base, amp, p_zero in PRODUCTS:
        serie = make_series(rng, base, amp, p_zero)
        unit_price = round(rng.uniform(0.8, 12.0), 2)

        parsed_rows.append({
            "product_code": code,
            "product_name": name,
            "series": [
                {"week": w, "units": u, "value": round(u * unit_price, 2)}
                for w, u in zip(week_str, serie)
            ],
        })

        mean = sum(serie) / len(serie)
        var = sum((x - mean) ** 2 for x in serie) / len(serie)
        prices[code] = unit_price
        demand_stats[code] = {"name": name, "weekly_mean": round(mean, 3),
                              "weekly_std": round(math.sqrt(var), 3)}

        # Participacion por sucursal (suma 1), estable por producto
        shares = [rng.uniform(0.1, 1.0) for _ in WAREHOUSES]
        s = sum(shares)
        shares = [x / s for x in shares]
        recent_weekly = sum(serie[-8:]) / 8

        stock_data[code] = {}
        warehouse_sale[code] = {}
        for wh, share in zip(WAREHOUSES, shares):
            # ventas por sucursal ~ participacion * total reciente (26 semanas)
            warehouse_sale[code][wh["code"]] = round(share * sum(serie[-26:]), 2)
            # stock actual ~ entre 0.5 y 6 semanas de cobertura de esa sucursal
            cover_weeks = rng.uniform(0.5, 6.0)
            stock_data[code][wh["code"]] = round(share * recent_weekly * cover_weeks, 1)

        abc_products.append({
            "product_code": code, "product_name": name,
            "sales_value": round(mean * unit_price * N_WEEKS, 2),
            "category": cat,
        })

    # Recalcular pct acumulado para abc_classification (orden por valor desc)
    abc_products.sort(key=lambda p: -p["sales_value"])
    total_val = sum(p["sales_value"] for p in abc_products)
    acc = 0.0
    for p in abc_products:
        acc += p["sales_value"]
        p["pct_of_total"] = round(p["sales_value"] / total_val, 6)
        p["cumulative_pct"] = round(acc / total_val, 6)

    counts = {c: sum(1 for p in abc_products if p["category"] == c) for c in ("A", "B", "C")}

    parsed = {
        "columns": ["product_code", "product_name", "series"],
        "row_count": len(parsed_rows),
        "rows": parsed_rows,
    }
    abc = {
        "ranking_start_date": FIRST_MONDAY.strftime("%Y-%m-%d"),
        "threshold_a": 0.80, "threshold_b": 0.95,
        "counts": counts, "products": abc_products,
    }
    return parsed, abc, stock_data, prices, demand_stats, warehouse_sale


def write_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  {path.relative_to(ROOT_DIR)}")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    parsed, abc, stock_data, prices, demand_stats, warehouse_sale = build()

    print("Escribiendo dataset de ejemplo (sintetico) en data/:")
    write_json(parsed, DATA_DIR / "parsed.json")
    write_json(abc, DATA_DIR / "abc_classification.json")
    write_json(stock_data, DATA_DIR / "stock_data.json")
    write_json(prices, DATA_DIR / "prices.json")
    write_json(demand_stats, DATA_DIR / "demand_statistics.json")
    write_json(warehouse_sale, DATA_DIR / "warehouse_sale.json")
    write_json(WAREHOUSES, DATA_DIR / "warehouses.json")

    print(f"\nListo: {len(PRODUCTS)} productos, {N_WEEKS} semanas, {len(WAREHOUSES)} sucursales.")
    print("Ahora puedes correr el pipeline sin base de datos (ver README, 'track de referencia').")


if __name__ == "__main__":
    main()
