"""
Extraccion de datos desde PostgreSQL (inventario.*)
Proyecto: Sistema inteligente de gestion de inventarios - Comercial La Feria

Este script se conecta DIRECTAMENTE a la base de datos y genera los
archivos de entrada que consumen train_forecasting.py, milp_reorder.py y
ga_transfers.py. Ejecutarlo primero, siempre que se quiera trabajar
con datos actualizados.

CAMBIO respecto a la version anterior:
  Antes se tomaba un numero fijo (TOP_PRODUCTS_N = 15) de productos por
  valor de venta, sin distinguir A/B/C. Ahora se implementa el modulo de
  categorizacion ABC dinamica comprometido en el alcance del proyecto
  (Seccion 4, "productos de categoria A y B"; Seccion 7, indicador
  "Cobertura de productos con pronostico automatizado: Categorias A y B").

  La clasificacion sigue el metodo clasico de Pareto sobre valor de venta
  acumulado (cantidad * precio), no sobre unidades:
    - Categoria A: productos cuyo valor acumulado cae dentro del primer
      80% del valor total de ventas.
    - Categoria B: el siguiente tramo, hasta 95% acumulado.
    - Categoria C: el 5% restante (cola larga, bajo valor individual).

  Estos umbrales (80/95) son los estandar en la literatura de gestion de
  inventarios (analisis ABC / regla de Pareto) y deben citarse en la
  seccion de metodologia si aun no esta hecho (p. ej. Silver, Pyke &
  Peterson, "Inventory Management and Production Planning and
  Scheduling").

Requisitos: pip install psycopg2-binary pandas python-dotenv

CONFIGURACION DE CREDENCIALES (.env):
  Las credenciales de conexion ya NO estan escritas en este archivo.
  Se leen desde un archivo ".env" en la raiz del proyecto (mismo nivel
  que manage.py / requirements.txt), que NUNCA se sube a Git.

  Pasos:
    1. Copia ".env.example" como ".env":  cp .env.example .env
    2. Rellena los valores reales en ".env"
    3. Verifica que ".env" este en tu .gitignore (ver instrucciones)
"""
import json
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Carga las variables del archivo .env ubicado en la raiz del proyecto
load_dotenv(ROOT_DIR / ".env")


def _get_env(name: str, required: bool = True, default=None):
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(
            f"Falta la variable de entorno '{name}'. "
            f"Revisa que exista un archivo .env en {ROOT_DIR} "
            f"(copia .env.example como .env y rellena los datos reales)."
        )
    return value


# =================================================================
# 1. CONFIGURACION DE CONEXION (leida desde .env, no hardcodeada)
# =================================================================
DB_CONFIG = {
    "host": _get_env("DB_HOST"),
    "port": int(_get_env("DB_PORT")),
    "dbname": _get_env("DB_NAME"),
    "user": _get_env("DB_USER"),
    "password": _get_env("DB_PASSWORD"),
}

RANKING_START_DATE = _get_env("RANKING_START_DATE", required=False, default="2025-06-01")
WAREHOUSE_SALES_START_DATE = _get_env("WAREHOUSE_SALES_START_DATE", required=False, default="2025-12-01")

# Umbrales de acumulado para clasificacion ABC (Pareto estandar)
ABC_THRESHOLD_A = 0.80   # hasta 80% acumulado -> categoria A
ABC_THRESHOLD_B = 0.95   # de 80% a 95% acumulado -> categoria B
                          # > 95% -> categoria C

# Que categorias entran al pipeline de forecasting (segun alcance del proyecto)
CATEGORIES_TO_FORECAST = ("A", "B")

# =================================================================


def connect():
    return psycopg2.connect(**DB_CONFIG)


def get_abc_classification(conn, start_date=RANKING_START_DATE):
    """
    Clasifica TODOS los productos vendidos desde start_date segun su
    participacion acumulada en el valor total de ventas (cantidad*precio).
    Devuelve una lista ordenada de mayor a menor valor, cada item con su
    categoria (A/B/C) y su porcentaje acumulado.
    """
    sql = """
        SELECT codigo_item, nombre_item, SUM(cantidad*precio) as venta_valor
        FROM inventario.ventas
        WHERE fecha >= %s AND tipo = 'Factura'
        GROUP BY codigo_item, nombre_item
        ORDER BY venta_valor DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (start_date,))
        rows = cur.fetchall()

    total_valor = sum(float(r[2]) for r in rows) if rows else 0.0

    resultado = []
    acumulado = 0.0
    for codigo, nombre, valor in rows:
        valor = float(valor)
        acumulado += valor
        pct_acumulado = acumulado / total_valor if total_valor > 0 else 0.0

        if pct_acumulado <= ABC_THRESHOLD_A:
            categoria = "A"
        elif pct_acumulado <= ABC_THRESHOLD_B:
            categoria = "B"
        else:
            categoria = "C"

        resultado.append({
            "product_code": codigo,
            "product_name": nombre,
            "sales_value": valor,
            "pct_of_total": valor / total_valor if total_valor > 0 else 0.0,
            "cumulative_pct": pct_acumulado,
            "category": categoria,
        })

    return resultado


def get_weekly_series(conn, codes):
    sql = """
        SELECT date_trunc('week', fecha)::date as semana,
               SUM(cantidad) as unidades, SUM(cantidad*precio) as valor
        FROM inventario.ventas
        WHERE codigo_item = %s AND tipo = 'Factura'
        GROUP BY 1 ORDER BY 1
    """
    result = {}
    with conn.cursor() as cur:
        for code in codes:
            cur.execute(sql, (code,))
            rows = cur.fetchall()
            result[code] = [
                {"week": r[0].strftime("%Y-%m-%d"), "units": float(r[1]), "value": float(r[2])}
                for r in rows
            ]
    return result


def get_stock(conn, codes):
    sql = """
        SELECT codigo_item, almacen, stock
        FROM inventario.stock_global
        WHERE codigo_item = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (codes,))
        rows = cur.fetchall()
    stock = {c: {} for c in codes}
    for code, warehouse, value in rows:
        stock[code][warehouse] = float(value)
    return stock


def get_prices(conn, codes):
    sql = """
        SELECT codigo_item, AVG(precio) as precio_promedio
        FROM inventario.ventas
        WHERE fecha >= %s AND tipo = 'Factura' AND codigo_item = ANY(%s)
        GROUP BY codigo_item
    """
    with conn.cursor() as cur:
        cur.execute(sql, (RANKING_START_DATE, codes))
        rows = cur.fetchall()
    return {r[0]: float(r[1]) for r in rows}


def get_sales_by_warehouse(conn, codes):
    sql = """
        SELECT codigo_item, almacen, SUM(cantidad) as unidades
        FROM inventario.ventas
        WHERE fecha >= %s AND tipo = 'Factura' AND codigo_item = ANY(%s)
        GROUP BY codigo_item, almacen
    """
    with conn.cursor() as cur:
        cur.execute(sql, (WAREHOUSE_SALES_START_DATE, codes))
        rows = cur.fetchall()
    result = {c: {} for c in codes}
    for code, warehouse, units in rows:
        result[code][warehouse] = float(units)
    return result


def get_warehouses(conn):
    sql = """
        SELECT codigo, nombre, dias_seguridad, prioridad_transferencia
        FROM inventario.almacenes
        WHERE activo = true
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    result = []
    for code, name, safety_days, priority in rows:
        routes = [r.strip() for r in priority.split(",")] if priority else []
        result.append({
            "code": code, "name": name,
            "safety_days": safety_days or 7,
            "transfer_priority": routes,
        })
    return result


def main():
    print("Connecting to PostgreSQL...")
    conn = connect()
    try:
        print(f"Classifying ALL products by ABC (Pareto sobre valor de venta desde {RANKING_START_DATE})...")
        classification = get_abc_classification(conn)

        counts = {"A": 0, "B": 0, "C": 0}
        for p in classification:
            counts[p["category"]] += 1
        print(f"  Total productos clasificados: {len(classification)}")
        print(f"  Categoria A: {counts['A']}  |  Categoria B: {counts['B']}  |  Categoria C: {counts['C']}")

        with open(DATA_DIR / "abc_classification.json", "w", encoding="utf-8") as f:
            json.dump({
                "ranking_start_date": RANKING_START_DATE,
                "threshold_a": ABC_THRESHOLD_A,
                "threshold_b": ABC_THRESHOLD_B,
                "counts": counts,
                "products": classification,
            }, f, ensure_ascii=False, indent=2)
        print("  Guardado en: data/abc_classification.json")

        # Productos que entran al pipeline de forecasting: categorias A y B
        top = [p for p in classification if p["category"] in CATEGORIES_TO_FORECAST]
        codes = [p["product_code"] for p in top]
        print(f"\nProductos seleccionados para forecasting (categorias {CATEGORIES_TO_FORECAST}): {len(codes)}")

        print("\nExtracting weekly sales series...")
        series = get_weekly_series(conn, codes)
        parsed = {
            "columns": ["product_code", "product_name", "series"],
            "row_count": len(top),
            "rows": [
                {"product_code": p["product_code"], "product_name": p["product_name"], "series": series[p["product_code"]]}
                for p in top
            ],
        }
        with open(DATA_DIR / "parsed.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)

        print("Extracting current stock by warehouse...")
        stock = get_stock(conn, codes)
        with open(DATA_DIR / "stock_data.json", "w", encoding="utf-8") as f:
            json.dump(stock, f, ensure_ascii=False, indent=2)

        print("Extracting average prices...")
        prices = get_prices(conn, codes)
        with open(DATA_DIR / "prices.json", "w", encoding="utf-8") as f:
            json.dump(prices, f, ensure_ascii=False, indent=2)

        print("Computing demand statistics (weekly mean/std dev)...")
        stats = {}
        for p in top:
            vals = [s["units"] for s in series[p["product_code"]]]
            mean = sum(vals) / len(vals) if vals else 0.0
            variance = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0
            stats[p["product_code"]] = {
                "name": p["product_name"],
                "weekly_mean": mean,
                "weekly_std": variance ** 0.5,
            }
        with open(DATA_DIR / "demand_statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print("Extracting sales by warehouse (for the genetic algorithm)...")
        sales_by_warehouse = get_sales_by_warehouse(conn, codes)
        with open(DATA_DIR / "warehouse_sale.json", "w", encoding="utf-8") as f:
            json.dump(sales_by_warehouse, f, ensure_ascii=False, indent=2)

        print("Extracting warehouse configuration and transfer routes...")
        warehouses = get_warehouses(conn)
        with open(DATA_DIR / "warehouses.json", "w", encoding="utf-8") as f:
            json.dump(warehouses, f, ensure_ascii=False, indent=2)

        print("\nDone. Archivos generados en data/:")
        print("  data/abc_classification.json  (clasificacion ABC completa del catalogo)")
        print("  data/parsed.json              (series semanales productos A+B)")
        print("  data/stock_data.json          (stock por producto y sucursal)")
        print("  data/prices.json              (precio de venta promedio por producto)")
        print("  data/demand_statistics.json   (media y desviacion de demanda semanal)")
        print("  data/warehouse_sale.json      (ventas por sucursal - algoritmo genetico)")
        print("  data/warehouses.json          (sucursales y rutas de transferencia)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()