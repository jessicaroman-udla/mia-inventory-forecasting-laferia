"""
MIA - Modelo de Inventario Automatizado
Comercial La Feria HLS CIA LTDA.

Desagrega el pronostico nacional por producto (pronostico_futuro_producto.csv)
hacia demanda por producto-sucursal, repartiendo proporcionalmente segun la
participacion historica de ventas de cada sucursal para ese producto
(ultimos 6 meses, tabla ventas).

Supuesto de diseno documentado: el pronostico se entrena a nivel nacional
agregado (ver train_forecasting_tiered.py); la desagregacion por sucursal
asume que el patron de reparto historico entre sucursales se mantiene
estable en el horizonte de pronostico. Debe declararse como supuesto en
la seccion 7.2 del documento de tesis.

Genera forecast_output.csv con las columnas que consume
optimizacion_milp_piloto.py: codigo_item, almacen, categoria,
modelo_ganador, demanda_pronosticada, mape.

No modifica el pipeline principal -> script standalone.
"""
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

ESQUEMA_BD = "inventario"

ROOT_DIR = Path(__file__).resolve().parents[2] if "__file__" in dir() else Path(".")

PRONOSTICO_PRODUCTO_PATH = ROOT_DIR / "pronostico_futuro_producto.csv"
RESULTS_A_PATH = ROOT_DIR / "model_comparison_results_A.csv"
RESULTS_B_PATH = ROOT_DIR / "model_comparison_results_B.csv"
OUTPUT_PATH = ROOT_DIR / "forecast_output.csv"

# Si una sucursal nunca vendio el producto en la ventana historica, no se
# le asigna demanda proyectada (queda fuera del reparto, no en cero forzado).
VENTANA_HISTORICA = "6 months"


def conectar():
    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(conn_str)


def cargar_participacion_por_sucursal(engine):
    """Participacion (%) de cada sucursal en las ventas historicas de cada producto."""
    print("Calculando participacion historica por sucursal-producto...")
    df = pd.read_sql(
        text(f"""
            SELECT codigo_item, almacen, SUM(cantidad) AS unidades_vendidas
            FROM {ESQUEMA_BD}.ventas
            WHERE fecha >= CURRENT_DATE - INTERVAL '{VENTANA_HISTORICA}'
              AND cantidad > 0
            GROUP BY codigo_item, almacen
        """),
        engine,
    )
    total_por_producto = df.groupby("codigo_item")["unidades_vendidas"].transform("sum")
    df["participacion"] = df["unidades_vendidas"] / total_por_producto
    return df[["codigo_item", "almacen", "participacion"]]


def cargar_mape_por_producto():
    mape_a = pd.read_csv(RESULTS_A_PATH)[["product_code", "best_mape"]] \
        if RESULTS_A_PATH.exists() else pd.DataFrame(columns=["product_code", "best_mape"])
    mape_b = pd.read_csv(RESULTS_B_PATH)[["product_code", "best_mape"]] \
        if RESULTS_B_PATH.exists() else pd.DataFrame(columns=["product_code", "best_mape"])
    return pd.concat([mape_a, mape_b], ignore_index=True).rename(
        columns={"product_code": "codigo_item", "best_mape": "mape"}
    )


def main():
    engine = conectar()

    pronostico = pd.read_csv(PRONOSTICO_PRODUCTO_PATH)
    pronostico = pronostico[pronostico["status"] == "OK"].copy()
    pronostico = pronostico.rename(columns={"product_code": "codigo_item"})

    participacion = cargar_participacion_por_sucursal(engine)
    mape = cargar_mape_por_producto()

    print(f"Productos con pronostico: {len(pronostico)}")

    df = pronostico.merge(participacion, on="codigo_item", how="inner")
    sin_participacion = set(pronostico["codigo_item"]) - set(df["codigo_item"])
    if sin_participacion:
        print(f"AVISO: {len(sin_participacion)} productos sin historial de ventas por "
              f"sucursal en los ultimos {VENTANA_HISTORICA} — quedan fuera del reparto.")

    df["demanda_pronosticada"] = df["demanda_pronosticada_total"] * df["participacion"]
    df = df.merge(mape, on="codigo_item", how="left")

    salida = df[["codigo_item", "almacen", "categoria", "modelo_ganador",
                 "demanda_pronosticada", "mape"]].copy()

    salida.to_csv(OUTPUT_PATH, index=False)
    print(f"\nListo. {len(salida)} registros producto-sucursal guardados en {OUTPUT_PATH}")
    print(f"Productos unicos: {salida['codigo_item'].nunique()}  |  "
          f"Sucursales: {salida['almacen'].nunique()}")


if __name__ == "__main__":
    main()
