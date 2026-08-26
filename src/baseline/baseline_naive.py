"""
baseline_naive.py

Modelo baseline (Naive Forecast) para el proyecto MIA - Comercial La Feria.
Punto de referencia mínimo frente al cual se evalúa si los modelos propuestos
(ARIMA, Prophet, LSTM, Holt-Winters) mejoran una estrategia simple de pronóstico.

Lógica del baseline:
    Demanda pronosticada (semana t) = Demanda observada en la última semana
    disponible del conjunto de entrenamiento (naive / random walk sin
    tendencia ni estacionalidad).

Requiere:
    - Conexión de solo lectura a la base PostgreSQL del proyecto (esquema `inventario`).
    - Variables de entorno definidas en un archivo `.env` (ver .env.example).

Salida:
    - resultados_baseline_detalle.csv   -> métricas por producto-sucursal
    - resultados_baseline_resumen.csv   -> medianas por categoría ABC (A/B/C)
    - Impresión en consola del resumen final

Uso:
    python baseline_naive.py
"""

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# 1. Configuración: carga de variables de entorno
# ---------------------------------------------------------------------------
load_dotenv()

DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    sys.exit(
        "ERROR: faltan variables de entorno. Verifica que exista un archivo .env "
        "con DB_USER, DB_PASSWORD y DB_NAME definidos (ver .env.example)."
    )

# Parámetros del experimento (ajusta si tu partición temporal cambia)
SEMANAS_TEST = int(os.environ.get("BASELINE_SEMANAS_TEST", 12))  # ventana de prueba, en semanas
ARCHIVO_DETALLE = "resultados_baseline_detalle.csv"
ARCHIVO_RESUMEN = "resultados_baseline_resumen.csv"

# ---------------------------------------------------------------------------
# 2. Conexión a la base de datos
# ---------------------------------------------------------------------------
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Conectando a la base de datos...")
with engine.connect() as conn:
    fecha_min, fecha_max = conn.execute(
        text(
            """
            SELECT MIN(fecha), MAX(fecha)
            FROM inventario.ventas
            WHERE tipo = 'Factura'
            """
        )
    ).one()

print(f"Rango de fechas disponible en 'ventas': {fecha_min} a {fecha_max}")

# Ventana total de trabajo: se replica el criterio de 13-14 meses del proyecto,
# usando la fecha máxima disponible como ancla.
fecha_max_d = fecha_max if isinstance(fecha_max, date) else fecha_max.date()
inicio_ventana = fecha_max_d - timedelta(days=14 * 30)  # aprox. 14 meses
fecha_corte_test = fecha_max_d - timedelta(weeks=SEMANAS_TEST)

print(f"Ventana de trabajo: {inicio_ventana} a {fecha_max_d}")
print(f"Corte train/test (ventana de prueba = {SEMANAS_TEST} semanas): {fecha_corte_test}")

# ---------------------------------------------------------------------------
# 3. Extracción de ventas semanales agregadas por producto-sucursal
# ---------------------------------------------------------------------------
query_ventas = text(
    """
    SELECT codigo_item, almacen, date_trunc('week', fecha)::date AS semana,
           SUM(cantidad) AS demanda
    FROM inventario.ventas
    WHERE tipo = 'Factura'
      AND fecha BETWEEN :inicio AND :fin
    GROUP BY codigo_item, almacen, date_trunc('week', fecha)
    ORDER BY codigo_item, almacen, semana
    """
)

print("Extrayendo ventas semanales agregadas (puede tardar unos segundos)...")
df = pd.read_sql(query_ventas, engine, params={"inicio": inicio_ventana, "fin": fecha_max_d})
df["semana"] = pd.to_datetime(df["semana"])
print(f"Registros semana-producto-sucursal extraídos: {len(df):,}")

# ---------------------------------------------------------------------------
# 4. Extracción de la clasificación ABC vigente (para reportar por categoría)
# ---------------------------------------------------------------------------
query_clasificacion = text(
    """
    SELECT DISTINCT ON (item_code) item_code, clasificacion_abc
    FROM inventario.analisis_avanzado
    WHERE clasificacion_abc IS NOT NULL
    ORDER BY item_code, fecha_analisis DESC
    """
)
df_clas = pd.read_sql(query_clasificacion, engine)
print(f"Productos con clasificación ABC vigente: {len(df_clas):,}")

# ---------------------------------------------------------------------------
# 5. Partición train/test y cálculo del pronóstico naive
# ---------------------------------------------------------------------------
corte = pd.Timestamp(fecha_corte_test)
df_train = df[df["semana"] < corte]
df_test = df[df["semana"] >= corte]

# Último valor observado en train, por producto-sucursal
ultimo_train = (
    df_train.sort_values("semana")
    .groupby(["codigo_item", "almacen"], as_index=False)
    .last()[["codigo_item", "almacen", "demanda"]]
    .rename(columns={"demanda": "naive_pred"})
)

comparacion = df_test.merge(ultimo_train, on=["codigo_item", "almacen"], how="inner")
print(f"Series producto-sucursal evaluadas (con historial en train y observación en test): "
      f"{comparacion[['codigo_item', 'almacen']].drop_duplicates().shape[0]:,}")


# ---------------------------------------------------------------------------
# 6. Cálculo de métricas (MAE, RMSE, MAPE) por producto-sucursal
# ---------------------------------------------------------------------------
def calcular_metricas(grupo: pd.DataFrame) -> pd.Series:
    real = grupo["demanda"].values
    pred = grupo["naive_pred"].values
    mae = np.mean(np.abs(real - pred))
    rmse = np.sqrt(np.mean((real - pred) ** 2))
    mask = real != 0
    mape = np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100 if mask.any() else np.nan
    return pd.Series({"MAE": mae, "RMSE": rmse, "MAPE": mape})


print("Calculando métricas por producto-sucursal...")
por_producto = (
    comparacion.groupby(["codigo_item", "almacen"])
    .apply(calcular_metricas)
    .reset_index()
)

# Cruce con clasificación ABC
por_producto = por_producto.merge(
    df_clas, left_on="codigo_item", right_on="item_code", how="left"
)
por_producto["clasificacion_abc"] = por_producto["clasificacion_abc"].fillna("Sin clasificar")

# ---------------------------------------------------------------------------
# 7. Resumen por categoría (mediana, igual convención que el resto del proyecto)
# ---------------------------------------------------------------------------
resumen = (
    por_producto.groupby("clasificacion_abc")
    .agg(
        n_series=("MAE", "count"),
        MAE_mediana=("MAE", "median"),
        RMSE_mediana=("RMSE", "median"),
        MAPE_mediana=("MAPE", "median"),
    )
    .reset_index()
    .sort_values("clasificacion_abc")
)

# ---------------------------------------------------------------------------
# 8. Guardar resultados y mostrar resumen
# ---------------------------------------------------------------------------
por_producto.to_csv(ARCHIVO_DETALLE, index=False)
resumen.to_csv(ARCHIVO_RESUMEN, index=False)

print("\n" + "=" * 60)
print("RESUMEN - Modelo baseline (Naive Forecast)")
print("=" * 60)
print(resumen.to_string(index=False))
print("\nArchivos generados:")
print(f"  - {ARCHIVO_DETALLE}  (detalle por producto-sucursal)")
print(f"  - {ARCHIVO_RESUMEN}  (medianas por categoría ABC)")
print("\nUsa estos valores para reemplazar los [pendiente] en la sección")
print("'D. Modelo baseline' del documento de tesis.")
