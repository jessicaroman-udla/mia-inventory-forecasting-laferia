"""
MIA - Modelo de Inventario Automatizado
Comercial La Feria HLS CIA LTDA.

Script de optimización MILP (piloto) para generación de recomendaciones
de reabastecimiento por producto-sucursal.

Ejecutar primero sobre un subconjunto piloto (categoría A, una sucursal)
para validar la formulación antes de escalar al catálogo completo.

Autor: Jessica Roman
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, LpBinary, lpSum, PULP_CBC_CMD

# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

SEED = 42
np.random.seed(SEED)

TABLA_PRONOSTICO = "forecast_output"

# --- Fuente del pronóstico en modo piloto: CSV local en vez de tabla en BD ---
# Columnas esperadas: codigo_item, almacen, categoria, modelo_ganador,
# demanda_pronosticada, mape
USAR_CSV_PRONOSTICO = True
RUTA_CSV_PRONOSTICO = "forecast_output.csv"  # <-- AJUSTAR a tu archivo real

# Filtros del piloto: reducir alcance antes de correr sobre todo el catálogo
PILOTO_CATEGORIA = None  # None = A+B juntas; o "A"/"B" para una sola
PILOTO_ALMACEN = None  # None = todas las sucursales; o ej. "CHOVE01" para una sola

# Supuestos de negocio documentados (ver justificación en sección 7.2 de la tesis)
HOLDING_COST_ANUAL_PCT = 0.20      # 20% anual del costo unitario
ORDER_COST_FIJO = 20.0             # costo fijo estimado por orden ($)
NIVEL_SERVICIO_DEFAULT = 0.95      # si no hay dato en analisis_avanzado
Z_SCORE_95 = 1.645                 # z para 95% de nivel de servicio (normal estándar)

HORIZONTE_DIAS = 30                # horizonte de planificación del piloto
CAPACIDAD_BUFFER = 1.5             # margen de holgura sobre el stock total actual

OUTPUT_CSV = "resultados_milp_piloto.csv"

# --- AJUSTAR si tus tablas viven bajo otro esquema (ya vimos que ventas
# está en 'inventario.'); confirma stock_global y productos también. ---
ESQUEMA_BD = "inventario"

# ============================================================
# CONEXIÓN Y CARGA DE DATOS
# ============================================================

def conectar():
    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(conn_str)


def cargar_datos(engine):
    print("Cargando stock actual...")
    stock = pd.read_sql(
        text(f"""
            SELECT codigo_item, almacen, stock, comprometido
            FROM {ESQUEMA_BD}.stock_global
        """),
        engine,
    )
    stock["disponible"] = (stock["stock"] - stock["comprometido"].fillna(0)).clip(lower=0)

    print("Cargando maestro de productos...")
    productos = pd.read_sql(
        text(f"""
            SELECT item_code AS codigo_item, item_name, lead_time,
                   pur_pack_un, itms_grp_cod
            FROM {ESQUEMA_BD}.productos
        """),
        engine,
    )

    print("Cargando costo unitario más reciente por producto (tabla ventas)...")
    # Se toma el costo de la venta más reciente por producto como proxy
    # de costo de adquisición vigente. Se limita a los últimos 6 meses
    # para evitar un sort completo sobre 3 años de histórico (evita el
    # crash de conexión por agotamiento de memoria/disco temporal).
    costos = pd.read_sql(
        text(f"""
            SELECT DISTINCT ON (codigo_item) codigo_item, costo, fecha
            FROM {ESQUEMA_BD}.ventas
            WHERE costo IS NOT NULL AND costo > 0
              AND fecha >= CURRENT_DATE - INTERVAL '6 months'
            ORDER BY codigo_item, fecha DESC
        """),
        engine,
    )[["codigo_item", "costo"]]

    print("Calculando desviación estándar histórica de ventas (respaldo de sigma)...")
    # Respaldo mientras se confirma la llave de unión con analisis_avanzado.
    # Igual que el query de costo, se limita la ventana para evitar un
    # agregado completo sobre 3 años de histórico.
    sigma_hist = pd.read_sql(
        text(f"""
            SELECT codigo_item, almacen, STDDEV(cantidad) AS sigma_demanda
            FROM {ESQUEMA_BD}.ventas
            WHERE fecha >= CURRENT_DATE - INTERVAL '6 months'
            GROUP BY codigo_item, almacen
        """),
        engine,
    )

    print(f"Cargando pronóstico...")
    if USAR_CSV_PRONOSTICO:
        try:
            pronostico = pd.read_csv(RUTA_CSV_PRONOSTICO)
        except FileNotFoundError:
            print(f"AVISO: no se encontró '{RUTA_CSV_PRONOSTICO}'.")
            print("Ajusta RUTA_CSV_PRONOSTICO a la ruta real de tu archivo de resultados.")
            raise
    else:
        try:
            pronostico = pd.read_sql(text(f"SELECT * FROM {TABLA_PRONOSTICO}"), engine)
        except Exception as e:
            print(f"AVISO: no se pudo leer '{TABLA_PRONOSTICO}' ({e}).")
            raise

    return stock, productos, costos, sigma_hist, pronostico


def construir_dataset(stock, productos, costos, sigma_hist, pronostico):
    df = pronostico.rename(columns={"demanda_pronosticada": "forecasted_demand"})

    if PILOTO_CATEGORIA is not None and "categoria" in df.columns:
        df = df[df["categoria"] == PILOTO_CATEGORIA]
    if PILOTO_ALMACEN is not None:
        df = df[df["almacen"] == PILOTO_ALMACEN]

    df = df.merge(stock[["codigo_item", "almacen", "disponible"]],
                   on=["codigo_item", "almacen"], how="left")
    df = df.merge(productos, on="codigo_item", how="left")
    df = df.merge(costos, on="codigo_item", how="left")
    df = df.merge(sigma_hist, on=["codigo_item", "almacen"], how="left")

    # Limpieza de nulos con valores conservadores
    df["disponible"] = df["disponible"].fillna(0)
    df["costo"] = df["costo"].fillna(df["costo"].median())
    df["sigma_demanda"] = df["sigma_demanda"].fillna(df["sigma_demanda"].median())
    # El campo maestro lead_time (SAP/PostgreSQL) no esta poblado a nivel de producto
    # (99.99% en 0). No se encontro proxy confiable por proveedor (todos son
    # distribuidores locales, no reflejan origen de la mercancia). Se usa itms_grp_cod
    # como proxy de origen: el grupo IMPORTACIONES (110) recibe el lead time importado
    # declarado (60-90 dias, punto medio 75); el resto del catalogo (abarrotes,
    # alimentos basicos, mayoritariamente de produccion/distribucion nacional) recibe
    # el lead time nacional declarado (7-15 dias, punto medio 11). Ver seccion 1.2.
    LEAD_TIME_NACIONAL_DIAS = 11
    LEAD_TIME_IMPORTADO_DIAS = 75
    GRUPO_IMPORTACIONES_COD = 110

    n_lead_time_faltante = (df["lead_time"].isna() | (df["lead_time"] == 0)).sum()
    if n_lead_time_faltante > 0:
        print(f"AVISO: {n_lead_time_faltante} productos sin lead_time en maestro (NULL o 0) "
              f"-- se aplico supuesto por grupo: {LEAD_TIME_IMPORTADO_DIAS} dias para "
              f"IMPORTACIONES (grupo {GRUPO_IMPORTACIONES_COD}), {LEAD_TIME_NACIONAL_DIAS} "
              f"dias para el resto del catalogo.")

    df["lead_time"] = df["lead_time"].replace(0, np.nan)
    lead_time_defecto = df["itms_grp_cod"].apply(
        lambda g: LEAD_TIME_IMPORTADO_DIAS if g == GRUPO_IMPORTACIONES_COD else LEAD_TIME_NACIONAL_DIAS
    )
    df["lead_time"] = df["lead_time"].fillna(lead_time_defecto)

    df["moq"] = df["pur_pack_un"].fillna(1).clip(lower=1)

    df["holding_cost"] = df["costo"] * HOLDING_COST_ANUAL_PCT / 365 * HORIZONTE_DIAS

    df = df.dropna(subset=["forecasted_demand", "codigo_item", "almacen"])
    return df.reset_index(drop=True)


def calcular_capacidad_sucursal(stock):
    # Supuesto CORREGIDO: la capacidad práctica se aproxima como la suma del
    # stock actual total por sucursal (todos los productos), con un margen
    # de holgura del 30% — no el máximo de un solo producto individual
    # (error anterior: eso subestimaba brutalmente la capacidad real y
    # volvía el problema infeasible).
    total_actual_por_sucursal = stock.groupby("almacen")["stock"].sum()
    cap = (total_actual_por_sucursal * CAPACIDAD_BUFFER).to_dict()
    return cap


# ============================================================
# MODELO MILP
# ============================================================

def resolver_milp(df, capacidad_sucursal):
    prob = LpProblem("Reabastecimiento_MIA", LpMinimize)

    idx = df.index.tolist()
    x = {i: LpVariable(f"x_{i}", lowBound=0, cat=LpInteger) for i in idx}
    y = {i: LpVariable(f"y_{i}", cat=LpBinary) for i in idx}

    M = 1_000_000  # cota superior grande para vínculo MOQ-orden
    df["punto_reorden"] = 0.0  # se completa dentro del loop de restricciones

    # Función objetivo: costo de adquisición + holding cost + costo fijo por orden
    prob += lpSum(
        df.loc[i, "costo"] * x[i]
        + df.loc[i, "holding_cost"] * (x[i] / 2)
        + ORDER_COST_FIJO * y[i]
        for i in idx
    )

    for i in idx:
        row = df.loc[i]
        nivel_servicio = row.get("nivel_servicio_configurado", NIVEL_SERVICIO_DEFAULT)
        if pd.isna(nivel_servicio):
            nivel_servicio = NIVEL_SERVICIO_DEFAULT
        z = Z_SCORE_95  # simplificación: usar 1.645 fijo (95%) salvo se calcule z(nivel_servicio)

        # Lead time explícito: distingue asimetría nacional (7-15 días)
        # vs. importado (60-90 días), tal como exige el alcance formal.
        lead_time = row["lead_time"]
        demanda_diaria = row["forecasted_demand"] / HORIZONTE_DIAS
        sigma_diaria = row["sigma_demanda"]  # aproximación: sigma histórica por transacción

        # Punto de reorden: demanda esperada durante el lead time + colchón
        # de seguridad escalado por sqrt(lead time), fórmula estándar de
        # gestión de inventarios (ver Vicente, 2025; Zabraoui et al., 2025).
        punto_reorden = demanda_diaria * lead_time + z * sigma_diaria * np.sqrt(lead_time)
        df.loc[i, "punto_reorden"] = punto_reorden

        # 1. Cobertura de demanda del horizonte + colchón para sobrevivir
        #    el próximo lead time sin caer bajo el punto de reorden.
        prob += row["disponible"] + x[i] >= row["forecasted_demand"] + punto_reorden, f"cobertura_{i}"

        # 2. Vínculo MOQ <-> orden
        prob += x[i] >= row["moq"] * y[i], f"moq_min_{i}"
        prob += x[i] <= M * y[i], f"moq_max_{i}"

    # 3. Capacidad por sucursal
    for almacen, grupo in df.groupby("almacen"):
        cap = capacidad_sucursal.get(almacen)
        if cap is None:
            continue
        prob += lpSum(df.loc[i, "disponible"] + x[i] for i in grupo.index) <= cap, f"capacidad_{almacen}"

    print(f"Resolviendo MILP sobre {len(idx)} pares producto-sucursal...")
    prob.solve(PULP_CBC_CMD(msg=False))

    if prob.status != 1:  # no óptimo -> diagnosticar antes de seguir
        print("\n=== DIAGNÓSTICO: problema no resuelto de forma óptima ===")

        # Pedido mínimo real que necesitaría cada producto (antes de resolver
        # el MILP), respetando MOQ, para ver si la capacidad alcanza incluso
        # en el escenario mínimo indispensable.
        def pedido_minimo(row):
            necesario = row["forecasted_demand"] + row["punto_reorden"] - row["disponible"]
            if necesario <= 0:
                return 0
            return max(necesario, row["moq"])

        df["_pedido_minimo_diag"] = df.apply(pedido_minimo, axis=1)

        print("Comparando disponible + pedido MÍNIMO necesario vs. capacidad por sucursal:")
        for almacen, grupo in df.groupby("almacen"):
            disponible_sum = grupo["disponible"].sum()
            minimo_sum = grupo["_pedido_minimo_diag"].sum()
            total_necesario = disponible_sum + minimo_sum
            cap = capacidad_sucursal.get(almacen)
            if cap is None:
                print(f"  {almacen}: capacidad=NO ENCONTRADA")
                continue
            estado = "!!! CAPACIDAD INSUFICIENTE incluso con pedido mínimo !!!" if total_necesario > cap else "ok"
            print(f"  {almacen}: disponible={disponible_sum:,.0f}  +pedido_minimo={minimo_sum:,.0f}  "
                  f"= {total_necesario:,.0f}  vs  capacidad={cap:,.0f}  -> {estado}")

        # Top 5 productos con mayor pedido mínimo individual, por si un solo
        # producto atípico (MOQ o punto de reorden desproporcionado) es el culpable.
        top5 = df.nlargest(5, "_pedido_minimo_diag")[
            ["codigo_item", "almacen", "disponible", "forecasted_demand",
             "punto_reorden", "moq", "_pedido_minimo_diag"]
        ]
        print("\nTop 5 productos con mayor pedido mínimo individual (posibles outliers):")
        print(top5.to_string(index=False))
        print("=== FIN DIAGNÓSTICO ===\n")

    df["cantidad_a_ordenar"] = [int(x[i].value() or 0) for i in idx]
    df["genera_orden"] = [int(y[i].value() or 0) for i in idx]

    return df, prob.status


# ============================================================
# MAIN
# ============================================================

def main():
    engine = conectar()
    stock, productos, costos, sigma_hist, pronostico = cargar_datos(engine)

    df = construir_dataset(stock, productos, costos, sigma_hist, pronostico)
    capacidad_sucursal = calcular_capacidad_sucursal(stock)

    print(f"Dataset piloto: {len(df)} pares producto-sucursal "
          f"(categoría={PILOTO_CATEGORIA}, almacén={PILOTO_ALMACEN or 'todos'})")

    resultado, status = resolver_milp(df, capacidad_sucursal)

    print(f"Estado de la solución: {status}")
    print(f"Órdenes generadas: {(resultado['genera_orden'] == 1).sum()} de {len(resultado)}")
    print(f"Costo total estimado: "
          f"{(resultado['costo'] * resultado['cantidad_a_ordenar']).sum():,.2f}")

    resultado.to_csv(OUTPUT_CSV, index=False)
    print(f"Resultados guardados en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()