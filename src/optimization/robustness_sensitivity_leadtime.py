"""
MIA - Modelo de Inventario Automatizado
Comercial La Feria HLS CIA LTDA.

ANEXO DE ROBUSTEZ - Prueba de sensibilidad (MILP): lead_time

Compara la solucion del MILP en dos escenarios sobre el MISMO subconjunto
de pares producto-sucursal:

  CASO A (bug original): lead_time = 0 para todos los productos, tal como
      llegaba desde SAP Business One / PostgreSQL antes de aplicar el
      supuesto por grupo (ver optimizacion_milp_piloto.py, construir_dataset).
  CASO B (pipeline corregido): lead_time con el supuesto por grupo
      (75 dias IMPORTACIONES / 11 dias resto del catalogo).

Objetivo: cuantificar cuanto cambia la solucion del solver ante esta
correccion de datos, como evidencia de sensibilidad del modelo para el
Anexo de Robustez (sensibilidad / estres / data drift).

No modifica optimizacion_milp_piloto.py -> se reutiliza por import.
Se corre sobre una MUESTRA (no el catalogo completo de 16,485 pares) por
tiempo de computo: el objetivo es demostrar sensibilidad, no repetir el
resultado de produccion.

Autor: Jessica Roman
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimizacion_milp_piloto as milp  # noqa: E402  (reutiliza el modulo real del pipeline)

# ============================================================
# CONFIGURACION DE LA PRUEBA
# ============================================================

DATA_DIR = milp.DATA_DIR

MUESTRA_N = 300          # pares producto-sucursal en la muestra de la prueba
SEED = 42                # misma semilla que el pipeline principal, para trazabilidad

OUTPUT_RESUMEN_CSV = DATA_DIR / "robustness_sensitivity_leadtime_resumen.csv"
OUTPUT_DETALLE_CSV = DATA_DIR / "robustness_sensitivity_leadtime_detalle.csv"
OUTPUT_LOG_TXT = DATA_DIR / "robustness_sensitivity_leadtime_summary.txt"


def preparar_dataset_muestra():
    """Carga los datos reales (misma fuente que produccion) y toma una
    muestra aleatoria fija para la prueba de sensibilidad."""
    engine = milp.conectar()
    stock, productos, costos, sigma_hist, pronostico = milp.cargar_datos(engine)
    df = milp.construir_dataset(stock, productos, costos, sigma_hist, pronostico)

    if len(df) > MUESTRA_N:
        df = df.sample(n=MUESTRA_N, random_state=SEED).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    capacidad_sucursal = milp.calcular_capacidad_sucursal(stock)
    return df, capacidad_sucursal


def correr_caso(df, capacidad_sucursal, forzar_lead_time_cero):
    """Corre el solver MILP real (resolver_milp) sobre una copia del dataset,
    forzando o no lead_time=0 segun el caso."""
    df_caso = df.copy()
    if forzar_lead_time_cero:
        df_caso["lead_time"] = 0  # reproduce el bug original tal como llegaba de SAP
    resultado, status = milp.resolver_milp(df_caso, capacidad_sucursal)
    return resultado, status


def resumir(resultado, status, etiqueta):
    ordenes = int((resultado["genera_orden"] == 1).sum())
    costo_total = float((resultado["costo"] * resultado["cantidad_a_ordenar"]).sum())
    reorden_medio = float(resultado["punto_reorden"].mean())
    reorden_max = float(resultado["punto_reorden"].max())

    print(f"\n--- {etiqueta} ---")
    print(f"Estado del solver (1=optimo): {status}")
    print(f"Ordenes generadas: {ordenes} de {len(resultado)}")
    print(f"Costo total estimado: {costo_total:,.2f}")
    print(f"Punto de reorden promedio: {reorden_medio:,.2f}  |  maximo: {reorden_max:,.2f}")

    return {
        "caso": etiqueta,
        "status_solver": status,
        "ordenes_generadas": ordenes,
        "total_pares": len(resultado),
        "costo_total_estimado": round(costo_total, 2),
        "punto_reorden_promedio": round(reorden_medio, 2),
        "punto_reorden_maximo": round(reorden_max, 2),
    }


def main():
    print(f"Preparando muestra de {MUESTRA_N} pares producto-sucursal (seed={SEED})...")
    DATA_DIR.mkdir(exist_ok=True)
    df, capacidad_sucursal = preparar_dataset_muestra()
    print(f"Muestra final: {len(df)} pares producto-sucursal.")

    resultado_bug, status_bug = correr_caso(df, capacidad_sucursal, forzar_lead_time_cero=True)
    resumen_bug = resumir(resultado_bug, status_bug, "CASO A: lead_time=0 (bug original)")

    resultado_fix, status_fix = correr_caso(df, capacidad_sucursal, forzar_lead_time_cero=False)
    resumen_fix = resumir(resultado_fix, status_fix, "CASO B: lead_time corregido (supuesto por grupo)")

    comparacion = pd.DataFrame([resumen_bug, resumen_fix])
    comparacion.to_csv(OUTPUT_RESUMEN_CSV, index=False)

    # Detalle por producto-sucursal: donde cambio mas el punto de reorden
    detalle = resultado_bug[["codigo_item", "almacen", "punto_reorden", "cantidad_a_ordenar"]].rename(
        columns={"punto_reorden": "reorden_bug", "cantidad_a_ordenar": "orden_bug"}
    ).merge(
        resultado_fix[["codigo_item", "almacen", "punto_reorden", "cantidad_a_ordenar"]].rename(
            columns={"punto_reorden": "reorden_fix", "cantidad_a_ordenar": "orden_fix"}
        ),
        on=["codigo_item", "almacen"],
    )
    detalle["diferencia_reorden"] = detalle["reorden_fix"] - detalle["reorden_bug"]
    detalle["diferencia_orden"] = detalle["orden_fix"] - detalle["orden_bug"]
    detalle = detalle.sort_values("diferencia_reorden", ascending=False)
    detalle.to_csv(OUTPUT_DETALLE_CSV, index=False)

    with open(OUTPUT_LOG_TXT, "w", encoding="utf-8") as f:
        f.write("ANEXO DE ROBUSTEZ - Prueba de sensibilidad: lead_time (MILP)\n")
        f.write(f"Muestra: {len(df)} pares producto-sucursal | seed={SEED}\n\n")
        f.write(comparacion.to_string(index=False))
        f.write("\n\nTop 10 productos con mayor cambio en punto de reorden "
                "(fix - bug):\n")
        f.write(detalle.head(10).to_string(index=False))
        f.write("\n\nTop 10 productos con MENOR cambio (posible insensibilidad "
                "residual, revisar):\n")
        f.write(detalle.tail(10).to_string(index=False))

    print(f"\nResumen guardado en {OUTPUT_RESUMEN_CSV}")
    print(f"Detalle producto-sucursal guardado en {OUTPUT_DETALLE_CSV}")
    print(f"Reporte de texto guardado en {OUTPUT_LOG_TXT}")


if __name__ == "__main__":
    main()
