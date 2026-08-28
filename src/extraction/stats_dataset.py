"""
stats_dataset.py

Resumen descriptivo y de calidad de datos de la tabla cruda `inventario.ventas`
(SOLO LECTURA). Sirve para llenar la seccion de descripcion del dataset y de
calidad de datos del documento.

Principios:
  - Reutiliza la MISMA conexion y el MISMO filtro que el pipeline real
    (`src/extraction/extract_data.py`): tabla `inventario.ventas`,
    `tipo = 'Factura'`.
  - TODO se calcula con SQL agregado en el servidor. Nunca se traen filas al
    cliente (la tabla tiene ~31 M de filas). Solo viajan los escalares del
    resultado.
  - NO modifica nada del pipeline: script nuevo, independiente. La sesion se
    abre en modo read-only.

Calcula:
  1. Filas totales en `inventario.ventas` y filas con `tipo = 'Factura'`.
  2. Rango de fechas MIN/MAX sobre las filas 'Factura'.
  3. Nº de productos (codigo_item distintos) y de sucursales (almacen distintos).
  4. Filas EXACTAMENTE duplicadas (todas las columnas iguales): conteo y %.
  5. Filas duplicadas por clave de negocio
     (codigo_item, almacen, fecha, cantidad, precio): conteo y %.
  6. % de nulos / faltantes por columna: fecha, almacen, codigo_item,
     cantidad, precio.

Uso:
    python src/extraction/stats_dataset.py
    python src/extraction/stats_dataset.py --skip-dups   # omite (4) y (5), las pesadas
"""
import argparse
import sys
import time
from pathlib import Path

# Reutiliza la conexion real del pipeline (carga .env + psycopg2). Importar NO
# ejecuta ninguna extraccion: extract_data.py solo corre sus queries dentro de
# main(), que aqui no se llama.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_data import connect  # noqa: E402

TABLA = "inventario.ventas"
FILTRO = "tipo = 'Factura'"
COLUMNAS_CALIDAD = ["fecha", "almacen", "codigo_item", "cantidad", "precio"]
CLAVE_NEGOCIO = ["codigo_item", "almacen", "fecha", "cantidad", "precio"]


def _faltante_expr(col: str) -> str:
    """Cuenta NULL y, para texto, tambien cadena vacia / solo espacios."""
    return (f"COUNT(*) FILTER (WHERE {FILTRO} AND "
            f"({col} IS NULL OR btrim({col}::text) = '')) AS {col}_faltante")


def _pct(parte, total) -> str:
    if not total:
        return "s/d"
    return f"{100 * parte / total:.4f}%"


def query_base(cur) -> dict:
    """Una sola pasada: conteos, rango de fechas, distintos y faltantes."""
    faltantes = ",\n            ".join(_faltante_expr(c) for c in COLUMNAS_CALIDAD)
    cur.execute(f"""
        SELECT
            COUNT(*)                                              AS filas_totales,
            COUNT(*) FILTER (WHERE {FILTRO})                      AS filas_factura,
            MIN(fecha) FILTER (WHERE {FILTRO})                    AS fecha_min,
            MAX(fecha) FILTER (WHERE {FILTRO})                    AS fecha_max,
            COUNT(DISTINCT codigo_item) FILTER (WHERE {FILTRO})   AS n_productos,
            COUNT(DISTINCT almacen)     FILTER (WHERE {FILTRO})   AS n_sucursales,
            {faltantes}
        FROM {TABLA}
    """)
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, cur.fetchone()))


def query_dup_exactas(cur) -> int:
    """Filas cuya combinacion de TODAS las columnas se repite."""
    cur.execute(f"""
        WITH f AS (SELECT * FROM {TABLA} WHERE {FILTRO})
        SELECT (SELECT COUNT(*) FROM f)
             - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM f) d)
    """)
    return cur.fetchone()[0]


def query_dup_clave(cur) -> int:
    """Filas 'extra' (mas alla de la primera) por clave de negocio repetida."""
    llave = ", ".join(CLAVE_NEGOCIO)
    cur.execute(f"""
        SELECT COALESCE(SUM(rep - 1), 0)
        FROM (
            SELECT COUNT(*) AS rep
            FROM {TABLA}
            WHERE {FILTRO}
            GROUP BY {llave}
            HAVING COUNT(*) > 1
        ) g
    """)
    return cur.fetchone()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dups", action="store_true",
                    help="omite las 2 consultas de duplicados (SELECT DISTINCT * / GROUP BY)")
    args = ap.parse_args()

    conn = connect()
    conn.set_session(readonly=True, autocommit=True)

    try:
        with conn.cursor() as cur:
            print(f"[1/3] Conteos, rango de fechas, distintos y faltantes sobre {TABLA} ...")
            t0 = time.time()
            base = query_base(cur)
            print(f"      ok ({time.time() - t0:.1f}s)")

            dup_exactas = dup_clave = None
            if args.skip_dups:
                print("[2/3] y [3/3] duplicados: OMITIDOS (--skip-dups)")
            else:
                print("[2/3] Filas exactamente duplicadas (SELECT DISTINCT *, puede tardar) ...")
                t0 = time.time()
                try:
                    dup_exactas = query_dup_exactas(cur)
                    print(f"      ok ({time.time() - t0:.1f}s)")
                except Exception as e:  # noqa: BLE001
                    print(f"      NO calculado: {e}")

                print("[3/3] Filas duplicadas por clave de negocio (GROUP BY, puede tardar) ...")
                t0 = time.time()
                try:
                    dup_clave = query_dup_clave(cur)
                    print(f"      ok ({time.time() - t0:.1f}s)")
                except Exception as e:  # noqa: BLE001
                    print(f"      NO calculado: {e}")
    finally:
        conn.close()

    n = base["filas_factura"]

    print("\n" + "=" * 72)
    print(f"RESUMEN - {TABLA}  (filtro del pipeline: {FILTRO})")
    print("=" * 72)
    print(f"Filas totales en la tabla .............. {base['filas_totales']:,}")
    print(f"Filas con tipo = 'Factura' ............. {n:,}")
    print(f"Rango de fechas (Factura) .............. {base['fecha_min']}  ->  {base['fecha_max']}")
    print(f"Productos distintos (codigo_item) ...... {base['n_productos']:,}")
    print(f"Sucursales distintas (almacen) ........ {base['n_sucursales']:,}")

    print("\n--- Duplicados (sobre filas 'Factura') ---")
    if dup_exactas is not None:
        print(f"Filas exactamente duplicadas .......... {dup_exactas:,}  ({_pct(dup_exactas, n)})")
    else:
        print("Filas exactamente duplicadas .......... (no calculado)")
    if dup_clave is not None:
        print(f"Filas duplicadas por clave de negocio . {dup_clave:,}  ({_pct(dup_clave, n)})")
        print(f"  clave = ({', '.join(CLAVE_NEGOCIO)})")
    else:
        print("Filas duplicadas por clave de negocio . (no calculado)")

    print("\n--- Nulos / faltantes por columna (sobre filas 'Factura') ---")
    print(f"{'columna':<14}{'faltantes':>14}{'%':>12}")
    for c in COLUMNAS_CALIDAD:
        falt = base[f"{c}_faltante"]
        print(f"{c:<14}{falt:>14,}{_pct(falt, n):>12}")
    print("=" * 72)


if __name__ == "__main__":
    main()
