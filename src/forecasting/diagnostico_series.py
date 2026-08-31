"""
diagnostico_series.py

Diagnóstico visual y estadístico para los productos con MAPE alto
detectados en train_forecasting.py:

  - C114053-0005  GS DETERGENTE GRANEL PRETEN (LDG)   MAPE=118.1%
  - A101003-0004  PALMA DE ORO ACEITE FUNDA 1LT        MAPE=17.7%
  - A111010-0040  GLORIA MATILDE ARROZ GRANEL          MAPE=19.1%
  - C113029-0120  COLGATE TRIPLE ACCION 60CC           MAPE=15.4%
  - A102012-0019  REAL TINAPA EN SALSA TOMATE AF 156GR MAPE=25.7%

Qué hace:
  1. Reutiliza la función de carga de datos de train_forecasting.py
     (para no duplicar la lógica de conexión / limpieza).
  2. Para cada producto problemático:
       - Grafica la serie completa (train+test) y la guarda como PNG.
       - Calcula % de semanas en cero, coeficiente de variación (CV),
         y cantidad de outliers (regla IQR 1.5x).
  3. Imprime un resumen en consola para pegar directo en la tesis.

Carga los datos exactamente igual que train_forecasting.py: lee
data/parsed.json, reindexa a semanas continuas (W-MON) y arma
un dict {codigo_producto: {"name": ..., "series": pd.Series}}.
Así la serie diagnosticada es 100% la misma que ve el modelo.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # para correr sin entorno gráfico
import matplotlib.pyplot as plt

# Coloca este script en la misma carpeta que train_forecasting.py
# (src/forecasting/). ROOT_DIR replica exactamente
# Path(__file__).resolve().parents[2] de train_forecasting.py.
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

# ============================================================
# CARGA DE DATOS (idéntica a train_forecasting.py)
# ============================================================
def load_products(root_dir: Path = ROOT_DIR) -> dict:
    data_path = root_dir / "data" / "parsed.json"
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    products = {}
    for row in raw["rows"]:
        df = pd.DataFrame(row["series"])
        df["week"] = pd.to_datetime(df["week"])
        df = df.sort_values("week").reset_index(drop=True)
        full_range = pd.date_range(df["week"].min(), df["week"].max(), freq="W-MON")
        df = df.set_index("week").reindex(full_range).fillna(0.0)
        df.index.name = "week"
        products[row["product_code"]] = {
            "name": row["product_name"],
            "series": df["units"].astype(float),
        }
    return products

PRODUCTOS_PROBLEMATICOS = {
    "C114053-0005": "GS DETERGENTE GRANEL PRETEN (LDG)",
    "A101003-0004": "PALMA DE ORO ACEITE FUNDA 1LT",
    "A111010-0040": "GLORIA MATILDE ARROZ GRANEL",
    "C113029-0120": "COLGATE TRIPLE ACCION 60CC",
    "A102012-0019": "REAL TINAPA EN SALSA TOMATE AF 156GR",
}

OUTPUT_DIR = DATA_DIR / "diagnostico_output"


def compute_stats(serie: pd.Series) -> dict:
    """Calcula métricas de diagnóstico sobre una serie de demanda semanal."""
    n = len(serie)
    n_ceros = int((serie == 0).sum())
    pct_ceros = 100 * n_ceros / n if n > 0 else np.nan

    media = serie.mean()
    std = serie.std()
    cv = std / media if media != 0 else np.nan

    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = serie[(serie < lim_inf) | (serie > lim_sup)]

    return {
        "n_semanas": n,
        "pct_semanas_cero": round(pct_ceros, 1),
        "media": round(media, 1),
        "std": round(std, 1),
        "cv": round(cv, 2) if not np.isnan(cv) else None,
        "n_outliers": len(outliers),
        "outliers_valores": outliers.tolist(),
        "clasificacion": clasificar_serie(pct_ceros, cv),
    }


def clasificar_serie(pct_ceros: float, cv: float) -> str:
    """Heurística simple para orientar la interpretación."""
    if pct_ceros >= 20:
        return "INTERMITENTE (muchas semanas en cero) -> MAPE no es la métrica adecuada"
    if cv is not None and cv > 1.0:
        return "ALTA VARIABILIDAD / posibles outliers -> revisar picos puntuales"
    return "REGULAR (no explica el MAPE alto por intermitencia/varianza obvia)"


def plot_series(codigo: str, nombre: str, serie: pd.Series):
    plt.figure(figsize=(11, 4))
    plt.plot(serie.index, serie.values, marker="o", markersize=3, linewidth=1)
    plt.axhline(serie.mean(), color="gray", linestyle="--", linewidth=1, label="Media")
    plt.title(f"{codigo} - {nombre}")
    plt.xlabel("Semana")
    plt.ylabel("Cantidad vendida")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{codigo}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    products = load_products(ROOT_DIR)
    print(f"Products loaded: {len(products)}")

    resumen = []

    for codigo, nombre in PRODUCTOS_PROBLEMATICOS.items():
        if codigo not in products:
            print(f"[!] {codigo} no encontrado en products, se omite.")
            continue

        serie = products[codigo]["series"]

        stats = compute_stats(serie)
        stats["codigo"] = codigo
        stats["nombre"] = nombre
        resumen.append(stats)

        png_path = plot_series(codigo, nombre, serie)

        print(f"\n=== {codigo} - {nombre} ===")
        print(f"  Semanas: {stats['n_semanas']}  |  % semanas en cero: {stats['pct_semanas_cero']}%")
        print(f"  Media: {stats['media']}  |  Std: {stats['std']}  |  CV: {stats['cv']}")
        print(f"  Outliers (IQR 1.5x): {stats['n_outliers']} -> {stats['outliers_valores']}")
        print(f"  Diagnóstico: {stats['clasificacion']}")
        print(f"  Gráfico guardado en: {png_path}")

    # Tabla resumen final (útil para pegar en la tesis)
    resumen_df = pd.DataFrame(resumen)[
        ["codigo", "nombre", "n_semanas", "pct_semanas_cero", "cv", "n_outliers", "clasificacion"]
    ]
    resumen_df.to_csv(os.path.join(OUTPUT_DIR, "resumen_diagnostico.csv"), index=False)
    print("\n================ RESUMEN ================")
    print(resumen_df.to_string(index=False))
    print(f"\nCSV guardado en: {OUTPUT_DIR}/resumen_diagnostico.csv")


def diagnosticar_train_test():
    """
    Segundo nivel de diagnóstico: compara el nivel (media) de train vs. test
    y superpone real vs. predicho por modelo, usando prediction_details.json
    (generado por train_forecasting.py, no hace falta reentrenar).
    """
    products = load_products(ROOT_DIR)

    with open(DATA_DIR / "prediction_details.json", "r", encoding="utf-8") as f:
        pred_details = json.load(f)

    resumen = []

    for codigo, nombre in PRODUCTOS_PROBLEMATICOS.items():
        if codigo not in products or codigo not in pred_details:
            print(f"[!] {codigo} no encontrado, se omite.")
            continue

        serie = products[codigo]["series"]
        det = pred_details[codigo]
        test_dates = pd.to_datetime(det["test_dates"])
        actual = np.array(det["actual"])
        n_test = len(test_dates)

        train = serie.iloc[:-n_test]
        test = serie.iloc[-n_test:]

        media_train, std_train = train.mean(), train.std()
        media_test = test.mean()
        shift_sigmas = (media_test - media_train) / std_train if std_train > 0 else np.nan

        stats = {
            "codigo": codigo,
            "nombre": nombre,
            "media_train": round(media_train, 1),
            "media_test": round(media_test, 1),
            "shift_pct": round(100 * (media_test - media_train) / media_train, 1),
            "shift_sigmas": round(shift_sigmas, 2),
        }
        resumen.append(stats)

        # --- Gráfico 1: serie completa con línea train/test ---
        plt.figure(figsize=(11, 4))
        plt.plot(train.index, train.values, label="Train", linewidth=1)
        plt.plot(test.index, test.values, label="Test (real)", linewidth=1.5, color="black")
        plt.axvline(test.index[0], color="red", linestyle="--", linewidth=1, label="Inicio test")
        plt.title(f"{codigo} - {nombre}: Train vs Test")
        plt.xlabel("Semana")
        plt.ylabel("Cantidad vendida")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{codigo}_train_test.png"), dpi=150)
        plt.close()

        # --- Gráfico 2: real vs predicho por modelo (solo test) ---
        plt.figure(figsize=(9, 4))
        plt.plot(test_dates, actual, label="Real", color="black", marker="o", linewidth=2)
        for modelo, color in [("ARIMA", "tab:blue"), ("Prophet", "tab:orange"), ("LSTM", "tab:green")]:
            if modelo in det:
                plt.plot(test_dates, det[modelo], label=modelo, linestyle="--", marker="x", color=color)
        plt.title(f"{codigo} - {nombre}: Real vs Predicho (test)")
        plt.xlabel("Semana")
        plt.ylabel("Cantidad vendida")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{codigo}_real_vs_pred.png"), dpi=150)
        plt.close()

        print(f"\n=== {codigo} - {nombre} ===")
        print(f"  Media train: {stats['media_train']}  |  Media test: {stats['media_test']}")
        print(f"  Cambio de nivel: {stats['shift_pct']}%  ({stats['shift_sigmas']} sigmas del train)")

    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(os.path.join(OUTPUT_DIR, "resumen_train_test.csv"), index=False)
    print("\n================ RESUMEN TRAIN vs TEST ================")
    print(resumen_df.to_string(index=False))
    print(f"\nCSV guardado en: {OUTPUT_DIR}/resumen_train_test.csv")


if __name__ == "__main__":
    main()
    diagnosticar_train_test()