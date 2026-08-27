"""
baseline_naive.py

Modelo baseline para el proyecto MIA - Comercial La Feria.
Punto de referencia minimo frente al cual se evalua si los modelos propuestos
(ARIMA, Prophet, LSTM, Holt-Winters) mejoran una estrategia trivial.

IMPORTANTE - comparabilidad:
    Este script reutiliza EXACTAMENTE la misma preparacion de datos y las
    mismas metricas que train_forecasting_tiered.py (importa load_data,
    split_series, mae, rmse, mape y TEST_WEEKS de ese modulo). Asi las
    cifras del baseline son directamente comparables con
    model_comparison_results_A.csv / _B.csv:
      - misma serie semanal nacional por producto (data/parsed.json)
      - misma reindexacion a semanas continuas W-MON con huecos = 0
      - misma particion cronologica (ultimas TEST_WEEKS = 12 semanas = test)
      - misma clasificacion ABC (data/abc_classification.json)
      - MAPE calculado solo sobre semanas con demanda real > 0

Dos baselines:
    Naive (random walk):   pronostico = ultimo valor observado en train.
    Seasonal naive:         pronostico(semana t) = valor de la semana t-52
                            (misma semana del anio anterior). Solo se calcula
                            si la serie tiene >= 52 + TEST_WEEKS semanas.

Entrada:  data/parsed.json, data/abc_classification.json
Salida:   data/model_comparison_results_baseline.csv   (detalle por producto)
          data/model_comparison_results_baseline_resumen.csv  (mediana por categoria)

Uso (rapido, sin base de datos):
    python src/baseline/baseline_naive.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reutiliza la logica del pipeline principal (mismo directorio que
# train_forecasting.py / _tiered.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting"))
from train_forecasting_tiered import (  # noqa: E402
    DATA_DIR,
    TEST_WEEKS,
    load_data,
    split_series,
    mae,
    rmse,
    mape,
)

OUTPUT_DETALLE = DATA_DIR / "model_comparison_results_baseline.csv"
OUTPUT_RESUMEN = DATA_DIR / "model_comparison_results_baseline_resumen.csv"

CATEGORIES = ("A", "B")  # las mismas que forecastea el pipeline principal
SEASONAL_PERIOD = 52     # semanas


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Random walk: repite el ultimo valor observado en train."""
    return np.full(horizon, float(train.iloc[-1]))


def seasonal_naive_forecast(series: pd.Series, horizon: int) -> np.ndarray | None:
    """Valor de la misma semana del anio anterior (t - 52)."""
    if len(series) < SEASONAL_PERIOD + horizon:
        return None
    # test = series[-horizon:]  ->  para el punto -horizon+k se usa -horizon+k-52
    ref = series.iloc[-horizon - SEASONAL_PERIOD: -SEASONAL_PERIOD]
    return np.clip(ref.values.astype(float), 0, None)


def evaluate_product(code: str, name: str, series: pd.Series, category: str) -> dict:
    train, test = split_series(series)
    horizon = len(test)
    y_true = test.values

    row = {
        "product_code": code,
        "product_name": name,
        "category": category,
        "n_train": len(train),
        "n_test": horizon,
    }

    pred_naive = naive_forecast(train, horizon)
    row["Naive_MAE"] = mae(y_true, pred_naive)
    row["Naive_RMSE"] = rmse(y_true, pred_naive)
    row["Naive_MAPE"] = mape(y_true, pred_naive)

    pred_snaive = seasonal_naive_forecast(series, horizon)
    if pred_snaive is not None:
        row["SeasonalNaive_MAE"] = mae(y_true, pred_snaive)
        row["SeasonalNaive_RMSE"] = rmse(y_true, pred_snaive)
        row["SeasonalNaive_MAPE"] = mape(y_true, pred_snaive)
    else:
        row["SeasonalNaive_MAE"] = np.nan
        row["SeasonalNaive_RMSE"] = np.nan
        row["SeasonalNaive_MAPE"] = np.nan

    # Compatibilidad con summarize_final_results.py
    row["selected_model"] = "Naive"
    row["best_mape"] = row["Naive_MAPE"]
    return row


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        for modelo in ("Naive", "SeasonalNaive"):
            m = sub[f"{modelo}_MAPE"].dropna()
            filas.append({
                "categoria": cat,
                "modelo": modelo,
                "n_series": len(sub),
                "n_mape_valido": len(m),
                "MAE_mediana": round(sub[f"{modelo}_MAE"].median(), 2),
                "RMSE_mediana": round(sub[f"{modelo}_RMSE"].median(), 2),
                "MAPE_mediana": round(m.median(), 2) if len(m) else np.nan,
                "MAPE_media": round(m.mean(), 2) if len(m) else np.nan,
                "pct_MAPE_bajo_15": round(100 * (m < 15).mean(), 1) if len(m) else np.nan,
            })
    return pd.DataFrame(filas)


def main():
    print("Cargando datos (data/parsed.json + data/abc_classification.json)...")
    products, category_by_code = load_data()

    seleccion = [
        (c, info) for c, info in products.items()
        if category_by_code.get(c) in CATEGORIES
    ]
    print(f"Productos A+B a evaluar: {len(seleccion)}  |  ventana de prueba: {TEST_WEEKS} semanas")

    filas = []
    for i, (code, info) in enumerate(seleccion, 1):
        serie = info["series"]
        if len(serie) <= TEST_WEEKS + 1:
            continue
        filas.append(evaluate_product(code, info["name"], serie, category_by_code[code]))
        if i % 500 == 0:
            print(f"  {i}/{len(seleccion)}")

    df = pd.DataFrame(filas)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DETALLE, index=False)

    resumen = resumir(df)
    resumen.to_csv(OUTPUT_RESUMEN, index=False)

    print("\n" + "=" * 72)
    print("RESUMEN - Modelo baseline (mediana por categoria ABC)")
    print("=" * 72)
    print(resumen.to_string(index=False))
    print(f"\nDetalle:  {OUTPUT_DETALLE}")
    print(f"Resumen:  {OUTPUT_RESUMEN}")
    print("\nComparar 'MAPE_mediana' de Naive/SeasonalNaive contra la mediana de "
          "best_mape en\nmodel_comparison_results_A.csv / _B.csv "
          "(usar src/forecasting/summarize_final_results.py).")


if __name__ == "__main__":
    main()
