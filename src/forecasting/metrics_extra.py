"""
metrics_extra.py

Relectura de los resultados de forecasting YA generados para calcular tres
metricas adicionales que pide el documento (punto 4): sMAPE, WAPE y MASE.

NO reentrena nada. Usa:
  - data/prediction_details_A.jsonl        (real vs predicho por semana, cat. A)
  - data/prediction_details_B_sample.jsonl (idem, muestra n=349 de cat. B)
  - data/model_comparison_results_A.csv / _B.csv / _B_sample.csv (modelo ganador y MAE)
  - data/model_comparison_results_baseline.csv (opcional, si existe)
  - data/parsed.json                        (serie de train, para el denominador de MASE
                                             y el real del test de cat. B completa)

Cobertura:
  - Categoria A (1876 productos): sMAPE + WAPE + MASE del modelo ganador.
  - Muestra B (349): sMAPE + WAPE + MASE del modelo ganador.
  - Categoria B completa (Holt-Winters): SOLO WAPE + MASE. El sMAPE necesita el
    valor predicho semana a semana y train_forecasting_tiered.py no lo guardo
    para Holt-Winters (solo guarda el MAE agregado). Para obtenerlo hay que
    volver a correr la parte B guardando las predicciones (barato).
  - Baseline Naive: sMAPE + WAPE + MASE (se recalcula la prediccion naive).

Definiciones (Hyndman & Koehler, 2006):
  sMAPE = (100/n) * sum( 2*|F-A| / (|A|+|F|) )         # 0-200 %, se omite |A|+|F|=0
  WAPE  = 100 * sum|A-F| / sum|A|                       # error total ponderado
  MASE  = MAE_test / MAE_naive_train                    # naive de 1 paso sobre el train
          MAE_naive_train = mean_{t>=2} |Y_t - Y_{t-1}|
  MASE_s (estacional, m=52): denominador = mean_{t>m} |Y_t - Y_{t-m}|

Salida:
  data/metrics_extra_detalle.csv   (por producto)
  data/metrics_extra_resumen.csv   (mediana por categoria y modelo)

Uso:
    python src/forecasting/metrics_extra.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_forecasting_tiered import DATA_DIR, TEST_WEEKS, load_data, split_series

SEASONAL_PERIOD = 52

PRED_A = DATA_DIR / "prediction_details_A.jsonl"
PRED_B_SAMPLE = DATA_DIR / "prediction_details_B_sample.jsonl"
RESULTS_A = DATA_DIR / "model_comparison_results_A.csv"
RESULTS_B = DATA_DIR / "model_comparison_results_B.csv"
RESULTS_B_SAMPLE = DATA_DIR / "model_comparison_results_B_sample.csv"
RESULTS_BASELINE = DATA_DIR / "model_comparison_results_baseline.csv"

OUT_DETALLE = DATA_DIR / "metrics_extra_detalle.csv"
OUT_RESUMEN = DATA_DIR / "metrics_extra_resumen.csv"


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def smape(actual, pred):
    a = np.abs(np.asarray(actual, dtype=float))
    f = np.abs(np.asarray(pred, dtype=float))
    denom = a + f
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2 * np.abs(np.asarray(pred) - np.asarray(actual))[mask] / denom[mask]) * 100)


def wape(actual, pred):
    a = np.asarray(actual, dtype=float)
    f = np.asarray(pred, dtype=float)
    tot = np.sum(np.abs(a))
    if tot == 0:
        return np.nan
    return float(np.sum(np.abs(a - f)) / tot * 100)


def naive_train_mae(train_values, m=1):
    y = np.asarray(train_values, dtype=float)
    if len(y) <= m:
        return np.nan
    d = np.mean(np.abs(y[m:] - y[:-m]))
    return float(d) if d > 0 else np.nan


def mase(mae_test, train_values, m=1):
    d = naive_train_mae(train_values, m)
    if d is None or np.isnan(d):
        return np.nan
    return float(mae_test / d)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def read_jsonl_predictions(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.update(rec)
    return out


def selected_model_map(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return dict(zip(df["product_code"].astype(str), df["selected_model"]))


# ---------------------------------------------------------------------------
# Bloques de calculo
# ---------------------------------------------------------------------------
def metrics_from_weekly(preds: dict, model_by_code: dict, products: dict,
                        category: str, source: str) -> list[dict]:
    """A y muestra B: hay real y predicho por semana -> las 3 metricas."""
    rows = []
    for code, det in preds.items():
        model = model_by_code.get(str(code))
        if model is None or model not in det:
            continue
        actual = np.asarray(det["actual"], dtype=float)
        pred = np.asarray(det[model], dtype=float)
        mae_test = float(np.mean(np.abs(actual - pred)))
        train = products[code]["series"].iloc[:-TEST_WEEKS].values if code in products else None
        rows.append({
            "product_code": code,
            "category": category,
            "source": source,
            "model": model,
            "n_test": len(actual),
            "MAE": round(mae_test, 3),
            "sMAPE": round(smape(actual, pred), 2),
            "WAPE": round(wape(actual, pred), 2),
            "MASE": round(mase(mae_test, train, 1), 3) if train is not None else np.nan,
            "MASE_s52": round(mase(mae_test, train, SEASONAL_PERIOD), 3) if train is not None else np.nan,
        })
    return rows


def metrics_b_full(products: dict, category_by_code: dict) -> list[dict]:
    """Cat. B completa (Holt-Winters): solo WAPE + MASE (no hay predicho semanal)."""
    df = pd.read_csv(RESULTS_B)
    rows = []
    for _, r in df.iterrows():
        code = str(r["product_code"])
        mae_test = r["HoltWinters_MAE"]
        if pd.isna(mae_test) or code not in products:
            continue
        serie = products[code]["series"]
        train, test = split_series(serie)
        sum_actual = float(np.sum(np.abs(test.values)))
        rows.append({
            "product_code": code,
            "category": "B",
            "source": "B_full",
            "model": "HoltWinters",
            "n_test": len(test),
            "MAE": round(float(mae_test), 3),
            "sMAPE": np.nan,  # requiere predicho semanal (no guardado para HW)
            "WAPE": round(mae_test * len(test) / sum_actual * 100, 2) if sum_actual > 0 else np.nan,
            "MASE": round(mase(mae_test, train.values, 1), 3),
            "MASE_s52": round(mase(mae_test, train.values, SEASONAL_PERIOD), 3),
        })
    return rows


def metrics_baseline(products: dict, category_by_code: dict) -> list[dict]:
    """Baseline naive: se recalcula la prediccion (ultimo valor de train)."""
    if not RESULTS_BASELINE.exists():
        return []
    df = pd.read_csv(RESULTS_BASELINE)
    rows = []
    for _, r in df.iterrows():
        code = str(r["product_code"])
        if code not in products:
            continue
        serie = products[code]["series"]
        train, test = split_series(serie)
        actual = test.values.astype(float)
        pred = np.full(len(actual), float(train.iloc[-1]))
        mae_test = float(np.mean(np.abs(actual - pred)))
        rows.append({
            "product_code": code,
            "category": r["category"],
            "source": "baseline",
            "model": "Naive",
            "n_test": len(actual),
            "MAE": round(mae_test, 3),
            "sMAPE": round(smape(actual, pred), 2),
            "WAPE": round(wape(actual, pred), 2),
            "MASE": round(mase(mae_test, train.values, 1), 3),
            "MASE_s52": round(mase(mae_test, train.values, SEASONAL_PERIOD), 3),
        })
    return rows


def _agg(g) -> pd.DataFrame:
    out = g.agg(
        n=("product_code", "count"),
        sMAPE_mediana=("sMAPE", "median"),
        WAPE_mediana=("WAPE", "median"),
        MASE_mediana=("MASE", "median"),
        MASE_s52_mediana=("MASE_s52", "median"),
        MASE_pct_bajo_1=("MASE", lambda s: round(100 * (s < 1).mean(), 1)),
    ).reset_index()
    return out


def resumen(df: pd.DataFrame) -> pd.DataFrame:
    por_modelo = _agg(df.groupby(["source", "category", "model"]))
    # Fila agregada por (source, category): el modelo ganador de cada producto
    overall = _agg(df.groupby(["source", "category"]))
    overall["model"] = "(seleccionado)"
    out = pd.concat([overall[por_modelo.columns], por_modelo], ignore_index=True)
    out = out.sort_values(["source", "category", "model"]).reset_index(drop=True)
    return out.round(2)


def main():
    print("Cargando serie base (data/parsed.json)...")
    products, category_by_code = load_data()

    rows = []

    print("Categoria A (prediction_details_A.jsonl)...")
    rows += metrics_from_weekly(
        read_jsonl_predictions(PRED_A), selected_model_map(RESULTS_A),
        products, "A", "A_full",
    )

    if PRED_B_SAMPLE.exists():
        print("Muestra B (prediction_details_B_sample.jsonl)...")
        rows += metrics_from_weekly(
            read_jsonl_predictions(PRED_B_SAMPLE), selected_model_map(RESULTS_B_SAMPLE),
            products, "B", "B_sample",
        )

    print("Categoria B completa (Holt-Winters, solo WAPE + MASE)...")
    rows += metrics_b_full(products, category_by_code)

    print("Baseline Naive...")
    rows += metrics_baseline(products, category_by_code)

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DETALLE, index=False)

    res = resumen(df)
    res.to_csv(OUT_RESUMEN, index=False)

    print("\n" + "=" * 90)
    print("RESUMEN - sMAPE / WAPE / MASE  (mediana por categoria y modelo)")
    print("=" * 90)
    print(res.to_string(index=False))
    print(f"\nDetalle: {OUT_DETALLE}")
    print(f"Resumen: {OUT_RESUMEN}")
    print("\nNota: sMAPE de 'B_full' queda vacio porque train_forecasting_tiered.py no guarda")
    print("la prediccion semana a semana de Holt-Winters (solo el MAE agregado).")


if __name__ == "__main__":
    main()
