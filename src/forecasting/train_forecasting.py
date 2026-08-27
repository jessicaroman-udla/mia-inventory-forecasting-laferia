"""
Pipeline de entrenamiento de modelos de pronostico de demanda (track de referencia).
Proyecto: Sistema inteligente de gestion de inventarios - Comercial La Feria

Replica la metodologia de la seccion 7.2 del documento capstone:
  - Comparacion ARIMA vs Prophet vs LSTM
  - Seleccion automatica del modelo de menor error (MAPE) por producto
  - Metricas: MAE, RMSE, MAPE
  - Particion cronologica (sin mezcla aleatoria)

Corre la comparacion completa sobre TODOS los productos de data/parsed.json
(sin distinguir A/B). Es la version simple para laptop / pocos productos; para
el catalogo completo A+B en servidor usar train_forecasting_tiered.py.

Entrada:  data/parsed.json
Salida:   data/model_comparison_results.csv
          data/prediction_details.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

PARSED_PATH = DATA_DIR / "parsed.json"
RESULTS_PATH = DATA_DIR / "model_comparison_results.csv"
PREDICTIONS_PATH = DATA_DIR / "prediction_details.json"

TEST_WEEKS = 12  # ultimas 12 semanas reservadas como conjunto de prueba


# ---------------------------------------------------------------
# 1. Carga y preparacion de datos
# ---------------------------------------------------------------
def load_products(parsed_path: Path = PARSED_PATH) -> dict:
    """Devuelve {codigo: {'name': str, 'series': pd.Series semanal}}."""
    with open(parsed_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    products = {}
    for row in raw["rows"]:
        df = pd.DataFrame(row["series"])
        df["week"] = pd.to_datetime(df["week"])
        df = df.sort_values("week").reset_index(drop=True)
        # Reindexar a semanas continuas (huecos = 0 -> sin venta esa semana)
        full_range = pd.date_range(df["week"].min(), df["week"].max(), freq="W-MON")
        df = df.set_index("week").reindex(full_range).fillna(0.0)
        df.index.name = "week"
        products[row["product_code"]] = {
            "name": row["product_name"],
            "series": df["units"].astype(float),
        }
    return products


def split_series(s: pd.Series):
    """Particion cronologica: las ultimas TEST_WEEKS semanas son test."""
    return s.iloc[:-TEST_WEEKS], s.iloc[-TEST_WEEKS:]


# ---------------------------------------------------------------
# 2. Metricas
# ---------------------------------------------------------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------
# 3. Modelos
# ---------------------------------------------------------------
def fit_arima(train: pd.Series, horizon: int):
    """ARIMA con seleccion automatica de orden (pmdarima.auto_arima)."""
    import pmdarima as pm

    model = pm.auto_arima(
        train.values,
        seasonal=True, m=52 if len(train) > 104 else 1,
        stepwise=True, suppress_warnings=True,
        error_action="ignore", max_p=3, max_q=3, max_P=1, max_Q=1,
    )
    return np.clip(model.predict(n_periods=horizon), 0, None)


def fit_prophet(train: pd.Series, horizon: int):
    """Prophet con estacionalidad anual aditiva."""
    from prophet import Prophet

    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(
        weekly_seasonality=False,
        yearly_seasonality=True,
        seasonality_mode="additive",
        interval_width=0.8,
    )
    m.fit(dfp)
    future = m.make_future_dataframe(periods=horizon, freq="W-MON")
    fcst = m.predict(future)
    return np.clip(fcst["yhat"].values[-horizon:], 0, None)


def fit_lstm(train: pd.Series, horizon: int, window: int = 8, epochs: int = 60):
    """LSTM univariado con ventana deslizante y prediccion recursiva."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input

    tf.random.set_seed(42)

    values = train.values.astype("float32")
    vmin, vmax = values.min(), values.max()
    rng = (vmax - vmin) if vmax > vmin else 1.0
    scaled = (values - vmin) / rng

    if len(scaled) <= window + 1:
        # Serie muy corta: respaldo simple (promedio movil)
        return np.full(horizon, values[-window:].mean())

    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i + window])
        y.append(scaled[i + window])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)

    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(32, activation="tanh"),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=8, verbose=0)

    window_vals = list(scaled[-window:])
    preds_scaled = []
    for _ in range(horizon):
        x_in = np.array(window_vals[-window:]).reshape(1, window, 1)
        p = model.predict(x_in, verbose=0)[0, 0]
        preds_scaled.append(p)
        window_vals.append(p)

    return np.clip(np.array(preds_scaled) * rng + vmin, 0, None)


MODELS = [("ARIMA", fit_arima), ("Prophet", fit_prophet), ("LSTM", fit_lstm)]


# ---------------------------------------------------------------
# 4. Entrenamiento y comparacion por producto
# ---------------------------------------------------------------
def evaluate_product(code: str, name: str, series: pd.Series):
    """Entrena los 3 modelos para un producto y devuelve (fila, detalle)."""
    train, test = split_series(series)
    horizon = len(test)
    print(f"\n=== {code} - {name} ===  (train={len(train)}, test={horizon})")

    row = {"product_code": code, "product_name": name,
           "n_train": len(train), "n_test": horizon}
    detail = {"test_dates": [d.strftime("%Y-%m-%d") for d in test.index],
              "actual": test.values.tolist()}

    for model_name, fit_fn in MODELS:
        try:
            pred = fit_fn(train, horizon)
            row[f"{model_name}_MAE"] = mae(test.values, pred)
            row[f"{model_name}_RMSE"] = rmse(test.values, pred)
            row[f"{model_name}_MAPE"] = mape(test.values, pred)
            detail[model_name] = np.asarray(pred).tolist()
            print(f"  {model_name:8s}-> MAE={row[f'{model_name}_MAE']:.1f}  "
                  f"RMSE={row[f'{model_name}_RMSE']:.1f}  MAPE={row[f'{model_name}_MAPE']:.1f}%")
        except Exception as e:  # noqa: BLE001 - se registra y se sigue con el resto
            print(f"  {model_name:8s}-> ERROR: {e}")
            row[f"{model_name}_MAE"] = row[f"{model_name}_RMSE"] = row[f"{model_name}_MAPE"] = np.nan

    candidates = {k: row.get(f"{k}_MAPE", np.nan) for k, _ in MODELS}
    candidates = {k: v for k, v in candidates.items() if not np.isnan(v)}
    best = min(candidates, key=candidates.get) if candidates else None
    row["selected_model"] = best
    row["best_mape"] = candidates.get(best, np.nan)
    print(f"  >> Modelo seleccionado: {best} (MAPE={row['best_mape']:.1f}%)")

    return row, detail


def main():
    products = load_products()
    print(f"Productos cargados: {len(products)}")

    results, predictions_detail = [], {}
    for code, info in products.items():
        row, detail = evaluate_product(code, info["name"], info["series"])
        results.append(row)
        predictions_detail[code] = detail

    results_df = pd.DataFrame(results)
    DATA_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions_detail, f, ensure_ascii=False, indent=2)

    print("\n\n================ RESUMEN FINAL ================")
    print(results_df[["product_code", "product_name", "ARIMA_MAPE", "Prophet_MAPE",
                      "LSTM_MAPE", "selected_model"]].to_string(index=False))
    print("\nDistribucion del modelo seleccionado:")
    print(results_df["selected_model"].value_counts())
    print(f"\nMAPE promedio del modelo seleccionado: {results_df['best_mape'].mean():.2f}%")
    print(f"Productos con MAPE < 15%: {(results_df['best_mape'] < 15).sum()} de {len(results_df)}")
    print(f"\nResultados guardados en {RESULTS_PATH} y {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
