"""
Pipeline de entrenamiento de modelos de pronostico de demanda
Proyecto: Sistema inteligente de gestion de inventarios - Comercial La Feria
Replica la metodologia de la seccion 7.2 del documento capstone:
  - Comparacion ARIMA vs Prophet vs LSTM
  - Seleccion automatica del modelo de menor error por producto
  - Metricas: MAE, RMSE, MAPE
  - Particion cronologica (sin mezcla aleatoria)

Datos: extraidos de inventario.ventas (PostgreSQL) para los 15 productos
categoria A de mayor venta, agregados a nivel semanal.
"""
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------
# 1. Carga y preparacion de datos
# ---------------------------------------------------------------
with open(ROOT_DIR / "data" / "parsed.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

products = {}
for row in raw["rows"]:
    df = pd.DataFrame(row["series"])
    df["week"] = pd.to_datetime(df["week"])
    df = df.sort_values("week").reset_index(drop=True)
    # Reindexar a semanas continuas (rellenar huecos con 0 = sin venta esa semana)
    full_range = pd.date_range(df["week"].min(), df["week"].max(), freq="W-MON")
    df = df.set_index("week").reindex(full_range).fillna(0.0)
    df.index.name = "week"
    products[row["product_code"]] = {
        "name": row["product_name"],
        "series": df["units"].astype(float),
    }

print(f"Products loaded: {len(products)}")
for code, info in products.items():
    print(f"  {code:16s} {info['name'][:40]:40s} {len(info['series'])} weeks")

# ---------------------------------------------------------------
# 2. Particion cronologica: train / test (ultimas 12 semanas = test)
# ---------------------------------------------------------------
TEST_WEEKS = 12

def split_series(s: pd.Series):
    train = s.iloc[:-TEST_WEEKS]
    test = s.iloc[-TEST_WEEKS:]
    return train, test

# ---------------------------------------------------------------
# 3. Metricas
# ---------------------------------------------------------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

# ---------------------------------------------------------------
# 4. Modelo ARIMA (seleccion automatica de orden con pmdarima)
# ---------------------------------------------------------------
def fit_arima(train: pd.Series, horizon: int):
    import pmdarima as pm
    model = pm.auto_arima(
        train.values,
        seasonal=True, m=52 if len(train) > 104 else 1,
        stepwise=True, suppress_warnings=True,
        error_action="ignore", max_p=3, max_q=3, max_P=1, max_Q=1,
    )
    forecast = model.predict(n_periods=horizon)
    return np.clip(forecast, 0, None)

# ---------------------------------------------------------------
# 5. Modelo Prophet
# ---------------------------------------------------------------
def fit_prophet(train: pd.Series, horizon: int):
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
    pred = fcst["yhat"].values[-horizon:]
    return np.clip(pred, 0, None)

# ---------------------------------------------------------------
# 6. Modelo LSTM (univariado, ventana deslizante)
# ---------------------------------------------------------------
def fit_lstm(train: pd.Series, horizon: int, window: int = 8, epochs: int = 60):
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

    # Prediccion recursiva multi-paso
    window_vals = list(scaled[-window:])
    preds_scaled = []
    for _ in range(horizon):
        x_in = np.array(window_vals[-window:]).reshape(1, window, 1)
        p = model.predict(x_in, verbose=0)[0, 0]
        preds_scaled.append(p)
        window_vals.append(p)

    preds = np.array(preds_scaled) * rng + vmin
    return np.clip(preds, 0, None)

# ---------------------------------------------------------------
# 7. Entrenamiento y comparacion por producto
# ---------------------------------------------------------------
results = []
predictions_detail = {}

for code, info in products.items():
    series = info["series"]
    name = info["name"]
    train, test = split_series(series)
    horizon = len(test)
    print(f"\n=== {code} - {name} ===  (train={len(train)}, test={horizon})")

    row = {"product_code": code, "product_name": name, "n_train": len(train), "n_test": horizon}
    preds_prod = {"test_dates": [d.strftime("%Y-%m-%d") for d in test.index], "actual": test.values.tolist()}

    # --- ARIMA ---
    try:
        pred_arima = fit_arima(train, horizon)
        row["ARIMA_MAE"] = mae(test.values, pred_arima)
        row["ARIMA_RMSE"] = rmse(test.values, pred_arima)
        row["ARIMA_MAPE"] = mape(test.values, pred_arima)
        preds_prod["ARIMA"] = pred_arima.tolist()
        print(f"  ARIMA   -> MAE={row['ARIMA_MAE']:.1f}  RMSE={row['ARIMA_RMSE']:.1f}  MAPE={row['ARIMA_MAPE']:.1f}%")
    except Exception as e:
        print(f"  ARIMA   -> ERROR: {e}")
        row["ARIMA_MAE"] = row["ARIMA_RMSE"] = row["ARIMA_MAPE"] = np.nan

    # --- Prophet ---
    try:
        pred_prophet = fit_prophet(train, horizon)
        row["Prophet_MAE"] = mae(test.values, pred_prophet)
        row["Prophet_RMSE"] = rmse(test.values, pred_prophet)
        row["Prophet_MAPE"] = mape(test.values, pred_prophet)
        preds_prod["Prophet"] = pred_prophet.tolist()
        print(f"  Prophet -> MAE={row['Prophet_MAE']:.1f}  RMSE={row['Prophet_RMSE']:.1f}  MAPE={row['Prophet_MAPE']:.1f}%")
    except Exception as e:
        print(f"  Prophet -> ERROR: {e}")
        row["Prophet_MAE"] = row["Prophet_RMSE"] = row["Prophet_MAPE"] = np.nan

    # --- LSTM ---
    try:
        pred_lstm = fit_lstm(train, horizon)
        row["LSTM_MAE"] = mae(test.values, pred_lstm)
        row["LSTM_RMSE"] = rmse(test.values, pred_lstm)
        row["LSTM_MAPE"] = mape(test.values, pred_lstm)
        preds_prod["LSTM"] = pred_lstm.tolist()
        print(f"  LSTM    -> MAE={row['LSTM_MAE']:.1f}  RMSE={row['LSTM_RMSE']:.1f}  MAPE={row['LSTM_MAPE']:.1f}%")
    except Exception as e:
        print(f"  LSTM    -> ERROR: {e}")
        row["LSTM_MAE"] = row["LSTM_RMSE"] = row["LSTM_MAPE"] = np.nan

    # --- Seleccion automatica del mejor modelo (menor MAPE) ---
    candidates = {k: row.get(f"{k}_MAPE", np.nan) for k in ["ARIMA", "Prophet", "LSTM"]}
    candidates = {k: v for k, v in candidates.items() if not np.isnan(v)}
    best = min(candidates, key=candidates.get) if candidates else None
    row["selected_model"] = best
    row["best_mape"] = candidates.get(best, np.nan)
    print(f"  >> Selected model: {best} (MAPE={row['best_mape']:.1f}%)")

    results.append(row)
    predictions_detail[code] = preds_prod

# ---------------------------------------------------------------
# 8. Guardar resultados
# ---------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(ROOT_DIR / "model_comparison_results.csv", index=False)

with open(ROOT_DIR / "prediction_details.json", "w", encoding="utf-8") as f:
    json.dump(predictions_detail, f, ensure_ascii=False, indent=2)

print("\n\n================ FINAL SUMMARY ================")
print(results_df[["product_code", "product_name", "ARIMA_MAPE", "Prophet_MAPE", "LSTM_MAPE", "selected_model"]].to_string(index=False))

print("\nSelected model distribution:")
print(results_df["selected_model"].value_counts())

print(f"\nAverage MAPE of the selected model: {results_df['best_mape'].mean():.2f}%")
print(f"Products with MAPE < 15% (project threshold): {(results_df['best_mape'] < 15).sum()} of {len(results_df)}")
