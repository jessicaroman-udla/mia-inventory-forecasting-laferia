  """
Entrenamiento de forecasting con tratamiento diferenciado por categoria ABC.
Proyecto: Sistema inteligente de gestion de inventarios - Comercial La Feria

  - Categoria A: comparacion completa ARIMA vs Prophet vs LSTM + seleccion
    automatica del mejor modelo por producto (igual que train_forecasting.py
    original, pero corriendo SOLO sobre categoria A).
  - Categoria B: metodo automatizado liviano (suavizado exponencial /
    Holt-Winters), sin comparar multiples modelos, para que sea viable en
    tiempo sobre miles de productos. Sigue siendo "pronostico automatizado"
    -> cumple el indicador de cobertura A+B del documento de alcance.

DISEÑADO PARA SERVIDOR:
  - Paraleliza por producto usando multiprocessing, aprovechando todos los
    nucleos disponibles (detectado automaticamente con cpu_count()).
  - CHECKPOINTING: cada producto procesado se guarda de inmediato en CSV.
    Si el proceso se corta (SSH caido, timeout, reinicio), solo hay que
    volver a correr el script: los productos ya hechos se saltan
    automaticamente.

USO RECOMENDADO EN SERVIDOR (para que siga corriendo aunque cierres la
sesion SSH):
    tmux new -s forecasting
    source venv/bin/activate
    python src/forecasting/train_forecasting_tiered.py
    # Ctrl+B luego D  -> sale del tmux sin matar el proceso
    # tmux attach -t forecasting  -> vuelve a conectarte a verlo despues

Requisitos adicionales sobre los que ya tenias:
    pip install statsmodels  (Holt-Winters para categoria B; probablemente
    ya lo tienes instalado porque pmdarima depende de statsmodels)
"""
import json
import os
import time
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[2]

RESULTS_A_PATH = ROOT_DIR / "model_comparison_results_A.csv"
RESULTS_B_PATH = ROOT_DIR / "model_comparison_results_B.csv"
PREDICTIONS_A_PATH = ROOT_DIR / "prediction_details_A.jsonl"

TEST_WEEKS = 12

# Deja 1 nucleo libre para no ahogar el servidor si corren otros procesos
N_WORKERS = max(1, cpu_count() - 1)


# ---------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------
# Carga de datos: parsed.json (series) + abc_classification.json (categoria)
# ---------------------------------------------------------------
def load_data():
    with open(ROOT_DIR / "data" / "parsed.json", "r", encoding="utf-8") as f:
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

    with open(ROOT_DIR / "data" / "abc_classification.json", "r", encoding="utf-8") as f:
        abc = json.load(f)
    category_by_code = {p["product_code"]: p["category"] for p in abc["products"]}

    return products, category_by_code


def split_series(s: pd.Series):
    train = s.iloc[:-TEST_WEEKS]
    test = s.iloc[-TEST_WEEKS:]
    return train, test


# ---------------------------------------------------------------
# Modelos - Categoria A (comparacion completa, igual que el pipeline original)
# ---------------------------------------------------------------
def fit_arima(train, horizon):
    import pmdarima as pm
    model = pm.auto_arima(
        train.values, seasonal=True, m=52 if len(train) > 104 else 1,
        stepwise=True, suppress_warnings=True, error_action="ignore",
        max_p=3, max_q=3, max_P=1, max_Q=1,
    )
    return np.clip(model.predict(n_periods=horizon), 0, None)


def fit_prophet(train, horizon):
    from prophet import Prophet
    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(weekly_seasonality=False, yearly_seasonality=True,
                seasonality_mode="additive", interval_width=0.8)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=horizon, freq="W-MON")
    fcst = m.predict(future)
    return np.clip(fcst["yhat"].values[-horizon:], 0, None)


def fit_lstm(train, horizon, window=8, epochs=60):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    tf.random.set_seed(42)

    values = train.values.astype("float32")
    vmin, vmax = values.min(), values.max()
    rng = (vmax - vmin) if vmax > vmin else 1.0
    scaled = (values - vmin) / rng

    if len(scaled) <= window + 1:
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


def process_product_A(args):
    """Worker: comparacion completa de 3 modelos para un producto categoria A."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    code, name, series_dict = args
    series = pd.Series(series_dict["values"], index=pd.to_datetime(series_dict["index"]))
    train, test = split_series(series)
    horizon = len(test)

    row = {"product_code": code, "product_name": name, "n_train": len(train), "n_test": horizon}
    preds = {"test_dates": [d.strftime("%Y-%m-%d") for d in test.index], "actual": test.values.tolist()}

    for model_name, fit_fn in [("ARIMA", fit_arima), ("Prophet", fit_prophet), ("LSTM", fit_lstm)]:
        try:
            pred = fit_fn(train, horizon)
            row[f"{model_name}_MAE"] = mae(test.values, pred)
            row[f"{model_name}_RMSE"] = rmse(test.values, pred)
            row[f"{model_name}_MAPE"] = mape(test.values, pred)
            preds[model_name] = pred.tolist()
        except Exception:
            row[f"{model_name}_MAE"] = row[f"{model_name}_RMSE"] = row[f"{model_name}_MAPE"] = np.nan

    candidates = {k: row.get(f"{k}_MAPE", np.nan) for k in ["ARIMA", "Prophet", "LSTM"]}
    candidates = {k: v for k, v in candidates.items() if not np.isnan(v)}
    best = min(candidates, key=candidates.get) if candidates else None
    row["selected_model"] = best
    row["best_mape"] = candidates.get(best, np.nan)

    return row, code, preds


# ---------------------------------------------------------------
# Modelo - Categoria B (metodo liviano automatizado: Holt-Winters)
# ---------------------------------------------------------------
def process_product_B(args):
    """Worker: suavizado exponencial (Holt-Winters) para un producto categoria B."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    code, name, series_dict = args
    series = pd.Series(series_dict["values"], index=pd.to_datetime(series_dict["index"]))
    train, test = split_series(series)
    horizon = len(test)

    row = {"product_code": code, "product_name": name, "n_train": len(train), "n_test": horizon}
    try:
        seasonal_periods = 52 if len(train) > 104 else None
        model = ExponentialSmoothing(
            train.values,
            trend="add",
            seasonal="add" if seasonal_periods else None,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit()
        pred = np.clip(model.forecast(horizon), 0, None)
        row["HoltWinters_MAE"] = mae(test.values, pred)
        row["HoltWinters_RMSE"] = rmse(test.values, pred)
        row["HoltWinters_MAPE"] = mape(test.values, pred)
        row["selected_model"] = "HoltWinters"
        row["best_mape"] = row["HoltWinters_MAPE"]
    except Exception:
        row["HoltWinters_MAE"] = row["HoltWinters_RMSE"] = row["HoltWinters_MAPE"] = np.nan
        row["selected_model"] = None
        row["best_mape"] = np.nan

    return row


# ---------------------------------------------------------------
# Checkpointing: saltar productos ya procesados, ir guardando incrementalmente
# ---------------------------------------------------------------
def load_done_codes(csv_path: Path) -> set:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return set(df["product_code"].astype(str))
    return set()


def append_row(csv_path: Path, row: dict):
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    print(f"Nucleos disponibles: {cpu_count()}  |  Workers a usar: {N_WORKERS}")
    products, category_by_code = load_data()

    codes_A = [c for c in products if category_by_code.get(c) == "A"]
    codes_B = [c for c in products if category_by_code.get(c) == "B"]
    print(f"Categoria A: {len(codes_A)} productos  |  Categoria B: {len(codes_B)} productos")

    done_A = load_done_codes(RESULTS_A_PATH)
    done_B = load_done_codes(RESULTS_B_PATH)
    pending_A = [c for c in codes_A if c not in done_A]
    pending_B = [c for c in codes_B if c not in done_B]
    print(f"Ya procesados -> A: {len(done_A)}  B: {len(done_B)}")
    print(f"Pendientes    -> A: {len(pending_A)}  B: {len(pending_B)}")

    def to_series_dict(code):
        s = products[code]["series"]
        return (code, products[code]["name"],
                {"values": s.values.tolist(), "index": [d.strftime("%Y-%m-%d") for d in s.index]})

    # --- Categoria A: comparacion completa, en paralelo ---
    if pending_A:
        print(f"\n=== Entrenando categoria A ({len(pending_A)} productos, comparacion completa) ===")
        args_list = [to_series_dict(c) for c in pending_A]
        t0 = time.time()
        with Pool(N_WORKERS) as pool:
            for i, (row, code, preds) in enumerate(pool.imap_unordered(process_product_A, args_list), 1):
                append_row(RESULTS_A_PATH, row)
                with open(PREDICTIONS_A_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({code: preds}, ensure_ascii=False) + "\n")
                if i % 25 == 0 or i == len(pending_A):
                    elapsed = time.time() - t0
                    mp = row.get("best_mape", float("nan"))
                    print(f"  [{i}/{len(pending_A)}] {row['product_code']} -> {row.get('selected_model')} "
                          f"(MAPE={mp:.1f}%)  |  {elapsed/60:.1f} min transcurridos")

    # --- Categoria B: metodo liviano, en paralelo ---
    if pending_B:
        print(f"\n=== Entrenando categoria B ({len(pending_B)} productos, Holt-Winters ligero) ===")
        args_list = [to_series_dict(c) for c in pending_B]
        t0 = time.time()
        with Pool(N_WORKERS) as pool:
            for i, row in enumerate(pool.imap_unordered(process_product_B, args_list), 1):
                append_row(RESULTS_B_PATH, row)
                if i % 200 == 0 or i == len(pending_B):
                    elapsed = time.time() - t0
                    mp = row.get("best_mape", float("nan"))
                    print(f"  [{i}/{len(pending_B)}] MAPE={mp:.1f}%  |  {elapsed/60:.1f} min transcurridos")

    print("\n================ RESUMEN ================")
    if RESULTS_A_PATH.exists():
        df_a = pd.read_csv(RESULTS_A_PATH)
        print(f"Categoria A: {len(df_a)} productos procesados. MAPE promedio: {df_a['best_mape'].mean():.2f}%")
        print(f"  Con MAPE < 15%: {(df_a['best_mape'] < 15).sum()} de {len(df_a)}")
    if RESULTS_B_PATH.exists():
        df_b = pd.read_csv(RESULTS_B_PATH)
        print(f"Categoria B: {len(df_b)} productos procesados. MAPE promedio: {df_b['best_mape'].mean():.2f}%")


if __name__ == "__main__":
    main()