"""
MIA - Modelo de Inventario Automatizado
Comercial La Feria HLS CIA LTDA.

Genera el pronostico FUTURO real por producto (no el backtest de
train_forecasting_tiered.py). Reutiliza el modelo ya seleccionado como
ganador en model_comparison_results_A.csv / _B.csv, lo reentrenna sobre
la serie completa (train+test) y predice hacia adelante HORIZON_WEEKS.

No modifica el pipeline principal de entrenamiento -> script standalone,
igual que los de explicabilidad.

Requiere: estar en el mismo directorio (src/forecasting/) que
train_forecasting_tiered.py, para reutilizar fit_arima/fit_prophet/fit_lstm.

USO EN SERVIDOR (puede tardar, igual que el entrenamiento original):
    tmux new -s pronostico_futuro
    source venv/bin/activate
    python src/forecasting/generar_pronostico_futuro.py
    # Ctrl+B luego D
"""
import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from train_forecasting_tiered import (
    fit_arima, fit_prophet, fit_lstm, load_data,
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT_DIR = Path(__file__).resolve().parents[2]

RESULTS_A_PATH = ROOT_DIR / "model_comparison_results_A.csv"
RESULTS_B_PATH = ROOT_DIR / "model_comparison_results_B.csv"
FUTURE_OUTPUT_PATH = ROOT_DIR / "pronostico_futuro_producto.csv"

HORIZON_WEEKS = 4  # ~30 dias, consistente con HORIZONTE_DIAS del MILP

N_WORKERS = max(1, cpu_count() - 1)

FIT_FUNCTIONS = {"ARIMA": fit_arima, "Prophet": fit_prophet, "LSTM": fit_lstm}


def load_done_codes(csv_path: Path) -> set:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return set(df["product_code"].astype(str))
    return set()


def append_row(csv_path: Path, row: dict):
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def process_product_future(args):
    code, name, series_dict, categoria, modelo_ganador = args
    series = pd.Series(series_dict["values"], index=pd.to_datetime(series_dict["index"]))

    row = {
        "product_code": code,
        "product_name": name,
        "categoria": categoria,
        "modelo_ganador": modelo_ganador,
    }

    try:
        if categoria == "A":
            fit_fn = FIT_FUNCTIONS.get(modelo_ganador)
            if fit_fn is None:
                raise ValueError(f"Modelo desconocido: {modelo_ganador}")
            pred = fit_fn(series, HORIZON_WEEKS)  # reentrenna sobre TODA la serie
        else:  # categoria B: Holt-Winters
            seasonal_periods = 52 if len(series) > 104 else None
            model = ExponentialSmoothing(
                series.values,
                trend="add",
                seasonal="add" if seasonal_periods else None,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            ).fit()
            pred = np.clip(model.forecast(HORIZON_WEEKS), 0, None)

        row["demanda_pronosticada_semanal"] = json.dumps(np.array(pred).tolist())
        row["demanda_pronosticada_total"] = float(np.sum(pred))
        row["horizon_weeks"] = HORIZON_WEEKS
        row["status"] = "OK"
    except Exception as e:
        row["demanda_pronosticada_semanal"] = None
        row["demanda_pronosticada_total"] = None
        row["horizon_weeks"] = HORIZON_WEEKS
        row["status"] = f"ERROR: {e}"

    return row


def main():
    print(f"Nucleos disponibles: {cpu_count()}  |  Workers a usar: {N_WORKERS}")
    products, category_by_code = load_data()

    resultados_a = pd.read_csv(RESULTS_A_PATH)[["product_code", "selected_model"]] \
        if RESULTS_A_PATH.exists() else pd.DataFrame(columns=["product_code", "selected_model"])
    resultados_b = pd.read_csv(RESULTS_B_PATH)[["product_code", "selected_model"]] \
        if RESULTS_B_PATH.exists() else pd.DataFrame(columns=["product_code", "selected_model"])

    modelo_por_codigo = {}
    for _, r in resultados_a.iterrows():
        modelo_por_codigo[r["product_code"]] = ("A", r["selected_model"])
    for _, r in resultados_b.iterrows():
        modelo_por_codigo[r["product_code"]] = ("B", r["selected_model"])

    done = load_done_codes(FUTURE_OUTPUT_PATH)
    pending = [c for c in products if c in modelo_por_codigo and c not in done]
    print(f"Productos con modelo ya seleccionado: {len(modelo_por_codigo)}")
    print(f"Ya procesados: {len(done)}  |  Pendientes: {len(pending)}")

    def to_args(code):
        s = products[code]["series"]
        categoria, modelo = modelo_por_codigo[code]
        return (
            code, products[code]["name"],
            {"values": s.values.tolist(), "index": [d.strftime("%Y-%m-%d") for d in s.index]},
            categoria, modelo,
        )

    if pending:
        args_list = [to_args(c) for c in pending]
        t0 = time.time()
        with Pool(N_WORKERS) as pool:
            for i, row in enumerate(pool.imap_unordered(process_product_future, args_list), 1):
                append_row(FUTURE_OUTPUT_PATH, row)
                if i % 50 == 0 or i == len(pending):
                    elapsed = time.time() - t0
                    print(f"  [{i}/{len(pending)}] {row['product_code']} -> "
                          f"{row['status']}  |  {elapsed / 60:.1f} min transcurridos")

    print(f"\nListo. Resultados en {FUTURE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
