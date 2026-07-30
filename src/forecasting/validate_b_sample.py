"""
Validacion de la estrategia diferenciada A/B: corre la comparacion completa
de modelos (ARIMA vs Prophet vs LSTM) sobre una MUESTRA representativa de
categoria B, para contrastar contra los resultados ya obtenidos con
Holt-Winters (metodo liviano usado en produccion para B).

JUSTIFICACION DEL TAMANO DE MUESTRA:
    Poblacion (N) = 3735 productos categoria B
    Nivel de confianza = 95% (Z = 1.96)
    Margen de error = 5%
    p = 0.5 (maxima varianza, escenario mas conservador)

    n = (N * Z^2 * p * (1-p)) / (e^2 * (N-1) + Z^2 * p * (1-p))
    n = 349 productos

USO:
    python src/forecasting/validate_b_sample.py

    Reutiliza load_data(), process_product_A(), split_series(), etc.
    desde train_forecasting_tiered.py (debe estar en el mismo directorio).

    Guarda resultados en model_comparison_results_B_sample.csv, con el
    mismo checkpointing que el script principal (si se corta, continua
    donde se quedo).
"""
import logging
import os
import random
import time
from multiprocessing import Pool, cpu_count

import pandas as pd

# Silenciar ruido de cmdstanpy/Prophet y TensorFlow/absl en el log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

from train_forecasting_tiered import (
    load_data,
    process_product_A,
    load_done_codes,
    append_row,
    ROOT_DIR,
)

SAMPLE_SIZE = 349
RANDOM_SEED = 42  # fijo para reproducibilidad -- documentar en la tesis

RESULTS_SAMPLE_PATH = ROOT_DIR / "model_comparison_results_B_sample.csv"
PREDICTIONS_SAMPLE_PATH = ROOT_DIR / "prediction_details_B_sample.jsonl"

N_WORKERS = max(1, cpu_count() - 1)


def main():
    print(f"Nucleos disponibles: {cpu_count()}  |  Workers a usar: {N_WORKERS}")
    products, category_by_code = load_data()

    codes_B = [c for c in products if category_by_code.get(c) == "B"]
    print(f"Poblacion categoria B: {len(codes_B)} productos")

    rng = random.Random(RANDOM_SEED)
    sample_codes = rng.sample(codes_B, min(SAMPLE_SIZE, len(codes_B)))
    print(f"Muestra seleccionada (seed={RANDOM_SEED}): {len(sample_codes)} productos")

    done = load_done_codes(RESULTS_SAMPLE_PATH)
    pending = [c for c in sample_codes if c not in done]
    print(f"Ya procesados: {len(done)}  |  Pendientes: {len(pending)}")

    def to_series_dict(code):
        s = products[code]["series"]
        return (code, products[code]["name"],
                {"values": s.values.tolist(), "index": [d.strftime("%Y-%m-%d") for d in s.index]})

    if pending:
        print(f"\n=== Comparacion completa sobre muestra B ({len(pending)} productos) ===")
        args_list = [to_series_dict(c) for c in pending]
        t0 = time.time()
        with Pool(N_WORKERS) as pool:
            for i, (row, code, preds) in enumerate(pool.imap_unordered(process_product_A, args_list), 1):
                append_row(RESULTS_SAMPLE_PATH, row)
                with open(PREDICTIONS_SAMPLE_PATH, "a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps({code: preds}, ensure_ascii=False) + "\n")
                if i % 25 == 0 or i == len(pending):
                    elapsed = time.time() - t0
                    mp = row.get("best_mape", float("nan"))
                    print(f"  [{i}/{len(pending)}] {row['product_code']} -> {row.get('selected_model')} "
                          f"(MAPE={mp:.1f}%)  |  {elapsed / 60:.1f} min transcurridos", flush=True)

    print("\n================ RESUMEN MUESTRA B (comparacion completa) ================")
    if RESULTS_SAMPLE_PATH.exists():
        df = pd.read_csv(RESULTS_SAMPLE_PATH)
        print(f"Productos procesados: {len(df)}")
        print(f"MAPE mediano: {df['best_mape'].median():.2f}%")
        print(f"MAPE promedio: {df['best_mape'].mean():.2f}%")
        print(f"Con MAPE < 15%: {(df['best_mape'] < 15).sum()} de {len(df)}")


if __name__ == "__main__":
    main()
