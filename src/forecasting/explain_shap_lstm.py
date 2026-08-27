"""
Explicabilidad LSTM con SHAP - Proyecto MIA
Comercial La Feria - insumo para Model Card / seccion 7.3-7.4

Responde la pregunta que pidio el tutor (clase semana 7, Shap/Line):
"de las ultimas `window` semanas de venta, cuales pesan mas en la
prediccion de la semana siguiente que hace el LSTM".

No modifica train_forecasting_tiered.py. Reentrena el LSTM SOLO para los
productos seleccionados (arquitectura y semilla identicas a fit_lstm() del
pipeline original, para que el resultado sea representativo) y corre SHAP
sobre ese modelo antes de descartarlo, ya que el pipeline principal no
persiste el modelo en disco.

Requisitos nuevos:
    pip install shap

Uso (desde la raiz del repo, con el venv activado):
    # Los 15 productos de categoria A donde LSTM fue el modelo ganador
    python src/forecasting/explain_shap_lstm.py --n-productos 15

    # Productos especificos
    python src/forecasting/explain_shap_lstm.py --codigos COD001,COD002

Salidas (en la raiz del repo):
    shap_lstm_results.json         -> importancia por lag, por producto
    shap_output/<codigo>.png       -> grafico de barras por producto
    shap_output/_resumen_global.png -> importancia promedio agregada
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PARSED_PATH = DATA_DIR / "parsed.json"
RESULTS_A_PATH = DATA_DIR / "model_comparison_results_A.csv"
OUTPUT_JSON = DATA_DIR / "shap_lstm_results.json"
OUTPUT_CHARTS_DIR = DATA_DIR / "shap_output"

# Deben coincidir exactamente con fit_lstm() en train_forecasting_tiered.py
WINDOW = 8
EPOCHS = 60
TEST_WEEKS = 12


# ---------------------------------------------------------------
# Misma carga de datos que el pipeline principal
# ---------------------------------------------------------------
def load_data():
    with open(PARSED_PATH, "r", encoding="utf-8") as f:
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


def split_series(s: pd.Series):
    train = s.iloc[:-TEST_WEEKS]
    test = s.iloc[-TEST_WEEKS:]
    return train, test


# ---------------------------------------------------------------
# Identico a fit_lstm() del pipeline original, pero devuelve el
# modelo y las ventanas (X) en vez de descartarlos.
# ---------------------------------------------------------------
def fit_lstm_explainable(train, window=WINDOW, epochs=EPOCHS):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    tf.random.set_seed(42)

    values = train.values.astype("float32")
    vmin, vmax = values.min(), values.max()
    rng = (vmax - vmin) if vmax > vmin else 1.0
    scaled = (values - vmin) / rng

    if len(scaled) <= window + 1:
        return None, None

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

    return model, X


# ---------------------------------------------------------------
# SHAP: importancia de cada una de las `window` semanas pasadas
# ---------------------------------------------------------------
def explain_product(code, name, train, background_size=30, n_instances=20):
    import shap

    model, X = fit_lstm_explainable(train)
    if model is None:
        return None

    rng = np.random.default_rng(42)

    # Background = referencia de "input normal" para SHAP
    n_bg = min(background_size, len(X))
    bg_idx = rng.choice(len(X), size=n_bg, replace=False)
    background = X[bg_idx]

    # Instancias a explicar: la ultima ventana (la que genera el forecast
    # real hacia adelante) + una muestra del resto para dar contexto general
    last_window = X[-1:]
    n_sample = min(n_instances, len(X))
    sample_idx = rng.choice(len(X), size=n_sample, replace=False)
    instances = np.concatenate([last_window, X[sample_idx]], axis=0)

    explainer = shap.GradientExplainer(model, background)
    raw = explainer.shap_values(instances)
    if isinstance(raw, list):
        raw = raw[0]
    sv = np.array(raw).reshape(instances.shape[0], WINDOW)  # (n_instancias, window)

    mean_abs_importance = np.abs(sv).mean(axis=0)
    lag_labels = [f"t-{WINDOW - i}" for i in range(WINDOW)]  # t-8 ... t-1

    return {
        "product_code": code,
        "product_name": name,
        "n_train_weeks": len(train),
        "lag_importance_mean_abs": dict(zip(lag_labels, mean_abs_importance.tolist())),
        "top_lag": lag_labels[int(np.argmax(mean_abs_importance))],
        "last_window_shap": dict(zip(lag_labels, np.abs(sv[0]).tolist())),
    }


# ---------------------------------------------------------------
# Graficos (mismo estilo simple que generate_charts.py)
# ---------------------------------------------------------------
def plot_product(result, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(result["lag_importance_mean_abs"].keys())
    values = list(result["lag_importance_mean_abs"].values())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="#1F3A7A")
    ax.set_title(f"Importancia SHAP por semana pasada - {result['product_code']}")
    ax.set_xlabel("Semana relativa (t-8 = hace 8 semanas, t-1 = la mas reciente)")
    ax.set_ylabel("|SHAP| promedio")
    fig.tight_layout()
    fig.savefig(out_dir / f"{result['product_code']}.png", dpi=150)
    plt.close(fig)


def plot_global_summary(results, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results[0]["lag_importance_mean_abs"].keys())
    matrix = np.array([[r["lag_importance_mean_abs"][l] for l in labels] for r in results])
    mean_by_lag = matrix.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, mean_by_lag, color="#1F3A7A")
    ax.set_title(f"Importancia SHAP promedio - {len(results)} productos (LSTM)")
    ax.set_xlabel("Semana relativa")
    ax.set_ylabel("|SHAP| promedio")
    fig.tight_layout()
    fig.savefig(out_dir / "_resumen_global.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# Seleccion de productos
# ---------------------------------------------------------------
def select_codes(n_productos, codigos_arg):
    if codigos_arg:
        return [c.strip() for c in codigos_arg.split(",") if c.strip()]

    if not RESULTS_A_PATH.exists():
        raise SystemExit(
            f"No encuentro {RESULTS_A_PATH}. Corre primero train_forecasting_tiered.py "
            f"(o pasa --codigos explicitamente)."
        )
    df = pd.read_csv(RESULTS_A_PATH)
    df_lstm = df[df["selected_model"] == "LSTM"].sort_values("best_mape")
    if df_lstm.empty:
        raise SystemExit("No hay productos con selected_model == 'LSTM' en model_comparison_results_A.csv")
    return df_lstm["product_code"].astype(str).head(n_productos).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-productos", type=int, default=15,
                         help="Cuantos productos (con LSTM como modelo ganador) explicar, "
                              "ordenados por mejor MAPE primero. Default: 15")
    parser.add_argument("--codigos", type=str, default=None,
                         help="Lista de codigos de producto separados por coma, en vez de "
                              "seleccionar automaticamente por MAPE")
    parser.add_argument("--background-size", type=int, default=30)
    parser.add_argument("--n-instances", type=int, default=20)
    args = parser.parse_args()

    codes = select_codes(args.n_productos, args.codigos)
    print(f"Productos a explicar ({len(codes)}): {codes}")

    products = load_data()
    OUTPUT_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, code in enumerate(codes, 1):
        if code not in products:
            print(f"  [{i}/{len(codes)}] {code} -> no encontrado en data/parsed.json, se omite")
            continue
        name = products[code]["name"]
        train, _ = split_series(products[code]["series"])
        print(f"  [{i}/{len(codes)}] {code} ({name}) -> entrenando LSTM + SHAP...")
        res = explain_product(code, name, train,
                               background_size=args.background_size,
                               n_instances=args.n_instances)
        if res is None:
            print(f"      historia insuficiente (<= {WINDOW + 1} semanas), se omite")
            continue
        results.append(res)
        plot_product(res, OUTPUT_CHARTS_DIR)
        print(f"      semana con mayor peso: {res['top_lag']}")

    if not results:
        print("No se genero ningun resultado.")
        return

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plot_global_summary(results, OUTPUT_CHARTS_DIR)

    top_lags = pd.Series([r["top_lag"] for r in results])
    print("\n================ RESUMEN ================")
    print(f"Productos explicados: {len(results)}")
    print("Semana con mayor peso, frecuencia entre los productos explicados:")
    print(top_lags.value_counts().to_string())
    print(f"\nGuardado: {OUTPUT_JSON}")
    print(f"Graficos: {OUTPUT_CHARTS_DIR}/")


if __name__ == "__main__":
    main()
