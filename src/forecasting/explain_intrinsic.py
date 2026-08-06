"""
Explicabilidad intrinseca: ARIMA, Prophet, Holt-Winters - Proyecto MIA
Comercial La Feria - insumo para Model Card / seccion 7.3-7.4

A diferencia de explain_shap_lstm.py (que necesita SHAP porque el LSTM es
caja negra), estos 3 modelos ya exponen internamente por que llegan a un
resultado -- solo hay que extraer esa informacion y presentarla:

  - ARIMA: el orden (p,d,q)(P,D,Q)m que elige auto_arima ya dice cuantas
    semanas pasadas usa (AR), si detecto estacionalidad anual (m=52) y
    cuanto corrige en base a errores anteriores (MA).
  - Prophet: se descompone nativamente en tendencia + estacionalidad anual.
    Se usa su propio plot_components() y se mide cuanta varianza aporta
    cada componente.
  - Holt-Winters: se descompone en nivel + tendencia + estacionalidad, con
    parametros de suavizado (alpha, beta, gamma) que el propio modelo ajusta.

No modifica train_forecasting_tiered.py ni train_forecasting.py. Reentrena
estos 3 modelos (deterministicos, sin necesidad de guardar nada) SOLO para
los productos de ejemplo que elijas.

Uso (desde la raiz del repo, con el venv activado; corre rapido, no
necesita servidor ni tmux):
    python src/forecasting/explain_intrinsic.py --n-productos 3
    python src/forecasting/explain_intrinsic.py --codigos COD001,COD002

Salidas (en la raiz del repo):
    intrinsic_explanations.json        -> texto + numeros por producto
    intrinsic_output/<codigo>_arima.png
    intrinsic_output/<codigo>_prophet.png
    intrinsic_output/<codigo>_holtwinters.png
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_A_PATH = ROOT_DIR / "model_comparison_results_A.csv"
OUTPUT_JSON = ROOT_DIR / "intrinsic_explanations.json"
OUTPUT_CHARTS_DIR = ROOT_DIR / "intrinsic_output"

TEST_WEEKS = 12


# ---------------------------------------------------------------
# Misma carga de datos que el resto del pipeline
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
    return products


def split_series(s: pd.Series):
    train = s.iloc[:-TEST_WEEKS]
    test = s.iloc[-TEST_WEEKS:]
    return train, test


# ---------------------------------------------------------------
# ARIMA - el orden elegido YA es la explicacion
# ---------------------------------------------------------------
def explain_arima(code, name, train, out_dir: Path):
    import pmdarima as pm

    model = pm.auto_arima(
        train.values, seasonal=True, m=52 if len(train) > 104 else 1,
        stepwise=True, suppress_warnings=True, error_action="ignore",
        max_p=3, max_q=3, max_P=1, max_Q=1,
    )
    order = model.order            # (p, d, q)
    seasonal_order = model.seasonal_order  # (P, D, Q, m)
    p, d, q = order
    P, D, Q, m = seasonal_order

    interpretation = []
    if p > 0:
        interpretation.append(
            f"Usa las ultimas {p} semana(s) de venta directamente para predecir (componente AR={p})."
        )
    if d > 0:
        interpretation.append(
            f"Le quito tendencia a la serie antes de modelar (diferenciacion d={d})."
        )
    if q > 0:
        interpretation.append(
            f"Corrige la prediccion en base a los ultimos {q} error(es) de prediccion anteriores (componente MA={q})."
        )
    if m > 1 and (P > 0 or D > 0 or Q > 0):
        interpretation.append(
            f"Detecto un patron que se repite cada {m} semanas (estacionalidad anual)."
        )
    else:
        interpretation.append("No detecto un patron estacional anual claro para este producto.")

    # Grafico simple: serie de train + orden elegido en el titulo
    _plot_series_with_title(
        train, f"ARIMA {order}x{seasonal_order} - {code}",
        out_dir / f"{code}_arima.png"
    )

    return {
        "order_pdq": list(order),
        "seasonal_order_PDQm": list(seasonal_order),
        "aic": float(model.aic()),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------
# Prophet - descomposicion nativa (tendencia + estacionalidad anual)
# ---------------------------------------------------------------
def explain_prophet(code, name, train, out_dir: Path):
    from prophet import Prophet

    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(weekly_seasonality=False, yearly_seasonality=True,
                seasonality_mode="additive", interval_width=0.8)
    m.fit(dfp)
    forecast = m.predict(dfp)  # componentes sobre el propio train

    trend_range = float(forecast["trend"].max() - forecast["trend"].min())
    yearly_range = float(forecast["yearly"].max() - forecast["yearly"].min()) if "yearly" in forecast else 0.0
    total = trend_range + yearly_range if (trend_range + yearly_range) > 0 else 1.0
    trend_pct = 100 * trend_range / total
    yearly_pct = 100 * yearly_range / total

    dominant = "la tendencia general" if trend_pct > yearly_pct else "la estacionalidad anual"
    interpretation = [
        f"Tendencia explica ~{trend_pct:.0f}% de la variacion capturada, estacionalidad anual ~{yearly_pct:.0f}%.",
        f"El componente dominante para este producto es {dominant}.",
    ]

    try:
        import matplotlib
        matplotlib.use("Agg")
        fig = m.plot_components(forecast)
        fig.suptitle(f"Componentes Prophet - {code}", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"{code}_prophet.png", dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass

    return {
        "trend_contribution_pct": trend_pct,
        "yearly_seasonality_contribution_pct": yearly_pct,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------
# Holt-Winters - nivel + tendencia + estacionalidad + parametros de suavizado
# ---------------------------------------------------------------
def explain_holtwinters(code, name, train, out_dir: Path):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    seasonal_periods = 52 if len(train) > 104 else None
    model = ExponentialSmoothing(
        train.values, trend="add",
        seasonal="add" if seasonal_periods else None,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit()

    alpha = float(model.params.get("smoothing_level", np.nan))
    beta = float(model.params.get("smoothing_trend", np.nan))
    gamma = float(model.params.get("smoothing_seasonal", np.nan))

    interpretation = []
    if not np.isnan(alpha):
        peso_reciente = "mucho peso a lo mas reciente" if alpha > 0.5 else "peso repartido entre historia reciente y pasada"
        interpretation.append(f"alpha={alpha:.2f} -> le da {peso_reciente} para estimar el nivel actual.")
    if not np.isnan(beta):
        interpretation.append(f"beta={beta:.2f} -> {'la tendencia se actualiza rapido' if beta > 0.3 else 'la tendencia cambia lento, es mas estable'}.")
    if seasonal_periods and not np.isnan(gamma):
        interpretation.append(f"gamma={gamma:.2f} -> el patron estacional anual {'se reajusta rapido con datos nuevos' if gamma > 0.3 else 'se mantiene bastante estable de un anio a otro'}.")
    else:
        interpretation.append("No se estimo componente estacional (historia insuficiente para un ciclo anual completo).")

    # Grafico: nivel/tendencia/estacionalidad si el modelo los expone
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        comps = {}
        for attr in ("level", "trend", "season"):
            val = getattr(model, attr, None)
            if val is not None:
                comps[attr] = np.asarray(val)

        n = len(comps) if comps else 1
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.2 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (attr, val) in zip(axes, comps.items()):
            ax.plot(val, color="#1F3A7A")
            ax.set_title(attr)
        if not comps:
            axes[0].plot(model.fittedvalues, color="#1F3A7A")
            axes[0].set_title("Valores ajustados (fittedvalues)")
        fig.suptitle(f"Componentes Holt-Winters - {code}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{code}_holtwinters.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    return {
        "smoothing_level_alpha": alpha,
        "smoothing_trend_beta": beta,
        "smoothing_seasonal_gamma": gamma,
        "seasonal_periods": seasonal_periods,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------
# Utilidad de grafico simple
# ---------------------------------------------------------------
def _plot_series_with_title(series: pd.Series, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(series.index, series.values, color="#1F3A7A")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# Seleccion de productos (mismo criterio que explain_shap_lstm.py)
# ---------------------------------------------------------------
def select_codes(n_productos, codigos_arg, source="A"):
    if codigos_arg:
        return [c.strip() for c in codigos_arg.split(",") if c.strip()]

    path = RESULTS_A_PATH if source == "A" else (ROOT_DIR / "model_comparison_results_B.csv")
    if not path.exists():
        raise SystemExit(
            f"No encuentro {path}. Corre primero train_forecasting_tiered.py "
            f"(o pasa --codigos explicitamente)."
        )
    df = pd.read_csv(path)
    df = df.sort_values("best_mape")
    return df["product_code"].astype(str).head(n_productos).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-productos", type=int, default=3,
                         help="Cuantos productos de ejemplo explicar (ordenados por mejor MAPE). Default: 3")
    parser.add_argument("--codigos", type=str, default=None,
                         help="Lista de codigos de producto separados por coma")
    parser.add_argument("--source", type=str, default="A", choices=["A", "B"],
                         help="De donde tomar los productos de ejemplo si no se pasan --codigos: "
                              "'A' = model_comparison_results_A.csv, 'B' = model_comparison_results_B.csv. "
                              "Usa 'B' para el model card de Holt-Winters (categoria B real). Default: A")
    args = parser.parse_args()

    codes = select_codes(args.n_productos, args.codigos, args.source)
    print(f"Productos a explicar ({len(codes)}): {codes}")

    products = load_data()
    OUTPUT_CHARTS_DIR.mkdir(exist_ok=True)

    all_results = []
    for i, code in enumerate(codes, 1):
        if code not in products:
            print(f"  [{i}/{len(codes)}] {code} -> no encontrado en data/parsed.json, se omite")
            continue
        name = products[code]["name"]
        train, _ = split_series(products[code]["series"])
        print(f"\n  [{i}/{len(codes)}] {code} ({name})")

        entry = {"product_code": code, "product_name": name}

        try:
            entry["ARIMA"] = explain_arima(code, name, train, OUTPUT_CHARTS_DIR)
            print(f"      ARIMA: orden {entry['ARIMA']['order_pdq']}x{entry['ARIMA']['seasonal_order_PDQm']}")
        except Exception as e:
            print(f"      ARIMA -> ERROR: {e}")
            entry["ARIMA"] = None

        try:
            entry["Prophet"] = explain_prophet(code, name, train, OUTPUT_CHARTS_DIR)
            print(f"      Prophet: tendencia {entry['Prophet']['trend_contribution_pct']:.0f}% / "
                  f"estacional {entry['Prophet']['yearly_seasonality_contribution_pct']:.0f}%")
        except Exception as e:
            print(f"      Prophet -> ERROR: {e}")
            entry["Prophet"] = None

        try:
            entry["HoltWinters"] = explain_holtwinters(code, name, train, OUTPUT_CHARTS_DIR)
            print(f"      Holt-Winters: alpha={entry['HoltWinters']['smoothing_level_alpha']:.2f}")
        except Exception as e:
            print(f"      Holt-Winters -> ERROR: {e}")
            entry["HoltWinters"] = None

        all_results.append(entry)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nGuardado: {OUTPUT_JSON}")
    print(f"Graficos: {OUTPUT_CHARTS_DIR}/")


if __name__ == "__main__":
    main()