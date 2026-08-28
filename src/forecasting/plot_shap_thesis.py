"""
plot_shap_thesis.py

Regenera, en alta resolucion (300 dpi) y con titulo descriptivo, los graficos
de importancia SHAP por semana de la ventana de entrada del LSTM, para
incluir en el documento de tesis.

NO recalcula SHAP: re-dibuja a partir de los valores ya guardados en
data/shap_lstm_results.json (salida de explain_shap_lstm.py, corrida con
shap==0.51.0 / GradientExplainer sobre la muestra de 15 productos categoria A
donde LSTM fue el modelo ganador). Asi las cifras coinciden exactamente con
las reportadas en el documento (t-1 domina en el 67% de los casos, t-2 20%,
t-3 13%).

Salida:  shap_output/shap_<codigo>.png   (raiz del proyecto)

Uso:
    python src/forecasting/plot_shap_thesis.py
    python src/forecasting/plot_shap_thesis.py --codigos A104060-0033,A111033-0238
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_JSON = ROOT_DIR / "data" / "shap_lstm_results.json"
OUT_DIR = ROOT_DIR / "shap_output"

LAG_ORDER = ["t-8", "t-7", "t-6", "t-5", "t-4", "t-3", "t-2", "t-1"]

# Trio por defecto: ilustra los tres desenlaces observados en la muestra.
DEFAULT_CODES = [
    "A104060-0033",  # top t-1 (caso mayoritario, 67%)
    "A111033-0238",  # top t-2 (20%)
    "A111010-0027",  # top t-3 (13%)
]


def plot_product(rec: dict, out_dir: Path) -> Path:
    code = rec["product_code"]
    name = rec["product_name"]
    imp = rec["lag_importance_mean_abs"]
    vals = np.array([imp[l] for l in LAG_ORDER], dtype=float)
    total = vals.sum()
    pct = 100 * vals / total if total > 0 else np.zeros_like(vals)
    top_lag = rec.get("top_lag", LAG_ORDER[int(np.argmax(vals))])

    colors = ["#C44E52" if l == top_lag else "#1F3A7A" for l in LAG_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(LAG_ORDER, vals, color=colors)
    for b, p in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{p:.0f}%", ha="center", va="bottom", fontsize=9.5)

    fig.suptitle(f"{code}  —  {name}", fontsize=13, fontweight="bold", y=0.975)
    ax.set_title(
        "Importancia SHAP (GradientExplainer) de cada semana de la ventana de entrada\n"
        f"sobre la predicción del LSTM  ·  n = {rec.get('n_train_weeks', '?')} semanas de entrenamiento",
        fontsize=8.5, color="#444444", pad=12)
    ax.set_xlabel("Semana relativa de entrada  (t-8 = hace 8 semanas · t-1 = la más reciente)",
                  fontsize=10)
    ax.set_ylabel("|SHAP| promedio", fontsize=10)
    ax.margins(y=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(top=0.83, bottom=0.12, left=0.12, right=0.96)

    out = out_dir / f"shap_{code}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codigos", type=str, default=None,
                    help="lista separada por comas; por defecto un trio representativo")
    args = ap.parse_args()

    if not RESULTS_JSON.exists():
        raise SystemExit(f"No encuentro {RESULTS_JSON}. Corre antes explain_shap_lstm.py.")

    results = {r["product_code"]: r for r in json.load(open(RESULTS_JSON, encoding="utf-8"))}
    codes = [c.strip() for c in args.codigos.split(",")] if args.codigos else DEFAULT_CODES

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Fuente: {RESULTS_JSON.relative_to(ROOT_DIR)}  ({len(results)} productos disponibles)\n")
    for code in codes:
        if code not in results:
            print(f"  [!] {code} no esta en la muestra SHAP, se omite")
            continue
        rec = results[code]
        out = plot_product(rec, OUT_DIR)
        print(f"  {out}")
        print(f"      producto : {rec['product_code']}  {rec['product_name']}")
        print(f"      top lag  : {rec['top_lag']}")


if __name__ == "__main__":
    main()
