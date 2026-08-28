"""
plot_real_vs_pred.py

Grafico de lineas real vs. predicho sobre las 12 semanas del conjunto de
prueba, a partir de las predicciones YA generadas (no reentrena):

  data/prediction_details_A.jsonl        -> categoria A (modelo ganador por producto)
  data/prediction_details_B_sample.jsonl -> muestra de categoria B (n=349)

Para cada corrida genera una figura con una grilla de productos elegidos a lo
largo de la distribucion de MAPE (p05, p20, p40, p60, p80, p95) o los codigos
que se pasen por --codigos.

Salida:
  data/charts/real_vs_pred_A.png
  data/charts/real_vs_pred_B_sample.png

Uso:
    python src/forecasting/plot_real_vs_pred.py                 # A y B_sample
    python src/forecasting/plot_real_vs_pred.py --source A --all-models
    python src/forecasting/plot_real_vs_pred.py --source A --codigos A103049-0096,C114046-0219
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_forecasting_tiered import DATA_DIR

CHARTS_DIR = DATA_DIR / "charts"

SOURCES = {
    "A": {
        "preds": DATA_DIR / "prediction_details_A.jsonl",
        "results": DATA_DIR / "model_comparison_results_A.csv",
        "title": "Categoria A - modelo seleccionado (ARIMA / Prophet / LSTM)",
        "out": CHARTS_DIR / "real_vs_pred_A.png",
    },
    "B_sample": {
        "preds": DATA_DIR / "prediction_details_B_sample.jsonl",
        "results": DATA_DIR / "model_comparison_results_B_sample.csv",
        "title": "Muestra categoria B (n=349) - modelo seleccionado",
        "out": CHARTS_DIR / "real_vs_pred_B_sample.png",
    },
}

MODEL_COLORS = {"ARIMA": "#4C72B0", "Prophet": "#DD8452", "LSTM": "#55A868",
                "HoltWinters": "#8172B2"}
QUANTILES = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]


def read_jsonl(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.update(json.loads(line))
    return out


def pick_products(results: pd.DataFrame, preds: dict, codigos, n):
    if codigos:
        return [c for c in codigos if c in preds]
    df = results.dropna(subset=["best_mape"])
    df = df[df["product_code"].isin(preds)]
    df = df.sort_values("best_mape").reset_index(drop=True)
    qs = QUANTILES if n is None else list(np.linspace(0.05, 0.95, n))
    idx = sorted({int(round(q * (len(df) - 1))) for q in qs})
    return df.loc[idx, "product_code"].tolist()


def plot_source(source: str, codigos=None, n=None, all_models=False):
    cfg = SOURCES[source]
    preds = read_jsonl(cfg["preds"])
    results = pd.read_csv(cfg["results"])
    meta = results.set_index("product_code")

    selected = pick_products(results, preds, codigos, n)
    if not selected:
        print(f"[{source}] sin productos para graficar")
        return

    ncols = 3 if len(selected) > 1 else 1
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows),
                             squeeze=False)

    for ax, code in zip(axes.flat, selected):
        det = preds[code]
        weeks = pd.to_datetime(det["test_dates"])
        actual = np.asarray(det["actual"], dtype=float)
        row = meta.loc[code]
        best = row["selected_model"]
        mape = row["best_mape"]
        name = str(row["product_name"])[:34]

        ax.plot(weeks, actual, color="black", marker="o", linewidth=2,
                markersize=4, label="Real", zorder=3)

        if all_models:
            for m in ("ARIMA", "Prophet", "LSTM", "HoltWinters"):
                if m in det and m != best:
                    ax.plot(weeks, det[m], color=MODEL_COLORS.get(m, "#999999"),
                            linewidth=1, linestyle=":", alpha=0.5, label=m)

        if best in det:
            ax.plot(weeks, det[best], color=MODEL_COLORS.get(best, "#C44E52"),
                    marker="s", markersize=3, linewidth=1.8, linestyle="--",
                    label=f"{best} (predicho)", zorder=2)

        ax.set_title(f"{code}  ·  {name}\n{best}  ·  MAPE {mape:.1f}%", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    for ax in axes.flat[len(selected):]:
        ax.set_visible(False)

    fig.suptitle(cfg["title"] + "  —  12 semanas de prueba", fontsize=12)
    fig.supylabel("Unidades vendidas por semana", fontsize=10)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg["out"], dpi=150)
    plt.close(fig)
    print(f"[{source}] {len(selected)} productos -> {cfg['out']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["A", "B_sample", "all"], default="all")
    ap.add_argument("--codigos", type=str, default=None,
                    help="lista separada por comas; ignora la seleccion por cuantiles")
    ap.add_argument("--n", type=int, default=None,
                    help="numero de productos a muestrear en la distribucion de MAPE")
    ap.add_argument("--all-models", action="store_true",
                    help="superpone los otros modelos en gris tenue")
    args = ap.parse_args()

    codigos = [c.strip() for c in args.codigos.split(",")] if args.codigos else None
    fuentes = ["A", "B_sample"] if args.source == "all" else [args.source]
    for s in fuentes:
        plot_source(s, codigos=codigos, n=args.n, all_models=args.all_models)


if __name__ == "__main__":
    main()
