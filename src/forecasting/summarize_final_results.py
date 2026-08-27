"""
Resumen final A/B - Proyecto MIA
Recalcula mediana, media y percentiles de MAPE sobre los resultados finales
(model_comparison_results_A.csv y _B.csv), para reportar de forma consistente
con el principio ya establecido en el proyecto: la mediana (no la media) es
la metrica correcta dado el sesgo por outliers de demanda intermitente.

Uso:
    python src/forecasting/summarize_final_results.py
"""
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


def summarize(path: Path, label: str):
    if not path.exists():
        print(f"{label}: no encontrado ({path.name})")
        return
    df = pd.read_csv(path)
    mape = df["best_mape"].dropna()
    print(f"\n=== {label} ({path.name}) ===")
    print(f"Productos: {len(df)}  |  Con MAPE valido: {len(mape)}")
    print(f"Mediana MAPE: {mape.median():.2f}%")
    print(f"Media MAPE:   {mape.mean():.2f}%")
    print(f"P25 / P75:    {mape.quantile(0.25):.2f}% / {mape.quantile(0.75):.2f}%")
    print(f"Con MAPE < 15%: {(mape < 15).sum()} de {len(mape)} ({100*(mape < 15).sum()/len(mape):.1f}%)")
    if "selected_model" in df.columns:
        print("Distribucion de modelo seleccionado:")
        print(df["selected_model"].value_counts(normalize=True).mul(100).round(1).to_string())


if __name__ == "__main__":
    summarize(DATA_DIR / "model_comparison_results_A.csv", "Categoria A (comparacion completa)")
    summarize(DATA_DIR / "model_comparison_results_B.csv", "Categoria B (Holt-Winters)")
    summarize(DATA_DIR / "model_comparison_results_B_sample.csv", "Muestra B (comparacion completa, n=349)")
    summarize(DATA_DIR / "model_comparison_results_baseline.csv", "Baseline Naive (A+B, ver src/baseline/)")
