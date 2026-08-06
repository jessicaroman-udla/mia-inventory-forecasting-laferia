"""
Significancia estadistica: comparacion completa vs. Holt-Winters (Categoria B)
Proyecto MIA - Comercial La Feria

Empareja, por product_code, el resultado de la comparacion completa
(ARIMA/Prophet/LSTM, corrida por validate_b_sample.py sobre la muestra de
349 productos) contra el resultado de Holt-Winters para esos MISMOS
productos (ya calculado en model_comparison_results_B.csv por el pipeline
tiered). Corre una prueba de Wilcoxon (pareada, no parametrica) sobre las
diferencias de MAPE.

No modifica ningun archivo de resultados existente, solo los lee.

Uso (desde la raiz del repo, con el venv activado; liviano, corre en
segundos, no necesita servidor ni tmux):
    python src/forecasting/compare_b_significance.py

Si el CSV de la muestra no se llama como se espera por defecto, indicalo:
    python src/forecasting/compare_b_significance.py --sample-csv NOMBRE_REAL.csv

Salida:
    Imprime el resumen en consola y guarda b_significance_comparison.csv
    con el detalle producto por producto (para citar en el documento si
    hace falta).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

# Nombre real, confirmado desde validate_b_sample.py (RESULTS_SAMPLE_PATH)
DEFAULT_SAMPLE_CANDIDATES = [
    "model_comparison_results_B_sample.csv",
]


def find_sample_csv(explicit_path: str | None) -> Path:
    if explicit_path:
        p = ROOT_DIR / explicit_path
        if not p.exists():
            sys.exit(f"No encuentro {p}. Verifica el nombre con:\n"
                      f"  ls -lt {ROOT_DIR}/*.csv | head -5")
        return p

    for name in DEFAULT_SAMPLE_CANDIDATES:
        p = ROOT_DIR / name
        if p.exists():
            return p

    sys.exit(
        "No encontre el CSV de la muestra automaticamente. Pasa el nombre exacto con "
        "--sample-csv, por ejemplo:\n"
        "  python src/forecasting/compare_b_significance.py --sample-csv mi_archivo.csv\n\n"
        "Para verlo en el servidor:\n"
        "  ssh -p 12980 -i ~/.ssh/id_ed25519_github_udla udla@186.4.222.67 "
        "\"ls -lt ~/mia-inventory-forecasting-laferia/*.csv | head -5\""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-csv", type=str, default=None,
                         help="CSV con la comparacion completa sobre la muestra de 349 "
                              "productos (salida de validate_b_sample.py)")
    parser.add_argument("--full-b-csv", type=str, default="model_comparison_results_B.csv",
                         help="CSV con los resultados de Holt-Winters para categoria B "
                              "(salida de train_forecasting_tiered.py)")
    args = parser.parse_args()

    sample_path = find_sample_csv(args.sample_csv)
    full_b_path = ROOT_DIR / args.full_b_csv
    if not full_b_path.exists():
        sys.exit(f"No encuentro {full_b_path}.")

    print(f"Muestra (comparacion completa): {sample_path.name}")
    print(f"Categoria B completa (Holt-Winters): {full_b_path.name}\n")

    df_sample = pd.read_csv(sample_path)
    df_b = pd.read_csv(full_b_path)

    # La columna con el MAPE del mejor de los 3 modelos (ARIMA/Prophet/LSTM)
    # en la muestra debe llamarse 'best_mape', igual que en model_comparison_results_A.csv
    if "best_mape" not in df_sample.columns:
        sys.exit(
            f"No encuentro la columna 'best_mape' en {sample_path.name}. "
            f"Columnas disponibles: {list(df_sample.columns)}\n"
            f"Ajusta el nombre de columna en el script si es distinto."
        )
    if "HoltWinters_MAPE" not in df_b.columns:
        sys.exit(
            f"No encuentro la columna 'HoltWinters_MAPE' en {full_b_path.name}. "
            f"Columnas disponibles: {list(df_b.columns)}"
        )

    df_sample = df_sample[["product_code", "product_name", "selected_model", "best_mape"]].rename(
        columns={"selected_model": "full_comparison_model", "best_mape": "full_comparison_mape"}
    )
    df_b = df_b[["product_code", "HoltWinters_MAPE"]]

    merged = df_sample.merge(df_b, on="product_code", how="inner")
    n_before = len(df_sample)
    n_matched = len(merged)
    print(f"Productos en la muestra: {n_before}  |  Emparejados con categoria B completa: {n_matched}")
    if n_matched < n_before:
        print(f"  ({n_before - n_matched} no se encontraron en {full_b_path.name} - se excluyen)")

    merged = merged.dropna(subset=["full_comparison_mape", "HoltWinters_MAPE"])
    n_valid = len(merged)
    print(f"Pares validos (sin NaN en ninguno de los dos MAPE): {n_valid}\n")

    if n_valid < 10:
        sys.exit("Muy pocos pares validos para una prueba estadistica confiable (<10).")

    diff = merged["HoltWinters_MAPE"] - merged["full_comparison_mape"]
    wins_full = int((diff > 0).sum())
    wins_hw = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    win_pct_full = 100 * wins_full / n_valid

    median_full = merged["full_comparison_mape"].median()
    median_hw = merged["HoltWinters_MAPE"].median()
    median_improvement_pp = median_hw - median_full

    from scipy.stats import wilcoxon
    stat, p_value = wilcoxon(merged["HoltWinters_MAPE"], merged["full_comparison_mape"])

    p25_full = merged["full_comparison_mape"].quantile(0.25)
    p75_full = merged["full_comparison_mape"].quantile(0.75)
    p25_hw = merged["HoltWinters_MAPE"].quantile(0.25)
    p75_hw = merged["HoltWinters_MAPE"].quantile(0.75)

    under15_full = int((merged["full_comparison_mape"] < 15).sum())
    under15_hw = int((merged["HoltWinters_MAPE"] < 15).sum())

    print("================ RESULTADO ================")
    print(f"Comparacion completa gana en: {wins_full}/{n_valid} ({win_pct_full:.1f}%)")
    print(f"Holt-Winters gana en:         {wins_hw}/{n_valid} ({100*wins_hw/n_valid:.1f}%)")
    print(f"Empates:                      {ties}/{n_valid} ({100*ties/n_valid:.1f}%)")
    print()
    print(f"{'Metrica':<28}{'Holt-Winters':>15}{'Mejor de 3':>15}")
    print(f"{'Mediana MAPE (%)':<28}{median_hw:>15.1f}{median_full:>15.1f}")
    print(f"{'Percentil 25 (%)':<28}{p25_hw:>15.1f}{p25_full:>15.1f}")
    print(f"{'Percentil 75 (%)':<28}{p75_hw:>15.1f}{p75_full:>15.1f}")
    print(f"{'MAPE < 15%':<28}{under15_hw:>10}/{n_valid} ({100*under15_hw/n_valid:.1f}%)"
          f"{under15_full:>7}/{n_valid} ({100*under15_full/n_valid:.1f}%)")
    print(f"Mejora mediana (pp):                  {median_improvement_pp:.2f}")
    print(f"\nWilcoxon signed-rank: statistic={stat:.2f}, p-value={p_value:.10f}")
    print("Diferencia estadisticamente significativa (p<0.05)"
          if p_value < 0.05 else "NO hay diferencia estadisticamente significativa (p>=0.05)")

    out_path = ROOT_DIR / "b_significance_comparison.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nDetalle guardado en: {out_path}")


if __name__ == "__main__":
    main()