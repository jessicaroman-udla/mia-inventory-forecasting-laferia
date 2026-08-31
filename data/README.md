# `data/` — datos y resultados generados

Esta carpeta contiene **todo** lo que produce el pipeline (entradas extraídas
de la base de datos y salidas de forecasting/optimización). **Nada de lo que
hay aquí se versiona en Git** (`.gitignore` excluye `data/*` salvo este
README), porque los datos reales de Comercial La Feria están bajo Acuerdo de
Confidencialidad (NDA).

Los scripts crean y leen los archivos aquí automáticamente; no hay que crear
nada a mano. La carpeta se regenera corriendo el pipeline (ver el README
principal).

## Entradas (las genera `src/extraction/extract_data.py` desde PostgreSQL)

> Sin acceso a la base de datos, `src/sample_data/generate_sample.py` escribe
> estos mismos archivos con datos **sintéticos** de ejemplo.

| Archivo | Contenido |
|---|---|
| `abc_classification.json` | Clasificación ABC (Pareto 80/95) de todo el catálogo vendido |
| `parsed.json` | Series de venta semanal por producto, categorías A y B |
| `stock_data.json` | Stock actual por producto y sucursal |
| `prices.json` | Precio de venta promedio por producto |
| `demand_statistics.json` | Media y desviación estándar de demanda semanal por producto |
| `warehouse_sale.json` | Unidades vendidas por producto y sucursal (para el algoritmo genético) |
| `warehouses.json` | Configuración de sucursales y rutas de transferencia |

## Salidas de forecasting (`src/forecasting/`)

| Archivo | Lo genera |
|---|---|
| `model_comparison_results.csv`, `prediction_details.json` | `train_forecasting.py` (track de referencia) |
| `model_comparison_results_A.csv`, `_B.csv`, `prediction_details_A.jsonl` | `train_forecasting_tiered.py` |
| `model_comparison_results_B_sample.csv`, `prediction_details_B_sample.jsonl` | `validate_b_sample.py` |
| `b_significance_comparison.csv` | `compare_b_significance.py` |
| `model_comparison_results_baseline.csv`, `model_comparison_results_baseline_resumen.csv` | `src/baseline/baseline_naive.py` |
| `metrics_extra_detalle.csv`, `metrics_extra_resumen.csv` | `metrics_extra.py` (sMAPE / WAPE / MASE) |
| `pronostico_futuro_producto.csv` | `generar_pronostico_futuro.py` |
| `shap_lstm_results.json`, `shap_output/` | `explain_shap_lstm.py` |
| `intrinsic_explanations.json`, `intrinsic_output/` | `explain_intrinsic.py` |
| `diagnostico_output/` | `diagnostico_series.py` |
| `charts/` (`.png`) | `generate_charts.py`, `plot_real_vs_pred.py` |

## Salidas de optimización (`src/optimization/`)

| Archivo | Lo genera |
|---|---|
| `resultado_milp_reorden.json` | `milp_reorder.py` (track de referencia) |
| `forecast_output.csv` | `desagregar_por_sucursal.py` |
| `pronosticos_recortados_log.csv` | `desagregar_por_sucursal.py` (auditoría de recortes) |
| `resultados_milp_piloto.csv` | `optimizacion_milp_piloto.py` |
| `resultado_ga_transferencias.json` | `ga_transfers.py` |
| `robustness_sensitivity_leadtime_*.csv` / `_summary.txt` | `robustness_sensitivity_leadtime.py` |
| `charts/` (`.png`) | `generate_optimization_charts.py` |

> Para enviar un resultado concreto a revisión sin versionar toda la carpeta:
> `git add -f data/<archivo>`.
