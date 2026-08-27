# Modelo baseline — MIA, Comercial La Feria

Baseline de referencia (sección D del checklist de tesis), comparable 1:1
contra ARIMA / Prophet / LSTM / Holt-Winters.

## Qué calcula

Sobre **exactamente los mismos datos y la misma metodología** que
`src/forecasting/train_forecasting_tiered.py` (importa `load_data`,
`split_series`, `mae`, `rmse`, `mape`, `TEST_WEEKS` de ese módulo):

| Baseline | Definición |
|---|---|
| **Naive** (random walk) | pronóstico = último valor observado en el train |
| **Seasonal naive** | pronóstico(semana *t*) = valor de la semana *t − 52* (misma semana del año anterior); solo si la serie tiene ≥ 64 semanas |

- Serie semanal **nacional por producto** (`data/parsed.json`), reindexada a
  semanas continuas con huecos = 0.
- Partición cronológica: últimas `TEST_WEEKS` (= 12) semanas como test.
- Clasificación ABC de `data/abc_classification.json`.
- MAPE solo sobre semanas con demanda real > 0.
- Categorías A y B (las que forecastea el pipeline principal).

## Ejecución

No necesita base de datos (usa los JSON ya extraídos). Corre en ~20 s.

```bash
python src/baseline/baseline_naive.py
```

## Salidas (en `data/`)

- `model_comparison_results_baseline.csv` — detalle por producto (MAE/RMSE/MAPE
  de Naive y Seasonal naive; mismo esquema que `model_comparison_results_A.csv`).
- `model_comparison_results_baseline_resumen.csv` — mediana por categoría ABC.

Para la tabla de la tesis, comparar `MAPE_mediana` del baseline contra la
mediana de `best_mape` en `model_comparison_results_A.csv` / `_B.csv`
(`python src/forecasting/summarize_final_results.py`).

## Reproducibilidad

- Determinístico (el naive no usa semilla).
- Los resultados dependen de la corrida de `extract_data.py` que generó
  `data/parsed.json` y `data/abc_classification.json`.
