# Modelo baseline (Naive Forecast) — MIA, Comercial La Feria

Script de referencia para calcular el modelo baseline exigido por el checklist
de tesis (sección D), comparable contra ARIMA/Prophet/LSTM/Holt-Winters.

## Instrucciones de ejecución

1. Instalar dependencias (idealmente en un entorno virtual):

   ```bash
   pip install -r requirements.txt
   ```

2. Configurar credenciales de conexión:

   ```bash
   cp .env.example .env
   ```

   Editar `.env` y completar `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`,
   `DB_PASSWORD` con los mismos valores que usa el proyecto Django
   (ver `settings.py` → `DATABASES`).

3. Ejecutar el script:

   ```bash
   python baseline_naive.py
   ```

4. Revisar los resultados generados:

   - `resultados_baseline_detalle.csv` — métricas (MAE, RMSE, MAPE) por
     combinación producto-sucursal.
   - `resultados_baseline_resumen.csv` — medianas agregadas por categoría
     ABC (A, B, C), listas para insertar en la Tabla de comparación
     baseline vs. modelo propuesto del documento de tesis.

## Parámetros ajustables

- `BASELINE_SEMANAS_TEST` (en `.env`, por defecto 12): tamaño de la ventana
  de prueba, en semanas. Debe coincidir con la ventana usada al evaluar
  ARIMA/Prophet/LSTM/Holt-Winters para que la comparación sea válida.

## Notas de reproducibilidad

- El script es de solo lectura sobre la base de datos (no modifica ni
  inserta registros).
- La clasificación ABC se toma del snapshot más reciente disponible en
  `inventario.analisis_avanzado`; si esta tabla se actualiza, los resultados
  pueden variar levemente respecto a corridas anteriores.
- Semilla aleatoria: no aplica (el naive forecast es determinístico, no
  requiere semilla).
