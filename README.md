# 🏪 Sistema Inteligente de Gestión de Inventarios con Forecasting Multi-Sucursal

> **Proyecto de Titulación — Maestría en Inteligencia Artificial Aplicada**  
> Facultad de Ingeniería y Ciencias Aplicadas

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.x-092E20?style=flat&logo=django&logoColor=white)](https://djangoproject.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-Academic-blue?style=flat)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Prototipo-yellow?style=flat)]()

---

## 📋 Descripción

Sistema web inteligente que integrará modelos de **forecasting de demanda** y algoritmos de **optimización matemática** para automatizar la gestión de inventarios en las sucursales de Comercial La Feria (Santo Domingo de los Tsáchilas, Ecuador).

El sistema extraerá datos históricos desde **PostgreSQL** (alimentado desde SAP Business One HANA), entrenará modelos predictivos sobre los patrones de venta y generará recomendaciones automáticas de reabastecimiento, transferencias inter-sucursales y alertas de stock crítico.

> Este repositorio contiene, por ahora, el **prototipo de la Sección 7.2** del documento capstone: los pipelines de forecasting y optimización ejecutables por línea de comandos (sin interfaz web todavía). Ver [Estado actual del prototipo](#-estado-actual-del-prototipo) más abajo.

---

## 👥 Equipo

| Rol | Nombre | Responsabilidad principal |
|-----|--------|--------------------------|
| Autor 1 | Lam Cheang Wiliam David | Desarrollo web, integración ETL, arquitectura |
| Autor 2 | Román Largo Jessica Johanna | Modelado ML, forecasting, optimización |
| Tutor | Criollo Caizaguano Luis Santiago | Dirección académica |

---

## 🎯 Objetivo General

Diseñar e implementar un sistema de gestión de inventarios con modelos de forecasting (ARIMA, Prophet, LSTM) y optimización entera para Comercial La Feria, a fin de reducir rupturas de stock y costos operativos mediante inteligencia artificial.

---

## ⚙️ Tecnologías

### Backend & Datos
- **Python 3.11+** — Lenguaje principal
- **Django 5.x** + Django REST Framework — Framework web (planificado)
- **PostgreSQL 16** — Base de datos principal
- **hdbcli** — Conector SAP Business One HANA (ETL, planificado)
- **Celery + Redis** — Tareas programadas y actualización de modelos (planificado)

### Forecasting & Machine Learning
- **statsmodels / pmdarima** — Modelos ARIMA
- **Prophet** (Meta) — Forecasting con estacionalidad
- **TensorFlow / Keras** — Red neuronal LSTM

### Optimización Matemática
- **PuLP** — Programación Lineal Entera Mixta (MILP) para punto de reorden y cantidad óptima de pedido por producto-sucursal
- **SQLAlchemy** — Conexión a PostgreSQL para el módulo de optimización (stock, maestro de productos, costos)

### Frontend & Visualización (planificado)
- **Django Templates** + **Bootstrap 5**
- **Plotly.js** / **Chart.js** — Dashboards interactivos y gráficos de KPIs

### DevOps & Herramientas
- **Git / GitHub** — Control de versiones
- **Fabric** — Automatización de deploy y ejecución del pipeline en el servidor (`fabfile.py`)
- **Sphinx** — Documentación técnica

---

## 🏗️ Arquitectura objetivo del sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    SAP Business One HANA                    │
│              (Datos históricos 2022-2024)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ ETL (hdbcli)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     PostgreSQL 16                           │
│           (Datos normalizados y procesados)                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────┐    ┌───────────────────────┐
│  Módulo Forecasting │    │  Módulo Optimización  │
│  ARIMA / Prophet /  │    │  MILP (PuLP)          │
│  LSTM / Holt-Winters│    │  → Punto de reorden y │
│  → Predicción       │    │    cantidad óptima de │
│    demanda futura   │    │    pedido por         │
│                      │    │    producto-sucursal  │
└─────────┬───────────┘    └──────────┬────────────┘
          └────────────┬──────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Django Web Application                    │
│   Dashboard │ Alertas │ Recomendaciones │ Reportes │ UAT    │
└─────────────────────────────────────────────────────────────┘
```

Los dos módulos de la mitad (Forecasting y Optimización) son lo que existe hoy como prototipo de línea de comandos en `src/`. La capa de PostgreSQL de entrada también existe (`src/extraction/extract_data.py`). El ETL desde SAP HANA y la aplicación web Django todavía no están implementados.

> El algoritmo genético de balanceo inter-sucursales (DEAP) que existió en una versión anterior del prototipo ya no forma parte del pipeline vigente: la asignación por sucursal la resuelve directamente el MILP piloto (`optimizacion_milp_piloto.py`) sobre demanda ya desagregada por sucursal (`desagregar_por_sucursal.py`).

---

## 🧹 Preprocesamiento y tratamiento de datos

El preprocesamiento se ejecuta como parte de `extract_data.py` y de la carga inicial de `train_forecasting.py` / `train_forecasting_tiered.py`, antes de entrenar cualquier modelo:

- **Consolidación y normalización.** Unificación de códigos de producto entre sucursales, normalización de unidades de medida y construcción de la serie de tiempo semanal por producto.
- **Tratamiento de vacíos.** Los días/semanas sin venta se rellenan explícitamente con cero (demanda nula, no dato faltante), diferenciándolos de huecos por cierre o incidencia operativa.
- **Corrección por quiebre de stock (demanda censurada).** Los períodos con existencia cero se tratan como demanda censurada: la venta observada subestima la demanda real. Sin esta corrección, los modelos aprenderían a reproducir las propias rupturas de stock que el sistema busca eliminar.
- **Clasificación ABC dinámica.** Categorización por rotación, margen y criticidad (método Pareto 80/95% de valor acumulado), ejecutada en `extract_data.py` (`ABC_THRESHOLD_A`, `ABC_THRESHOLD_B`).
- **Partición temporal.** División cronológica en entrenamiento/prueba (`TEST_WEEKS`, 12 semanas por defecto), sin mezcla aleatoria, para preservar el orden temporal.
- **Escalado.** Normalización min-max de las series para el LSTM, con reversión del escalado al generar las predicciones.

El módulo de optimización (`src/optimization/`) aplica su propio tratamiento adicional sobre los datos de stock/demanda, ya con los productos identificados a nivel producto-sucursal:

- **Winsorizing de la desviación estándar de demanda (`sigma_demanda`).** Capada al percentil 99 antes de calcular el colchón de seguridad del punto de reorden, porque ventas puntuales de alto volumen (posible B2B) en un subconjunto pequeño de productos distorsionaban el stock de seguridad hasta niveles absurdos (ver `optimizacion_milp_piloto.py`).
- **Exclusión de productos a granel.** Se venden por peso/volumen pero el maestro los registra en unidades (`UN`), generando demandas y puntos de reorden 100-1000x superiores a lo real; se excluyen tanto en `desagregar_por_sucursal.py` como en `optimizacion_milp_piloto.py` hasta que se implemente un tratamiento correcto de unidad de venta.
- **Tope de sensatez sobre el pronóstico.** `desagregar_por_sucursal.py` recorta cualquier pronóstico que supere 3x la venta histórica reciente del producto, como salvaguarda ante inestabilidad de ARIMA/Prophet en productos con historial corto o muy volátil.

### Privacidad y anonimización

Los datos provienen de `inventario.ventas` en PostgreSQL (alimentado desde SAP Business One HANA) y corresponden a operaciones B2B y de inventario interno, sujetas a la Ley Orgánica de Protección de Datos Personales (LOPDP) de Ecuador. El pipeline aplica:

- **Minimización de datos**: solo se extraen los campos necesarios para forecasting y optimización (código de producto, cantidad, fecha, sucursal) — no se descargan datos de contacto ni identificadores personales de clientes.
- **Control de acceso**: las credenciales de conexión viven exclusivamente en `.env` (excluido de Git vía `.gitignore`); el acceso a los JSON/CSV de resultados está restringido al equipo del proyecto (ver sección "🔒 Privacidad y Datos" más abajo).
- **Sin escritura de vuelta al ERP**: el sistema únicamente lee de SAP HANA; no existe riesgo de alterar datos de producción del ERP desde este repositorio.

---

## 🔍 Explicabilidad de los modelos

Además de las métricas de error (MAE/RMSE/MAPE), el proyecto documenta **por qué** cada modelo llega a su resultado:

- **ARIMA y Prophet** son intrínsecamente explicables: el orden `(p,d,q)(P,D,Q)m` de ARIMA y la descomposición tendencia/estacionalidad de Prophet ya exponen su lógica de decisión sin herramientas externas.
- **Holt-Winters** se explica mediante sus parámetros de suavizado (alpha=nivel, beta=tendencia, gamma=estacionalidad).
- **LSTM**, al ser una red neuronal, requiere una herramienta externa: se usa **SHAP** (`GradientExplainer`) para estimar qué semanas pasadas (dentro de la ventana de 8 semanas de entrada) pesan más en cada predicción.

Ver `explain_shap_lstm.py` y `explain_intrinsic.py` en la tabla de módulos arriba. Los model cards resultantes (resumen de 1 página por estrategia, con limitaciones y riesgos éticos identificados) se documentan en el capstone, sección 7.3.

---

## ✅ Estado actual del prototipo

El prototipo ejecutable de este repositorio cubre el flujo de datos → forecasting → optimización descrito en la Sección 7.2 del documento capstone, organizado en tres módulos dentro de `src/`:

| Carpeta | Script | Qué hace | Genera |
|---|---|---|---|
| `src/extraction/` | `extract_data.py` | Se conecta a PostgreSQL (credenciales vía `.env`), clasifica **todo** el catálogo vendido por método ABC/Pareto (80%/95% de valor acumulado) y descarga los datos de los productos en categorías A y B | `data/abc_classification.json`, `data/parsed.json`, `stock_data.json`, `prices.json`, `demand_statistics.json`, `warehouse_sale.json`, `warehouses.json` |
| `src/forecasting/` | `train_forecasting.py` | Entrena ARIMA, Prophet y LSTM por producto y selecciona el mejor (comparación completa sobre todos los productos de `data/parsed.json`, sin distinguir A/B) | `model_comparison_results.csv`, `prediction_details.json` |
| `src/forecasting/` | `train_forecasting_tiered.py` | Versión pensada para servidor: paraleliza por producto (`multiprocessing`) y hace checkpointing incremental. Aplica comparación completa ARIMA/Prophet/LSTM solo a categoría A, y suavizado exponencial (Holt-Winters) —más liviano— a categoría B, para poder cubrir miles de productos. Expone `fit_arima`/`fit_prophet`/`fit_lstm`/`load_data` que reutilizan otros scripts del pipeline | `model_comparison_results_A.csv`, `model_comparison_results_B.csv`, `prediction_details_A.jsonl` |
| `src/forecasting/` | `validate_b_sample.py` | Valida empíricamente la estrategia diferenciada A/B: corre la comparación completa de 3 modelos sobre una muestra estadísticamente representativa (n=349, 95% confianza, 5% margen) de categoría B, para contrastar contra Holt-Winters | `model_comparison_results_B_sample.csv`, `prediction_details_B_sample.jsonl` |
| `src/forecasting/` | `compare_b_significance.py` | Prueba de significancia estadística (Wilcoxon signed-rank) entre la comparación completa (muestra B) y Holt-Winters, sobre los productos coincidentes | consola + `b_significance_comparison.csv` |
| `src/forecasting/` | `summarize_final_results.py` | Recalcula mediana, media y percentiles de MAPE sobre `model_comparison_results_A.csv`, `_B.csv` y `_B_sample.csv` (la mediana, no la media, es la métrica de reporte correcta dado el sesgo por outliers de demanda intermitente) | consola |
| `src/forecasting/` | `explain_shap_lstm.py` | Explicabilidad post-hoc del LSTM (SHAP GradientExplainer): qué semanas pasadas (t-1...t-8) pesan más en la predicción, para una muestra de productos donde LSTM fue el modelo ganador | `shap_lstm_results.json`, carpeta `shap_output/` |
| `src/forecasting/` | `explain_intrinsic.py` | Explicabilidad intrínseca de ARIMA (orden p,d,q), Prophet (descomposición tendencia/estacionalidad) y Holt-Winters (parámetros de suavizado alpha/beta/gamma), para una muestra de productos de categoría A o B | `intrinsic_explanations.json`, carpeta `intrinsic_output/` |
| `src/forecasting/` | `generar_pronostico_futuro.py` | Genera el pronóstico **futuro real** (no el backtest): retoma el modelo ganador de cada producto en `model_comparison_results_A/B.csv`, lo reentrena sobre la serie completa (train+test) y predice `HORIZON_WEEKS` (4 semanas) hacia adelante, a nivel nacional agregado | `pronostico_futuro_producto.csv` |
| `src/optimization/` | `desagregar_por_sucursal.py` | Desagrega el pronóstico nacional (`pronostico_futuro_producto.csv`) a nivel producto-sucursal, repartiendo según la participación histórica de ventas de cada sucursal (últimos 6 meses); aplica un tope de sensatez (3x la venta histórica reciente) y excluye productos a **granel** (venta por peso, mal representada en unidades) | `forecast_output.csv` (+ `pronosticos_recortados_log.csv` si hubo recortes) |
| `src/optimization/` | `optimizacion_milp_piloto.py` | Optimizador MILP (PuLP) piloto: para cada par producto-sucursal en `forecast_output.csv`, calcula el punto de reorden (demanda esperada durante el lead time + colchón de seguridad `z·σ·√lead_time`) y resuelve la cantidad óptima a ordenar sujeta a MOQ y a la capacidad de bodega por sucursal. Lee stock/costos/sigma directamente de PostgreSQL (SQLAlchemy); usa lead time por defecto según grupo de producto (importado vs. nacional) cuando falta en el maestro, winsoriza `sigma_demanda` al percentil 99, excluye productos a granel, y si el solver no llega a óptimo imprime un diagnóstico de qué sucursal/producto está causando la infactibilidad | `resultados_milp_piloto.csv` |
| `src/optimization/` | `robustness_sensitivity_leadtime.py` | Anexo de robustez: reutiliza `optimizacion_milp_piloto.py` por import y compara, sobre la misma muestra de 300 pares producto-sucursal, la solución del MILP con `lead_time=0` (bug original tal como llegaba de SAP) contra la solución con el lead time corregido por grupo — cuantifica cuánto cambia el punto de reorden y el plan de compra ante esa corrección de datos | `robustness_sensitivity_leadtime_resumen.csv`, `_detalle.csv`, `_summary.txt` |

Todos los scripts resuelven las rutas de datos relativas a la raíz del proyecto (no al directorio desde donde se invoque `python`), así que pueden ejecutarse desde cualquier ubicación.

> `train_forecasting.py` (comparación completa para todo lo que haya en `parsed.json`) y `train_forecasting_tiered.py` (tratamiento diferenciado A/B, paralelo, con checkpointing) son **dos variantes del mismo paso** del pipeline: la primera es la más simple para correr en una laptop con pocos productos; la segunda es la pensada para correr en un servidor sobre el catálogo completo de categorías A+B, y es la que reutilizan `validate_b_sample.py` y `generar_pronostico_futuro.py`.
>
> `stock_data.json`, `prices.json`, `demand_statistics.json`, `warehouse_sale.json` y `warehouses.json` los sigue generando `extract_data.py`, pero **ningún script vigente del módulo de optimización los consume**: `optimizacion_milp_piloto.py` y `desagregar_por_sucursal.py` consultan esos mismos datos directamente en PostgreSQL. Quedan como remanente de una versión anterior del pipeline (que usaba `milp_reorder.py` y un algoritmo genético de transferencias, `ga_transfers.py`, ambos retirados).

---

## 🚀 Cómo ejecutar el prototipo localmente

### Paso 1 — Instalar Python

Necesita **Python 3.10, 3.11 o 3.12** (TensorFlow todavía no soporta bien 3.13 en todos los sistemas).

```bash
python3 --version
```

Si no lo tiene, descárguelo de [python.org/downloads](https://www.python.org/downloads/).

### Paso 2 — Crear un entorno virtual (recomendado)

Esto evita conflictos con otros proyectos de Python que tenga instalados.

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Sabrá que funcionó porque su terminal mostrará `(venv)` al inicio de la línea.

### Paso 3 — Instalar las librerías necesarias

Con el entorno virtual activado:

```bash
pip install -r requirements.txt
```

Esto puede tardar 5-10 minutos la primera vez (TensorFlow y Prophet son paquetes grandes).

Si `prophet` falla en Windows, instale primero:
```bash
pip install pystan==2.19.1.1
pip install prophet
```

### Paso 4 — Configurar la conexión a su base de datos

Las credenciales **ya no se escriben dentro del código**: se leen desde un archivo `.env` en la raíz del proyecto, que nunca se sube a Git (está en `.gitignore`).

1. Copie la plantilla:

   ```bash
   cp .env.example .env
   ```

2. Abra `.env` y complete con sus datos reales (los mismos que usa para conectarse con pgAdmin, DBeaver, o la herramienta que normalmente usa para ver la base `inventario`):

   ```
   DB_HOST=TU_HOST_O_IP
   DB_PORT=5432
   DB_NAME=TU_BASE_DE_DATOS
   DB_USER=TU_USUARIO
   DB_PASSWORD=TU_PASSWORD

   RANKING_START_DATE=2025-06-01
   WAREHOUSE_SALES_START_DATE=2025-12-01
   ```

   - `RANKING_START_DATE`: desde qué fecha se calcula la clasificación ABC y el precio promedio de cada producto.
   - `WAREHOUSE_SALES_START_DATE`: desde qué fecha se calculan las ventas por sucursal (usadas por el algoritmo genético de transferencias). Ambas tienen un valor por defecto si se dejan vacías.

> Si la base de datos está en un servidor remoto (no en su propia computadora), debe tener el puerto accesible desde su red, o usar un túnel SSH.

### Paso 5 — Ejecutar el pipeline completo, en este orden

```bash
python src/extraction/extract_data.py
python src/forecasting/train_forecasting_tiered.py      # o train_forecasting.py en una laptop, con pocos productos
python src/forecasting/generar_pronostico_futuro.py
python src/optimization/desagregar_por_sucursal.py
python src/optimization/optimizacion_milp_piloto.py
```

Cada script imprime su progreso en pantalla. `train_forecasting_tiered.py` es el que más tarda (varios minutos a horas, según el tamaño del catálogo, porque entrena hasta 3 modelos por cada producto de categoría A). `optimizacion_milp_piloto.py` necesita que `forecast_output.csv` ya exista (lo genera el paso anterior) y se conecta directamente a PostgreSQL para leer stock, maestro de productos y costos — no basta con haber corrido `extract_data.py`.

Recomendado dentro de `tmux` en un servidor, para que el entrenamiento siga corriendo aunque se cierre la sesión SSH:

```bash
tmux new -s forecasting
source venv/bin/activate
python src/forecasting/train_forecasting_tiered.py
# Ctrl+B, luego D -> sale de tmux sin matar el proceso
# tmux attach -t forecasting -> vuelve a conectarse para ver el progreso
```

**Pasos opcionales** (no forman parte del pipeline mínimo, pero son los que respaldan la metodología y la explicabilidad en el documento capstone):

```bash
python src/forecasting/validate_b_sample.py        # valida la estrategia diferenciada A/B sobre una muestra de categoría B
python src/forecasting/compare_b_significance.py   # prueba de Wilcoxon sobre esa muestra
python src/forecasting/summarize_final_results.py  # resumen de MAPE (mediana/media/percentiles)
python src/forecasting/explain_intrinsic.py --n-productos 3
python src/forecasting/explain_shap_lstm.py --n-productos 15
python src/optimization/robustness_sensitivity_leadtime.py   # anexo de robustez del MILP
```

Si algún producto-sucursal sale infactible en `optimizacion_milp_piloto.py`, el propio script imprime un diagnóstico en consola (comparación disponible + pedido mínimo vs. capacidad por sucursal, y el top 5 de productos con mayor pedido mínimo) — no hace falta un script aparte.

### Paso 6 — Ver los resultados

Al terminar, en la raíz del proyecto encontrará:

- `data/abc_classification.json` — clasificación ABC de todo el catálogo (útil para revisar cuántos productos entraron en cada categoría)
- `model_comparison_results_A.csv` / `model_comparison_results_B.csv` (o `model_comparison_results.csv` si usó `train_forecasting.py`) — ábralos con Excel
- `pronostico_futuro_producto.csv` — pronóstico a 4 semanas por producto, a nivel nacional
- `forecast_output.csv` — el mismo pronóstico desagregado a nivel producto-sucursal
- `resultados_milp_piloto.csv` — plan de compra sugerido: cantidad a ordenar por producto-sucursal

Y, si corrió los pasos opcionales: `model_comparison_results_B_sample.csv`, `b_significance_comparison.csv`, `intrinsic_explanations.json` + carpeta `intrinsic_output/`, `shap_lstm_results.json` + carpeta `shap_output/`, y `robustness_sensitivity_leadtime_resumen.csv` / `_detalle.csv` / `_summary.txt`.

### Personalizar la corrida

Puede ajustar estos parámetros directamente en cada script:

- **`ABC_THRESHOLD_A`** y **`ABC_THRESHOLD_B`** en `src/extraction/extract_data.py`: umbrales de valor acumulado (por defecto 80% / 95%, estándar en análisis ABC) que definen qué productos caen en cada categoría
- **`CATEGORIES_TO_FORECAST`** en `src/extraction/extract_data.py`: qué categorías se descargan para el pipeline de forecasting (por defecto `("A", "B")`)
- **`RANKING_START_DATE`** y **`WAREHOUSE_SALES_START_DATE`** en `.env`: ventanas de fechas usadas para la clasificación ABC/precios y para las ventas por sucursal (ver [Paso 4](#paso-4--configurar-la-conexión-a-su-base-de-datos))
- **`TEST_WEEKS`** en `src/forecasting/train_forecasting.py` / `train_forecasting_tiered.py`: cuántas semanas usar como conjunto de prueba (por defecto 12)
- **`HORIZON_WEEKS`** en `src/forecasting/generar_pronostico_futuro.py`: semanas hacia adelante a pronosticar (por defecto 4, consistente con `HORIZONTE_DIAS` del MILP)
- **`VENTANA_HISTORICA`** y **`TOPE_MULTIPLICADOR`** en `src/optimization/desagregar_por_sucursal.py`: ventana usada para calcular la participación histórica por sucursal, y el múltiplo de venta histórica reciente por encima del cual se recorta un pronóstico inestable
- **`PILOTO_CATEGORIA`** y **`PILOTO_ALMACEN`** en `src/optimization/optimizacion_milp_piloto.py`: filtros para acotar el piloto (por defecto `None` = todas las categorías A+B y todas las sucursales)
- **`HOLDING_COST_ANUAL_PCT`**, **`ORDER_COST_FIJO`**, **`NIVEL_SERVICIO_DEFAULT`**, **`HORIZONTE_DIAS`** y **`CAPACIDAD_BUFFER`** en `src/optimization/optimizacion_milp_piloto.py`: supuestos de costo, nivel de servicio y capacidad de bodega — ajústelos a los valores reales de la empresa
- **`LEAD_TIME_NACIONAL_DIAS`** / **`LEAD_TIME_IMPORTADO_DIAS`** / **`GRUPO_IMPORTACIONES_COD`** en `src/optimization/optimizacion_milp_piloto.py`: supuesto de lead time por grupo de producto, usado cuando el maestro no trae el dato (caso mayoritario hoy)

### Problemas comunes

| Error | Causa probable | Solución |
|---|---|---|
| `RuntimeError: Falta la variable de entorno...` | No existe `.env`, o le falta alguna variable | Repita el Paso 4: `cp .env.example .env` y complete los valores |
| `could not connect to server` | Datos de conexión incorrectos, o el puerto no está abierto | Verifique host/puerto/usuario/password en `.env`; pruebe primero conectarse con pgAdmin |
| `ModuleNotFoundError` | El entorno virtual no está activado, o falta instalar requirements | Repita el Paso 2 y 3 |
| `prophet` no instala en Windows | Falta un compilador de C++ | Instale "Visual Studio Build Tools" o use WSL (Windows Subsystem for Linux) |
| El script de LSTM tarda mucho | TensorFlow corriendo solo en CPU | Es normal; cada producto tarda ~10-15 segundos en entrenar |

---

## 🚢 Despliegue y ejecución remota (Fabric)

Correr `train_forecasting_tiered.py` sobre el catálogo completo conviene hacerlo en un servidor (no en la laptop), y repetir a mano los pasos de SSH + `tmux` cada vez es tedioso. `fabfile.py` automatiza ese flujo desde la máquina local.

### Requisitos previos

- Autenticación SSH por llave hacia el servidor, configurada una sola vez (`ssh-keygen` + `ssh-copy-id`), para no tener que escribir contraseña en cada comando.
- `pip install fabric` en su máquina local (ya está en `requirements.txt`).
- Editar las variables al inicio de `fabfile.py` (`HOST`, `PORT`, `USER`, `SSH_KEY`, `PROJECT_DIR`) con los datos reales del servidor. El repositorio debe existir previamente en `PROJECT_DIR` del servidor, con su propio `venv` y su propio `.env` (ver [Paso 4](#paso-4--configurar-la-conexión-a-su-base-de-datos)).

### Comandos disponibles

Ejecútelos desde la raíz del proyecto, en su máquina local:

| Comando | Qué hace |
|---|---|
| `fab deploy` | `git pull` + `pip install -r requirements.txt` en el servidor |
| `fab extract` | Corre `extract_data.py` en el servidor (regenera `parsed.json` y `abc_classification.json`) |
| `fab train` | Lanza `train_forecasting_tiered.py` dentro de una sesión `tmux` en background; si ya hay una corriendo, no la duplica |
| `fab status` | Muestra si el entrenamiento sigue activo y las últimas líneas de `training.log` |
| `fab attach` | Se conecta en vivo a la sesión `tmux` de entrenamiento (`Ctrl+B`, luego `D` para salir sin matar el proceso) |
| `fab stop` | Detiene la sesión de entrenamiento |
| `fab validate_b` | Lanza `validate_b_sample.py` en `tmux` (comparación completa sobre la muestra de categoría B) |
| `fab status_validate_b` | Muestra si esa validación sigue activa y las últimas líneas de `validate_b.log` |
| `fab explain_shap` | Lanza `explain_shap_lstm.py` en `tmux`; acepta `n=<cantidad>` o `codigos=<COD1,COD2>` (ej. `fab explain-shap:n=25`) |
| `fab status_explain_shap` | Muestra si esa explicabilidad sigue activa y las últimas líneas de `shap_lstm.log` |

No hay tareas de Fabric para `generar_pronostico_futuro.py`, `desagregar_por_sucursal.py` ni `optimizacion_milp_piloto.py` todavía — se corren manualmente por SSH (son más rápidos que el entrenamiento y no suelen necesitar `tmux`).

Flujo típico de una corrida completa en el servidor:

```bash
fab deploy       # trae el código más reciente e instala dependencias nuevas
fab extract      # regenera los datos de entrada
fab train        # lanza el entrenamiento en background (tmux)
fab status        # repetir cada tanto para chequear progreso
fab attach        # opcional, para ver el log en vivo
```

---

## 📁 Estructura del proyecto (actual)

```
mia-inventory-forecasting-laferia/
│
├── src/
│   ├── extraction/
│   │   └── extract_data.py                    # PostgreSQL → clasificación ABC + JSON de entrada
│   ├── forecasting/
│   │   ├── train_forecasting.py                # ARIMA / Prophet / LSTM + selección automática
│   │   ├── train_forecasting_tiered.py          # Igual, pero paralelo/checkpointed y diferenciado A/B
│   │   ├── validate_b_sample.py                 # Valida la estrategia A/B sobre una muestra de categoría B
│   │   ├── compare_b_significance.py            # Prueba de Wilcoxon sobre esa muestra
│   │   ├── summarize_final_results.py           # Resumen de MAPE (mediana/media/percentiles)
│   │   ├── explain_intrinsic.py                 # Explicabilidad ARIMA/Prophet/Holt-Winters
│   │   ├── explain_shap_lstm.py                 # Explicabilidad LSTM (SHAP)
│   │   └── generar_pronostico_futuro.py         # Pronóstico futuro real (4 semanas), no backtest
│   └── optimization/
│       ├── desagregar_por_sucursal.py           # Pronóstico nacional → demanda por producto-sucursal
│       ├── optimizacion_milp_piloto.py          # MILP (PuLP): punto de reorden y pedido óptimo
│       └── robustness_sensitivity_leadtime.py   # Anexo de robustez: sensibilidad del MILP a lead_time
│
├── data/
│   ├── abc_classification.json        # Clasificación ABC de todo el catálogo (extract_data.py)
│   └── parsed.json                    # Series semanales por producto categoría A+B
│
├── fabfile.py                         # Automatiza deploy + ejecución del pipeline en el servidor (Fabric)
├── requirements.txt
├── .env.example                       # Plantilla de variables de entorno (sin datos reales)
├── .gitignore
└── README.md
```

Los JSON/CSV de entrada y de resultados (`stock_data.json`, `prices.json`, `demand_statistics.json`, `warehouse_sale.json`, `warehouses.json`, `model_comparison_results_A.csv`, `model_comparison_results_B.csv`, `prediction_details_A.jsonl`, `pronostico_futuro_producto.csv`, `forecast_output.csv`, `resultados_milp_piloto.csv`) se guardan en la raíz del proyecto cada vez que corre los scripts. Los pasos opcionales de validación/explicabilidad/robustez generan además `model_comparison_results_B_sample.csv`, `b_significance_comparison.csv`, `intrinsic_explanations.json` (+ `intrinsic_output/`), `shap_lstm_results.json` (+ `shap_output/`) y `robustness_sensitivity_leadtime_*`. El archivo `.env` con las credenciales reales **no se sube a Git** (ver `.gitignore`).

### Estructura objetivo (aplicación web completa, planificada)

```
├── core/                    # Configuración principal Django
├── inventory/               # App: gestión de inventarios (modelos de BD)
├── forecasting/             # App: wrapper web de src/forecasting
├── optimization/            # App: wrapper web de src/optimization
├── etl/                     # Extracción SAP HANA → PostgreSQL
├── dashboard/               # App: frontend y visualizaciones
├── reports/                 # Generación de reportes automáticos
├── tests/                   # Pruebas unitarias e integración
├── docs/                    # Documentación Sphinx
├── requirements-dev.txt
├── .env.example
└── manage.py
```

---

## 📊 Módulos del Sistema

| Módulo | Descripción | Estado |
|--------|-------------|--------|
| Clasificación ABC (Pareto) | Categorización dinámica del catálogo por valor de venta acumulado (80/95) | ✅ Prototipo funcional |
| Extracción PostgreSQL | Descarga de datos históricos y generación de JSON de entrada (credenciales vía `.env`) | ✅ Prototipo funcional |
| Forecasting ARIMA / Prophet / LSTM | Comparación y selección automática por MAPE (categoría A) | ✅ Prototipo funcional |
| Forecasting Holt-Winters (tiered) | Método liviano automatizado para categoría B, paralelo y con checkpointing | ✅ Prototipo funcional |
| Validación A/B y significancia | Comparación completa sobre muestra representativa de B (n=349) + prueba de Wilcoxon | ✅ Prototipo funcional |
| Explicabilidad (intrínseca + SHAP) | Model cards: orden ARIMA, componentes Prophet/Holt-Winters, importancia SHAP del LSTM | ✅ Prototipo funcional |
| Pronóstico futuro + desagregación | Pronóstico a 4 semanas por producto (nacional) y reparto a producto-sucursal | ✅ Prototipo funcional |
| Optimizador MILP piloto | Punto de reorden y cantidad óptima de pedido por producto-sucursal (PuLP), con lead time por grupo y diagnóstico de infactibilidad | ✅ Prototipo funcional |
| Anexo de robustez (MILP) | Sensibilidad de la solución del MILP ante el supuesto de lead time | ✅ Prototipo funcional |
| Algoritmo Genético (transferencias) | Balanceo inter-sucursales (DEAP) | ⛔ Retirado del pipeline vigente |
| ETL SAP HANA → PostgreSQL | Extracción y carga automatizada de datos históricos | 🔲 Pendiente |
| Dashboard Web (Django) | Visualización de KPIs, alertas y recomendaciones | 🔲 Pendiente |
| Pruebas UAT | Validación con usuarios reales (SUS) | 🔲 Pendiente |

---

## 📅 Cronograma

| Fase | Semanas | Actividades principales |
|------|---------|------------------------|
| 1 — Análisis | Sem 1-2 | EDA, entrevistas, requerimientos (SRS) |
| 2 — Diseño | Sem 3-4 | Arquitectura, BD, diseño de modelos |
| 3 — Desarrollo | Sem 5-8 | ETL, forecasting, optimización, dashboard |
| 4 — Validación | Sem 9-10 | Métricas MAE/RMSE/MAPE, pruebas UAT |
| 5 — Piloto | Sem 11-12 | Piloto en sucursal, capacitación, informe final |

---

## 📏 Métricas de Evaluación

### Modelos de Forecasting
- **MAE** — Error Absoluto Medio
- **RMSE** — Raíz del Error Cuadrático Medio
- **MAPE** — Error Porcentual Absoluto Medio (objetivo: < 15% en categoría A)

### Pruebas de Usabilidad (UAT)
- **SUS** (System Usability Scale) — objetivo: ≥ 68 puntos
- 5 usuarios reales, 2 sesiones de 90 minutos

### KPIs Operativos (piloto)
- Reducción de rupturas de stock: objetivo ≥ 30%
- Reducción de tiempo de decisión: objetivo ≥ 50%

---

## 🔒 Privacidad y Datos

> ⚠️ **Los datos reales de Comercial La Feria usados para generar los archivos de este repositorio están protegidos bajo Acuerdo de Confidencialidad (NDA).**

Antes de compartir este repositorio públicamente, revise que los JSON/CSV generados por los scripts (stock, precios, ventas por almacén, clasificación ABC, resultados de forecasting/optimización, etc.) no queden incluidos en el control de versiones ni se distribuyan fuera del equipo del proyecto. El archivo `.env` con las credenciales de conexión ya está excluido vía `.gitignore`, pero **la carpeta `data/` y los JSON/CSV de resultados en la raíz del proyecto NO están en `.gitignore`** y deben revisarse manualmente antes de cualquier `git add`.

---

## 📄 Licencia

Este proyecto es de uso académico exclusivo, desarrollado como trabajo de titulación de la Maestría en Inteligencia Artificial Aplicada. Todos los derechos sobre los datos pertenecen a Comercial La Feria. Los algoritmos y el código desarrollado son propiedad intelectual de los autores y la universidad.

---

## 📬 Contacto

- **Autor 1:** Lam Cheang Wiliam David
- **Autor 2:** Román Largo Jessica Johanna
- **Tutor:** Criollo Caizaguano Luis Santiago
- **Institución:** Facultad de Ingeniería y Ciencias Aplicadas
