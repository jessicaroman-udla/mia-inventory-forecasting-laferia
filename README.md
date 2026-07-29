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
- **PuLP** — Programación Lineal Entera Mixta (MILP) para cantidades óptimas de pedido
- **DEAP** — Algoritmo Genético para balanceo inter-sucursales

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
│  LSTM               │    │  Algoritmo Genético   │
│  → Predicción       │    │  → Órdenes óptimas y  │
│    demanda futura   │    │    transferencias     │
└─────────┬───────────┘    └──────────┬────────────┘
          └────────────┬──────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Django Web Application                    │
│   Dashboard │ Alertas │ Recomendaciones │ Reportes │ UAT   │
└─────────────────────────────────────────────────────────────┘
```

Los dos módulos de la mitad (Forecasting y Optimización) son lo que existe hoy como prototipo de línea de comandos en `src/`. La capa de PostgreSQL de entrada también existe (`src/extraction/extract_data.py`). El ETL desde SAP HANA y la aplicación web Django todavía no están implementados.

---

## ✅ Estado actual del prototipo

El prototipo ejecutable de este repositorio cubre el flujo de datos → forecasting → optimización descrito en la Sección 7.2 del documento capstone, organizado en tres módulos dentro de `src/`:

| Carpeta | Script | Qué hace | Genera |
|---|---|---|---|
| `src/extraction/` | `extract_data.py` | Se conecta a PostgreSQL (credenciales vía `.env`), clasifica **todo** el catálogo vendido por método ABC/Pareto (80%/95% de valor acumulado) y descarga los datos de los productos en categorías A y B | `data/abc_classification.json`, `data/parsed.json`, `stock_data.json`, `prices.json`, `demand_statistics.json`, `warehouse_sale.json`, `warehouses.json` |
| `src/forecasting/` | `train_forecasting.py` | Entrena ARIMA, Prophet y LSTM por producto y selecciona el mejor (comparación completa sobre todos los productos de `data/parsed.json`, sin distinguir A/B) | `model_comparison_results.csv`, `prediction_details.json` |
| `src/forecasting/` | `train_forecasting_tiered.py` | Versión pensada para servidor: paraleliza por producto (`multiprocessing`) y hace checkpointing incremental. Aplica comparación completa ARIMA/Prophet/LSTM solo a categoría A, y suavizado exponencial (Holt-Winters) —más liviano— a categoría B, para poder cubrir miles de productos | `model_comparison_results_A.csv`, `model_comparison_results_B.csv`, `prediction_details_A.jsonl` |
| `src/forecasting/` | `diagnostico_series.py` | Diagnóstico de los productos con MAPE más alto: gráfico de la serie, % de semanas en cero, coeficiente de variación, outliers (regla IQR) y comparación train vs. test / real vs. predicho | Carpeta `diagnostico_output/` con `.png` y 2 CSV de resumen |
| `src/forecasting/` | `generate_charts.py` | Genera los gráficos de comparación de modelos (a partir de `model_comparison_results.csv`) | 3 archivos `.png` |
| `src/optimization/` | `milp_reorder.py` | Optimizador MILP (PuLP): punto de reorden y cantidad óptima de pedido | `resultado_milp_reorden.json` |
| `src/optimization/` | `ga_transfers.py` | Algoritmo genético (DEAP): balanceo de transferencias entre sucursales | `resultado_ga_transferencias.json` |
| `src/optimization/` | `generate_optimization_charts.py` | Genera los gráficos del MILP y del algoritmo genético | 2 archivos `.png` |

Todos los scripts resuelven las rutas de datos relativas a la raíz del proyecto (no al directorio desde donde se invoque `python`), así que pueden ejecutarse desde cualquier ubicación.

> `train_forecasting.py` (comparación completa para todo lo que haya en `parsed.json`) y `train_forecasting_tiered.py` (tratamiento diferenciado A/B, paralelo, con checkpointing) son **dos variantes del mismo paso** del pipeline: la primera es la más simple para correr en una laptop con pocos productos; la segunda es la pensada para correr en un servidor sobre el catálogo completo de categorías A+B. `diagnostico_series.py` es una herramienta auxiliar de análisis (no forma parte del pipeline principal), útil cuando `train_forecasting.py` reporta un MAPE alto en algún producto.

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
python src/forecasting/train_forecasting.py
python src/forecasting/generate_charts.py
python src/optimization/milp_reorder.py
python src/optimization/ga_transfers.py
python src/optimization/generate_optimization_charts.py
```

Cada script imprime su progreso en pantalla. `train_forecasting.py` es el que más tarda (varios minutos, porque entrena 3 modelos por cada producto de categoría A+B).

**Alternativa para correr sobre todo el catálogo (categorías A+B) en un servidor:**

```bash
python src/forecasting/train_forecasting_tiered.py
```

Este script reemplaza a `train_forecasting.py` cuando el número de productos es grande: paraleliza el entrenamiento entre núcleos y guarda cada producto procesado de inmediato (checkpointing), así que si se corta la conexión SSH basta con volver a correrlo — retoma donde se quedó. Recomendado dentro de `tmux` para que siga corriendo aunque se cierre la sesión:

```bash
tmux new -s forecasting
source venv/bin/activate
python src/forecasting/train_forecasting_tiered.py
# Ctrl+B, luego D -> sale de tmux sin matar el proceso
# tmux attach -t forecasting -> vuelve a conectarse para ver el progreso
```

**Diagnóstico opcional** si algún producto sale con MAPE alto en `model_comparison_results.csv`:

```bash
python src/forecasting/diagnostico_series.py
```

### Paso 6 — Ver los resultados

Al terminar, en la raíz del proyecto encontrará:

- `data/abc_classification.json` — clasificación ABC de todo el catálogo (útil para revisar cuántos productos entraron en cada categoría)
- `model_comparison_results.csv` (o `model_comparison_results_A.csv` / `_B.csv` si usó la versión `tiered`) — ábralo con Excel
- `grafico_comparacion_mape.png`, `grafico_distribucion_modelos.png`, `grafico_forecast_vs_real.png`
- `resultado_milp_reorden.json`, `grafico_plan_compra_milp.png`
- `resultado_ga_transferencias.json`, `grafico_dias_inventario_ga.png`

### Personalizar la corrida

Puede ajustar estos parámetros directamente en cada script:

- **`ABC_THRESHOLD_A`** y **`ABC_THRESHOLD_B`** en `src/extraction/extract_data.py`: umbrales de valor acumulado (por defecto 80% / 95%, estándar en análisis ABC) que definen qué productos caen en cada categoría
- **`CATEGORIES_TO_FORECAST`** en `src/extraction/extract_data.py`: qué categorías se descargan para el pipeline de forecasting (por defecto `("A", "B")`)
- **`RANKING_START_DATE`** y **`WAREHOUSE_SALES_START_DATE`** en `.env`: ventanas de fechas usadas para la clasificación ABC/precios y para las ventas por sucursal (ver [Paso 4](#paso-4--configurar-la-conexión-a-su-base-de-datos))
- **`TEST_WEEKS`** en `src/forecasting/train_forecasting.py` / `train_forecasting_tiered.py`: cuántas semanas usar como conjunto de prueba (por defecto 12)
- **`CAPITAL_BUDGET`** y **`WAREHOUSE_CAPACITY_UNITS`** en `src/optimization/milp_reorder.py`: ajústelos a los valores reales de la empresa
- **`LEAD_TIME_DAYS`** en `src/optimization/milp_reorder.py`: reemplace el supuesto de 10 días si consigue el dato real de lead time
- **`SAFETY_DAYS`** y **`EXCESS_DAYS`** en `src/optimization/ga_transfers.py`: rango objetivo de días de inventario por sucursal (por defecto 7-30 días)

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
| `fab status` | Muestra si el entrenamiento sigue activo y las últimas 30 líneas de `training.log` |
| `fab attach` | Se conecta en vivo a la sesión `tmux` para ver el progreso (`Ctrl+B`, luego `D` para salir sin matar el proceso) |
| `fab stop` | Detiene la sesión de entrenamiento |

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
│   │   └── extract_data.py            # PostgreSQL → clasificación ABC + JSON de entrada
│   ├── forecasting/
│   │   ├── train_forecasting.py       # ARIMA / Prophet / LSTM + selección automática
│   │   ├── train_forecasting_tiered.py # Igual, pero paralelo/checkpointed y diferenciado A/B
│   │   ├── diagnostico_series.py      # Diagnóstico de productos con MAPE alto
│   │   └── generate_charts.py         # Gráficos de comparación de modelos
│   └── optimization/
│       ├── milp_reorder.py            # MILP (PuLP): punto de reorden y pedido óptimo
│       ├── ga_transfers.py            # Algoritmo genético (DEAP): transferencias
│       └── generate_optimization_charts.py
│
├── data/
│   ├── abc_classification.json        # Clasificación ABC de todo el catálogo (extract_data.py)
│   └── parsed.json                    # Series semanales por producto categoría A+B
│
├── prototipo_comercial_la_feria.ipynb # Notebook equivalente, para correr en Colab
├── fabfile.py                         # Automatiza deploy + ejecución del pipeline en el servidor (Fabric)
├── requirements.txt
├── .env.example                       # Plantilla de variables de entorno (sin datos reales)
├── .gitignore
└── README.md
```

Los JSON/CSV de entrada y de resultados (`stock_data.json`, `prices.json`, `demand_statistics.json`, `warehouse_sale.json`, `warehouses.json`, `prediction_details.json`, `model_comparison_results.csv`, `resultado_milp_reorden.json`, `resultado_ga_transferencias.json`) y los `.png` generados se guardan en la raíz del proyecto cada vez que corre los scripts. La variante `train_forecasting_tiered.py` genera además `model_comparison_results_A.csv`, `model_comparison_results_B.csv` y `prediction_details_A.jsonl`; `diagnostico_series.py` guarda todo dentro de `diagnostico_output/`. El archivo `.env` con las credenciales reales **no se sube a Git** (ver `.gitignore`).

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
| Diagnóstico de series | Análisis de productos con MAPE alto (intermitencia, CV, outliers, cambios de nivel train/test) | ✅ Prototipo funcional |
| Optimizador MILP | Cantidades óptimas de pedido (PuLP) | ✅ Prototipo funcional |
| Algoritmo Genético | Balanceo inter-sucursales (DEAP) | ✅ Prototipo funcional |
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

Antes de compartir este repositorio públicamente, revise que los JSON/CSV generados por `src/extraction/extract_data.py` (stock, precios, ventas por almacén, clasificación ABC, etc.) no queden incluidos en el control de versiones ni se distribuyan fuera del equipo del proyecto. El archivo `.env` con las credenciales de conexión ya está excluido vía `.gitignore`, pero los JSON/CSV de resultados se generan en la raíz del proyecto y deben revisarse manualmente antes de cualquier `git add`.

---

## 📄 Licencia

Este proyecto es de uso académico exclusivo, desarrollado como trabajo de titulación de la Maestría en Inteligencia Artificial Aplicada. Todos los derechos sobre los datos pertenecen a Comercial La Feria. Los algoritmos y el código desarrollado son propiedad intelectual de los autores y la universidad.

---

## 📬 Contacto

- **Autor 1:** Lam Cheang Wiliam David
- **Autor 2:** Román Largo Jessica Johanna
- **Tutor:** Criollo Caizaguano Luis Santiago
- **Institución:** Facultad de Ingeniería y Ciencias Aplicadas
