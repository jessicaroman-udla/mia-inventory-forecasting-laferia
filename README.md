# 🏪 Sistema Inteligente de Gestión de Inventarios con Forecasting Multi-Sucursal

> **Proyecto de Titulación — Maestría en Inteligencia Artificial Aplicada**  
> Facultad de Ingeniería y Ciencias Aplicadas

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-Academic-blue?style=flat)](#-licencia)
[![Status](https://img.shields.io/badge/Status-Prototipo-yellow?style=flat)]()

---

## 📋 Descripción

Sistema que integra modelos de **forecasting de demanda** y algoritmos de
**optimización matemática** para automatizar la gestión de inventarios en las
sucursales de Comercial La Feria (Santo Domingo de los Tsáchilas, Ecuador).

El sistema extrae datos históricos desde **PostgreSQL** (alimentado desde SAP
Business One HANA), entrena modelos predictivos sobre los patrones de venta y
genera recomendaciones de reabastecimiento, transferencias inter-sucursales y
alertas de stock crítico.

> Este repositorio contiene el **prototipo de línea de comandos** de la
> Sección 7.2 del documento capstone: los pipelines de forecasting y
> optimización, ejecutables sin interfaz web. La aplicación web Django y el ETL
> desde SAP HANA todavía no están implementados.

---

## 👥 Equipo

| Rol | Nombre | Responsabilidad principal |
|-----|--------|---------------------------|
| Autor 1 | Lam Cheang Wiliam David | Desarrollo web, integración ETL, arquitectura |
| Autor 2 | Román Largo Jessica Johanna | Modelado ML, forecasting, optimización |
| Tutor | Criollo Caizaguano Luis Santiago | Dirección académica |

---

## 🎯 Objetivo General

Diseñar e implementar un sistema de gestión de inventarios con modelos de
forecasting (ARIMA, Prophet, LSTM, Holt-Winters) y optimización entera para
Comercial La Feria, a fin de reducir rupturas de stock y costos operativos.

---

## ⚙️ Tecnologías

| Área | Librerías |
|---|---|
| Datos / BD | `psycopg2`, `sqlalchemy`, `python-dotenv` |
| Forecasting | `pmdarima` (ARIMA), `prophet`, `tensorflow-cpu` (LSTM), `statsmodels` (Holt-Winters) |
| Explicabilidad | `shap` (LSTM), descomposición intrínseca de ARIMA/Prophet/Holt-Winters |
| Optimización | `pulp` (MILP), `deap` (algoritmo genético) |
| Estadística | `scipy` (Wilcoxon), `numpy`, `pandas` |
| Visualización | `matplotlib` |
| Deploy | `fabric` (SSH + tmux) |

---

## 🏗️ Arquitectura objetivo del sistema

```
        SAP Business One HANA  ──ETL (pendiente)──►  PostgreSQL 16
                                                         │
                          ┌──────────────────────────────┴───────────────┐
                          ▼                                              ▼
                 Módulo Forecasting                            Módulo Optimización
          ARIMA / Prophet / LSTM / Holt-Winters               MILP (PuLP) + AG (DEAP)
              → demanda futura por producto                 → órdenes y transferencias
                          └──────────────────────┬───────────────────────┘
                                                 ▼
                           Django Web App  (dashboard / alertas — pendiente)
```

Hoy existen como prototipo los dos módulos centrales (`src/forecasting/`,
`src/optimization/`) y la extracción desde PostgreSQL (`src/extraction/`).

---

## 📁 Estructura del proyecto

```
mia-inventory-forecasting-laferia/
│
├── src/
│   ├── sample_data/
│   │   └── generate_sample.py             # dataset sintético de ejemplo (corre el pipeline SIN base de datos)
│   ├── extraction/
│   │   ├── extract_data.py                # PostgreSQL → clasificación ABC + JSON de entrada
│   │   └── stats_dataset.py               # resumen descriptivo y de calidad de `inventario.ventas` (solo lectura)
│   ├── forecasting/
│   │   ├── train_forecasting.py           # [referencia] ARIMA/Prophet/LSTM sobre todo parsed.json
│   │   ├── train_forecasting_tiered.py    # [principal] A: 3 modelos · B: Holt-Winters · paralelo + checkpoint
│   │   ├── validate_b_sample.py           # valida la estrategia B sobre una muestra (n=349)
│   │   ├── compare_b_significance.py      # prueba de Wilcoxon: 3 modelos vs Holt-Winters
│   │   ├── summarize_final_results.py     # mediana/percentiles de MAPE de A y B
│   │   ├── metrics_extra.py               # métricas adicionales (sMAPE, WAPE, MASE) sobre resultados ya generados
│   │   ├── generar_pronostico_futuro.py   # pronóstico FUTURO real (no backtest)
│   │   ├── explain_shap_lstm.py           # explicabilidad post-hoc del LSTM (SHAP)
│   │   ├── explain_intrinsic.py           # explicabilidad intrínseca ARIMA/Prophet/Holt-Winters
│   │   ├── diagnostico_series.py          # diagnóstico de productos con MAPE alto
│   │   ├── generate_charts.py             # gráficos de comparación de modelos
│   │   ├── plot_real_vs_pred.py           # gráfico real vs. predicho sobre el conjunto de prueba
│   │   └── plot_shap_thesis.py            # re-dibuja los gráficos SHAP en alta resolución para la tesis
│   ├── baseline/
│   │   ├── baseline_naive.py              # baseline Naive / Seasonal naive (mismos datos y métricas que el pipeline)
│   │   └── README_baseline.md
│   └── optimization/
│       ├── milp_reorder.py                # [referencia] MILP sobre 15 productos A (sin BD)
│       ├── desagregar_por_sucursal.py     # pronóstico nacional → demanda por producto-sucursal
│       ├── optimizacion_milp_piloto.py    # [principal] MILP por producto-sucursal (con BD)
│       ├── ga_transfers.py                # algoritmo genético de transferencias inter-sucursales
│       ├── robustness_sensitivity_leadtime.py  # anexo de robustez: sensibilidad al lead time
│       └── generate_optimization_charts.py
│
├── data/                    # TODO lo generado (entradas y resultados) — NO se versiona (NDA)
│   └── README.md            # descripción de cada archivo que vive aquí
│
├── docs/
│   ├── INSTRUCCIONES_EJECUCION.md   # guía reproducible paso a paso SIN base de datos ni credenciales
│   └── img/                         # capturas de ejemplo para la documentación
│
├── fabfile.py               # deploy + ejecución remota del pipeline (Fabric)
├── requirements.txt
├── .env.example             # plantilla de variables de entorno
├── LICENSE                  # licencia de uso académico
├── CITATION.cff             # cómo citar el trabajo
├── AUTHORS.md               # autores y responsabilidades
└── README.md
```

Todos los scripts resuelven las rutas relativas a la raíz del proyecto
(vía `Path(__file__)`), así que se pueden ejecutar desde cualquier ubicación.
Cada script lee y escribe **siempre dentro de `data/`**.

---

## 🔀 Los dos "tracks" del pipeline

El mismo paso (forecasting, MILP) tiene dos implementaciones según la escala:

| | **Track de referencia** (laptop) | **Track principal** (servidor) |
|---|---|---|
| Forecasting | `train_forecasting.py` | `train_forecasting_tiered.py` |
| MILP | `milp_reorder.py` | `desagregar_por_sucursal.py` → `optimizacion_milp_piloto.py` |
| Escala | 15 productos A, demanda nacional | catálogo A+B, por producto-sucursal |
| Necesita BD activa al correr | solo para `extract_data.py` | sí (los scripts consultan PostgreSQL en vivo) |
| Paralelo / checkpointing | no | sí (`multiprocessing`, reanudable) |
| Para qué sirve | leer y entender la metodología | generar los resultados de la tesis |

El algoritmo genético (`ga_transfers.py`) y el baseline
(`src/baseline/baseline_naive.py`) son comunes a ambos tracks.

---

## 🧹 Preprocesamiento y tratamiento de datos

Se ejecuta dentro de `extract_data.py` y de la carga de los scripts de
entrenamiento, antes de entrenar cualquier modelo:

- **Consolidación y normalización.** Unificación de códigos de producto entre
  sucursales, normalización de unidades y construcción de la serie semanal.
- **Tratamiento de vacíos.** Las semanas sin venta se rellenan con cero
  (demanda nula, no dato faltante).
- **Detección de atípicos.** Picos por ventas mayoristas o promociones se
  identifican por rango intercuartílico (ver `diagnostico_series.py`).
- **Demanda censurada.** Los períodos con stock cero se tratan como demanda
  subestimada; sin esta corrección los modelos aprenderían a reproducir las
  propias rupturas de stock.
- **Clasificación ABC dinámica.** Pareto 80/95 % de valor de venta acumulado
  (`ABC_THRESHOLD_A`, `ABC_THRESHOLD_B` en `extract_data.py`).
- **Partición temporal.** División cronológica train/test (`TEST_WEEKS`, 12
  semanas por defecto), sin mezcla aleatoria.
- **Escalado.** Normalización min-max para el LSTM, revertida al predecir.
- **Exclusión de productos a granel** (`GRANEL` en la descripción): la unidad
  de medida `UN` no representa el volumen real vendido por peso.

### Privacidad y anonimización

Los datos provienen de `inventario.ventas` en PostgreSQL y están sujetos a la
Ley Orgánica de Protección de Datos Personales (LOPDP) de Ecuador. El pipeline
aplica **minimización** (solo código, cantidad, fecha, sucursal — sin datos de
cliente), **control de acceso** (credenciales solo en `.env`) y **sin escritura
de vuelta al ERP** (solo lectura de SAP HANA).

---

## 🔍 Explicabilidad de los modelos

- **ARIMA / Prophet / Holt-Winters** son intrínsecamente explicables: el orden
  `(p,d,q)(P,D,Q)m`, la descomposición tendencia/estacionalidad y los
  parámetros de suavizado (α, β, γ) exponen su lógica sin herramientas externas
  → `explain_intrinsic.py`.
- **LSTM** requiere una herramienta externa: **SHAP** (`GradientExplainer`)
  para estimar qué semanas pasadas (ventana de 8) pesan más en cada predicción
  → `explain_shap_lstm.py`.

---

## 🚀 Cómo ejecutar el prototipo

> **¿Solo quieres reproducir los resultados sin acceso a la base de datos?**
> Sigue [`docs/INSTRUCCIONES_EJECUCION.md`](docs/INSTRUCCIONES_EJECUCION.md): usa
> un dataset sintético (`src/sample_data/generate_sample.py`) y corre el
> pipeline completo sin PostgreSQL ni credenciales. El resto de esta sección
> describe la ejecución con datos reales.

### 1. Requisitos

Python **3.10, 3.11 o 3.12** (TensorFlow aún no soporta bien 3.13).

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt     # 5-10 min la primera vez
```

### 2. Configurar la conexión a la base de datos

```bash
cp .env.example .env
```

Edita `.env` con los datos reales de la base `inventario` (los mismos de
pgAdmin/DBeaver). Ver los comentarios dentro de `.env.example` para cada
variable. `.env` **nunca se sube a Git**.

### 3. Ejecutar

**Track de referencia (laptop, pocos productos):**

```bash
python src/extraction/extract_data.py
python src/forecasting/train_forecasting.py
python src/forecasting/generate_charts.py
python src/optimization/milp_reorder.py
python src/optimization/ga_transfers.py
python src/optimization/generate_optimization_charts.py
```

**Track principal (servidor, catálogo completo A+B):**

```bash
python src/extraction/extract_data.py
python src/forecasting/train_forecasting_tiered.py      # el paso más largo
python src/forecasting/summarize_final_results.py       # resumen de MAPE de A y B
python src/forecasting/generar_pronostico_futuro.py     # pronóstico futuro real
python src/optimization/desagregar_por_sucursal.py      # → data/forecast_output.csv
python src/optimization/optimizacion_milp_piloto.py     # → data/resultados_milp_piloto.csv
python src/optimization/ga_transfers.py                 # → data/resultado_ga_transferencias.json
```

Pasos opcionales de baseline / validación / explicabilidad / robustez:

```bash
python src/baseline/baseline_naive.py                   # baseline Naive / Seasonal naive (sin BD)
python src/forecasting/metrics_extra.py                 # sMAPE, WAPE y MASE sobre los resultados ya generados
python src/forecasting/plot_real_vs_pred.py             # gráfico real vs. predicho del conjunto de prueba
python src/forecasting/validate_b_sample.py
python src/forecasting/compare_b_significance.py
python src/forecasting/explain_shap_lstm.py
python src/forecasting/plot_shap_thesis.py              # re-dibuja los SHAP en alta resolución (no recalcula)
python src/forecasting/explain_intrinsic.py --source A
python src/forecasting/diagnostico_series.py
python src/optimization/robustness_sensitivity_leadtime.py
```

### 4. Ver los resultados

Todo queda dentro de **`data/`** (ver `data/README.md` para el detalle):

| Archivo | Qué contiene |
|---|---|
| `data/model_comparison_results_A.csv` / `_B.csv` | MAPE por producto y modelo seleccionado |
| `data/model_comparison_results_baseline*.csv` | MAE/RMSE/MAPE de Naive y Seasonal naive (detalle + resumen) |
| `data/metrics_extra_detalle.csv` / `_resumen.csv` | sMAPE, WAPE y MASE por producto y agregados |
| `data/pronostico_futuro_producto.csv` | demanda proyectada por producto |
| `data/resultados_milp_piloto.csv` | punto de reorden y cantidad a ordenar por producto-sucursal |
| `data/resultado_ga_transferencias.json` | plan de transferencias entre sucursales |
| `data/robustness_sensitivity_leadtime_summary.txt` | anexo de robustez (sensibilidad al lead time) |
| `data/charts/*.png` | gráficos de forecasting y de optimización |

### Parámetros configurables

| Parámetro | Archivo | Por defecto |
|---|---|---|
| `ABC_THRESHOLD_A` / `ABC_THRESHOLD_B` | `extract_data.py` | 0.80 / 0.95 |
| `CATEGORIES_TO_FORECAST` | `extract_data.py` | `("A", "B")` |
| `RANKING_START_DATE` / `WAREHOUSE_SALES_START_DATE` | `.env` | 2025-06-01 / 2025-12-01 |
| `TEST_WEEKS` | `train_forecasting*.py` | 12 |
| `HORIZON_WEEKS` | `generar_pronostico_futuro.py` | 4 |
| `CAPITAL_BUDGET` / `WAREHOUSE_CAPACITY_UNITS` | `milp_reorder.py` | 40 000 / 400 000 |
| `HOLDING_COST_ANUAL_PCT`, `ORDER_COST_FIJO`, `Z_SCORE_95` | `optimizacion_milp_piloto.py` | 0.20 / 20.0 / 1.645 |
| `SAFETY_DAYS` / `EXCESS_DAYS` | `ga_transfers.py` | 7 / 30 |

### Problemas comunes

| Error | Solución |
|---|---|
| `RuntimeError: Falta la variable de entorno...` | `cp .env.example .env` y completar los valores |
| `could not connect to server` | verificar host/puerto/usuario en `.env`; probar primero con pgAdmin |
| `ModuleNotFoundError` | activar el venv y `pip install -r requirements.txt` |
| `FileNotFoundError: data/...` | correr antes el script que genera ese archivo (ver tabla de arriba) |
| `prophet` no instala en Windows | instalar "Visual Studio Build Tools" o usar WSL |

---

## 🚢 Despliegue y ejecución remota (Fabric)

El track principal conviene correrlo en un servidor. `fabfile.py` automatiza
el flujo SSH + `tmux`. Los datos del servidor se leen de `.env`
(`DEPLOY_HOST`, `DEPLOY_PORT`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`,
`DEPLOY_PROJECT_DIR`).

| Comando | Qué hace |
|---|---|
| `fab deploy` | `git pull` + `pip install -r requirements.txt` en el servidor |
| `fab extract` | corre `extract_data.py` en el servidor |
| `fab train` | lanza `train_forecasting_tiered.py` en `tmux` (background, no duplica sesión) |
| `fab status` | estado del entrenamiento + últimas líneas de `data/logs/training.log` |
| `fab attach` | se conecta en vivo a la sesión `tmux` (`Ctrl+B`, luego `D` para salir) |
| `fab stop` | detiene la sesión |
| `fab validate-b` / `fab explain-shap` | lanzan esos scripts en `tmux` |

---

## 📏 Métricas de evaluación

- **Forecasting:** MAE, RMSE y **MAPE** (objetivo < 15 % en categoría A). La
  **mediana** de MAPE es la métrica de reporte (no la media), por el sesgo de
  la demanda intermitente.
- **Métricas complementarias** (`metrics_extra.py`): sMAPE, WAPE y MASE
  (Hyndman & Koehler, 2006), calculadas sobre los resultados ya generados sin
  reentrenar.
- **Baseline** (`src/baseline/baseline_naive.py`): Naive (random walk) y
  Seasonal naive, sobre exactamente los mismos datos, partición y métricas que
  el pipeline principal. Es el punto de referencia mínimo contra el que se
  contrasta si ARIMA/Prophet/LSTM/Holt-Winters aportan mejora
  (ver `src/baseline/README_baseline.md`).
- **Significancia estadística** (`compare_b_significance.py`): prueba de
  Wilcoxon pareada para la estrategia diferenciada A/B.
- **Usabilidad (UAT, pendiente):** SUS ≥ 68, 5 usuarios, 2 sesiones.
- **KPIs operativos (piloto, pendiente):** reducción de rupturas ≥ 30 %,
  reducción de tiempo de decisión ≥ 50 %.

---

## 🔒 Privacidad y datos

> ⚠️ **Los datos reales de Comercial La Feria están protegidos bajo Acuerdo de
> Confidencialidad (NDA).**

Todo lo que produce el pipeline se escribe en `data/`, que está **excluida de
Git** (`.gitignore` ignora `data/*` salvo `data/README.md`). El archivo `.env`
con las credenciales también está excluido. Para enviar un resultado concreto
a revisión: `git add -f data/<archivo>`.

---

## 📄 Licencia

Uso académico exclusivo (trabajo de titulación, Maestría en Inteligencia
Artificial Aplicada). Los datos pertenecen a Comercial La Feria; el código es
propiedad intelectual de los autores y la universidad. Ver [`LICENSE`](LICENSE).

Para citar este trabajo, ver [`CITATION.cff`](CITATION.cff).

---

## 📬 Contacto

- **Autor 1:** Lam Cheang Wiliam David
- **Autor 2:** Román Largo Jessica Johanna
- **Tutor:** Criollo Caizaguano Luis Santiago
- **Institución:** Facultad de Ingeniería y Ciencias Aplicadas
