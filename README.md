# 🏪 Sistema Inteligente de Gestión de Inventarios con Forecasting Multi-Sucursal

> **Proyecto de Titulación — Maestría en Inteligencia Artificial Aplicada**  
> Facultad de Ingeniería y Ciencias Aplicadas

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.x-092E20?style=flat&logo=django&logoColor=white)](https://djangoproject.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-Academic-blue?style=flat)](LICENSE)
[![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow?style=flat)]()

---

## 📋 Descripción

Sistema web inteligente que integra modelos de **forecasting de demanda** y algoritmos de **optimización matemática** para automatizar la gestión de inventarios en las sucursales de Comercial La Feria (Santo Domingo de los Tsáchilas, Ecuador).

El sistema extrae datos históricos desde **SAP Business One HANA**, entrena modelos predictivos sobre los patrones de venta y genera recomendaciones automáticas de reabastecimiento, transferencias inter-sucursales y alertas de stock crítico.

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
- **Django 5.x** + Django REST Framework — Framework web
- **PostgreSQL 16** — Base de datos principal
- **hdbcli** — Conector SAP Business One HANA (ETL)
- **Celery + Redis** — Tareas programadas y actualización de modelos

### Forecasting & Machine Learning
- **statsmodels** — Modelos ARIMA
- **Prophet v1.1** (Meta) — Forecasting con estacionalidad
- **TensorFlow / Keras 2.x** — Red neuronal LSTM

### Optimización Matemática
- **PuLP 2.x** — Programación Lineal Entera Mixta (MILP) para cantidades óptimas de pedido
- **DEAP 1.4** — Algoritmo Genético para balanceo inter-sucursales
- **SciPy** — Simulated Annealing (método de comparación)

### Frontend & Visualización
- **Django Templates** + **Bootstrap 5**
- **Plotly.js** — Dashboards interactivos
- **Chart.js** — Gráficos de KPIs

### DevOps & Herramientas
- **Git / GitHub** — Control de versiones
- **Sphinx** — Documentación técnica
- **Trello** — Gestión de tareas (Kanban)

---

## 🏗️ Arquitectura del Sistema

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
│  → Predicción       │    │  Simulated Annealing  │
│    demanda futura   │    │  → Órdenes óptimas    │
└─────────┬───────────┘    └──────────┬────────────┘
          └────────────┬──────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Django Web Application                    │
│   Dashboard │ Alertas │ Recomendaciones │ Reportes │ UAT   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura del Proyecto

```
mia-inventory-forecasting-laferia/
│
├── 📂 core/                    # Configuración principal Django
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── 📂 inventory/               # App: gestión de inventarios
│   ├── models.py               # Modelos de BD (Producto, Sucursal, Stock)
│   ├── views.py
│   └── serializers.py
│
├── 📂 forecasting/             # App: modelos predictivos
│   ├── arima_model.py
│   ├── prophet_model.py
│   ├── lstm_model.py
│   └── model_selector.py       # Selección automática por MAPE
│
├── 📂 optimization/            # App: algoritmos de optimización
│   ├── milp_solver.py          # PuLP — cantidades óptimas de pedido
│   ├── genetic_algorithm.py    # DEAP — balanceo inter-sucursales
│   └── simulated_annealing.py  # SciPy — método de comparación
│
├── 📂 etl/                     # Extracción SAP HANA → PostgreSQL
│   ├── hana_connector.py
│   ├── transformers.py
│   └── tasks.py                # Tareas Celery programadas
│
├── 📂 dashboard/               # App: frontend y visualizaciones
│   ├── views.py
│   └── templates/
│       └── dashboard/
│
├── 📂 reports/                 # Generación de reportes automáticos
│
├── 📂 tests/                   # Pruebas unitarias e integración
│   ├── test_forecasting.py
│   ├── test_optimization.py
│   └── test_etl.py
│
├── 📂 docs/                    # Documentación Sphinx
│
├── 📂 notebooks/               # Jupyter Notebooks — EDA y experimentos
│   ├── 01_eda_ventas.ipynb
│   ├── 02_arima_experiments.ipynb
│   ├── 03_prophet_experiments.ipynb
│   └── 04_lstm_experiments.ipynb
│
├── 📂 data/                    # Datos de ejemplo anonimizados (sin datos reales)
│   └── sample_data.csv
│
├── requirements.txt
├── requirements-dev.txt
├── .env.example                # Variables de entorno (sin credenciales reales)
├── .gitignore
├── manage.py
└── README.md
```

---

## 🚀 Instalación y Configuración Local

### Prerrequisitos
- Python 3.11+
- PostgreSQL 16
- Redis (para Celery)
- Git

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/mia-inventory-forecasting-laferia.git
cd mia-inventory-forecasting-laferia
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus credenciales locales
```

### 4. Configurar base de datos

```bash
python manage.py migrate
python manage.py createsuperuser
```

### 5. Ejecutar el servidor de desarrollo

```bash
python manage.py runserver
```

Acceder en: `http://localhost:8000`

---

## 📊 Módulos del Sistema

| Módulo | Descripción | Estado |
|--------|-------------|--------|
| ETL SAP HANA | Extracción y carga de datos históricos | 🔲 Pendiente |
| EDA | Análisis exploratorio de datos | 🔲 Pendiente |
| Categorización ABC | Clasificación dinámica de productos | 🔲 Pendiente |
| Forecasting ARIMA | Modelo estadístico de series temporales | 🔲 Pendiente |
| Forecasting Prophet | Modelo con estacionalidad y festividades | 🔲 Pendiente |
| Forecasting LSTM | Red neuronal recurrente | 🔲 Pendiente |
| Optimizador MILP | Cantidades óptimas de pedido (PuLP) | 🔲 Pendiente |
| Algoritmo Genético | Balanceo inter-sucursales (DEAP) | 🔲 Pendiente |
| Dashboard Web | Visualización de KPIs y alertas | 🔲 Pendiente |
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

> ⚠️ **Este repositorio NO contiene datos reales de Comercial La Feria.**

Los datos históricos están protegidos bajo Acuerdo de Confidencialidad (NDA). El repositorio incluye únicamente:
- Datos de ejemplo anonimizados en `/data/sample_data.csv`
- Scripts ETL sin credenciales (usar `.env`)
- Notebooks con datos sintéticos para experimentación

---

## 📄 Licencia

Este proyecto es de uso académico exclusivo, desarrollado como trabajo de titulación de la Maestría en Inteligencia Artificial Aplicada. Todos los derechos sobre los datos pertenecen a Comercial La Feria. Los algoritmos y el código desarrollado son propiedad intelectual de los autores y la universidad.

---

## 📬 Contacto

- **Autor 1:** Lam Cheang Wiliam David
- **Autor 2:** Román Largo Jessica Johanna
- **Tutor:** Criollo Caizaguano Luis Santiago
- **Institución:** Facultad de Ingeniería y Ciencias Aplicadas
