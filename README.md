# SMARTUR - Sistema de Recomendación Híbrido (Motor Yelp)

SMARTUR es un sistema de recomendación híbrido que combina **Filtrado Colaborativo (Pearson + KNN)** con **Random Forest contextual** para generar sugerencias personalizadas usando el dataset real de Yelp.

## Algoritmos

- **Correlación de Pearson** centrada para medir similitud usuario-usuario.
- **K-Nearest Neighbors (KNN)** para encontrar perfiles similares.
- **Random Forest Regressor** con features de ubicación, categorías y popularidad para ranking contextual.
- **Fusión Híbrida** con peso α configurable: `score = α·CF + (1−α)·RF`.

## Requisitos

- Python 3.8+
- ~4GB RAM disponible (la matriz User-Item es grande)

## Estructura

```
MODELO/
├── data/                    # CSVs procesados + JSON originales de Yelp
├── models/                  # Modelos entrenados (.joblib)
├── src/                     # Motor de recomendación
│   ├── engine.py            # Pearson + KNN (matriz de utilidad)
│   ├── cf.py                # Predicción CF por vecinos
│   ├── rf_model.py          # Random Forest contextual (categorías + ubicación)
│   ├── fusion.py            # Combinación híbrida + filtrado por contexto
│   ├── evaluate.py          # Evaluación RMSE/MAE por componente
│   ├── optimize_alpha.py    # Grid search para α óptimo
│   ├── main.py              # Punto de entrada CLI
│   ├── api.py               # API REST (FastAPI)
│   └── pre_procesamiento.py # Filtrado de datos Yelp JSON → CSV
├── tests/                   # Tests
├── descargar_yelp.py        # Descarga automatizada del dataset
└── requirements.txt         # Dependencias
```

## Instalación

```bash
git clone https://github.com/tinnlaroli/SMARTUR.git
cd SMARTUR
python -m venv modelo
.\modelo\Scripts\activate        # Windows
# source modelo/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

## Preparación de Datos

```bash
# 1. Descargar dataset de Yelp (~4GB)
python descargar_yelp.py

# 2. Mover los JSON a data/ y filtrar
cd src
python pre_procesamiento.py
```

## Ejecución

```bash
cd src

# Modo consola (prueba rápida)
python main.py

# Evaluación de métricas (RMSE/MAE por componente)
python evaluate.py

# Optimización de α
python optimize_alpha.py

# Modo API (servidor)
uvicorn api:app --host 0.0.0.0 --port 8000
```

Swagger UI disponible en: `http://localhost:8000/docs`

## Métricas de Rendimiento

Evaluadas sobre 1,000 muestras del 20% de test set (66K usuarios, 5K negocios):

| Componente | RMSE | MAE |
|---|---|---|
| Baseline (media global) | 1.3690 | 1.1208 |
| CF (Pearson + KNN) | 1.4165 | 1.1361 |
| RF (Random Forest, 19 features) | 1.2746 | 1.0322 |
| **Hibrido (alpha=0.1)** | **1.2755** | **1.0353** |

Alpha optimizado via grid search (0.0 a 1.0). Ejecuta `python evaluate.py` o `python optimize_alpha.py` para regenerar.
