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

---

## Consumo desde el frontend

Base URL (ejemplo): `http://localhost:8000`. La API usa CORS abierto (`allow_origins=["*"]`), así que el front puede llamar desde cualquier origen.

### 1. Health check

| | |
|---|---|
| **Método** | `GET` |
| **URL** | `{base}/health` |
| **Headers** | Ninguno requerido |
| **Body** | No aplica |

**Respuesta 200:**  
`{ "status": "ok", "engine_ready": true, "rf_ready": true, "users_count": 66115 }`

---

### 2. Recomendaciones (GET) — sin contexto del formulario

| | |
|---|---|
| **Método** | `GET` |
| **URL** | `{base}/recommend/{user_id}` |
| **Query params** | `alpha` (number, 0–1, opcional, default 0.1), `top_n` (number, 1–50, opcional, default 5) |
| **Headers** | Ninguno requerido |
| **Body** | No aplica |

**Ejemplo URL:**  
`GET http://localhost:8000/recommend/mh_-eMZ6K5RLWhZyISBhwA?alpha=0.1&top_n=5`

**Respuesta 200:** mismo esquema que el POST (ver abajo).

---

### 3. Recomendaciones (POST) — con contexto del formulario (recomendado para el flujo completo)

| | |
|---|---|
| **Método** | `POST` |
| **URL** | `{base}/recommend/{user_id}` |
| **Headers** | `Content-Type: application/json` |
| **Body** | JSON con el siguiente formato. Todos los campos del body son opcionales. |

**Formato del body (JSON):**

```json
{
  "alpha": 0.1,
  "top_n": 5,
  "context": {
    "tiposTurismo": ["naturaleza", "gastronomico", "cultural", "aventura", "rural"],
    "pref_outdoor": true
  }
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `alpha` | number | Peso del filtrado colaborativo (0–1). Por defecto 0.1. |
| `top_n` | number | Cantidad de recomendaciones a devolver (1–50). Por defecto 5. |
| `context` | object \| null | Contexto del formulario del usuario. Si no se envía, no se aplica filtro por categorías. |
| `context.tiposTurismo` | string[] | Array de tipos de turismo elegidos. Valores admitidos: `"naturaleza"`, `"aventura"`, `"gastronomico"`, `"cultural"`, `"rural"`. Se mapean a categorías Yelp internamente. |
| `context.pref_outdoor` | boolean | Si es `true`, se priorizan negocios con categorías outdoor (Parks, Hiking, Nature, etc.). |

**Ejemplo mínimo (sin contexto):**  
`POST http://localhost:8000/recommend/mh_-eMZ6K5RLWhZyISBhwA`  
Body: `{}` o `{ "alpha": 0.1, "top_n": 5 }`

**Ejemplo con contexto:**  
`POST http://localhost:8000/recommend/mh_-eMZ6K5RLWhZyISBhwA`  
Body:  
`{ "alpha": 0.1, "top_n": 5, "context": { "tiposTurismo": ["gastronomico"], "pref_outdoor": false } }`

**Respuesta 200 (GET y POST):**

```json
{
  "user_id": "mh_-eMZ6K5RLWhZyISBhwA",
  "recommendations": [
    {
      "item_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
      "title": "Nombre del negocio",
      "score": 4.098,
      "pred_cf": 3.809,
      "pred_rf": 4.772
    }
  ],
  "alpha": 0.1
}
```

| Campo de cada ítem | Tipo | Descripción |
|--------------------|------|-------------|
| `item_id` | string | ID del negocio (business_id). |
| `title` | string | Nombre del negocio. |
| `score` | number | Puntuación final (combinación CF + RF). |
| `pred_cf` | number | Puntuación del modelo colaborativo. |
| `pred_rf` | number | Puntuación del modelo contextual (Random Forest). |

**Errores:** 503 si los modelos no están cargados; 500 si falla la recomendación (detalle en el body).

---

## Puntos de mejora identificados

- **Caché de predicciones:** Con 19 features en el RF y un pool de hasta 100 candidatos por request, la latencia puede crecer. Opciones: Redis para resultados por `(user_id, context_hash)` o `@lru_cache` en memoria para usuarios/contextos frecuentes, con TTL para no desactualizar mucho.
- **Métricas de ranking:** RMSE/MAE miden error de rating, no si el Top 3 “le atina” al usuario. Añadir en `evaluate.py` métricas tipo Precision@K, Recall@K o “hit rate en Top 3” (proporción de veces que un ítem realmente valorado alto por el usuario aparece en el Top 3 recomendado) daría una señal más alineada con el uso turístico.
- **Limpieza de datos:** Revisar en los CSVs de Yelp: duplicados de reviews (mismo user_id + business_id), negocios sin `categories` o con `categories` vacío (el RF rellena con 0), y outliers en `latitude`/`longitude` o `review_count` para evitar que distorsionen el RF.
- **Dataset:** El split 80/20 es fijo (random_state=42). Valorar validación cruzada o un split temporal (por fecha) si las reseñas tienen fecha, para medir mejor generalización. Los CSVs `items.csv`, `users.csv`, `ratings.csv`, `pairs_feedback.csv` no se usan en el motor actual; se pueden eliminar o documentar como legacy.
- **Cold start:** Usuarios nuevos solo reciben populares; ítems nuevos dependen del RF. Una política explícita (ej. mostrar “trending” o “nuevos en tu zona”) mejoraría la experiencia cuando no hay historial.
- **CORS y seguridad:** En producción conviene restringir `allow_origins` a los dominios del front y, si aplica, añadir rate limiting y autenticación.
