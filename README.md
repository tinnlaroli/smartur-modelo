# SMARTUR Recommender System

Sistema de recomendación híbrido que combina modelos cognitivos basados en contenido, filtrado colaborativo y ranking de preferencias mediante aprendizaje por pares.

## Descripción

SMARTUR es un pipeline de recomendación que integra múltiples técnicas de machine learning para generar recomendaciones personalizadas. El sistema combina:

- Modelo cognitivo basado en contenido (TF-IDF y similitud coseno)
- Filtrado colaborativo (ALS y LightFM)
- Ranking de preferencias (Random Forest para comparaciones por pares)
- Fusión híbrida de modelos con pesos configurables

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## Instalación

### 1. Crear entorno virtual

```bash
python -m venv venv
```

### 2. Activar entorno virtual

**Linux/Mac:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Estructura de Datos

Coloca tus archivos CSV en el directorio `data/` con la siguiente estructura:

### items.csv

Columnas requeridas: `item_id`, `nombre`, `tipo`, `categoria`, `subcategoria`, `precio_promedio`, `ubicacion`, `valoracion_promedio`, `caracteristicas`

### users.csv

Columnas requeridas: `user_id`, `edad`, `genero`, `ubicacion`, `preferencias`, `historial_busquedas`

### ratings.csv

Columnas requeridas: `user_id`, `item_id`, `rating`, `fecha`, `contexto`

### pairs_feedback.csv

Columnas requeridas: `user_id`, `item_a`, `item_b`, `seleccionado`, `fecha`

## Flujo de Ejecución

Sigue estos pasos en orden para configurar y ejecutar el sistema:

### 1. Preprocesamiento de datos

Prepara y normaliza los datos de entrada:

```bash
python -m src.preprocess
```

Este paso genera las características preprocesadas para usuarios e items, y crea la matriz de interacciones.

### 2. Construcción del modelo cognitivo

Construye la matriz de similitud basada en contenido:

```bash
python -m src.cognitive
```

### 3. Entrenamiento del modelo de ranking

Entrena el modelo Random Forest para comparaciones por pares:

```bash
python -m src.rf_model
```

Esto generará `models/rf_model.joblib` si hay datos suficientes en `pairs_feedback.csv`.

### 4. Iniciar la API

Ejecuta el servidor de la API REST:

```bash
python -m src.api
```

O usando uvicorn (si está configurado):

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

La API estará disponible en `http://localhost:5000` (Flask) o `http://localhost:8000` (uvicorn).

## Uso de la API

### Obtener recomendaciones para un usuario

```bash
GET http://localhost:5000/recommend/{user_id}?top_n=10
```

**Ejemplo:**

```bash
curl http://localhost:5000/recommend/1?top_n=10
```

### Obtener items similares

```bash
GET http://localhost:5000/recommend/similar/{item_id}?top_n=10
```

### Enviar feedback

```bash
POST http://localhost:5000/feedback
Content-Type: application/json

{
  "user_id": 1,
  "item_id": 5,
  "rating": 4,
  "feedback": "positive"
}
```

### Health check

```bash
GET http://localhost:5000/health
```

## Estructura del Proyecto

```text
SMARTUR/
├── data/                  # Archivos CSV de entrada
│   ├── items.csv
│   ├── users.csv
│   ├── ratings.csv
│   └── pairs_feedback.csv
├── models/                # Modelos entrenados guardados
│   ├── rf_model.joblib
│   └── scalers_and_encoders.pkl
├── notebooks/             # Notebooks de desarrollo
│   └── dev_pipeline.ipynb
├── src/                   # Código fuente
│   ├── api.py            # API REST Flask
│   ├── cf.py             # Filtrado colaborativo
│   ├── cognitive.py      # Modelo cognitivo
│   ├── fusion.py         # Fusión híbrida
│   ├── preprocess.py     # Preprocesamiento
│   ├── rf_model.py       # Modelo de ranking
│   └── evaluate.py       # Evaluación de modelos
├── tests/                # Tests unitarios
│   └── test_module.py
├── requirements.txt       # Dependencias Python
└── README.md             # Este archivo
```

## Testing

Ejecuta los tests con pytest:

```bash
pytest -q
```

Para ejecutar tests con más detalle:

```bash
pytest -v
```

## Configuración y Ajustes

### Parámetros del modelo

Los siguientes parámetros pueden ajustarse en los módulos correspondientes:

- **Filtrado colaborativo**: número de factores, iteraciones, regularización (en `src/cf.py`)
- **Modelo cognitivo**: parámetros de TF-IDF, número de características (en `src/cognitive.py`)
- **Fusión híbrida**: pesos de combinación de modelos (en `src/fusion.py`)
- **Random Forest**: número de estimadores, profundidad máxima (en `src/rf_model.py`)

### Pesos de fusión por defecto

- Modelo cognitivo: 0.3
- Filtrado colaborativo: 0.4
- Ranking: 0.3

Estos valores pueden ajustarse en `src/fusion.py` según el rendimiento observado.

## Notas de Producción

Para un entorno de producción, considera implementar:

- Generación de candidatos más sofisticada (FAISS para búsqueda vectorial)
- Filtros de proximidad geográfica
- Caché de recomendaciones para mejorar rendimiento
- Sistema de logging estructurado
- Monitoreo de métricas de recomendación
- Re-entrenamiento periódico de modelos
- Manejo robusto de usuarios nuevos (cold start)

## Troubleshooting

### Error al cargar modelos

Asegúrate de haber ejecutado los pasos de preprocesamiento y entrenamiento antes de iniciar la API.

### Error de imports

Si encuentras errores de imports, verifica que estés ejecutando desde el directorio raíz del proyecto y que el entorno virtual esté activado.

### Datos insuficientes

Algunos modelos requieren una cantidad mínima de datos. Si no hay suficientes ratings o pares de feedback, algunos componentes pueden no entrenarse correctamente.

## Licencia

[Especificar licencia si aplica]

## Contacto

[Información de contacto del proyecto]
