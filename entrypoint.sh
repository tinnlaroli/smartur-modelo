#!/bin/sh
set -e

DATA_DIR="/app/data"
MODELS_DIR="/app/models"
BUNDLED_MODELS="/app/_bundled/models"
REVIEWS_CSV="$DATA_DIR/data_reviews_limpio.csv"
BUSINESS_CSV="$DATA_DIR/data_negocios_limpio.csv"

mkdir -p "$DATA_DIR" "$MODELS_DIR"

if [ -d "$BUNDLED_MODELS" ] && [ -z "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
  echo "[bootstrap] Sembrando artefactos base en models/"
  cp -r "$BUNDLED_MODELS/." "$MODELS_DIR/"
fi

if [ "${AUTO_BOOTSTRAP:-1}" = "1" ]; then
  if [ ! -f "$REVIEWS_CSV" ] || [ ! -f "$BUSINESS_CSV" ]; then
    echo "[bootstrap] Datos de entrenamiento no encontrados."
    echo "[bootstrap] Paso 1/2: descargando dataset Yelp..."
    python /app/descargar_yelp.py
    echo "[bootstrap] Paso 2/2: preprocesando datos..."
    python pre_procesamiento.py
    echo "[bootstrap] Datos listos."
  else
    echo "[bootstrap] CSVs de entrenamiento ya presentes."
  fi
else
  if [ ! -f "$REVIEWS_CSV" ] || [ ! -f "$BUSINESS_CSV" ]; then
    echo "[bootstrap] ERROR: faltan CSVs en $DATA_DIR y AUTO_BOOTSTRAP=0" >&2
    exit 1
  fi
fi

echo "[bootstrap] Iniciando API (entrena RF/GBM en primer arranque si faltan modelos)..."
exec uvicorn api:app --host 0.0.0.0 --port 8000
