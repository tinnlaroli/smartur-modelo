# SMARTUR - Sistema de Recomendación (Motor Yelp)

SMARTUR es un pipeline de recomendación híbrido que integra múltiples técnicas de machine learning para generar sugerencias personalizadas utilizando el dataset de Yelp. El sistema combina modelos cognitivos, filtrado colaborativo y ranking de preferencias para ofrecer resultados precisos.

## Algoritmos y Características

El núcleo del motor de recomendación se basa en los siguientes algoritmos:

- **Coeficiente de Correlación de Pearson** para medir la similitud usuario-ítem.
- **K-Nearest Neighbors (KNN)** para encontrar perfiles similares.
- **Random Forest Regressor** para el ranking de preferencias basado en contexto.

## Requisitos Previos

- Python 3.8 o superior
- Git

## Estructura del Proyecto

```text
SMARTUR/
├── data/                  # Almacena los archivos JSON originales de Yelp y los CSV procesados.
├── models/                # Modelos entrenados exportados en formato .joblib.
├── src/recommendation/    # Lógica central del motor, fusión de modelos y definiciones de la API.
├── descargar_yelp.py      # Script para la descarga automatizada del dataset de Yelp.
└── requirements.txt       # Dependencias del proyecto.
```

## Guía de Instalación Paso a Paso

### 1. Clonar el Repositorio y Preparar el Entorno

Primero, clona el repositorio en tu máquina local y accede al directorio del proyecto:

```bash
git clone https://github.com/tinnlaroli/SMARTUR.git
cd SMARTUR
```

Crea un entorno virtual (en este ejemplo lo llamaremos `modelo`):

```bash
python -m venv modelo
```

Activa el entorno virtual:

**Windows:**

```bash
.\modelo\Scripts\activate
```

**Linux/macOS:**

```bash
source modelo/bin/activate
```

Instala las dependencias requeridas para el proyecto:

```bash
pip install -r requirements.txt
```

### 2. Descargar el Dataset de Yelp

Para alimentar el modelo con datos reales, es necesario descargar el dataset oficial de Yelp. Ejecuta el siguiente script desde la raíz del proyecto:

```bash
python descargar_yelp.py
```

> **Nota:** El dataset tiene un peso aproximado de 4GB comprimido. Una vez finalizada la descarga, asegúrate de extraer y mover los archivos `yelp_academic_dataset_business.json` y `yelp_academic_dataset_review.json` a la carpeta `data/` del proyecto.

### 3. Pre-procesamiento y Limpieza de Datos

Para evitar saturar la memoria RAM y optimizar los datos crudos (~5GB), es necesario filtrarlos a un formato estandarizado y más ligero. Dirígete a la carpeta del motor y ejecuta el script de limpieza:

```bash
cd src/recommendation
python pre_procesamiento.py
```

Este proceso generará los archivos `data_negocios_limpio.csv` y `data_reviews_limpio.csv` dentro de la carpeta `data/`.

## Ejecución del Modelo

### Modo Consola (Prueba Rápida)

Para verificar que el motor basado en Pearson y el Random Forest están funcionando correctamente a través de la consola, puedes ejecutar el script principal:

```bash
# Asegúrate de estar en el directorio src/recommendation
python main.py
```

### Modo API (Servidor de Producción)

Para levantar el servicio web (útil para ser consumido por un frontend o aplicación cliente), inicia el servidor FastAPI utilizando Uvicorn:

```bash
# Asegúrate de estar en el directorio src/recommendation
uvicorn api:app --host 0.0.0.0 --port 8000
```

Una vez que el servidor esté en ejecución, podrás acceder a la documentación interactiva de la API (Swagger UI) en el navegador: `http://localhost:8000/docs`

## Métricas de Rendimiento

El sistema ha sido evaluado con el dataset de Yelp, alcanzando las siguientes métricas de validación:

- **RMSE (Error Cuadrático Medio):** ~1.30
- **MAE (Error Absoluto Medio):** ~1.06
