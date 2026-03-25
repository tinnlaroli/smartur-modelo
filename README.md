# SMARTUR - Sistema de Recomendación Híbrido (v4 True ML Contextual)

SMARTUR es un sistema de recomendación híbrido de grado industrial que combina **Filtrado Colaborativo (Pearson + KNN)** con una arquitectura **True Machine Learning Contextual (Random Forest de Interacciones Cruzadas)** para generar sugerencias personalizadas hiper-precisas usando el dataset real de Yelp.

## Novedades de la versión v4

- **Generación de Contextos Sintéticos**: El Random Forest ya no usa reglas estáticas (sistemas expertos). Ahora entrena simulando millones de perfiles de turistas virtuales, descubriendo por sí solo el impacto de buscar "restaurantes caros para turistas solos" vs "lugares con rampa para familias".
- **Filtros de Poda Duros**: Restricciones infalibles para cuando el turista pide un hotel (`needs_hotel`) o pide que no haya lugares de comida (`pref_food=false`).
- **Nuevos Metadatos de Yelp**: Extrae nativamente `GoodForKids` y determina el ambiente (`Ambience` romántico/íntimo).

## Estructura

```bash
MODELO/
├── data/                    # CSVs procesados + JSON originales de Yelp
├── models/                  # Modelos entrenados (.joblib)
├── src/                     # Motor de recomendación
│   ├── engine.py            # Pearson + KNN (matriz de utilidad)
│   ├── cf.py                # Predicción CF por vecinos
│   ├── rf_model.py          # Cerebro ML (Simulador Sintético + RF Cruzado)
│   ├── context_encoder.py   # Transformador JSON React -> Vector Numérico
│   ├── fusion.py            # Filtros Duros de Poda y Combinación Híbrida 
│   ├── evaluate.py          # Evaluación RMSE/MAE + Ranking Metrics (NDCG, Precision)
│   ├── optimize_alpha.py    # Grid search para α óptimo
│   ├── api.py               # API REST (FastAPI)
│   └── pre_procesamiento.py # NLP y Extracción de JSON Yelp → CSV
├── tests/                   # Tests
├── descargar_yelp.py        # Descarga automatizada del dataset
└── requirements.txt         # Dependencias
```

## Ejecución de API

```bash
cd src
# Si cambiaste los datos extraídos o es la primera vez:
python pre_procesamiento.py

# Levantar el servidor 
# (Entrenará automáticamente el Random Forest v4 si no detecta el pre-compilado en /models)
python api.py
```

Swagger UI disponible en: `http://localhost:8000/docs`

---

## Consumo desde el Frontend

### Recomendaciones (POST) — Payload de Contexto Completo

| | |
|---|---|
| **Método** | `POST` |
| **URL** | `{base}/recommend/{user_id}` |
| **Headers** | `Content-Type: application/json` |

**Formato del body (JSON):**

```json
{
  "alpha": 0.2,
  "top_n": 5,
  "context": {
    "presupuesto_bucket": "medio", 
    "edad_range": "35-44", 
    "tiposTurismo": ["cultural", "gastronomico"], 
    
    "group_type": "familia", 
    "wants_tours": true, 
    "needs_hotel": false, 
    "pref_food": true, 
    
    "requiere_accesibilidad": true,
    "pref_outdoor": false
  }
}
```

| Campo | Descripción |
|-------|-------------|
| `context.presupuesto_bucket` | `"bajo"`, `"medio"`, `"alto"`, `"premium"`. |
| `context.edad_range` | `"18-24"`, `"25-34"`, `"35-44"`, `"45-54"`, `"55+"`. |
| `context.tiposTurismo` | Lista que puede incluir: `"naturaleza"`, `"aventura"`, `"gastronomico"`, `"cultural"`, `"rural"`. |
| `context.group_type` | `"solo"`, `"pareja"`, `"familia"`, `"amigos"`. Modifica dinámicamente el puntaje (ej. familias chocan positivamente con negocios `is_good_for_kids`). |
| `context.wants_tours` | `true/false`. Si es true, el modelo de IA bonificará negocios con categoría "Tours". |
| `context.needs_hotel` | `true/false`. **Filtro Duro**: Si es true, la recomendación elimina cualquier negocio que *no* sea hotel. |
| `context.pref_food` | `true/false`. **Filtro Duro**: Si es false, elimina tajantemente lugares de comida. (Por defecto es true). |
| `context.requiere_accesibilidad` | `true/false`. **Filtro Duro**: Si es true, solo retorna lugares donde `WheelchairAccessible` sea positivo. |
| `context.pref_outdoor` | `true/false`. **Filtro Duro**: Prioriza fuertemente el `OutdoorSeating` |

**Respuesta 200:**

```json
{
  "user_id": "usuario_test",
  "recommendations": [
    {
      "item_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
      "title": "Restaurante Local",
      "score": 4.098,
      "pred_cf": 3.809,
      "pred_rf": 4.772
    }
  ],
  "alpha": 0.2
}
```
