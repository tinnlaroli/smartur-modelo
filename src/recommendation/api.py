import os
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importamos tus nuevos módulos
from engine import SmarturEngine
from rf_model import SmarturContextModel
from fusion import recommend_hybrid

# Configuración de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartur-api")

app = FastAPI(title="SMARTUR Recommender API v2", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Modelos de Datos (Pydantic) ---
class RecItem(BaseModel):
    business_id: str
    final_score: float
    cf_part: float
    rf_part: float

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecItem]
    alpha: float

# --- Estado Global de los Modelos ---
# Inicializamos como None para cargarlos al arrancar
engine = None
context_model = None

@app.on_event("startup")
def startup_event():
    """Carga los modelos pesados una sola vez al iniciar la API"""
    global engine, context_model
    try:
        logger.info("Cargando Motor de Pearson (Engine)...")
        engine = SmarturEngine()
        engine.prepare_pearson_matrix()
        
        logger.info("Cargando Modelo de Contexto (Random Forest)...")
        context_model = SmarturContextModel()
        # Entrenamos con la data cargada en el engine
        context_model.train(engine.train_data)
        
        logger.info("¡SMARTUR v2 listo para recibir peticiones!")
    except Exception as e:
        logger.error(f"Error crítico en el arranque: {e}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine_ready": engine is not None and engine.user_item_matrix is not None,
        "rf_ready": context_model is not None and context_model.model is not None,
        "users_count": len(engine.user_item_matrix.index) if engine else 0
    }

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendation(
    user_id: str, 
    alpha: float = Query(0.7, ge=0.0, le=1.0)
):
    if engine is None or context_model is None:
        raise HTTPException(status_code=503, detail="Los modelos se están cargando o fallaron.")

    try:
        # Usamos la lógica de fusión que ya probaste en main.py
        recs = recommend_hybrid(user_id, engine, context_model, alpha=alpha)
        
        # Formatear respuesta según el modelo Pydantic
        format_recs = [RecItem(**r) for r in recs]
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=format_recs,
            alpha=alpha
        )
    except Exception as e:
        logger.error(f"Error al recomendar para {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno al generar recomendaciones.")

# Ejecución (opcional, mejor usar uvicorn en terminal)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)