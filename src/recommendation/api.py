import os
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importamos tus módulos
from engine import SmarturEngine
from rf_model import SmarturContextModel
from fusion import recommend_hybrid

# Configuración de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartur-api")

app = FastAPI(title="SMARTUR Recommender API v2", version="2.0")

# CORS - Asegúrate de que permita el puerto de tu React (ej. 5173 o 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class RecItem(BaseModel):
    item_id: str      
    title: str        
    score: float      
    pred_cf: float    
    pred_rf: float    

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecItem]
    alpha: float

class RecommendRequest(BaseModel):
    alpha: float = 0.7
    context: Optional[Dict[str, Any]] = None

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
        context_model.train(engine.train_data)
        
        logger.info("¡SMARTUR v2 listo para recibir peticiones!")
    except Exception as e:
        logger.error(f"Error crítico en el arranque: {e}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine_ready": engine is not None,
        "rf_ready": context_model is not None,
        "users_count": len(engine.user_item_matrix.index) if engine else 0
    }

# Endpoint GET (para pruebas rápidas sin formulario)
@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendation(
    user_id: str, 
    alpha: float = Query(0.7, ge=0.0, le=1.0)
):
    if engine is None or context_model is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados.")

    try:
        recs = recommend_hybrid(user_id, engine, context_model, alpha=alpha)
        format_recs = [RecItem(**r) for r in recs]
        return RecommendationResponse(user_id=user_id, recommendations=format_recs, alpha=alpha)
    except Exception as e:
        logger.error(f"Error en GET recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NUEVO: Endpoint POST para el formulario de React
@app.post("/recommend/{user_id}", response_model=RecommendationResponse)
def post_recommendation(user_id: str, payload: RecommendRequest):
    """
    Recibe el contexto del MultiStepForm (presupuesto, tipos de turismo, etc.)
    """
    if engine is None or context_model is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados.")

    try:
        logger.info(f"Recomendación POST para usuario: {user_id}")
        
        # Pasamos el payload.context a tu nueva lógica de fusion.py
        recs = recommend_hybrid(
            user_id=user_id, 
            engine=engine, 
            context_model=context_model, 
            alpha=payload.alpha,
            context=payload.context # <-- Aquí viajan los datos de Step1 a Step4
        )
        
        format_recs = [RecItem(**r) for r in recs]
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=format_recs,
            alpha=payload.alpha
        )
    except Exception as e:
        logger.error(f"Error en POST recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)