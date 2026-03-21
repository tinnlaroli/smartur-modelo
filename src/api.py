import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

"""
SMARTUR API Base File
Define los Endpoints de recomendación mediante FastAPI.
Conecta los flujos entre el Engine de Pearson y el Modelo Contextual de RF.
"""

from engine import SmarturEngine
from rf_model import SmarturContextModel
from fusion import recommend_hybrid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartur-api")

engine = None
context_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos pesados una sola vez al iniciar la API."""
    global engine, context_model
    try:
        logger.info("Cargando Motor de Pearson (Engine)...")
        engine = SmarturEngine()
        engine.prepare_pearson_matrix()

        logger.info("Cargando Modelo de Contexto (Random Forest)...")
        context_model = SmarturContextModel()
        if not context_model.load():
            logger.info("Modelo de Random Forest no encontrado, entrenando ahora por única vez...")
            context_model.train(engine.train_data)

        logger.info("SMARTUR v2 listo para recibir peticiones.")
    except Exception as e:
        logger.error(f"Error crítico en el arranque: {e}")
    yield


app = FastAPI(title="SMARTUR Recommender API v2", version="2.1", lifespan=lifespan)

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
    alpha: float = 0.1
    context: Optional[Dict[str, Any]] = None
    top_n: int = 5


@app.get("/health")
def health():
    """
    Endpoint pasivo para sondear la disponibilidad y estado de carga de la API.
    Aporta métricas rápidas del estado interno del Engine y el RandomForest en RAM.
    """
    return {
        "status": "ok",
        "engine_ready": engine is not None,
        "rf_ready": context_model is not None,
        "users_count": engine.user_item_matrix.shape[0] if engine and engine.user_item_matrix is not None else 0,
    }


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendation(
    user_id: str,
    alpha: float = Query(0.1, ge=0.0, le=1.0),
    top_n: int = Query(5, ge=1, le=50),
):
    if engine is None or context_model is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados.")

    try:
        recs = recommend_hybrid(
            user_id, engine, context_model, alpha=alpha, top_n=top_n
        )
        return RecommendationResponse(
            user_id=user_id,
            recommendations=[RecItem(**r) for r in recs],
            alpha=alpha,
        )
    except Exception as e:
        logger.error(f"Error en GET recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/{user_id}", response_model=RecommendationResponse)
def post_recommendation(user_id: str, payload: RecommendRequest):
    if engine is None or context_model is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados.")

    try:
        logger.info(f"Recomendación POST para usuario: {user_id}")
        recs = recommend_hybrid(
            user_id=user_id,
            engine=engine,
            context_model=context_model,
            alpha=payload.alpha,
            context=payload.context,
            top_n=payload.top_n,
        )
        return RecommendationResponse(
            user_id=user_id,
            recommendations=[RecItem(**r) for r in recs],
            alpha=payload.alpha,
        )
    except Exception as e:
        logger.error(f"Error en POST recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train-rf")
def train_rf():
    """
    Llamado bajo demanda de re-entrenamiento del Random Forest contextual.
    Esto vuelve a generar el árbol utilizando datos actuales de la Base e ignora el archivo local previo, 
    creando un fichero `.joblib` en disco en los procesos intermedios que quedará para los futuros arranques.
    """
    if engine is None or context_model is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados.")
    try:
        logger.info("Forzando entrenamiento del modelo Random Forest...")
        context_model.train(engine.train_data)
        return {"status": "ok", "message": "Modelo entrenado y guardado correctamente."}
    except Exception as e:
        logger.error(f"Error entrenando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
