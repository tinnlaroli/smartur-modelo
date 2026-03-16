import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    return {
        "status": "ok",
        "engine_ready": engine is not None,
        "rf_ready": context_model is not None,
        "users_count": len(engine.user_item_matrix.index) if engine else 0,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
