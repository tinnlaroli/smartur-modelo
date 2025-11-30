# src/api.py
import os
import logging
from typing import List, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.cognitive import load_user_sim
from src.fusion import recommend_top3, _load_rf_model, candidate_pool_by_popularity_and_category

# Config logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartur-api")

# Paths (ajusta si cambias estructura)
ITEMS_CSV = "data/items.csv"
USERS_CSV = "data/users.csv"
RATINGS_CSV = "data/ratings.csv"
RF_MODEL_PATH = "models/rf_model.joblib"
USERS_LIST_NPY = "models/users_list.npy"
USER_SIM_NPY = "models/user_cog_sim.npy"

# Path al PDF subido (ruta local que mencionaste)
# Nota: según el flujo, el sistema transformará este path a una URL pública cuando despliegues.
MANUAL_LOCAL_PATH = "/mnt/data/AISE-29-AISE210101-1.pdf"

app = FastAPI(title="SMARTUR Recommender API", version="0.1")

# CORS (ajusta orígenes en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Response models
class RecItem(BaseModel):
    item_id: int
    title: Optional[str]
    score: float
    pred_cf: float
    pred_rf: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecItem]
    alpha: float
    candidate_pool: int

# Load static csvs at startup (keeps memory; reload endpoint could be added)
def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

# Attempt to load artifacts on startup (if missing, endpoints will return helpful error)
try:
    items_df = load_csv_safe(ITEMS_CSV)
    users_df = load_csv_safe(USERS_CSV)
    ratings_df = load_csv_safe(RATINGS_CSV)
    rf_model, rf_features = None, None
    try:
        rf_model, rf_features = _load_rf_model(RF_MODEL_PATH)
        logger.info("RF model cargado ok.")
    except Exception as e:
        logger.warning(f"RF model no cargado: {e}")
    users_list = np.load(USERS_LIST_NPY, allow_pickle=True) if os.path.exists(USERS_LIST_NPY) else np.array([])
    user_sim = np.load(USER_SIM_NPY) if os.path.exists(USER_SIM_NPY) else np.array([[]])
    logger.info(f"Items: {len(items_df)}, Users: {len(users_df)}, Ratings: {len(ratings_df)}")
except FileNotFoundError as e:
    # delays startup but still allow server to run with errors handled per endpoint
    logger.warning(f"CSV faltante: {e}")
    items_df = pd.DataFrame()
    users_df = pd.DataFrame()
    ratings_df = pd.DataFrame()
    rf_model, rf_features = None, None
    users_list = np.array([])
    user_sim = np.array([[]])

@app.get("/health")
def health():
    return {
        "status": "ok",
        "items_loaded": bool(not items_df.empty),
        "users_loaded": bool(not users_df.empty),
        "ratings_loaded": bool(not ratings_df.empty),
        "rf_loaded": rf_model is not None,
        "cog_loaded": users_list.size > 0 and user_sim.size > 0,
    }

@app.get("/manual")
def manual():
    """
    Devuelve la ruta local del PDF que subiste.
    La infraestructura que despliegue la API puede mapear este path a una URL pública.
    """
    if os.path.exists(MANUAL_LOCAL_PATH):
        return {"local_path": MANUAL_LOCAL_PATH}
    raise HTTPException(status_code=404, detail="Manual no encontrado en el path esperado.")

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend_endpoint(
    user_id: int,
    alpha: float = Query(0.6, ge=0.0, le=1.0, description="Peso para pred_cf. score = alpha*pred_cf + (1-alpha)*pred_rf"),
    candidates: int = Query(200, ge=10, le=1000, description="Número de candidatos a generar antes de rankear"),
    k_cf: int = Query(20, ge=1, le=200, description="Vecinos CF (k)"),
):
    # Basic validations
    if users_df.empty or items_df.empty or ratings_df.empty:
        raise HTTPException(status_code=500, detail="Datasets no cargados. Ejecuta preprocessing y carga CSVs en data/")

    if user_id not in users_df['user_id'].values:
        raise HTTPException(status_code=404, detail=f"Usuario {user_id} no encontrado")

    if rf_model is None:
        raise HTTPException(status_code=500, detail="Modelo RF no disponible. Entrena y coloca models/rf_model.joblib")

    if users_list.size == 0 or user_sim.size == 0:
        raise HTTPException(status_code=500, detail="Artefactos cognitivos no disponibles. Ejecuta src.cognitive")

    # Candidate pool
    try:
        candidate_ids = candidate_pool_by_popularity_and_category(user_id, items_df, ratings_df, users_df, top_n=candidates)
    except Exception as e:
        logger.exception("Error generando candidate pool")
        candidate_ids = items_df['item_id'].tolist()[:candidates]

    # Use recommend_top3 (batch-optimized)
    try:
        recs = recommend_top3(
            user_id=user_id,
            ratings_df=ratings_df,
            users_df=users_df,
            items_df=items_df,
            users_list=users_list,
            sim_matrix=user_sim,
            rf_model=rf_model,
            rf_features=rf_features,
            alpha=alpha,
            candidate_pool_fn=lambda uid: candidate_ids,
            top_n_candidates=len(candidate_ids),
            k_cf=k_cf
        )
    except Exception as e:
        logger.exception("Error al generar recomendaciones")
        raise HTTPException(status_code=500, detail=str(e))

    # Build response
    out = []
    for r in recs:
        out.append(RecItem(
            item_id=int(r['item_id']),
            title=r.get('title', None),
            score=float(r['score']),
            pred_cf=float(r['pred_cf']),
            pred_rf=float(r['pred_rf'])
        ))

    return RecommendationResponse(user_id=user_id, recommendations=out, alpha=alpha, candidate_pool=len(candidate_ids))


from pydantic import BaseModel
from typing import Any, Dict, Optional

# ----------------------------
# Request model for POST /recommend/{user_id}
# ----------------------------
class RecommendPayload(BaseModel):
    alpha: float = 0.6
    candidates: int = 200
    k_cf: int = 20
    context: Optional[Dict[str, Any]] = None

@app.post("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend_endpoint_post(user_id: int, payload: RecommendPayload):
    """
    Acepta un body JSON con {alpha, candidates, k_cf, context}
    y devuelve la misma estructura que el GET.
    (Actualmente reusa la lógica de recommend_top3)
    """
    # validaciones similares al GET
    if users_df.empty or items_df.empty or ratings_df.empty:
        raise HTTPException(status_code=500, detail="Datasets no cargados. Ejecuta preprocessing y carga CSVs en data/")

    if user_id not in users_df['user_id'].values:
        raise HTTPException(status_code=404, detail=f"Usuario {user_id} no encontrado")

    if rf_model is None:
        raise HTTPException(status_code=500, detail="Modelo RF no disponible. Entrena y coloca models/rf_model.joblib")

    if users_list.size == 0 or user_sim.size == 0:
        raise HTTPException(status_code=500, detail="Artefactos cognitivos no disponibles. Ejecuta src.cognitive")

    # Log del contexto recibido (útil para debug)
    logger.info(f"[POST /recommend] user_id={user_id} alpha={payload.alpha} candidates={payload.candidates} k_cf={payload.k_cf} context_keys={list((payload.context or {}).keys())}")

    # Candidate pool — por ahora reusamos la misma función; si quieres, en el futuro
    # puedes usar 'payload.context' para crear pools-contextuales.
    try:
        candidate_ids = candidate_pool_by_popularity_and_category(user_id, items_df, ratings_df, users_df, top_n=payload.candidates)
    except Exception as e:
        logger.exception("Error generando candidate pool")
        candidate_ids = items_df['item_id'].tolist()[:payload.candidates]

    # Ejecutar recommend_top3 igual que GET
    try:
        recs = recommend_top3(
            user_id=user_id,
            ratings_df=ratings_df,
            users_df=users_df,
            items_df=items_df,
            users_list=users_list,
            sim_matrix=user_sim,
            rf_model=rf_model,
            rf_features=rf_features,
            alpha=payload.alpha,
            candidate_pool_fn=lambda uid: candidate_ids,
            top_n_candidates=len(candidate_ids),
            k_cf=payload.k_cf,
            # si quieres que recommend_top3 acepte context en el futuro, agrega parámetro aquí
        )
    except Exception as e:
        logger.exception("Error al generar recomendaciones (POST)")
        raise HTTPException(status_code=500, detail=str(e))

    out = []
    for r in recs:
        out.append(RecItem(
            item_id=int(r['item_id']),
            title=r.get('title', None),
            score=float(r['score']),
            pred_cf=float(r['pred_cf']),
            pred_rf=float(r['pred_rf'])
        ))

    return RecommendationResponse(user_id=user_id, recommendations=out, alpha=payload.alpha, candidate_pool=len(candidate_ids))


# Run with: uvicorn src.api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting SMARTUR API on http://0.0.0.0:8000")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, log_level="info")
