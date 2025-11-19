# src/api.py
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from typing import List
from cognitive import load_user_sim
from fusion import recommend_top3
import uvicorn

app = FastAPI(title="SMARTUR Recommender API")

# Load model artifacts at startup
RF_MODEL_FILE = "models/rf_model.joblib"
ITEMS_CSV = "data/items.csv"
USERS_CSV = "data/users.csv"
RATINGS_CSV = "data/ratings.csv"

print("Cargando artefactos...")

try:
    rf_model = joblib.load(RF_MODEL_FILE)
except Exception:
    rf_model = None
    print("Warning: RF model no cargado, coloca models/rf_model.joblib")

items = pd.read_csv(ITEMS_CSV)
users = pd.read_csv(USERS_CSV)
ratings = pd.read_csv(RATINGS_CSV)

try:
    users_list, user_sim = load_user_sim()
except Exception:
    users_list, user_sim = np.array([]), np.array([[]])
    print("Warning: user_cog_sim no encontrado. Ejecuta cognitive.py primero.")

def candidate_pool_simple(user_id, top_n=200):
    # heurística simple: seleccionar top popular items (por número de reviews) y items en mismo tipo
    if 'N' in items.columns:
        pop = items.sort_values('N', ascending=False)['item_id'].values[:top_n].tolist()
    else:
        pop = items['item_id'].values[:top_n].tolist()
    return pop

@app.get("/recommend/{user_id}", response_model=List[dict])
def recommend(user_id: int, alpha: float = 0.6):
    if rf_model is None:
        raise HTTPException(status_code=500, detail="RF model no cargado en servidor.")
    if user_id not in users['user_id'].values:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")
    candidates = candidate_pool_simple(user_id)
    top3 = recommend_top3(user_id, candidates, ratings, users, items, users_list, user_sim, rf_model, alpha=alpha)
    return top3

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
