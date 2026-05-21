# AGENTS.md — SMARTUR v4

## Project summary
Hybrid recommendation system (Collaborative Filtering via Pearson + KNN, and contextual Random Forest) using the Yelp dataset. Exposes a FastAPI on port 8000.

## Key commands

```bash
# Install deps
pip install -r requirements.txt

# Download Yelp dataset (via kagglehub)
python descargar_yelp.py

# Pre-process raw Yelp JSON → cleaned CSVs (run when data changes or first time)
cd src && python pre_procesamiento.py

# Start API (auto-trains RF if no model in /models)
cd src && python api.py
# Swagger: http://localhost:8000/docs

# Run CLI recommendation demo
cd src && python main.py

# Run evaluation (RMSE, MAE, NDCG, Precision, Hit Rate)
cd src && python evaluate.py

# Optimize alpha via grid search
cd src && python optimize_alpha.py

# Run tests
pytest          # configured: pythonpath=.,src  testpaths=tests
```

## Architecture

```
src/
  api.py             — FastAPI entrypoint (uvicorn). Imports from sibling modules directly.
  engine.py          — CF engine: loads CSVs, builds sparse user-item matrix, KNN model
  cf.py              — Pearson prediction from KNN neighbors
  rf_model.py        — Random Forest contextual model with synthetic user simulation
  context_encoder.py — Transforms React form JSON → numeric features (budget, age, tourism types, group type, match features)
  fusion.py          — Two-stage pipeline: retrieval (KNN pool) → hard/soft filters → RF ranking → α-blended final score
  pre_procesamiento.py — NLP + extraction: Yelp JSON → data_negocios_limpio.csv, data_reviews_limpio.csv
  evaluate.py        — RMSE/MAE + ranking metrics (NDCG@K, Precision@K, Hit Rate@K)
  optimize_alpha.py  — Grid search for optimal α weight
```

## Data flow

1. `descargar_yelp.py` → raw JSON in `data/`
2. `pre_procesamiento.py` → cleaned CSVs in `data/` (gitignored)
3. `api.py` startup → loads CSVs via `engine.py`, trains/loads RF from `models/rf_context_yelp.joblib`
4. `POST /recommend/{user_id}` → `fusion.py` hybrid pipeline → JSON response

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/recommend/{user_id}` | Hybrid CF+RF recommendation with context |
| GET | `/metrics` | Returns `models/algorithm_metrics.json` — latest RMSE/MAE/NDCG per algorithm. Used by PLATAFORMA ML dashboard. Returns 404 if no metrics file exists yet. |
| GET | `/health` | Liveness check |

## Important gotchas

- **Working directory**: All src scripts must be run from `src/` directory (relative imports like `from engine import SmarturEngine`). The Dockerfile sets `WORKDIR /app/src` for this reason.
- **pytest.ini**: `pythonpath = . src` so tests can import from `src/` without `cd src`.
- **Large Yelp JSON files**: `data/*.json` and `data/*_limpio.csv` are gitignored. Models (`models/*.joblib`) are also gitignored. First run or fresh clone requires data download + preprocessing + RF training.
- **RF training is slow**: The Random Forest trains on ~80k interactions with 35+ features. Takes several minutes on first start. API auto-trains if `models/rf_context_yelp.joblib` is missing.
- **Default alpha = 0.2**: README recommends α=0.2 (80% RF weight, 20% CF). Code default in POST payload is 0.1.
- **CORS**: API has `allow_origins=["*"]` — permissive for local dev.
- **Context fields**: The POST `/recommend/{user_id}` endpoint accepts `context` dict with fields: `presupuesto_bucket`, `edad_range`, `tiposTurismo`, `group_type`, `wants_tours`, `needs_hotel`, `pref_food`, `requiere_accesibilidad`, `pref_outdoor`.
- **Hard filters** (`fusion.py:filtro_duro`): `needs_hotel` eliminates non-hotels; `pref_food=false` eliminates food places; `requiere_accesibilidad` eliminates non-accessible venues; `pref_outdoor` prioritizes outdoor seating.

## Docker

```bash
docker compose up --build   # image: smartur-api:local, port 8000
# First start may take several minutes if RF model is missing (healthcheck start_period: 300s)
```

## Tests

Single test file `tests/test_module.py` with basic instantiation tests. Tests gracefully handle missing data files (catch `FileNotFoundError`). For full tests, ensure cleaned CSVs exist in `data/`.
