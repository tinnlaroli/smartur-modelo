# src/fusion.py
import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from src.cf import predict_cf_for_user_item
from src.cognitive import load_user_sim

MODEL_PATH = "models/rf_model.joblib"
os.makedirs("models", exist_ok=True)

def _load_rf_model(path: str = MODEL_PATH):
    """
    Load RF model saved by rf_model.py. It may be a dict {'model':..., 'features':...}
    or a raw sklearn estimator.
    Returns (model, features_list)
    """
    bundle = joblib.load(path)
    if isinstance(bundle, dict):
        model = bundle.get("model", None)
        features = bundle.get("features", None)
        # backward compatibility: some versions saved model directly
        if model is None and "estimator" in bundle:
            model = bundle["estimator"]
    else:
        model = bundle
        features = getattr(model, "feature_names_in_", None)
        if features is not None:
            features = list(features)
    if model is None:
        raise ValueError(f"No se encontró un modelo válido en {path}")
    return model, features

def candidate_pool_by_popularity_and_category(user_id: int, items_df: pd.DataFrame, ratings_df: pd.DataFrame,
                                            users_df: pd.DataFrame, top_n: int = 200) -> List[int]:
    """
    Simple candidate generation:
    - top by popularity (N)
    - plus items from user's top categories (if 'type' exists)
    Returns a list of candidate item_ids (unique) up to top_n
    """
    items = items_df.copy()
    # popularity
    if 'N' in items.columns:
        pop = items.sort_values('N', ascending=False)['item_id'].tolist()
    else:
        pop = items['item_id'].tolist()

    candidates = []
    # add top popular
    candidates.extend(pop[:top_n])

    # try to extend by user's favored types if available
    try:
        # infer user's top rated types
        rated = ratings_df[ratings_df['user_id'] == user_id].merge(items_df, on='item_id', how='left')
        if 'type' in rated.columns:
            top_types = rated.groupby('type')['rating'].mean().sort_values(ascending=False).index.tolist()[:3]
            for t in top_types:
                t_items = items_df[items_df['type'] == t]['item_id'].tolist()
                for it in t_items:
                    if it not in candidates:
                        candidates.append(int(it))
                    if len(candidates) >= top_n:
                        break
                if len(candidates) >= top_n:
                    break
    except Exception:
        pass

    # ensure uniqueness and limit
    unique_cands = []
    for it in candidates:
        if it not in unique_cands:
            unique_cands.append(int(it))
        if len(unique_cands) >= top_n:
            break
    return unique_cands

def score_candidate(user_id: int, item_id: int,
                    ratings_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame,
                    users_list: np.ndarray, sim_matrix: np.ndarray,
                    rf_model, rf_features: List[str],
                    alpha: float = 0.6, k_cf: int = 20) -> Tuple[float, float, float]:
    """
    Compute (score, pred_cf, pred_rf) for a single (user,item).
    score = alpha * pred_cf + (1-alpha) * pred_rf
    """
    # CF prediction (fallback to global mean inside function if needed)
    pred_cf = predict_cf_for_user_item(user_id, item_id, ratings_df, users_list, sim_matrix, k=k_cf)
    # RF prediction: build feature vector aligned with rf_features
    feat = {}
    # item features
    for c in ['R','K','logN','price','lat','lon']:
        feat[f'item_{c}'] = float(items_df.loc[items_df['item_id']==item_id, c].values[0]) if c in items_df.columns and not items_df.loc[items_df['item_id']==item_id, c].empty else 0.0
    # user simple features
    feat['user_edad'] = float(users_df.loc[users_df['user_id']==user_id, 'edad'].values[0]) if 'edad' in users_df.columns and not users_df.loc[users_df['user_id']==user_id, 'edad'].empty else 30.0
    feat['pred_cf'] = float(pred_cf)

    # include one-hot columns if present in features
    X_row = pd.DataFrame([feat])
    if rf_features is not None:
        # add missing features as zero
        for f in rf_features:
            if f not in X_row.columns:
                X_row[f] = 0.0
        X_row = X_row[rf_features]
    else:
        # If we don't have features list, rely on model.feature_names_in_ if available
        try:
            X_row = X_row[rf_model.feature_names_in_]
        except Exception:
            pass

    pred_rf = float(np.clip(rf_model.predict(X_row.values)[0], 1.0, 5.0))
    score = alpha * pred_cf + (1.0 - alpha) * pred_rf
    return score, float(pred_cf), float(pred_rf)

def recommend_top3(user_id: int,
                ratings_df: pd.DataFrame,
                users_df: pd.DataFrame,
                items_df: pd.DataFrame,
                users_list: np.ndarray,
                sim_matrix: np.ndarray,
                rf_model=None,
                rf_features: List[str]=None,
                alpha: float = 0.6,
                candidate_pool_fn = None,
                top_n_candidates: int = 200,
                k_cf:int = 20) -> List[Dict[str, Any]]:
    """
    Main function to recommend Top-3 items for a user.
    - candidate_pool_fn: function(user_id) -> List[item_id]; if None use popularity+category
    - returns list of dicts: [{'item_id':..., 'title':..., 'score':..., 'pred_cf':..., 'pred_rf':...}, ...]
    """
    # load RF if not provided
    if rf_model is None:
        rf_model, rf_features = _load_rf_model()

    # candidate generation
    if candidate_pool_fn is None:
        candidate_ids = candidate_pool_by_popularity_and_category(user_id, items_df, ratings_df, users_df, top_n=top_n_candidates)
    else:
        candidate_ids = candidate_pool_fn(user_id)

    scored = []
    for it in candidate_ids:
        try:
            sc, pcf, prf = score_candidate(user_id, it, ratings_df, users_df, items_df, users_list, sim_matrix, rf_model, rf_features, alpha=alpha, k_cf=k_cf)
            title = items_df.loc[items_df['item_id']==it, 'title'].values[0] if 'title' in items_df.columns and not items_df.loc[items_df['item_id']==it, 'title'].empty else ""
            scored.append({'item_id': int(it), 'title': str(title), 'score': float(sc), 'pred_cf': float(pcf), 'pred_rf': float(prf)})
        except Exception:
            # skip problematic items
            continue

    scored_sorted = sorted(scored, key=lambda x: x['score'], reverse=True)
    return scored_sorted[:3]

# Quick debug helper
if __name__ == "__main__":
    # quick smoke test if run directly
    import numpy as np, pandas as pd, joblib
    from src.cognitive import load_user_sim
    print("DEBUG: running a quick smoke test of fusion.recommend_top3")
    items = pd.read_csv("data/items.csv")
    users = pd.read_csv("data/users.csv")
    ratings = pd.read_csv("data/ratings.csv")
    users_list, sim = load_user_sim()
    rf_model, rf_features = _load_rf_model()
    u = int(users['user_id'].iloc[0])
    recs = recommend_top3(u, ratings, users, items, users_list, sim, rf_model=rf_model, rf_features=rf_features, alpha=0.6)
    print("Top-3 (fusion):", recs)
