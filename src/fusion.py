# src/fusion.py
import joblib
import numpy as np
import pandas as pd
from cf import predict_cf_for_user_item

RF_MODEL_FILE = "models/rf_model.joblib"

def score_candidate(user_id, item_id, ratings_df, users_df, items_df, users_list, sim_matrix, rf_model, alpha=0.6):
    pred_cf = predict_cf_for_user_item(user_id, item_id, ratings_df, users_list, sim_matrix, k=20)
    # build RF row
    item_row = items_df.loc[items_df['item_id'] == item_id]
    user_row = users_df.loc[users_df['user_id'] == user_id]
    feat = {}
    for c in ['R','K','logN','price','lat','lon']:
        feat[f'item_{c}'] = float(item_row[c].values[0]) if c in item_row.columns else 0.0
    feat['user_edad'] = float(user_row['edad'].values[0]) if 'edad' in user_row.columns else 30.0
    feat['pred_cf'] = float(pred_cf)
    X_row = pd.DataFrame([feat])
    # align columns to rf_model
    try:
        X_row = X_row[rf_model.feature_names_in_]
    except Exception:
        # if mismatch, try to add missing cols with zeros
        for c in rf_model.feature_names_in_:
            if c not in X_row.columns:
                X_row[c] = 0.0
        X_row = X_row[rf_model.feature_names_in_]
    pred_rf = rf_model.predict(X_row.values)[0]
    score = alpha * pred_cf + (1 - alpha) * pred_rf
    return float(score), float(pred_cf), float(pred_rf)

def recommend_top3(user_id, candidate_item_ids, ratings_df, users_df, items_df, users_list, sim_matrix, rf_model, alpha=0.6):
    scored = []
    for it in candidate_item_ids:
        score, cf, rf = score_candidate(user_id, it, ratings_df, users_df, items_df, users_list, sim_matrix, rf_model, alpha=alpha)
        scored.append((it, score, cf, rf))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    top3 = scored_sorted[:3]
    return [{'item_id': s[0], 'score': s[1], 'pred_cf': s[2], 'pred_rf': s[3]} for s in top3]
