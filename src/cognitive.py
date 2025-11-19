# src/cognitive.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

ARTIFACT = "models/scalers_and_encoders.pkl"
USERS_LIST_FILE = "models/users_list.npy"
USER_COG_SIM_FILE = "models/user_cog_sim.npy"
USER_COG_DF_FILE = "models/user_cog_df.csv"

def build_user_cognitive_pattern(pairs_feedback: pd.DataFrame, items_df: pd.DataFrame, features=None) -> pd.DataFrame:
    """
    pairs_feedback: cols ['user_id','item_a','item_b']
    items_df: must contain 'item_id' and feature columns (e.g., R,K,logN)
    returns: DataFrame with columns ['user_id', feat1, feat2, ...]
    """
    if features is None:
        features = ['R','K','logN']
    # Ensure item_id dtype comparable
    items_df = items_df.copy()
    items_df['item_id'] = items_df['item_id'].astype(int)
    # Filter features presence
    for f in features:
        if f not in items_df.columns:
            items_df[f] = 0.0

    items_map = items_df.set_index('item_id')[features].to_dict('index')
    user_acc = {}
    counts = {}
    missing_pairs = 0
    for _, row in pairs_feedback.iterrows():
        try:
            u = int(row['user_id'])
            a = int(row['item_a'])
            b = int(row['item_b'])
        except Exception:
            # skip malformed rows
            continue
        if a not in items_map or b not in items_map:
            missing_pairs += 1
            continue
        va = np.array([items_map[a][f] for f in features], dtype=float)
        vb = np.array([items_map[b][f] for f in features], dtype=float)
        v = (va + vb) / 2.0
        if u not in user_acc:
            user_acc[u] = v.copy()
            counts[u] = 1
        else:
            user_acc[u] += v
            counts[u] += 1
    rows = []
    for u, sumv in user_acc.items():
        avgv = sumv / counts[u]
        row = {'user_id': int(u)}
        for i, f in enumerate(features):
            row[f] = float(avgv[i])
        rows.append(row)
    user_cog_df = pd.DataFrame(rows).sort_values('user_id').reset_index(drop=True)
    # diagnostics
    total_pairs = len(pairs_feedback)
    used_pairs = sum(counts.values()) if counts else 0
    print(f"[cognitive] total pairs rows: {total_pairs}, used pairs: {used_pairs}, missing pairs (items not found): {missing_pairs}")
    print(f"[cognitive] users with cognitive pattern: {len(user_cog_df)}")
    return user_cog_df

def compute_user_cogsim(user_cog_df: pd.DataFrame):
    if user_cog_df is None or user_cog_df.empty:
        raise ValueError("user_cog_df vacío — asegúrate de generar pairs_feedback o que item_ids existan en items.csv")
    users = user_cog_df['user_id'].values
    X = user_cog_df.drop(columns=['user_id']).values
    sim = cosine_similarity(X)
    # persist small artifacts
    os.makedirs("models", exist_ok=True)
    np.save(USERS_LIST_FILE, users)
    np.save(USER_COG_SIM_FILE, sim)
    user_cog_df.to_csv(USER_COG_DF_FILE, index=False)
    print(f"[cognitive] saved users list ({len(users)}) -> {USERS_LIST_FILE}")
    print(f"[cognitive] saved sim matrix {sim.shape} -> {USER_COG_SIM_FILE}")
    print(f"[cognitive] saved user_cog_df -> {USER_COG_DF_FILE}")
    return users, sim

def load_user_sim():
    users = np.load(USERS_LIST_FILE, allow_pickle=True)
    sim = np.load(USER_COG_SIM_FILE)
    return users, sim

if __name__ == "__main__":
    # quick run for dev
    from src.preprocess import load_csvs, preprocess_items
    items, users, ratings, pairs = load_csvs()
    items_p, _ = preprocess_items(items)
    # ensure pairs have correct columns
    expected_cols = {'user_id','item_a','item_b'}
    if not expected_cols.issubset(set(pairs.columns)):
        raise ValueError(f"pairs_feedback.csv debe contener columnas: {expected_cols}. Columnas encontradas: {pairs.columns.tolist()}")
    user_cog = build_user_cognitive_pattern(pairs, items_p)
    users_list, sim = compute_user_cogsim(user_cog)
    print("Users:", len(users_list), "Sim matrix shape:", sim.shape)
