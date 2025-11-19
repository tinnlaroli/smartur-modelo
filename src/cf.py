# src/cf.py
import numpy as np
import pandas as pd

def predict_cf_for_user_item(user_id, item_id, ratings_df, users_list, sim_matrix, k=20, shrink=10):
    """
    Predice la calificación esperada (1–5) para un usuario-item
    usando Filtrado Colaborativo basado en usuarios con similitud cognitiva.

    user_id: id del usuario destino
    item_id: id del ítem a predecir
    ratings_df: DataFrame con columnas [user_id, item_id, rating]
    users_list: array de user_ids con vector cognitivo
    sim_matrix: matriz de similitud (len(users_list) x len(users_list))
    k: cantidad de vecinos más similares
    shrink: factor para suavizar similitudes (penaliza baja cantidad de ratings)
    """
    # Si el usuario no está en la matriz cognitiva → devuelve la media global
    if user_id not in users_list:
        return ratings_df["rating"].mean()

    # Índice del usuario
    idx_u = np.where(users_list == user_id)[0][0]
    sims = sim_matrix[idx_u]

    # Filtrar usuarios que calificaron ese item
    mask = ratings_df["item_id"] == item_id
    item_ratings = ratings_df[mask]
    if item_ratings.empty:
        return ratings_df["rating"].mean()

    users_rated = item_ratings["user_id"].values
    # Filtrar solo usuarios presentes en users_list
    valid_users = [u for u in users_rated if u in users_list]
    if len(valid_users) == 0:
        return ratings_df["rating"].mean()

    idxs = [np.where(users_list == u)[0][0] for u in valid_users]
    user_sims = sims[idxs]
    user_ratings = item_ratings[item_ratings["user_id"].isin(valid_users)]["rating"].values

    # Seleccionar top-k vecinos (en valor absoluto)
    if len(user_sims) > k:
        topk_idx = np.argsort(user_sims)[-k:]
        user_sims = user_sims[topk_idx]
        user_ratings = user_ratings[topk_idx]

    # Aplicar suavizado (shrinkage)
    user_sims = user_sims / (1 + shrink / (np.abs(user_sims) + 1e-9))

    # Si la suma de similitudes es casi cero → devolver media global
    denom = np.sum(np.abs(user_sims))
    if denom < 1e-6:
        return ratings_df["rating"].mean()

    # Predicción ponderada
    pred = np.dot(user_sims, user_ratings) / denom
    return float(np.clip(pred, 1, 5))


def predict_cf_for_user(user_id, ratings_df, users_list, sim_matrix, items, k=20):
    """
    Devuelve predicciones CF para todos los items no calificados por el usuario.
    """
    rated_items = ratings_df.loc[ratings_df["user_id"] == user_id, "item_id"].tolist()
    unrated_items = [i for i in items["item_id"].tolist() if i not in rated_items]

    preds = []
    for i in unrated_items:
        r = predict_cf_for_user_item(user_id, i, ratings_df, users_list, sim_matrix, k)
        preds.append((i, r))
    return sorted(preds, key=lambda x: x[1], reverse=True)
