import numpy as np


def predict_cf_pearson(user_id, item_id, engine, k=20):
    """
    Predicción CF usando Correlación de Pearson:
    Pred = Media_u + Σ(Sim · (Rating_v − Media_v)) / Σ|Sim|
    """
    if user_id not in engine.user_item_matrix.index or \
       item_id not in engine.user_item_matrix.columns:
        return engine.train_data['stars'].mean()

    user_vector = engine.matrix_centered.loc[[user_id]]
    n_neighbors = min(k + 1, len(engine.user_item_matrix))
    distances, indices = engine.knn_model.kneighbors(
        user_vector, n_neighbors=n_neighbors
    )

    neighbor_indices = engine.user_item_matrix.index[indices[0]]
    similarities = 1 - distances[0]

    weighted_sum = 0.0
    sim_sum = 0.0

    for i, neighbor_id in enumerate(neighbor_indices):
        if neighbor_id == user_id:
            continue

        rating = engine.user_item_matrix.loc[neighbor_id, item_id]
        if rating > 0:
            diff = engine.matrix_centered.loc[neighbor_id, item_id]
            weighted_sum += similarities[i] * diff
            sim_sum += abs(similarities[i])

    user_mean = float(engine.user_means[user_id])
    if np.isnan(user_mean):
        user_mean = float(engine.train_data['stars'].mean())

    if sim_sum == 0:
        return user_mean

    prediction = user_mean + (weighted_sum / sim_sum)
    result = float(np.clip(prediction, 1, 5))
    return result if not np.isnan(result) else user_mean
