import numpy as np


def predict_cf_pearson(user_id, item_id, engine, k=20):
    """
    Predicción utilizando Filtrado Colaborativo Basado en Memoria (K-NN user-based approach).
    Predice la puntuación (1 a 5 estrellas) matemática extrapolada que un `user_id` le otorgaría
    a un `item_id` determinado, en base as su similitud vectorial (con coseno geométrico) respecto
    a una vecindad con los K-usuarios más similares a su perfil.

    Fórmula Matemática general:
        Pred = Media_u + Σ(Sim · (Rating_v − Media_v)) / Σ|Sim|

    Args:
        user_id (str): Identificador textual en la db del usuario objetivo.
        item_id (str): Identificador de Yelp o negocio a calificar.
        engine (SmarturEngine): Referencia general en memoria al Motor construido con scikit Sparse matrix.
        k (int): Cantidad máxima de vecinos a buscar.
        
    Returns:
        float: Una predicción de puntaje inferido delimitado (clip) entre 1.0 y 5.0. 
               Devuelve por defecto la media del individuo si no hay datos de vecinos disponibles.
    """
    user_idx = engine.get_user_idx(user_id)
    item_idx = engine.get_biz_idx(item_id)

    if user_idx is None or item_idx is None:
        # Si el usuario o el item solicitado sufren de cold-start total
        return engine.train_data['stars'].mean()

    user_vector = engine.matrix_centered[user_idx]
    n_neighbors = min(k + 1, engine.matrix_centered.shape[0])
    distances, indices = engine.knn_model.kneighbors(
        user_vector, n_neighbors=n_neighbors
    )

    neighbor_indices = indices[0]
    similarities = 1 - distances[0]

    weighted_sum = 0.0
    sim_sum = 0.0

    for i, neighbor_id in enumerate(neighbor_indices):
        if neighbor_id == user_idx:
            continue

        rating = engine.user_item_matrix[neighbor_id, item_idx]
        if rating > 0:
            diff = engine.matrix_centered[neighbor_id, item_idx]
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
