import numpy as np
import pandas as pd

def predict_cf_pearson(user_id, item_id, engine, k=20):
    """
    Predicción CF usando Correlación de Pearson:
    Pred = Media_u + (Suma(Sim * (Rating_v - Media_v)) / Suma(|Sim|))
    """
    if user_id not in engine.user_item_matrix.index or item_id not in engine.user_item_matrix.columns:
        return engine.train_data['stars'].mean()

    # Encontrar vecinos que calificaron el item
    user_vector = engine.matrix_centered.loc[[user_id]]
    distances, indices = engine.knn_model.kneighbors(user_vector, n_neighbors=k+1)
    
    neighbor_indices = engine.user_item_matrix.index[indices[0]]
    similarities = 1 - distances[0] # Convertir distancia a similitud
    
    weighted_sum = 0
    sim_sum = 0
    
    for i, neighbor_id in enumerate(neighbor_indices):
        if neighbor_id == user_id: continue
        
        rating = engine.user_item_matrix.loc[neighbor_id, item_id]
        if rating > 0:
            # Usamos el dato centrado (Rating - Media)
            diff = engine.matrix_centered.loc[neighbor_id, item_id]
            weighted_sum += similarities[i] * diff
            sim_sum += abs(similarities[i])
            
    if sim_sum == 0:
        return engine.user_means[user_id]
        
    prediction = engine.user_means[user_id] + (weighted_sum / sim_sum)
    return float(np.clip(prediction, 1, 5))