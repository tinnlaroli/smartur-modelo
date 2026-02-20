from cf import predict_cf_pearson

def recommend_hybrid(user_id, engine, context_model, alpha=0.7):
    """
    Sistema Híbrido:
    1. KNN de Pearson genera 50 candidatos.
    2. Random Forest califica esos 50 según contexto.
    3. Fusión ponderada.
    """
    # 1. Generar Pool de Candidatos (KNN)
    candidate_ids = engine.get_candidate_pool(user_id, top_n=50)
    
    recommendations = []
    
    # 2. Obtener predicciones del Random Forest para todo el pool
    rf_scores = context_model.predict_context(candidate_ids)
    rf_map = dict(zip(candidate_ids, rf_scores))
    
    for biz_id in candidate_ids:
        # Predicción Colaborativa (Pearson)
        score_cf = predict_cf_pearson(user_id, biz_id, engine)
        
        # Predicción Contextual (RF)
        score_rf = rf_map.get(biz_id, 3.0)
        
        # Fusión matemática
        # score = alpha * Pearson + (1-alpha) * Random Forest
        final_score = (alpha * score_cf) + ((1 - alpha) * score_rf)
        
        recommendations.append({
            'business_id': biz_id,
            'final_score': round(final_score, 2),
            'cf_part': round(score_cf, 2),
            'rf_part': round(score_rf, 2)
        })
    
    # Ordenar y devolver Top 3
    return sorted(recommendations, key=lambda x: x['final_score'], reverse=True)[:3]