from cf import predict_cf_pearson

# Mapeo de categorías: Frontend (Las Altas Montañas) -> Backend (Yelp Dataset)
MAPEO_CATEGORIAS = {
    'naturaleza': ['Parks', 'Botanical Gardens', 'Hiking', 'Landmarks & Historical Buildings', 'Lakes'],
    'aventura': ['Active Life', 'Hiking', 'Rafting', 'Mountain Biking', 'Tours'],
    'gastronomico': ['Restaurants', 'Food', 'Cafes', 'Traditional Mexican', 'Bakeries'],
    'cultural': ['Museums', 'Art Galleries', 'Arts & Entertainment', 'Historical Tours', 'Festivals'],
    'rural': ['Hotels', 'Bed & Breakfast', 'Campgrounds', 'Farm Stays', 'Guest Houses']
}

def filtrar_candidatos_por_contexto(biz_df, context):
    """Filtra el DataFrame de negocios basado en el contexto del formulario React"""
    if not context:
        return biz_df

    filtered_df = biz_df.copy()

    # 1. Filtro de Categorías (Step 2 del Form)
    if 'tiposTurismo' in context and context['tiposTurismo']:
        yelp_categories = []
        for tipo in context['tiposTurismo']:
            yelp_categories.extend(MAPEO_CATEGORIAS.get(tipo, []))
        
        if yelp_categories:
            # Buscamos coincidencias en la columna 'categories' de Yelp
            mask = filtered_df['categories'].str.contains('|'.join(yelp_categories), case=False, na=False)
            filtered_df = filtered_df[mask]

    # 2. Filtro Outdoor (Step 2: preferencia_lugar === 'aire')
    if context.get('pref_outdoor') is True:
        outdoor_keywords = ['Parks', 'Outdoor', 'Hiking', 'Nature', 'Lakes']
        mask = filtered_df['categories'].str.contains('|'.join(outdoor_keywords), case=False, na=False)
        filtered_df = filtered_df[mask]

    return filtered_df

def recommend_hybrid(user_id, engine, context_model, alpha=0.7, context=None):
    """
    Sistema Híbrido actualizado para SMARTUR v2:
    1. Genera pool inicial vía KNN (Pearson).
    2. Filtra candidatos usando el contexto del formulario de React.
    3. Re-rankea con Random Forest.
    """
    # 1. Pool inicial de candidatos (Pearson/KNN)
    candidate_ids = engine.get_candidate_pool(user_id, top_n=100)
    biz_candidates = engine.df_biz[engine.df_biz['business_id'].isin(candidate_ids)]
    
    # 2. Aplicar filtros del formulario
    refined_df = filtrar_candidatos_por_contexto(biz_candidates, context)
    
    # Si el filtro vacía la lista, volvemos a los candidatos originales para no dar 0 resultados
    if refined_df.empty:
        final_ids = candidate_ids
    else:
        final_ids = refined_df['business_id'].tolist()
    
    # 3. Predicciones del Random Forest para el pool refinado
    rf_scores = context_model.predict_context(final_ids)
    rf_map = dict(zip(final_ids, rf_scores))
    
    
    recommendations = []
    for biz_id in final_ids:
        score_cf = predict_cf_pearson(user_id, biz_id, engine)
        score_rf = rf_map.get(biz_id, 3.0)
        
        final_score = (alpha * score_cf) + ((1 - alpha) * score_rf)
        
        # Obtenemos el nombre del negocio
        nombre_negocio = engine.df_biz[engine.df_biz['business_id'] == biz_id]['name'].values[0]
        
        # Sincronizamos las llaves con el Modal de React y el modelo RecItem
        recommendations.append({
            'item_id': str(biz_id), 
            'title': str(nombre_negocio), 
            'score': float(round(final_score, 3)),  
            'pred_cf': float(round(score_cf, 3)),   
            'pred_rf': float(round(score_rf, 3))    
        })
    
    # Ordenamos por la nueva llave 'score'
    return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:3]