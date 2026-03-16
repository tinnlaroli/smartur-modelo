from cf import predict_cf_pearson

MAPEO_CATEGORIAS = {
    'naturaleza': ['Parks', 'Botanical Gardens', 'Hiking', 'Landmarks & Historical Buildings', 'Lakes'],
    'aventura': ['Active Life', 'Hiking', 'Rafting', 'Mountain Biking', 'Tours'],
    'gastronomico': ['Restaurants', 'Food', 'Cafes', 'Traditional Mexican', 'Bakeries'],
    'cultural': ['Museums', 'Art Galleries', 'Arts & Entertainment', 'Historical Tours', 'Festivals'],
    'rural': ['Hotels', 'Bed & Breakfast', 'Campgrounds', 'Farm Stays', 'Guest Houses']
}


def filtrar_candidatos_por_contexto(biz_df, context):
    """Filtra el DataFrame de negocios basado en el contexto del formulario React."""
    if not context:
        return biz_df

    filtered_df = biz_df.copy()

    if 'tiposTurismo' in context and context['tiposTurismo']:
        yelp_categories = []
        for tipo in context['tiposTurismo']:
            yelp_categories.extend(MAPEO_CATEGORIAS.get(tipo, []))
        if yelp_categories:
            mask = filtered_df['categories'].str.contains(
                '|'.join(yelp_categories), case=False, na=False
            )
            filtered_df = filtered_df[mask]

    if context.get('pref_outdoor') is True:
        outdoor_keywords = ['Parks', 'Outdoor', 'Hiking', 'Nature', 'Lakes']
        mask = filtered_df['categories'].str.contains(
            '|'.join(outdoor_keywords), case=False, na=False
        )
        filtered_df = filtered_df[mask]

    return filtered_df


def recommend_hybrid(user_id, engine, context_model, alpha=0.1, context=None, top_n=5):
    """
    Sistema Híbrido SMARTUR v2:
    1. Pool inicial vía KNN (Pearson).
    2. Filtra candidatos por contexto del formulario React.
    3. Re-rankea con RF + CF ponderados por alpha.
    """
    candidate_ids = engine.get_candidate_pool(user_id, top_n=100)
    biz_candidates = engine.df_biz[engine.df_biz['business_id'].isin(candidate_ids)]

    refined_df = filtrar_candidatos_por_contexto(biz_candidates, context)

    if refined_df.empty:
        final_ids = candidate_ids
    else:
        final_ids = refined_df['business_id'].tolist()

    rf_scores = context_model.predict_context(final_ids)
    rf_map = dict(zip(final_ids, rf_scores))

    recommendations = []
    for biz_id in final_ids:
        score_cf = predict_cf_pearson(user_id, biz_id, engine)
        score_rf = rf_map.get(biz_id, 3.0)

        final_score = (alpha * score_cf) + ((1 - alpha) * score_rf)

        name_match = engine.df_biz.loc[
            engine.df_biz['business_id'] == biz_id, 'name'
        ]
        nombre = name_match.values[0] if len(name_match) > 0 else 'Desconocido'

        recommendations.append({
            'item_id': str(biz_id),
            'title': str(nombre),
            'score': float(round(final_score, 3)),
            'pred_cf': float(round(score_cf, 3)),
            'pred_rf': float(round(score_rf, 3)),
        })

    return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_n]
