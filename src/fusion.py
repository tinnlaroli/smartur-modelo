"""
SMARTUR Fusion v3: Pipeline de dos etapas (Retrieval → Ranking).

Fase A (Retrieval):
  1. Pool amplio de ~200 candidatos vía KNN/Pearson
  2. Filtro duro: elimina ítems que violen restricciones binarias
  3. Filtro suave: prioriza categorías según tiposTurismo

Fase B (Ranking):
  1. RF contextual re-rankea con vector [Item + User + Interaction]
  2. Blend final: α × CF_score + (1-α) × RF_contextual_score
"""

from cf import predict_cf_pearson
from context_encoder import MAPEO_CATEGORIAS

# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------

def filtro_duro(biz_df, context):
    """
    Elimina candidatos que violen restricciones binarias del usuario.
    Son condiciones de 'no-negociable' (deal-breakers).
    """
    if not context:
        return biz_df

    filtered = biz_df.copy()

    # Accesibilidad: si el usuario la requiere, eliminar negocios sin ella
    if context.get('requiere_accesibilidad') is True:
        if 'is_accessible' in filtered.columns:
            filtered = filtered[filtered['is_accessible'] == 1]

    # Outdoor: si el usuario lo prefiere como requisito duro
    if context.get('pref_outdoor') is True:
        if 'outdoor' in filtered.columns:
            filtered = filtered[filtered['outdoor'] == 1]

    return filtered


def filtrar_candidatos_por_contexto(biz_df, context):
    """Filtro suave: prioriza negocios cuyas categorías coincidan con tiposTurismo."""
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
            # Si ningún candidato coincide, mantener todos (degradación graciosa)
            if mask.any():
                filtered_df = filtered_df[mask]

    return filtered_df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def recommend_hybrid(user_id, engine, context_model, alpha=0.1, context=None, top_n=5):
    """
    Sistema Híbrido SMARTUR v3 (Retrieval + Ranking).

    1. Retrieval: pool de 200 candidatos vía KNN.
    2. Filtro duro: restricciones binarias (accesibilidad, outdoor).
    3. Filtro suave: categorías de turismo.
    4. Ranking: RF contextual con vector [User + Item + Interaction].
    5. Blend: α × CF + (1-α) × RF_contextual.
    """
    # ── Fase A: Retrieval ────────────────────────────────────────────────
    candidate_ids = engine.get_candidate_pool(user_id, top_n=200)
    biz_candidates = engine.df_biz[engine.df_biz['business_id'].isin(candidate_ids)]

    # Filtro duro (restricciones binarias)
    refined_df = filtro_duro(biz_candidates, context)

    # Filtro suave (categorías de turismo)
    refined_df = filtrar_candidatos_por_contexto(refined_df, context)

    # Fallback: si todos los filtros vaciaron la lista, usar pool crudo
    if refined_df.empty:
        final_ids = candidate_ids
    else:
        final_ids = refined_df['business_id'].tolist()

    # ── Fase B: Ranking ──────────────────────────────────────────────────
    # RF contextual: usa vector completo si hay contexto del turista
    rf_scores = context_model.predict_with_context(final_ids, user_context=context)
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
