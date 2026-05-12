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

import pandas as pd

from cf import predict_cf_pearson
from context_encoder import MAPEO_CATEGORIAS
from poi_repository import fetch_all_items

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

    # Hotel: si el usuario lo necesita como base
    if context.get('needs_hotel') is True:
        mask = filtered['categories'].str.contains('Hotels & Travel|Hotels', case=False, na=False)
        filtered = filtered[mask]

    # Comida: si el usuario NO quiere comida
    if context.get('pref_food') is False:
        mask = ~filtered['categories'].str.contains('Restaurants|Food', case=False, na=False)
        filtered = filtered[mask]

    return filtered


def _diversify(recs, biz_cat_lookup, top_n, max_per_main_cat=2):
    """
    Re-rankea la lista ordenada por score para que el top-N tenga
    máximo max_per_main_cat ítems con la misma categoría principal.
    Mantiene el orden de relevancia y solo desplaza los duplicados al final.
    """
    cat_counts = {}
    selected = []
    overflow = []

    for rec in recs:
        cats_str = str(biz_cat_lookup.get(rec['item_id'], ''))
        main_cat = cats_str.split(',')[0].strip() if cats_str else 'Unknown'
        if cat_counts.get(main_cat, 0) < max_per_main_cat:
            selected.append(rec)
            cat_counts[main_cat] = cat_counts.get(main_cat, 0) + 1
        else:
            overflow.append(rec)
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        selected.extend(overflow[:top_n - len(selected)])
    return selected[:top_n]


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

def recommend_hybrid(user_id, engine, context_model, alpha=0.4, context=None, top_n=5):
    """
    Sistema Híbrido SMARTUR (Retrieval + Ranking).

    Modo producción (local POIs disponibles):
      - Candidatos: SOLO POIs de la BD local (Altas Montañas, Veracruz)
      - Ranking: 100% RF contextual (alpha=0 automático, CF no aplica)

    Modo desarrollo/fallback (sin BD local):
      - Candidatos: 200 negocios Yelp vía KNN
      - Ranking: α × CF + (1-α) × RF
    """
    # ── Fase A: Retrieval ────────────────────────────────────────────────
    try:
        local_pois = fetch_all_items()
        local_biz = context_model.prepare_local_items(local_pois)
    except Exception:
        local_biz = pd.DataFrame()

    if not local_biz.empty:
        # Producción: POIs locales como pool exclusivo
        local_biz = context_model._add_category_features(local_biz)
        biz_candidates = local_biz
        effective_alpha = 0.0  # CF no tiene señal para POIs locales
    else:
        # Fallback desarrollo: candidatos Yelp KNN
        candidate_ids = engine.get_candidate_pool(user_id, top_n=200)
        biz_candidates = engine.df_biz[engine.df_biz['business_id'].isin(candidate_ids)]
        effective_alpha = alpha

    # Filtro duro (restricciones binarias)
    refined_df = filtro_duro(biz_candidates, context)

    # Filtro suave (categorías de turismo)
    refined_df = filtrar_candidatos_por_contexto(refined_df, context)

    # Fallback: si los filtros vaciaron la lista, usar pool sin filtrar
    if refined_df.empty:
        refined_df = biz_candidates

    final_ids = refined_df['business_id'].tolist()

    # ── Fase B: Ranking ──────────────────────────────────────────────────
    ref_df = local_biz if not local_biz.empty else engine.df_biz
    rf_scores = context_model.predict_with_context(final_ids, user_context=context, df_biz_override=ref_df)
    rf_map = dict(zip(final_ids, rf_scores))

    biz_cat_lookup = ref_df.set_index('business_id')['categories'].to_dict()
    all_biz_names = ref_df.set_index('business_id')['name'].to_dict()
    biz_kind_lookup = ref_df.set_index('business_id')['kind'].to_dict() if 'kind' in ref_df.columns else {}
    local_id_set = set(local_biz['business_id']) if not local_biz.empty else set()
    matrix_col_set = set(engine.user_item_matrix_columns)

    recommendations = []

    for biz_id in final_ids:
        if biz_id in local_id_set:
            score_cf = 4.0
        elif biz_id in matrix_col_set:
            score_cf = predict_cf_pearson(user_id, biz_id, engine)
        else:
            score_cf = engine.train_data['stars'].mean()
        score_rf = rf_map.get(biz_id, 3.0)

        final_score = (effective_alpha * score_cf) + ((1 - effective_alpha) * score_rf)
        nombre = all_biz_names.get(biz_id, 'Desconocido')
        kind = biz_kind_lookup.get(biz_id, 'poi')

        recommendations.append({
            'item_id': str(biz_id),
            'title': str(nombre),
            'score': float(round(final_score, 3)),
            'pred_cf': float(round(score_cf, 3)),
            'pred_rf': float(round(score_rf, 3)),
            'kind': kind,
        })

    sorted_recs = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    return _diversify(sorted_recs, biz_cat_lookup, top_n)
