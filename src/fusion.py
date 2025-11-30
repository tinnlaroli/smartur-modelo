# src/fusion.py
import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from src.cf import predict_cf_for_user_item
from src.cognitive import load_user_sim

MODEL_PATH = "models/rf_model.joblib"
os.makedirs("models", exist_ok=True)

def _load_rf_model(path: str = MODEL_PATH):
    """
    Load RF model saved by rf_model.py. It may be a dict {'model':..., 'features':...}
    or a raw sklearn estimator.
    Returns (model, features_list)
    """
    bundle = joblib.load(path)
    if isinstance(bundle, dict):
        model = bundle.get("model", None)
        features = bundle.get("features", None)
        # backward compatibility: some versions saved model directly
        if model is None and "estimator" in bundle:
            model = bundle["estimator"]
    else:
        model = bundle
        features = getattr(model, "feature_names_in_", None)
        if features is not None:
            features = list(features)
    if model is None:
        raise ValueError(f"No se encontró un modelo válido en {path}")
    return model, features

# src/fusion.py - Reemplazar función completa

def candidate_pool_by_popularity_and_category(
    user_id: int, 
    items_df: pd.DataFrame, 
    ratings_df: pd.DataFrame,
    users_df: pd.DataFrame, 
    context: dict = None,
    top_n: int = 200
) -> List[int]:
    """
    Generación de candidatos con filtrado contextual.
    context puede contener:
    - tiposTurismo: list[str]
    - presupuesto_daily: float
    - actividad_level: int (1-5)
    - pref_outdoor: bool
    - accesibilidad: str ('si'/'no')
    - group_type: str
    """
    items = items_df.copy()
    
    print(f"[candidate_pool] Iniciando con {len(items)} items")
    
    # ============================================
    # 1. FILTRAR POR TIPOS DE TURISMO
    # ============================================
    if context and 'tiposTurismo' in context and context['tiposTurismo']:
        tipos_usuario = context['tiposTurismo']
        
        # Mapeo: tiposTurismo (formulario) -> type (items.csv)
        type_mapping = {
            'naturaleza': ['atracción', 'tour'],
            'gastronomico': ['restaurante'],
            'cultural': ['atracción'],
            'aventura': ['tour'],
            'rural': ['hotel', 'tour']
        }
        
        allowed_types = set()
        for t in tipos_usuario:
            if t in type_mapping:
                allowed_types.update(type_mapping[t])
        
        if allowed_types and 'type' in items.columns:
            items = items[items['type'].isin(allowed_types)]
            print(f"[candidate_pool] Después de filtrar por tipos {allowed_types}: {len(items)} items")
    
    # ============================================
    # 2. FILTRAR POR PRESUPUESTO
    # ============================================
    if context and 'presupuesto_daily' in context and 'price' in items.columns:
        budget = context['presupuesto_daily']
        
        # Porcentaje adaptativo según tipos en el pool actual
        if 'type' in items.columns and len(items) > 0:
            types_present = items['type'].unique()
            # Lógica: hoteles son más caros, restaurantes/atracciones más accesibles
            if 'hotel' in types_present:
                pct = 1.5  # Hoteles pueden ser hasta 150% del budget diario
            elif 'tour' in types_present:
                pct = 0.8  # Tours 80%
            elif 'restaurante' in types_present:
                pct = 0.75  # Restaurantes 75%
            elif 'atracción' in types_present:
                pct = 0.5  # Atracciones 50%
            else:
                pct = 0.7  # Default 70%
        else:
            pct = 0.7
        
        max_price = budget * pct
        
        print(f"[DEBUG] Budget: {budget}, Max price: {max_price:.0f} ({pct*100:.0f}% para tipos: {types_present if 'type' in items.columns else 'N/A'})")
        print(f"[DEBUG] Price range ANTES: {items['price'].min():.2f} - {items['price'].max():.2f}")
        print(f"[DEBUG] Items ANTES del filtro: {len(items)}")
        
        items = items[items['price'] <= max_price]
        
        print(f"[DEBUG] Items DESPUÉS del filtro: {len(items)}")
        print(f"[candidate_pool] Después de filtrar por presupuesto (<= {max_price:.0f} MXN): {len(items)} items")
        
    # ============================================
    # 3. FILTRAR POR NIVEL DE ACTIVIDAD
    # ============================================
    if context and 'actividad_level' in context and 'actividad_level' in items.columns:
        user_activity = context['actividad_level']
        # Tolerancia: ±1 nivel
        items = items[
            (items['actividad_level'] >= user_activity - 1) & 
            (items['actividad_level'] <= user_activity + 1)
        ]
        print(f"[candidate_pool] Después de filtrar por actividad (±1 de {user_activity}): {len(items)} items")
    
    # ============================================
    # 4. FILTRAR POR PREFERENCIA OUTDOOR
    # ============================================
    if context and context.get('pref_outdoor') and 'outdoor' in items.columns:
        items = items[items['outdoor'] == 1]
        print(f"[candidate_pool] Después de filtrar outdoor: {len(items)} items")
    
    # ============================================
    # 5. FILTRAR POR ACCESIBILIDAD
    # ============================================
    if context and context.get('accesibilidad') == 'si' and 'accesible' in items.columns:
        items = items[items['accesible'] == 1]
        print(f"[candidate_pool] Después de filtrar accesibilidad: {len(items)} items")
    
    # ============================================
    # 6. BOOST POR GROUP_TYPE (usando tags)
    # ============================================
    if context and 'group_type' in context and 'tags' in items.columns:
        group = context['group_type']
        items['priority'] = 0
        
        if group == 'familia':
            items['priority'] = items['tags'].str.contains('familiar', case=False, na=False).astype(int) * 2
        elif group == 'pareja':
            items['priority'] = items['tags'].str.contains('romántico', case=False, na=False).astype(int) * 2
        elif group == 'amigos':
            items['priority'] = items['tags'].str.contains('aventura|excursión', case=False, na=False).astype(int) * 2
        
        items = items.sort_values(['priority', 'N'], ascending=[False, False])
        print(f"[candidate_pool] Boost aplicado por group_type '{group}'")
    else:
        # Sin boost, ordenar solo por popularidad
        if 'N' in items.columns:
            items = items.sort_values('N', ascending=False)
    
    # ============================================
    # 7. LIMITAR A top_n
    # ============================================
    candidates = items['item_id'].tolist()[:top_n]
    
    # Si después de filtros quedan muy pocos, rellenar con populares
    if len(candidates) < 3:
        print(f"[candidate_pool] WARNING: Solo {len(candidates)} candidatos. Rellenando con populares...")
        all_items = items_df.sort_values('N', ascending=False)['item_id'].tolist()
        for item_id in all_items:
            if item_id not in candidates:
                candidates.append(item_id)
            if len(candidates) >= top_n:
                break
    
    print(f"[candidate_pool] Candidatos finales: {len(candidates)}")
    return candidates

def score_candidate(user_id: int, item_id: int,
                    ratings_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame,
                    users_list: np.ndarray, sim_matrix: np.ndarray,
                    rf_model, rf_features: List[str],
                    alpha: float = 0.6, k_cf: int = 20) -> Tuple[float, float, float]:
    """
    Compute (score, pred_cf, pred_rf) for a single (user,item).
    score = alpha * pred_cf + (1-alpha) * pred_rf
    """
    # CF prediction (fallback to global mean inside function if needed)
    pred_cf = predict_cf_for_user_item(user_id, item_id, ratings_df, users_list, sim_matrix, k=k_cf)
    # RF prediction: build feature vector aligned with rf_features
    feat = {}
    # item features
    for c in ['R','K','logN','price','lat','lon']:
        feat[f'item_{c}'] = float(items_df.loc[items_df['item_id']==item_id, c].values[0]) if c in items_df.columns and not items_df.loc[items_df['item_id']==item_id, c].empty else 0.0
    # user simple features
    feat['user_edad'] = float(users_df.loc[users_df['user_id']==user_id, 'edad'].values[0]) if 'edad' in users_df.columns and not users_df.loc[users_df['user_id']==user_id, 'edad'].empty else 30.0
    feat['pred_cf'] = float(pred_cf)

    # include one-hot columns if present in features
    X_row = pd.DataFrame([feat])
    if rf_features is not None:
        # add missing features as zero
        for f in rf_features:
            if f not in X_row.columns:
                X_row[f] = 0.0
        X_row = X_row[rf_features]
    else:
        # If we don't have features list, rely on model.feature_names_in_ if available
        try:
            X_row = X_row[rf_model.feature_names_in_]
        except Exception:
            pass

    pred_rf = float(np.clip(rf_model.predict(X_row.values)[0], 1.0, 5.0))
    score = alpha * pred_cf + (1.0 - alpha) * pred_rf
    return score, float(pred_cf), float(pred_rf)

# src/fusion.py - Modificar recommend_top3

def recommend_top3(user_id: int,
                ratings_df: pd.DataFrame,
                users_df: pd.DataFrame,
                items_df: pd.DataFrame,
                users_list: np.ndarray,
                sim_matrix: np.ndarray,
                rf_model=None,
                rf_features: List[str]=None,
                alpha: float = 0.6,
                candidate_pool_fn = None,
                top_n_candidates: int = 200,
                k_cf:int = 20,
                context: dict = None) -> List[Dict[str, Any]]:  # <-- NUEVO PARÁMETRO
    """
    Main function to recommend Top-3 items for a user.
    - context: dict con campos del formulario para filtrado contextual
    """
    if rf_model is None:
        rf_model, rf_features = _load_rf_model()

    # candidate generation
    if candidate_pool_fn is None:
        candidate_ids = candidate_pool_by_popularity_and_category(
            user_id, 
            items_df, 
            ratings_df, 
            users_df, 
            context=context,  # <-- PASAR CONTEXT
            top_n=top_n_candidates
        )
    else:
        candidate_ids = candidate_pool_fn(user_id)

    scored = []
    for it in candidate_ids:
        try:
            sc, pcf, prf = score_candidate(
                user_id, it, ratings_df, users_df, items_df, 
                users_list, sim_matrix, rf_model, rf_features, 
                alpha=alpha, k_cf=k_cf
            )
            title = items_df.loc[items_df['item_id']==it, 'title'].values[0] if 'title' in items_df.columns and not items_df.loc[items_df['item_id']==it, 'title'].empty else ""
            scored.append({
                'item_id': int(it), 
                'title': str(title), 
                'score': float(sc), 
                'pred_cf': float(pcf), 
                'pred_rf': float(prf)
            })
        except Exception:
            continue

    scored_sorted = sorted(scored, key=lambda x: x['score'], reverse=True)
    return scored_sorted[:3]


# Quick debug helper
if __name__ == "__main__":
    # quick smoke test if run directly
    import numpy as np, pandas as pd, joblib
    from src.cognitive import load_user_sim
    print("DEBUG: running a quick smoke test of fusion.recommend_top3")
    items = pd.read_csv("data/items.csv")
    users = pd.read_csv("data/users.csv")
    ratings = pd.read_csv("data/ratings.csv")
    users_list, sim = load_user_sim()
    rf_model, rf_features = _load_rf_model()
    u = int(users['user_id'].iloc[0])
    recs = recommend_top3(u, ratings, users, items, users_list, sim, rf_model=rf_model, rf_features=rf_features, alpha=0.6)
    print("Top-3 (fusion):", recs)
