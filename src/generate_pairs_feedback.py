# src/generate_pairs_from_ratings.py
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

DATA_DIR = "data"
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")
ITEMS_CSV = os.path.join(DATA_DIR, "items.csv")
PAIRS_CSV = os.path.join(DATA_DIR, "pairs_feedback.csv")

MIN_RATING_SIMILAR = 4.0   # umbral para considerar item "alto"
MAX_PAIRS_PER_USER = 20    # límite de pares que generaremos por usuario
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

if not os.path.exists(RATINGS_CSV):
    raise FileNotFoundError("ratings.csv no encontrado. Genera ratings primero.")
if not os.path.exists(ITEMS_CSV):
    raise FileNotFoundError("items.csv no encontrado. Genera items primero.")

ratings = pd.read_csv(RATINGS_CSV)
items_df = pd.read_csv(ITEMS_CSV)

# filtrar solo ratings altos
high = ratings[ratings['rating'] >= MIN_RATING_SIMILAR]

pairs = []
# NUEVO: Agrupar por usuario Y contexto similar
for uid, grp in high.groupby('user_id'):
    # Obtener items que calificó alto
    items_rated = grp['item_id'].unique().tolist()
    if len(items_rated) < 2:
        continue
    
    # NUEVO: Priorizar pares de items del mismo tipo/contexto
    items_info = items_df[items_df['item_id'].isin(items_rated)]
    
    # Generar pares priorizando items similares
    pairs_list = []
    for type_name, type_group in items_info.groupby('type')['item_id']:
        type_items = type_group.tolist()
        if len(type_items) >= 2:
            # Pares del mismo tipo tienen más probabilidad
            for i in range(len(type_items)):
                for j in range(i+1, len(type_items)):
                    pairs_list.append((type_items[i], type_items[j], 2.0))  # peso 2x
    
    # También algunos pares cross-type
    for i in range(len(items_rated)):
        for j in range(i+1, len(items_rated)):
            if (items_rated[i], items_rated[j]) not in [(p[0], p[1]) for p in pairs_list]:
                pairs_list.append((items_rated[i], items_rated[j], 1.0))  # peso normal
    
    # Muestrear con pesos
    if pairs_list:
        weights = np.array([p[2] for p in pairs_list])
        max_pairs = min(MAX_PAIRS_PER_USER, len(pairs_list))
        # Normalizar pesos para probabilidades
        weights_normalized = weights / weights.sum()
        selected_idx = np.random.choice(len(pairs_list), size=max_pairs, replace=False, p=weights_normalized)
        selected = [pairs_list[i] for i in selected_idx]
        
        # asignar timestamps recientes aleatorios
        for a, b, _ in selected:
            days_ago = random.randint(0, 90)
            ts = (datetime.now() - timedelta(days=days_ago, hours=random.randint(0,23), minutes=random.randint(0,59))).strftime("%Y-%m-%d %H:%M:%S")
            pairs.append({'user_id': int(uid), 'item_a': int(a), 'item_b': int(b), 'timestamp': ts})

pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv(PAIRS_CSV, index=False, encoding='utf-8')
print(f"✔ pairs_feedback.csv creado con {len(pairs_df)} pares -> {PAIRS_CSV}")
