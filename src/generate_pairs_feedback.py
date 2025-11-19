# src/generate_pairs_from_ratings.py
import pandas as pd
import os
import random
from datetime import datetime, timedelta

DATA_DIR = "data"
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")
PAIRS_CSV = os.path.join(DATA_DIR, "pairs_feedback.csv")

MIN_RATING_SIMILAR = 4.0   # umbral para considerar item "alto"
MAX_PAIRS_PER_USER = 20    # límite de pares que generaremos por usuario
SEED = 42

random.seed(SEED)

if not os.path.exists(RATINGS_CSV):
    raise FileNotFoundError("ratings.csv no encontrado. Genera ratings primero.")

ratings = pd.read_csv(RATINGS_CSV)
# filtrar solo ratings altos
high = ratings[ratings['rating'] >= MIN_RATING_SIMILAR]

pairs = []
for uid, grp in high.groupby('user_id'):
    items = grp['item_id'].unique().tolist()
    if len(items) < 2:
        continue
    # número de pares a generar: combinaciones limitadas
    max_pairs = min(MAX_PAIRS_PER_USER, int(len(items)*(len(items)-1)/2))
    # generar pares aleatorios sin repetición
    all_pairs = []
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            all_pairs.append((items[i], items[j]))
    random.shuffle(all_pairs)
    selected = all_pairs[:max_pairs]
    # asignar timestamps recientes aleatorios
    for a,b in selected:
        days_ago = random.randint(0, 90)
        ts = (datetime.now() - timedelta(days=days_ago, hours=random.randint(0,23), minutes=random.randint(0,59))).strftime("%Y-%m-%d %H:%M:%S")
        pairs.append({'user_id': int(uid), 'item_a': int(a), 'item_b': int(b), 'timestamp': ts})

pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv(PAIRS_CSV, index=False, encoding='utf-8')
print(f"✔ pairs_feedback.csv creado con {len(pairs_df)} pares -> {PAIRS_CSV}")
