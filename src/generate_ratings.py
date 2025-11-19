# src/generate_ratings.py
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# ---------- Config ----------
DATA_DIR = "data"
ITEMS_CSV = os.path.join(DATA_DIR, "items.csv")
USERS_CSV = os.path.join(DATA_DIR, "users.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")

NUM_USERS_IF_NONE = 200        # usuarios a generar si no hay users.csv
AVG_RATINGS_PER_USER = 25     # media de ratings por usuario (Poisson)
MAX_RATINGS_PER_USER = 200
MIN_RATING = 1
MAX_RATING = 5
DAYS_HISTORY = 365            # generar timestamps en los últimos 365 días
SEED = 42
# ----------------------------

random.seed(SEED)
np.random.seed(SEED)

# Carga items
if not os.path.exists(ITEMS_CSV):
    raise FileNotFoundError(f"No se encontró {ITEMS_CSV}. Genera items.csv primero.")

items = pd.read_csv(ITEMS_CSV)
if 'item_id' not in items.columns:
    raise ValueError("items.csv debe contener la columna 'item_id'")

item_ids = items['item_id'].tolist()

# Si no hay users.csv, lo generamos
if not os.path.exists(USERS_CSV):
    print("No encontré users.csv — generando users.csv sintético...")
    users = []
    for uid in range(1, NUM_USERS_IF_NONE + 1):
        edad = int(np.clip(int(np.random.normal(35, 12)), 18, 75))
        genero = random.choice(['M','F','O'])
        home_country = random.choice(['MX','US','ES','AR','CO','CL'])
        interests = random.choice(['naturaleza', 'gastronomía', 'cultura', 'aventura', 'relax'])
        users.append({'user_id': uid, 'edad': edad, 'genero': genero, 'home_country': home_country, 'interests': interests})
    users_df = pd.DataFrame(users)
    users_df.to_csv(USERS_CSV, index=False, encoding='utf-8')
    print(f"✔ users.csv creado con {len(users_df)} usuarios")
else:
    users_df = pd.read_csv(USERS_CSV)
    print(f"✔ users.csv cargado con {len(users_df)} usuarios")

# Construir distribución de popularidad para muestrear items (usa N si existe, sino R)
if 'N' in items.columns and items['N'].sum() > 0:
    pop_weights = items['N'].astype(float).values
else:
    # fallback: usar R (rating) para dar peso a más valorados
    pop_weights = items['R'].astype(float).values
# evitar ceros
pop_weights = np.clip(pop_weights, a_min=0.01, a_max=None)
pop_p = pop_weights / pop_weights.sum()

# Bias por item (centrado en 0), derivado de R (map R [1..5] -> -0.5..0.5)
item_bias_map = {}
for _, row in items.iterrows():
    r = float(row.get('R', 3.5))
    # si R ya está normalizado (0..1) detectamos rango y desnormalizamos aprox:
    if r <= 1.1:  # parece normalizado a [0,1]
        # map 0..1 -> 1..5
        r = r*4 + 1
    bias = (r - 3.0) / 4.0  # rango aproximado -0.5 .. +0.5
    item_bias_map[int(row['item_id'])] = bias

# Generar ratings
rows = []
total_interactions = 0
for _, urow in users_df.iterrows():
    uid = int(urow['user_id'])
    # nº de ratings por usuario: Poisson alrededor de AVG_RATINGS_PER_USER, con min 5
    n = np.random.poisson(AVG_RATINGS_PER_USER)
    n = int(np.clip(n, 5, MAX_RATINGS_PER_USER))
    # usuario tiene un sesgo personal (usuario más crítico o generoso)
    user_bias = np.random.normal(0.0, 0.35)  # desviación estándar razonable
    # elegir n items, con reemplazo = False
    choices = np.random.choice(item_ids, size=min(n, len(item_ids)), replace=False, p=pop_p)
    for it in choices:
        base = 3.0  # centro
        ibias = item_bias_map.get(int(it), 0.0)
        raw = base + ibias + user_bias + np.random.normal(0.0, 0.6)  # ruido
        # redondear a medio punto opcional o a .1; luego mapear a int 1..5
        r = float(np.round(raw * 2) / 2.0)  # round to 0.5 steps
        r = max(MIN_RATING, min(MAX_RATING, r))
        # timestamp aleatorio
        days_ago = random.randint(0, DAYS_HISTORY)
        ts = (datetime.now() - timedelta(days=days_ago, hours=random.randint(0,23), minutes=random.randint(0,59))).strftime("%Y-%m-%d %H:%M:%S")
        rows.append({'user_id': uid, 'item_id': int(it), 'rating': r, 'timestamp': ts})
    total_interactions += len(choices)

ratings_df = pd.DataFrame(rows)
# Asegurar que hay variedad y formato correcto
ratings_df = ratings_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

ratings_df.to_csv(RATINGS_CSV, index=False, encoding='utf-8')
print(f"✔ ratings.csv creado con {len(ratings_df)} filas (≈{total_interactions} interacciones)")
print("Rutas:")
print(" -", USERS_CSV)
print(" -", RATINGS_CSV)
