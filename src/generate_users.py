# src/generate_users.py
import pandas as pd
import numpy as np
import random
import os

DATA_DIR = "data"
USERS_CSV = os.path.join(DATA_DIR, "users.csv")
ITEMS_CSV = os.path.join(DATA_DIR, "items.csv")

# Parámetros
NUM_USERS = 500           # cambia según necesites
SEED = 42
COUNTRIES = ['MX','US','ES','AR','CO','CL','GB','FR']
AGE_MEAN = 35
AGE_SD = 12
INTEREST_POOL = [
    'naturaleza','gastronomía','cultura','aventura','relax',
    'historia','fotografía','familia','deportes','vida_nocturna'
]

random.seed(SEED)
np.random.seed(SEED)

# Si existe items.csv, extrae tags para hacer intereses más realistas
tags_by_type = {}
if os.path.exists(ITEMS_CSV):
    items = pd.read_csv(ITEMS_CSV)
    if 'tags' in items.columns:
        # recolecta tags únicos y su frecuencia
        all_tags = []
        for t in items['tags'].dropna().astype(str):
            parts = [s.strip() for s in t.split(',') if s.strip()]
            all_tags.extend(parts)
        if all_tags:
            tag_counts = pd.Series(all_tags).value_counts()
            # usar top tags como interés candidates
            INTEREST_POOL = tag_counts.index.tolist()[:20]

# Generar usuarios
users = []
for uid in range(1, NUM_USERS + 1):
    edad = int(np.clip(int(np.random.normal(AGE_MEAN, AGE_SD)), 18, 75))
    genero = random.choices(['M','F','O'], weights=[0.48,0.48,0.04])[0]
    home_country = random.choice(COUNTRIES)
    # asignar 1-3 intereses (evitar duplicados)
    n_interests = random.choices([1,2,3],[0.5,0.35,0.15])[0]
    interests = random.sample(INTEREST_POOL, k=min(n_interests, len(INTEREST_POOL)))
    # para coherencia, guarda intereses como string separada por ';'
    interest_str = ';'.join(interests)
    users.append({
        'user_id': uid,
        'edad': edad,
        'genero': genero,
        'home_country': home_country,
        'interests': interest_str
    })

df = pd.DataFrame(users)
df.to_csv(USERS_CSV, index=False, encoding='utf-8')
print(f"✔ users.csv generado con {len(df)} usuarios -> {USERS_CSV}")
