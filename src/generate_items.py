# generate_items.py - VERSIÓN MEJORADA

import pandas as pd
import numpy as np
import random

# Mapeo de tipos a características contextuales
TYPES_CONFIG = {
    "hotel": {
        "tags_pool": ["familiar", "romántico", "ecológico", "descanso", "panorama"],
        "price_range": (800, 3500),
        "actividad_level": [1, 2],  # bajo-medio
    },
    "restaurante": {
        "tags_pool": ["gastronomía", "tradicional", "familiar", "romántico", "urbano"],
        "price_range": (150, 800),
        "actividad_level": [2, 3],
    },
    "atracción": {
        "tags_pool": ["cultura", "historia", "naturaleza", "vista", "familiar"],
        "price_range": (50, 400),
        "actividad_level": [2, 3, 4],
    },
    "tour": {
        "tags_pool": ["aventura", "naturaleza", "extremo", "excursión", "cultural"],
        "price_range": (300, 2000),
        "actividad_level": [3, 4, 5],  # medio-alto
    },
    "tienda": {
        "tags_pool": ["artesanía", "local", "souvenir", "gastronomía"],
        "price_range": (50, 500),
        "actividad_level": [1, 2],
    }
}

items = []
for i in range(1, 201):
    t = random.choice(list(TYPES_CONFIG.keys()))
    config = TYPES_CONFIG[t]
    
    # Rating y reviews
    R = round(np.random.uniform(3.5, 5.0), 1)
    K = i
    N = random.randint(20, 500)
    
    # Precio según tipo
    price = round(np.random.uniform(*config["price_range"]), 2)
    
    # Coordenadas (Córdoba, Veracruz)
    lat = round(18.85 + np.random.uniform(-0.05, 0.05), 4)
    lon = round(-97.13 + np.random.uniform(-0.05, 0.05), 4)
    
    # Tags contextuales (2-3 tags)
    n_tags = random.randint(2, 3)
    tags = random.sample(config["tags_pool"], min(n_tags, len(config["tags_pool"])))
    tag_str = ",".join(tags)
    
    # NUEVO: Nivel de actividad (para filtrar por actividad_level)
    actividad = random.choice(config["actividad_level"])
    
    # NUEVO: Indicadores booleanos
    outdoor = 1 if any(x in tags for x in ["naturaleza", "excursión", "panorama"]) else 0
    accesible = 1 if random.random() < 0.3 else 0  # 30% son accesibles
    
    desc = f"Lugar de tipo {t} con enfoque en {' y '.join(tags[:2])}."
    title = f"{t.capitalize()} {i:03d}"
    
    items.append([
        i, title, t, R, K, N, price, lat, lon, 
        tag_str, desc, actividad, outdoor, accesible
    ])

df = pd.DataFrame(items, columns=[
    "item_id", "title", "type", "R", "K", "N", "price", "lat", "lon",
    "tags", "description", "actividad_level", "outdoor", "accesible"
])

df.to_csv("data/items.csv", index=False, encoding="utf-8")
print(f"✓ items.csv generado con {len(df)} filas y campos contextuales")