# generate_users.py - VERSIÓN MEJORADA

import pandas as pd
import numpy as np
import random
import os

DATA_DIR = "data"
USERS_CSV = os.path.join(DATA_DIR, "users.csv")

NUM_USERS = 500
SEED = 42
COUNTRIES = ['MX','US','ES','AR','CO','CL','GB','FR']
AGE_MEAN = 35
AGE_SD = 12

# Intereses alineados con tiposTurismo del formulario
INTEREST_POOL = [
    'naturaleza', 'gastronomía', 'cultura', 'aventura', 'rural',
    'historia', 'fotografía', 'familia', 'deportes', 'relax'
]

# Tipos de viajero (alineado con group_type)
GROUP_TYPES = ['solo', 'pareja', 'familia', 'amigos']

random.seed(SEED)
np.random.seed(SEED)

users = []
for uid in range(1, NUM_USERS + 1):
    edad = int(np.clip(int(np.random.normal(AGE_MEAN, AGE_SD)), 18, 75))
    genero = random.choices(['M','F','O'], weights=[0.48, 0.48, 0.04])[0]
    home_country = random.choice(COUNTRIES)
    
    # Intereses (1-3)
    n_interests = random.choices([1, 2, 3], [0.5, 0.35, 0.15])[0]
    interests = random.sample(INTEREST_POOL, k=min(n_interests, len(INTEREST_POOL)))
    interest_str = ';'.join(interests)
    
    # NUEVO: Preferencia de viaje típica
    typical_group = random.choice(GROUP_TYPES)
    
    # NUEVO: Presupuesto típico (low/med/high)
    budget_pref = random.choices(['low', 'med', 'high'], weights=[0.3, 0.5, 0.2])[0]
    
    # NUEVO: Nivel de actividad preferido (1-5)
    actividad_pref = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.2, 0.4, 0.2, 0.1])[0]
    
    users.append({
        'user_id': uid,
        'edad': edad,
        'genero': genero,
        'home_country': home_country,
        'interests': interest_str,
        'typical_group': typical_group,
        'budget_pref': budget_pref,
        'actividad_pref': actividad_pref,
    })

df = pd.DataFrame(users)
df.to_csv(USERS_CSV, index=False, encoding='utf-8')
print(f"✓ users.csv generado con {len(df)} usuarios y perfiles contextuales")