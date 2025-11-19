# generate_items.py

'''
Este script genera un archivo CSV con 200 items aleatorios para el proyecto SMARTUR.

Los items tienen las siguientes columnas:
- item_id: ID del item
- title: Título del item (tipo + número de item)
- type: Tipo del item
- R: Rating del item
- K: Número de reviews del item
- N: Número de reviews del item
- price: Precio del item
- lat: Latitud del item
- lon: Longitud del item
- tags: Tags del item
- description: Descripción del item
'''
import pandas as pd
import numpy as np
import random

types = ["hotel","restaurante","atracción","tour","tienda"]
tags_pool = {
    "hotel": ["descanso","panorama","romántico","ecológico"],
    "restaurante": ["gastronomía","tradicional","urbano","familiar"],
    "atracción": ["cultura","experiencia","historia","vista"],
    "tour": ["aventura","naturaleza","excursión","extremo"],
    "tienda": ["artesanía","local","souvenir","gastronomía"]
}

items = []
for i in range(1,201):
    t = random.choice(types)
    R = round(np.random.uniform(3.5,5.0),1)
    K = i
    N = random.randint(20,500)
    price = round(np.random.uniform(100,1500),2)
    lat = round(18.85 + np.random.uniform(-0.05,0.05),4)
    lon = round(-97.13 + np.random.uniform(-0.05,0.05),4)
    tag = ",".join(random.sample(tags_pool[t],2))
    desc = f"Lugar de tipo {t} con enfoque en {tag.replace(',',' y ')}."
    title = f"{t.capitalize()} {i:03d}"
    items.append([i,title,t,R,K,N,price,lat,lon,tag,desc])

df = pd.DataFrame(items, columns=["item_id","title","type","R","K","N","price","lat","lon","tags","description"])
df.to_csv("data/items.csv", index=False, encoding="utf-8")
print("items.csv generado con", len(df), "filas")
