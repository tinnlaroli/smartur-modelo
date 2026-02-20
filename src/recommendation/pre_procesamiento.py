import pandas as pd
import json
import os

def filtrar_yelp(limite_registros=100000):
    # Definimos las rutas relativas correctamente
    # ../.. sube de 'recommendation' a 'src' y luego a la raíz 'SMARTUR'
    ruta_business = '../../data/yelp_academic_dataset_business.json'
    ruta_reviews = '../../data/yelp_academic_dataset_review.json'
    
    print(f"Buscando negocios en: {os.path.abspath(ruta_business)}")
    
    print("Filtrando negocios de interés...")
    negocios = []
    # Usamos la ruta corregida
    with open(ruta_business, 'r', encoding='utf-8') as f:
        for line in f:
            biz = json.loads(line)
            cat = str(biz.get('categories', ''))
            if any(term in cat for term in ['Tourism', 'Hotels', 'Restaurants', 'Local Services']):
                negocios.append(biz)
    
    df_biz = pd.DataFrame(negocios)
    biz_ids = set(df_biz['business_id'])
    print(f"Negocios encontrados: {len(df_biz)}")

    print(f"Filtrando reviews (limite: {limite_registros})...")
    reviews = []
    count = 0
    # Usamos la ruta corregida
    with open(ruta_reviews, 'r', encoding='utf-8') as f:
        for line in f:
            rev = json.loads(line)
            if rev['business_id'] in biz_ids:
                reviews.append(rev)
                count += 1
            if count >= limite_registros:
                break
    
    df_rev = pd.DataFrame(reviews)
    
    # Guardamos los resultados también en la carpeta data
    df_biz.to_csv('../../data/data_negocios_limpio.csv', index=False)
    df_rev.to_csv('../../data/data_reviews_limpio.csv', index=False)
    print("¡Listo! Archivos CSV creados en la carpeta /data.")

if __name__ == "__main__":
    filtrar_yelp()