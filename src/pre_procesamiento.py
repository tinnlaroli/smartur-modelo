import pandas as pd
import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_DIR, '..', 'data')


def _parse_bool_attr(attributes, key):
    """Extrae un atributo booleano del dict de attributes de Yelp.
    Yelp almacena estos valores como strings: 'True', 'False', 'None', u'none'."""
    if not attributes or not isinstance(attributes, dict):
        return 0
    val = attributes.get(key)
    if val is None:
        return 0
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, str):
        return 1 if val.strip().lower() == 'true' else 0
    return 0


def _parse_price_level(attributes):
    """Extrae RestaurantsPriceRange2 (1-4) del dict de attributes. Default: 2."""
    if not attributes or not isinstance(attributes, dict):
        return 2
    val = attributes.get('RestaurantsPriceRange2')
    if val is None:
        return 2
    try:
        level = int(val)
        return max(1, min(4, level))
    except (ValueError, TypeError):
        return 2


def filtrar_yelp(limite_registros=100000):
    ruta_business = os.path.join(_DATA, 'yelp_academic_dataset_business.json')
    ruta_reviews = os.path.join(_DATA, 'yelp_academic_dataset_review.json')

    print(f"Buscando negocios en: {os.path.abspath(ruta_business)}")

    print("Filtrando negocios de interés...")
    negocios = []
    with open(ruta_business, 'r', encoding='utf-8') as f:
        for line in f:
            biz = json.loads(line)
            cat = str(biz.get('categories', ''))
            if any(term in cat for term in ['Tourism', 'Hotels', 'Restaurants', 'Local Services']):
                # Extraer atributos enriquecidos del negocio
                attrs = biz.get('attributes') or {}
                biz['price_level'] = _parse_price_level(attrs)
                biz['is_accessible'] = _parse_bool_attr(attrs, 'WheelchairAccessible')
                biz['outdoor'] = _parse_bool_attr(attrs, 'OutdoorSeating')
                negocios.append(biz)

    df_biz = pd.DataFrame(negocios)

    # Asegurar que las columnas enriquecidas existan con tipos correctos
    for col, default in [('price_level', 2), ('is_accessible', 0), ('outdoor', 0)]:
        if col not in df_biz.columns:
            df_biz[col] = default
        df_biz[col] = df_biz[col].fillna(default).astype(int)

    biz_ids = set(df_biz['business_id'])
    print(f"Negocios encontrados: {len(df_biz)}")

    print(f"Filtrando reviews (limite: {limite_registros})...")
    reviews = []
    count = 0
    with open(ruta_reviews, 'r', encoding='utf-8') as f:
        for line in f:
            rev = json.loads(line)
            if rev['business_id'] in biz_ids:
                reviews.append(rev)
                count += 1
            if count >= limite_registros:
                break

    df_rev = pd.DataFrame(reviews)

    df_biz.to_csv(os.path.join(_DATA, 'data_negocios_limpio.csv'), index=False)
    df_rev.to_csv(os.path.join(_DATA, 'data_reviews_limpio.csv'), index=False)
    print("Listo. CSVs creados en data/.")
    print(f"  Columnas enriquecidas: price_level, is_accessible, outdoor")


if __name__ == "__main__":
    filtrar_yelp()
