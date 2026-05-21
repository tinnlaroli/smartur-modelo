import logging
import os
import json
import pandas as pd
import psycopg2

_logger = logging.getLogger("smartur-api")


def _get_env(name, default=None):
    value = os.getenv(name)
    return value if value not in (None, '') else default


# Category/context mappings for tourist services and traveler profiles
_SERVICE_TYPE_CATEGORIES = {
    'hotel': {
        'categories_raw': 'hotel, accommodation, Hotels & Travel',
        'categories_mapped': ['rural'],
        'outdoor': False,
    },
    'restaurant': {
        'categories_raw': 'restaurant, food, gastronomy, local food, Restaurants',
        'categories_mapped': ['gastronomy'],
        'outdoor': False,
    },
    'tour': {
        'categories_raw': 'tour, hiking, nature, adventure, Tours',
        'categories_mapped': ['nature'],
        'outdoor': True,
    },
}

_INTEREST_MAP = {
    'naturaleza': 'naturaleza', 'nature': 'naturaleza',
    'cultura': 'cultural', 'culture': 'cultural', 'cultural': 'cultural',
    'gastronomía': 'gastronomico', 'gastronomia': 'gastronomico',
    'gastronomy': 'gastronomico', 'gastronomico': 'gastronomico',
    'aventura': 'aventura', 'adventure': 'aventura',
    'rural': 'rural',
}

_ACTIVITY_BUDGET = {1: 'bajo', 2: 'bajo', 3: 'medio', 4: 'alto', 5: 'premium'}


def get_poi_connection():
    return psycopg2.connect(
        host=_get_env('POI_DB_HOST', 'localhost'),
        port=int(_get_env('POI_DB_PORT', 5432)),
        database=_get_env('POI_DB_NAME', 'smartur'),
        user=_get_env('POI_DB_USER', 'postgres'),
        password=_get_env('POI_DB_PASSWORD', os.getenv('DB_PASSWORD', '12345678')),
        options='-c client_encoding=UTF8',
    )


def fetch_pois(active_only=True):
    where = 'WHERE is_active = TRUE' if active_only else ''
    query = f'''
        SELECT id, name, categories_raw, categories_mapped,
               price_level, is_accessible, outdoor, latitude, longitude
        FROM point_of_interest {where}
    '''
    with get_poi_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)

    if 'categories_raw' in df.columns:
        df['categories_raw'] = df['categories_raw'].apply(
            lambda v: v.encode('utf-8', errors='ignore').decode('utf-8')
            if isinstance(v, str)
            else (v if isinstance(v, str) else '')
        )

    if 'categories_mapped' in df.columns:
        def _normalize(val):
            if isinstance(val, list):
                return val
            if val is None:
                return []
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    return parsed if isinstance(parsed, list) else []
                except (json.JSONDecodeError, TypeError):
                    return []
            return []
        df['categories_mapped'] = df['categories_mapped'].apply(_normalize)
    else:
        df['categories_mapped'] = [[] for _ in range(len(df))]

    df['is_accessible'] = df['is_accessible'].apply(lambda v: 1 if v else 0)
    df['outdoor'] = df['outdoor'].apply(lambda v: 1 if v else 0)
    df['price_level'] = df['price_level'].fillna(2).astype(int)

    return df


def fetch_tourist_services(active_only=True):
    """Tourist services as a POI-compatible DataFrame with prefixed IDs (svc_N)."""
    where = 'WHERE ts.active = TRUE' if active_only else ''
    query = f'''
        SELECT ts.id_service AS id, ts.name, ts.service_type,
               l.latitude, l.longitude
        FROM tourist_service ts
        LEFT JOIN location l ON ts.id_location = l.id_location
        {where}
    '''
    with get_poi_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df['id'] = 'svc_' + df['id'].astype(str)
    df['categories_raw'] = df['service_type'].map(
        {k: v['categories_raw'] for k, v in _SERVICE_TYPE_CATEGORIES.items()}
    ).fillna('service')
    df['categories_mapped'] = df['service_type'].map(
        {k: v['categories_mapped'] for k, v in _SERVICE_TYPE_CATEGORIES.items()}
    ).apply(lambda x: x if isinstance(x, list) else [])
    df['outdoor'] = df['service_type'].map(
        {k: 1 if v['outdoor'] else 0 for k, v in _SERVICE_TYPE_CATEGORIES.items()}
    ).fillna(0).astype(int)
    df['price_level'] = 2
    df['is_accessible'] = 0
    df['latitude'] = df['latitude'].fillna(0).astype(float)
    df['longitude'] = df['longitude'].fillna(0).astype(float)
    df['kind'] = 'svc'
    return df


def fetch_all_items(active_only=True):
    """Unified candidate pool: POIs + tourist services."""
    pois = fetch_pois(active_only)
    pois['kind'] = 'poi'
    try:
        services = fetch_tourist_services(active_only)
    except Exception:
        services = pd.DataFrame()
    if services.empty:
        return pois
    return pd.concat([pois, services], ignore_index=True)


def fetch_real_interactions(min_events: int = 2) -> pd.DataFrame:
    """
    Builds a user-item implicit rating matrix from live SMARTUR interaction data.

    Signal weighting:
      explicit star rating (user_rating)  → raw value 1-5 (dominates when present)
      favorite (user_favorite)             → +4.0
      visit    (user_visit)                → +2.5
      detail_open (user_interaction)       → +0.4 per open, capped at 5
      dwell    (user_interaction)          → +0.03 per second up to 120 s
      skip     (user_interaction)          → -0.5 per skip, capped at 3

    Returns DataFrame[user_id, item_id, implicit_score] clipped to [1, 5].
    Returns empty DataFrame on failure so callers can fall back gracefully.
    """
    import numpy as np

    sql = """
        SELECT
            combined.user_id::text,
            combined.place_kind || '_' || combined.place_id::text AS item_id,
            MAX(combined.explicit_rating) AS explicit_rating,
            SUM(CASE WHEN combined.signal = 'favorite' THEN 1 ELSE 0 END) AS favorites,
            SUM(CASE WHEN combined.signal = 'visit'    THEN 1 ELSE 0 END) AS visits,
            SUM(CASE WHEN combined.signal = 'detail_open' THEN 1 ELSE 0 END) AS opens,
            MAX(CASE WHEN combined.signal = 'dwell' THEN combined.dwell_ms ELSE 0 END) AS max_dwell_ms,
            SUM(CASE WHEN combined.signal = 'skip'   THEN 1 ELSE 0 END) AS skips
        FROM (
            SELECT user_id, place_kind, place_id,
                   'favorite'    AS signal,
                   NULL::smallint AS explicit_rating,
                   NULL::int      AS dwell_ms
            FROM user_favorite
            WHERE is_active = TRUE
              AND place_kind IN ('svc', 'poi')

            UNION ALL

            SELECT user_id, place_kind, place_id,
                   'visit', NULL, NULL
            FROM user_visit
            WHERE place_kind IN ('svc', 'poi')

            UNION ALL

            SELECT user_id, place_kind, place_id,
                   event_type, NULL, dwell_ms
            FROM user_interaction
            WHERE event_type IN ('detail_open', 'dwell', 'skip')
              AND place_kind IN ('svc', 'poi')

            UNION ALL

            SELECT user_id, place_kind, place_id,
                   'rating', rating, NULL
            FROM user_rating
        ) combined
        GROUP BY combined.user_id, combined.place_kind, combined.place_id
        HAVING COUNT(*) >= %s
    """
    try:
        with get_poi_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (min_events,))
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
    except Exception as exc:
        _logger.warning(f"fetch_real_interactions: DB query failed — {exc}")
        return pd.DataFrame(columns=["user_id", "item_id", "implicit_score"])

    if not rows:
        return pd.DataFrame(columns=["user_id", "item_id", "implicit_score"])

    df = pd.DataFrame(rows, columns=cols)

    explicit   = df["explicit_rating"].fillna(0).astype(float)
    favorites  = df["favorites"].fillna(0).astype(float).clip(0, 1)
    visits     = df["visits"].fillna(0).astype(float).clip(0, 1)
    opens      = df["opens"].fillna(0).astype(float).clip(0, 5)
    dwell_sec  = (df["max_dwell_ms"].fillna(0).astype(float) / 1000).clip(0, 120)
    skips      = df["skips"].fillna(0).astype(float).clip(0, 3)

    implicit_base = (
        favorites * 4.0
        + visits   * 2.5
        + opens    * 0.4
        + dwell_sec * 0.03
        - skips    * 0.5
    ).clip(1, 5)

    has_explicit = explicit > 0
    df["implicit_score"] = np.where(has_explicit, explicit, implicit_base).clip(1, 5)

    return df[["user_id", "item_id", "implicit_score"]]


def fetch_traveler_profile(user_id):
    """
    Reads traveler_profile for a numeric user_id and returns a context dict
    compatible with ContextEncoder. Returns None if not found or user_id is non-numeric.
    """
    try:
        uid = int(user_id)
    except (ValueError, TypeError):
        return None

    query = '''
        SELECT age_range, interests, activity_level, travel_type, has_accessibility
        FROM traveler_profile
        WHERE user_id = %s AND is_active = TRUE
        LIMIT 1
    '''
    with get_poi_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (uid,))
            row = cur.fetchone()
    if not row:
        return None

    age_range, interests, activity_level, travel_type, has_accessibility = row

    tipos = []
    if interests:
        for interest in interests:
            mapped = _INTEREST_MAP.get(interest.lower().strip())
            if mapped and mapped not in tipos:
                tipos.append(mapped)

    return {
        'edad_range': age_range or '25-34',
        'tiposTurismo': tipos,
        'presupuesto_bucket': _ACTIVITY_BUDGET.get(activity_level, 'medio'),
        'group_type': (travel_type or 'solo').lower(),
        'requiere_accesibilidad': bool(has_accessibility),
    }