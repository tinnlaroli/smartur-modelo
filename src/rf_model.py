import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from context_encoder import ContextEncoder, MAPEO_CATEGORIAS

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_DIR, '..', 'data')
_MODELS = os.path.join(_DIR, '..', 'models')
TOP_N_CATEGORIES = 15

# Categorías turísticas garantizadas en el feature set aunque no estén en el top-15 por frecuencia.
# Necesarias para que el RF aprenda señales de turismo de naturaleza/aventura/cultural
# cuando el dataset base es mayoritariamente gastronómico (ej. Yelp).
TOURISM_ANCHOR_CATEGORIES = [
    # Yelp tourism categories (always needed for fallback dev mode)
    'Parks', 'Hiking', 'Museums', 'Hotels', 'Hotels & Travel',
    'Arts & Entertainment', 'Active Life', 'Tours',
    'Landmarks & Historical Buildings', 'Bed & Breakfast',
    'Campgrounds', 'Zoos', 'Botanical Gardens',
    # Local POI categories — Altas Montañas, Veracruz (categories_mapped + categories_raw)
    'nature', 'culture', 'gastronomy',
    'park', 'viewpoint', 'waterfall', 'volcano', 'mountain',
    'hacienda', 'cathedral', 'sanctuary', 'market', 'museum',
    'history', 'monument', 'zocalo', 'botanical garden',
]


class SmarturContextModel:
    """
    Modelo contextual SMARTUR v4 basado en Random Forest Regressor.
    True Machine Learning Contextual: 
    El modelo aprende las interacciones dinámicamente mediante la inyección de 
    datos de contexto de usuario sintéticos durante el entrenamiento.
    """
    def __init__(self, business_path=None):
        if business_path is None:
            business_path = os.path.join(_DATA, 'data_negocios_limpio.csv')
        self.df_biz = pd.read_csv(business_path)
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1
        )
        self.top_categories = []
        self.numeric_features = [
            'review_count', 'is_open',
            'price_level', 'is_accessible', 'outdoor',
            'is_good_for_kids', 'is_romantic'
        ]
        self.cat_features = []
        self.features = []  # Item + Context (todas agregadas al train)
        self.is_fitted = False
        self.encoder = ContextEncoder()
        self._local_defaults = {
            'review_count': 10,
            'is_open': 1,
            'is_good_for_kids': 0,
            'is_romantic': 0,
        }

    def _extract_top_categories(self, df):
        """Encuentra las N categorías más frecuentes + anclas turísticas garantizadas."""
        all_cats = (
            df['categories']
            .dropna()
            .str.split(',')
            .explode()
            .str.strip()
        )
        top_by_freq = all_cats.value_counts().head(TOP_N_CATEGORIES).index.tolist()
        present = set(top_by_freq)
        anchors = [c for c in TOURISM_ANCHOR_CATEGORIES if c not in present]
        self.top_categories = top_by_freq + anchors
        self.cat_features = [f'cat_{c}' for c in self.top_categories]

    def _add_category_features(self, df):
        """Agrega columnas binarias para las top categorías."""
        result = df.copy()
        if 'review_count' in result.columns:
            result['review_count'] = np.log1p(result['review_count'].fillna(0))
        for cat in self.top_categories:
            result[f'cat_{cat}'] = (
                result['categories']
                .fillna('')
                .str.contains(cat, case=False, regex=False)
                .astype(int)
            )
        for col in self.numeric_features:
            if col not in result.columns:
                result[col] = 0
        return result

    def prepare_local_items(self, poi_df):
        """Normaliza POIs locales al esquema del modelo."""
        if poi_df is None or poi_df.empty:
            return pd.DataFrame(columns=self.df_biz.columns)

        df = poi_df.copy()
        df['business_id'] = df['id'].astype(str)
        df['name'] = df['name'].fillna('Local POI')

        def _join_categories(row):
            parts = []
            raw = row.get('categories_raw')
            if isinstance(raw, str) and raw.strip():
                parts.append(raw.strip())
            mapped = row.get('categories_mapped', [])
            if isinstance(mapped, list) and mapped:
                parts.extend(mapped)
            return ','.join(parts) if parts else ''

        df['categories'] = df.apply(_join_categories, axis=1)

        if 'price_level' not in df.columns:
            df['price_level'] = 2
        if 'is_accessible' not in df.columns:
            df['is_accessible'] = 0
        if 'outdoor' not in df.columns:
            df['outdoor'] = 0
        if 'latitude' not in df.columns:
            df['latitude'] = 0
        if 'longitude' not in df.columns:
            df['longitude'] = 0

        df['price_level'] = df['price_level'].fillna(2).astype(int)
        df['is_accessible'] = df['is_accessible'].fillna(0).astype(int)
        df['outdoor'] = df['outdoor'].fillna(0).astype(int)
        df['latitude'] = df['latitude'].fillna(0).astype(float)
        df['longitude'] = df['longitude'].fillna(0).astype(float)

        for col, default in self._local_defaults.items():
            if col not in df.columns:
                df[col] = default
            df[col] = df[col].fillna(default)

        return df

    def _simulate_user_contexts(self, df):
        """
        Infiere el perfil de usuario desde su historial de ratings en lugar de asignarlo
        aleatoriamente. Tipos de turismo y tipo de grupo se derivan de las categorías y
        atributos de los negocios que el usuario calificó con ≥4 estrellas, haciendo que
        los match features sean correlacionados con los ratings reales.
        """
        print("Inferiendo contextos de usuario desde historial de ratings...")
        result = df.copy()
        np.random.seed(42)
        n = len(df)
        liked_mask = df['stars_user'] >= 4
        cats_lower = df['categories'].fillna('').str.lower()

        # 1. Inferir user_budget desde precio promedio de items valorados ≥4
        if 'price_level' in df.columns:
            user_mean_price = (
                df[liked_mask]
                .groupby('user_id')['price_level']
                .mean().round().fillna(2).astype(int)
            )
        else:
            user_mean_price = pd.Series(dtype=int)
        result['user_budget'] = result['user_id'].map(user_mean_price).fillna(2).astype(int)

        # 2. Inferir tipos de turismo desde categorías de items valorados ≥4
        for t in self.encoder.tourism_types:
            cats_to_match = [c.lower() for c in MAPEO_CATEGORIAS.get(t, [])]
            if not cats_to_match:
                result[f'user_tur_{t}'] = 0
                continue
            biz_has_tur = cats_lower.str.contains('|'.join(cats_to_match), regex=True, na=False)
            user_tur_map = (liked_mask & biz_has_tur).groupby(df['user_id']).any().astype(int)
            result[f'user_tur_{t}'] = result['user_id'].map(user_tur_map).fillna(0).astype(int)

        # 3. Inferir group_type desde items románticos / kid-friendly valorados ≥4
        romantic_col = df['is_romantic'] if 'is_romantic' in df.columns else pd.Series(0, index=df.index)
        kids_col = df['is_good_for_kids'] if 'is_good_for_kids' in df.columns else pd.Series(0, index=df.index)
        likes_romantic_mask = liked_mask & (romantic_col == 1)
        likes_kids_mask = liked_mask & (kids_col == 1)
        romantic_map = likes_romantic_mask.groupby(df['user_id']).any()
        kids_map = likes_kids_mask.groupby(df['user_id']).any()

        user_is_romantic = result['user_id'].map(romantic_map).fillna(False)
        user_is_kids = result['user_id'].map(kids_map).fillna(False)

        rand_fallback = np.random.choice(['solo', 'amigos'], size=n, p=[0.4, 0.6])
        groups = np.where(
            user_is_kids, 'familia',
            np.where(user_is_romantic & ~user_is_kids, 'pareja', rand_fallback)
        )
        for g in self.encoder.group_types:
            result[f'user_group_{g}'] = (groups == g).astype(int)

        # 4. Age range — no inferible desde Yelp, mantener aleatorio
        result['user_age_range'] = np.random.randint(1, 6, size=n)

        # 5. Preferencias booleanas (no inferibles desde Yelp con certeza)
        result['user_requires_accessibility'] = np.random.binomial(1, 0.05, size=n)
        result['user_pref_outdoor'] = np.random.binomial(1, 0.2, size=n)
        result['user_wants_tours'] = np.random.binomial(1, 0.15, size=n)
        result['user_needs_hotel'] = np.random.binomial(1, 0.05, size=n)
        result['user_pref_food'] = np.random.binomial(1, 0.9, size=n)

        # 6. Match features (ahora correlacionados con ratings reales)
        result['budget_delta'] = (result['user_budget'] - result.get('price_level', 2)).abs()

        overlap = np.zeros(n)
        for t in self.encoder.tourism_types:
            mask_user = result[f'user_tur_{t}'] == 1
            cats_to_match = [c.lower() for c in MAPEO_CATEGORIAS.get(t, [])]
            if cats_to_match:
                mask_biz = cats_lower.str.contains('|'.join(cats_to_match), regex=True, na=False)
                overlap += (mask_user & mask_biz).astype(int)
        result['interest_overlap'] = overlap

        is_good_kids = result.get('is_good_for_kids', np.zeros(n))
        result['kids_match'] = ((result['user_group_familia'] == 1) & (is_good_kids == 1)).astype(int)

        is_rom = result.get('is_romantic', np.zeros(n))
        result['romantic_match'] = ((result['user_group_pareja'] == 1) & (is_rom == 1)).astype(int)

        result['tours_match'] = (
            (result['user_wants_tours'] == 1) & cats_lower.str.contains('tours', na=False)
        ).astype(int)

        return result

    def train(self, reviews_df):
        train_df = reviews_df.merge(self.df_biz, on='business_id', suffixes=('_user', '_biz'))

        # 1. Pipeline de Item
        self._extract_top_categories(train_df)
        train_df = self._add_category_features(train_df)

        # 2. Pipeline de User (Generación Sintética)
        train_df = self._simulate_user_contexts(train_df)

        # 3. Features combinadas: Item + User + Match
        self.features = self.numeric_features + self.cat_features + self.encoder.all_context_feature_names

        X = train_df[self.features].fillna(0)
        y = train_df['stars_user']

        print(f"RF (True ML Contextual): entrenando sobre {X.shape[0]} interacciones con {X.shape[1]} variables cruzadas.")
        print(f"Features en red: {self.features}")
        
        self.model.fit(X, y)

        os.makedirs(_MODELS, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'top_categories': self.top_categories,
            'features': self.features,
        }, os.path.join(_MODELS, 'rf_context_yelp.joblib'))
        
        self.is_fitted = True
        print("Random Forest Contextual (ML) entrenado y guardado.")

    def load(self, model_path=None):
        """Carga modelo Random Forest con features combinadas desde disco."""
        if model_path is None:
            model_path = os.path.join(_MODELS, 'rf_context_yelp.joblib')
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            self.model = data['model']
            self.top_categories = data['top_categories']
            self.features = data['features']
            self.is_fitted = True
            self.cat_features = [f'cat_{c}' for c in self.top_categories]
            print("Random Forest Contextual cargado desde disco.")
            return True
        return False

    def predict_context(self, business_ids, df_biz_override=None):
        """Fallback si no mandan ningún contexto."""
        return self.predict_with_context(business_ids, user_context=None, df_biz_override=df_biz_override)

    def predict_with_context(self, business_ids, user_context=None, df_biz_override=None):
        """
        Inferencia Pura ML de 2nda Etapa:
        Procesa el dataset crudo en vectores ampliados [Item] + [User] + [Match]
        y los inyecta al modelo Random Forest pre-entrenado.
        El modelo toma la decisión final evaluando todas las colisiones sin sumar/restar manualmente.
        """
        ref_df = df_biz_override if df_biz_override is not None else self.df_biz
        biz_indexed = ref_df.set_index('business_id')
        biz_features = biz_indexed.reindex(business_ids).reset_index()
        biz_features = self._add_category_features(biz_features)

        # Codificamos al usuario real (React Form)
        # Si no hay contexto, el encoder emite defaults limpios ('cold start usuario medio')
        user_feats = self.encoder.encode_user(user_context)

        # Generamos la colisión de Match cruzando contra la lista de businesess obtenidos
        context_rows = []
        for _, biz_row in biz_features.iterrows():
            match_feats = self.encoder.compute_match_features(user_feats, biz_row)
            context_rows.append({**user_feats, **match_feats})

        context_df = pd.DataFrame(context_rows, index=biz_features.index)

        # Unimos las tablas ItemFeature | UserFeature | MatchFeature
        combined_df = pd.concat([biz_features, context_df], axis=1)

        # Aseguramos un DataFrame rígido con las mismas features de `self.features` original
        for col in self.features:
            if col not in combined_df.columns:
                combined_df[col] = 0

        X = combined_df[self.features].fillna(0)

        # Predicción pura dictaminada 100% por el Random Forest
        preds = self.model.predict(X)
        
        return np.clip(preds, 1, 5)
