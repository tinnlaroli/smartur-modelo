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
            'review_count', 'latitude', 'longitude', 'is_open',
            'price_level', 'is_accessible', 'outdoor',
            'is_good_for_kids', 'is_romantic'
        ]
        self.cat_features = []
        self.features = []  # Item + Context (todas agregadas al train)
        self.is_fitted = False
        self.encoder = ContextEncoder()

    def _extract_top_categories(self, df):
        """Encuentra las N categorías más frecuentes en el dataset."""
        all_cats = (
            df['categories']
            .dropna()
            .str.split(',')
            .explode()
            .str.strip()
        )
        self.top_categories = all_cats.value_counts().head(TOP_N_CATEGORIES).index.tolist()
        self.cat_features = [f'cat_{c}' for c in self.top_categories]

    def _add_category_features(self, df):
        """Agrega columnas binarias para las top categorías."""
        result = df.copy()
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

    def _simulate_user_contexts(self, df):
        """
        Inyecta perfiles sintéticos de usuario mediante simulación vectorizada.
        Crea las verdaderas interacciones (Cross-Features) para que el RF aprenda
        nativamente su peso real.
        """
        print("Vectorizando generación de contextos sintéticos de usuario...")
        result = df.copy()
        np.random.seed(42)
        n = len(df)

        # 1. Inferir user_budget (basado en likes)
        user_likes = df[df['stars_user'] >= 4]
        if 'price_level' in df.columns:
            user_mean_price = user_likes.groupby('user_id')['price_level'].mean().round().fillna(2).astype(int)
        else:
            user_mean_price = pd.Series(dtype=int)
            
        result['user_budget'] = result['user_id'].map(user_mean_price).fillna(2).astype(int)
        
        # 2. Generar aleatoriamente features sintéticas de usuario
        result['user_age_range'] = np.random.randint(1, 6, size=n)
        
        for t in self.encoder.tourism_types:
            result[f'user_tur_{t}'] = np.random.binomial(1, 0.2, size=n)
            
        # Group types (one-hot)
        groups = np.random.choice(self.encoder.group_types, size=n, p=[0.3, 0.4, 0.2, 0.1])
        for g in self.encoder.group_types:
            result[f'user_group_{g}'] = (groups == g).astype(int)
            
        # Preferences
        result['user_requires_accessibility'] = np.random.binomial(1, 0.05, size=n)
        result['user_pref_outdoor'] = np.random.binomial(1, 0.2, size=n)
        result['user_wants_tours'] = np.random.binomial(1, 0.15, size=n)
        result['user_needs_hotel'] = np.random.binomial(1, 0.05, size=n)
        result['user_pref_food'] = np.random.binomial(1, 0.9, size=n)

        # 3. Match features (vectorizado cruzado Item x User)
        result['budget_delta'] = (result['user_budget'] - result.get('price_level', 2)).abs()
        
        cats_lower = result['categories'].fillna('').str.lower()
        overlap = np.zeros(n)
        
        for t in self.encoder.tourism_types:
            mask_user_has_tur = (result[f'user_tur_{t}'] == 1)
            cats_to_match = [c.lower() for c in MAPEO_CATEGORIAS.get(t, [])]
            mask_biz_has_tur = cats_lower.str.contains('|'.join(cats_to_match), regex=True, na=False)
            overlap += (mask_user_has_tur & mask_biz_has_tur).astype(int)
            
        result['interest_overlap'] = overlap

        is_good_kids = result.get('is_good_for_kids', np.zeros(n))
        result['kids_match'] = ((result['user_group_familia'] == 1) & (is_good_kids == 1)).astype(int)

        is_rom = result.get('is_romantic', np.zeros(n))
        result['romantic_match'] = ((result['user_group_pareja'] == 1) & (is_rom == 1)).astype(int)

        result['tours_match'] = ((result['user_wants_tours'] == 1) & cats_lower.str.contains('tours', na=False)).astype(int)

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

    def predict_context(self, business_ids):
        """Fallback si no mandan ningún contexto."""
        return self.predict_with_context(business_ids, user_context=None)

    def predict_with_context(self, business_ids, user_context=None):
        """
        Inferencia Pura ML de 2nda Etapa: 
        Procesa el dataset crudo en vectores ampliados [Item] + [User] + [Match]
        y los inyecta al modelo Random Forest pre-entrenado.
        El modelo toma la decisión final evaluando todas las colisiones sin sumar/restar manualmente.
        """
        biz_indexed = self.df_biz.set_index('business_id')
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
