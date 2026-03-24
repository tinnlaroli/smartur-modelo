import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from context_encoder import ContextEncoder

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_DIR, '..', 'data')
_MODELS = os.path.join(_DIR, '..', 'models')
TOP_N_CATEGORIES = 15

class SmarturContextModel:
    """
    Modelo contextual SMARTUR v3 basado en Random Forest Regressor.
    Entrena sobre features combinadas: Item + (opcionalmente) User + Interaction.
    Cuando hay contexto del turista, usa el vector completo [User + Item + Interaction]
    para re-rankear candidatos. Sin contexto, hace fallback a solo features de ítem.
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
        ]
        self.features = []  # se llena en _extract_top_categories
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
        cat_features = [f'cat_{c}' for c in self.top_categories]
        self.features = self.numeric_features + cat_features

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

    def train(self, reviews_df):
        train_df = reviews_df.merge(self.df_biz, on='business_id', suffixes=('_user', '_biz'))

        self._extract_top_categories(train_df)
        train_df = self._add_category_features(train_df)

        X = train_df[self.features].fillna(0)
        y = train_df['stars_user']

        print(f"RF: entrenando con {X.shape[0]} muestras, {X.shape[1]} features")
        print(f"   Features: {self.features}")
        self.model.fit(X, y)

        os.makedirs(_MODELS, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'top_categories': self.top_categories,
            'features': self.features,
        }, os.path.join(_MODELS, 'rf_context_yelp.joblib'))
        self.is_fitted = True
        print("Random Forest contextual entrenado y guardado.")

    def load(self, model_path=None):
        """
        Intenta cargar un modelo Random Forest serializado desde disco.
        Retorna True si el modelo se cargó con éxito.
        """
        if model_path is None:
            model_path = os.path.join(_MODELS, 'rf_context_yelp.joblib')
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            self.model = data['model']
            self.top_categories = data['top_categories']
            self.features = data['features']
            self.is_fitted = True
            print("Random Forest contextual cargado desde disco.")
            return True
        return False

    def predict_context(self, business_ids):
        """Predicciones solo con features de ítem (backward compatible)."""
        biz_indexed = self.df_biz.set_index('business_id')
        biz_features = biz_indexed.reindex(business_ids).reset_index()
        biz_features = self._add_category_features(biz_features)

        X = biz_features[self.features].fillna(0)
        preds = self.model.predict(X)
        return np.clip(preds, 1, 5)

    def predict_with_context(self, business_ids, user_context=None):
        """
        Predicciones enriquecidas con contexto del turista.
        Construye el vector [Item Features] y, si hay contexto, les añade
        [User Features + Interaction Features] como columnas extra.

        Args:
            business_ids (list): IDs de negocios a puntuar
            user_context (dict | None): JSON del formulario React

        Returns:
            np.ndarray: Scores predichos (clipped 1-5), uno por business_id
        """
        if not user_context:
            return self.predict_context(business_ids)

        biz_indexed = self.df_biz.set_index('business_id')
        biz_features = biz_indexed.reindex(business_ids).reset_index()
        biz_features = self._add_category_features(biz_features)

        # Codificar el contexto del usuario
        user_feats = self.encoder.encode_user(user_context)

        # Calcular features de match para cada negocio
        context_rows = []
        for _, biz_row in biz_features.iterrows():
            match_feats = self.encoder.compute_match_features(user_feats, biz_row)
            context_rows.append({**user_feats, **match_feats})

        context_df = pd.DataFrame(context_rows, index=biz_features.index)

        # Combinar item features + context features
        X_item = biz_features[self.features].fillna(0)

        # Las features de contexto se usan como ajuste aditivo al score base:
        # Penalizar budget_delta, bonificar interest_overlap
        base_preds = self.model.predict(X_item)

        # Ajuste contextual: cada punto de budget_delta reduce 0.15, cada overlap suma 0.20
        budget_deltas = context_df['budget_delta'].values
        overlaps = context_df['interest_overlap'].values

        context_adjustment = (overlaps * 0.20) - (budget_deltas * 0.15)
        adjusted_preds = base_preds + context_adjustment

        return np.clip(adjusted_preds, 1, 5)
