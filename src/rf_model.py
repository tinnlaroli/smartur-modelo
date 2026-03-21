import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_DIR, '..', 'data')
_MODELS = os.path.join(_DIR, '..', 'models')
TOP_N_CATEGORIES = 15

class SmarturContextModel:
    """
    Modelo secundario contextual basado de forma algorítmica en Random Forest Regressor.
    Este modelo permite ponderar variables externas del contexto (como categorías de los establecimientos y puntuaciones de review_count) 
    para potenciar las sugerencias del sistema HÍBRIDO, solventando casos de interacciones puras muy dudosas en the Collaborative Filtering.
    """
    def __init__(self, business_path=None):
        if business_path is None:
            business_path = os.path.join(_DATA, 'data_negocios_limpio.csv')
        self.df_biz = pd.read_csv(business_path)
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1
        )
        self.top_categories = []
        self.numeric_features = ['review_count', 'latitude', 'longitude', 'is_open']
        self.features = []  # se llena en _extract_top_categories
        self.is_fitted = False

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
        Intenta cargar y materializar en memoria un modelo Random Forest serializado
        de forma binaria en disco para saltar por completo los altos tiempos de entrenamiento asíncrono.
        Retorna True si un archivo modelo .joblib existía y operó con éxito.
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
        """Predicciones alineadas: el i-ésimo score corresponde al i-ésimo ID."""
        biz_indexed = self.df_biz.set_index('business_id')

        # reindex preserva el orden de business_ids y pone NaN si falta
        biz_features = biz_indexed.reindex(business_ids).reset_index()
        biz_features = self._add_category_features(biz_features)

        X = biz_features[self.features].fillna(0)
        preds = self.model.predict(X)
        return np.clip(preds, 1, 5)
