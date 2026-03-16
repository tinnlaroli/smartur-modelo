import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_DIR, '..', 'data')


class SmarturEngine:
    def __init__(self, reviews_path=None, business_path=None):
        if reviews_path is None:
            reviews_path = os.path.join(_DATA, 'data_reviews_limpio.csv')
        if business_path is None:
            business_path = os.path.join(_DATA, 'data_negocios_limpio.csv')

        self.df = pd.read_csv(reviews_path)
        self.train_data, self.test_data = train_test_split(
            self.df, test_size=0.2, random_state=42
        )
        self.df_biz = pd.read_csv(business_path)

        self.user_item_matrix = None
        self.user_means = None
        self.knn_model = None

    def prepare_pearson_matrix(self):
        """Construye la matriz de utilidad con centrado correcto para Pearson.

        Missing entries quedan en 0 tras centrar (= sin señal para KNN),
        en vez del viejo fillna(0) que las dejaba en -media (= señal falsa).
        """
        raw = self.train_data.pivot_table(
            index='user_id', columns='business_id', values='stars'
        )
        self.user_means = raw.mean(axis=1)

        # Mask booleana: True donde hay rating real
        has_rating = raw.notna().values

        # Rellenar NaN con 0 y convertir a float32 para ahorrar RAM
        self.user_item_matrix = raw.fillna(0).astype(np.float32)
        del raw

        # Centrar usando numpy puro (evita copias de DataFrame)
        centered = self.user_item_matrix.values - self.user_means.values[:, np.newaxis]
        centered[~has_rating] = 0.0  # missing → 0 ("sin señal"), NO -media
        del has_rating

        self.matrix_centered = pd.DataFrame(
            centered.astype(np.float32),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns,
        )
        del centered

        self.knn_model = NearestNeighbors(
            metric='correlation', algorithm='brute', n_jobs=-1
        )
        self.knn_model.fit(self.matrix_centered)

        n_u = self.user_item_matrix.shape[0]
        n_i = self.user_item_matrix.shape[1]
        print(f"Engine listo: {n_u} usuarios y {n_i} negocios.")

    def get_candidate_pool(self, user_id, top_n=50):
        """Pool inicial de candidatos mediante KNN."""
        if user_id not in self.user_item_matrix.index:
            return (
                self.train_data.groupby('business_id')['stars']
                .count()
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )

        k = min(top_n, len(self.user_item_matrix) - 1)
        distances, indices = self.knn_model.kneighbors(
            self.matrix_centered.loc[[user_id]], n_neighbors=k
        )

        similar_users = self.user_item_matrix.index[indices[0]]
        candidates = self.train_data[self.train_data['user_id'].isin(similar_users)]
        return (
            candidates.groupby('business_id')['stars']
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
