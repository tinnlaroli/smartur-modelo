# pearson + knn
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

class SmarturEngine:
    def __init__(self, 
                reviews_path='../../data/data_reviews_limpio.csv',
                business_path='../../data/data_negocios_limpio.csv'):
        
        # 1. Carga de Reseñas y Split (80/20)
        self.df = pd.read_csv(reviews_path)
        self.train_data, self.test_data = train_test_split(self.df, test_size=0.2, random_state=42)
        
        # 2. CARGA DE NEGOCIOS (Crucial para el filtrado contextual de SMARTUR)
        # Esto soluciona el error: 'SmarturEngine' object has no attribute 'df_biz'
        self.df_biz = pd.read_csv(business_path) 
        
        self.user_item_matrix = None
        self.user_means = None
        self.knn_model = None

    def prepare_pearson_matrix(self):
        """Construye la matriz de utilidad y entrena el KNN con Correlación de Pearson"""
        # Crear matriz User-Item (pivot table) en float32 para ahorrar RAM
        self.user_item_matrix = self.train_data.pivot_table(
            index='user_id', 
            columns='business_id', 
            values='stars'
        ).fillna(0).astype(np.float32)
        
        # Centrado de datos para Pearson: r_centered = r_ui - media_u
        self.user_means = self.user_item_matrix.replace(0, np.nan).mean(axis=1).fillna(0)
        self.matrix_centered = self.user_item_matrix.sub(self.user_means, axis=0)
        
        # Entrenar KNN con métrica de correlación (Pearson puro)
        self.knn_model = NearestNeighbors(metric='correlation', algorithm='brute', n_jobs=-1)
        self.knn_model.fit(self.matrix_centered)
        
        print(f"Engine listo: {self.user_item_matrix.shape[0]} usuarios y {self.user_item_matrix.shape[1]} negocios.")

    def get_candidate_pool(self, user_id, top_n=50):
        """Genera el pool inicial de candidatos mediante KNN para filtrar luego en fusion.py"""
        if user_id not in self.user_item_matrix.index:
            # Cold start: Devolver negocios más populares de Yelp
            return self.train_data.groupby('business_id')['stars'].count().sort_values(ascending=False).head(top_n).index.tolist()

        distances, indices = self.knn_model.kneighbors(
            self.matrix_centered.loc[[user_id]], 
            n_neighbors=top_n
        )
        
        # Obtener usuarios similares e identificar sus negocios favoritos
        similar_users = self.user_item_matrix.index[indices[0]]
        candidates = self.train_data[self.train_data['user_id'].isin(similar_users)]
        return candidates.groupby('business_id')['stars'].mean().sort_values(ascending=False).head(top_n).index.tolist()