import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_DIR, '..', 'data')


class SmarturEngine:
    """
    Motor principal de Inteligencia para Recomendaciones (Basado en Filtrado Colaborativo K-Nearest Neighbors).
    Se encarga de cargar la información histórica de interacciones ('reviews') y datos de negocios ('business'),
    y genera una matriz matemática optimizada de Utilidad Esparcida (User-Item Sparse Matrix).
    """

    def __init__(self, reviews_path=None, business_path=None):
        """
        Inicializa el motor cargando los datos de disco y particionando para entrenamiento/test.
        
        Args:
            reviews_path (str, opcional): Ruta al CSV que contiene las reseñas (interacciones).
            business_path (str, opcional): Ruta al CSV que contiene los detalles de los negocios.
        """
        if reviews_path is None:
            reviews_path = os.path.join(_DATA, 'data_reviews_limpio.csv')
        if business_path is None:
            business_path = os.path.join(_DATA, 'data_negocios_limpio.csv')

        self.df = pd.read_csv(reviews_path)
        self.train_data, self.test_data = train_test_split(
            self.df, test_size=0.2, random_state=42
        )
        self.df_biz = pd.read_csv(business_path)

        # Matriz Sparse (Comprimida) Original de los ratings exactos
        self.user_item_matrix = None
        # Matriz Sparse Centrada respecto a la media de cada usuario (esencial para el cálculo tipo Pearson)
        self.matrix_centered = None
        # Índices y nombres de columnas mapeados (pandas Categories)
        self.user_item_matrix_index = None
        self.user_item_matrix_columns = None
        # Media de calificación (rating) por individuo
        self.user_means = None
        # Objeto NearestNeighbors entrenado de scikit-learn
        self.knn_model = None
        # Diccionarios de mapeo rápido entre ID de texto a índice entero en la Matriz
        self._user_idx_map = None
        self._biz_idx_map = None

    def get_user_idx(self, user_id):
        """Devuelve el índice matricial interno (entero posicional) de un ID de usuario en base al mapa cacheado."""
        if self._user_idx_map is None:
            self._user_idx_map = {u: i for i, u in enumerate(self.user_item_matrix_index)}
        return self._user_idx_map.get(user_id)

    def get_biz_idx(self, biz_id):
        """Devuelve el índice matricial interno (entero posicional) de un ID de negocio en base al mapa cacheado."""
        if self._biz_idx_map is None:
            self._biz_idx_map = {b: i for i, b in enumerate(self.user_item_matrix_columns)}
        return self._biz_idx_map.get(biz_id)

    def prepare_pearson_matrix(self):
        """
        Construye la matriz matemática de Utilidad optimizada utilizando scipy.sparse.csr_matrix.
        Este formato previene ineficiencias y el error `MemoryError` de memoria RAM cuando existen
        decenas de miles de combinaciones usuario-ítem (evitando guardar los ceros y celdas vacías explícitamente).
        
        Además, prepara el centrado matemático de los datos (la calificación restada por la media personal del usuario).
        Para los datos centrados explícitamente, la correlación de Pearson y la Distancia del Coseno ('cosine')
        miden la misma relación geométrica de similitud; por lo tanto, usamos cosine que sí soporta estas matrices sparse.
        """
        cats_u = self.train_data['user_id'].astype('category')
        cats_i = self.train_data['business_id'].astype('category')
        
        row = cats_u.cat.codes
        col = cats_i.cat.codes
        data = self.train_data['stars'].values.astype(np.float32)
        
        n_u = len(cats_u.cat.categories)
        n_i = len(cats_i.cat.categories)
        
        self.user_item_matrix_index = cats_u.cat.categories
        self.user_item_matrix_columns = cats_i.cat.categories
        
        # 1. Construir matriz estructural csr_matrix de valor original no modificado (ratings exactos 1 a 5)
        self.user_item_matrix = csr_matrix((data, (row, col)), shape=(n_u, n_i))
        
        # Calcular medias por usuario
        user_sums = np.bincount(row, weights=data)
        user_counts = np.bincount(row)
        user_sums = np.pad(user_sums, (0, n_u - len(user_sums)), 'constant')
        user_counts = np.pad(user_counts, (0, n_u - len(user_counts)), 'constant')
        
        user_means = np.divide(user_sums, user_counts, out=np.zeros_like(user_sums), where=user_counts!=0)
        self.user_means = pd.Series(user_means, index=self.user_item_matrix_index)
        
        # Centrado
        centered_data = data - user_means[row]
        self.matrix_centered = csr_matrix((centered_data, (row, col)), shape=(n_u, n_i))
        
        # Cosine distance equivale a Pearson correlation para datos centrados.
        self.knn_model = NearestNeighbors(
            metric='cosine', algorithm='brute', n_jobs=-1
        )
        self.knn_model.fit(self.matrix_centered)
        
        print(f"Engine listo: {n_u} usuarios y {n_i} negocios.")

    def get_candidate_pool(self, user_id, top_n=50):
        """
        Calcula un abanico (pool) inicial rápido de negocios como recomendaciones candidatas 
        para un usuario, apoyándose de los usuarios más similares encontrados mediante K-NN (K-vecinos más cercanos).
        
        Args:
            user_id (str): La ID única tipo UUID del usuario.
            top_n (int): La cantidad deseada de recomendaciones candidatas base.
            
        Returns:
            list: Una lista que contiene los IDs en texto de negocios que podrían gustarle al usuario.
            Si el usuario no tiene historial, se le retornan los tópicos más populares de forma global.
        """
        if self.user_item_matrix_index is None or user_id not in self.user_item_matrix_index:
            # Cold-start: Usuario nuevo sin ningún historial de interacciones en el dataset
            return (
                self.train_data.groupby('business_id')['stars']
                .count()
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )

        k = min(top_n, self.user_item_matrix.shape[0] - 1)
        user_idx = self.get_user_idx(user_id)
        
        query = self.matrix_centered[user_idx]
        distances, indices = self.knn_model.kneighbors(
            query, n_neighbors=k
        )

        similar_users = self.user_item_matrix_index[indices[0]]
        candidates = self.train_data[self.train_data['user_id'].isin(similar_users)]
        return (
            candidates.groupby('business_id')['stars']
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
