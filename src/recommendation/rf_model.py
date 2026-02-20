import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

class SmarturContextModel:
    def __init__(self, business_path='../../data/data_negocios_limpio.csv'):
        # Cargamos los datos originales de negocios
        self.df_biz = pd.read_csv(business_path)
        self.model = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1)
        
        # Usamos los nombres ORIGINALES que están en el CSV de negocios
        self.features = ['stars', 'review_count', 'latitude', 'longitude']

    def train(self, reviews_df):
        # Unir reviews con datos de negocios
        train_df = reviews_df.merge(self.df_biz, on='business_id')
        
        # Renombramos 'stars_y' a 'stars' temporalmente para que coincida con self.features
        # stars_x es el rating del usuario (target)
        # stars_y es el rating del negocio (feature)
        train_df = train_df.rename(columns={'stars_y': 'stars', 'stars_x': 'user_rating'})
        
        X = train_df[self.features].fillna(0)
        y = train_df['user_rating'] 
        
        print(f"Entrenando con {X.shape[0]} muestras y las características: {self.features}")
        self.model.fit(X, y)
        
        # Guardamos el modelo
        joblib.dump(self.model, '../../models/rf_context_yelp.joblib')
        print("Random Forest contextual entrenado con éxito.")

    def predict_context(self, business_ids):
        # Aquí self.df_biz ya tiene la columna 'stars', así que funcionará perfectamente
        biz_features = self.df_biz[self.df_biz['business_id'].isin(business_ids)]
        X = biz_features[self.features].fillna(0)
        return self.model.predict(X)