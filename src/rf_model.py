# src/rf_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

from src.cognitive import load_user_sim
from src.cf import predict_cf_for_user_item

MODEL_PATH = "models/rf_model.joblib"
os.makedirs("models", exist_ok=True)


def build_training_dataset(ratings: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame, users_list, sim, k=20) -> pd.DataFrame:
    """
    Combina ratings, users, items y predicciones CF en un solo dataset de entrenamiento.
    """
    print("[rf_model] Construyendo dataset supervisado...")
    df = ratings.merge(users, on="user_id", how="left")
    df = df.merge(items, on="item_id", how="left")

    print("[rf_model] Calculando predicciones CF (puede tardar un poco)...")
    df["pred_cf"] = [
        predict_cf_for_user_item(u, i, ratings, users_list, sim, k)
        for u, i in zip(df["user_id"], df["item_id"])
    ]

    # Variables numéricas que usaremos para predecir
    num_cols = ["edad", "R", "K", "logN", "price", "pred_cf"]

    # Añadimos columnas one-hot si existen
    for c in df.columns:
        if c.startswith("gen_") or c.startswith("type_"):
            num_cols.append(c)

    df = df[num_cols + ["rating"]].fillna(0)
    print(f"[rf_model] Dataset final: {df.shape[0]} filas, {len(num_cols)} features.")
    return df, num_cols


def train_rf(df: pd.DataFrame, features: list[str]) -> RandomForestRegressor:
    """
    Entrena el modelo Random Forest y muestra métricas básicas.
    """
    X = df[features]
    y = df["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[rf_model] Entrenando Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    print(f"[rf_model] Evaluación: MAE={mae:.4f}  RMSE={rmse:.4f}")
    return model


if __name__ == "__main__":
    from src.preprocess import load_csvs, preprocess_items, preprocess_users

    print("=== Entrenando modelo RF híbrido SMARTUR ===")
    items, users, ratings, pairs = load_csvs()
    items_p, _ = preprocess_items(items)
    users_p, _ = preprocess_users(users)
    users_list, sim = load_user_sim()

    df, features = build_training_dataset(ratings, users_p, items_p, users_list, sim)
    rf_model = train_rf(df, features)

    joblib.dump({"model": rf_model, "features": features}, MODEL_PATH)
    print(f"✔ Modelo guardado en {MODEL_PATH}")
