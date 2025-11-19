# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
from typing import Tuple

MODEL_ARTIFACT = "models/scalers_and_encoders.pkl"

def safe_log1p_series(s: pd.Series) -> pd.Series:
    return np.log1p(s.fillna(0).astype(float))

def _make_onehot_encoder(**kwargs):
    """
    Crea OneHotEncoder compatible con múltiples versiones de sklearn.
    En versiones nuevas el argumento se llama sparse_output; antes era sparse.
    """
    try:
        # prefer sparse_output si está disponible
        return OneHotEncoder(sparse_output=False, **{k: v for k, v in kwargs.items()})
    except TypeError:
        # versiones antiguas usan 'sparse'
        return OneHotEncoder(sparse=False, **{k: v for k, v in kwargs.items()})

def preprocess_items(items: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    df = items.copy()
    # ensure required cols
    for c in ['R','K','N']:
        if c not in df.columns:
            df[c] = 0.0
    df['logN'] = safe_log1p_series(df['N'])
    # fill price if missing
    if 'price' not in df.columns:
        df['price'] = 0.0
    df[['R','K','logN','price']] = df[['R','K','logN','price']].astype(float).fillna(0.0)

    scaler = MinMaxScaler()
    df[['R','K','logN','price']] = scaler.fit_transform(df[['R','K','logN','price']])

    # category / type one-hot
    enc = None
    if 'type' in df.columns:
        enc = _make_onehot_encoder(handle_unknown='ignore')
        cat = enc.fit_transform(df[['type']].fillna('unknown'))
        # cat may be numpy array
        cat_names = [f"type_{c}" for c in enc.categories_[0]]
        cat_df = pd.DataFrame(cat, columns=cat_names, index=df.index)
        df = pd.concat([df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)

    artifacts = {'scaler_items': scaler, 'enc_type': enc}
    joblib.dump(artifacts, MODEL_ARTIFACT)
    return df, artifacts

def preprocess_users(users: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    df = users.copy()
    # basic fills
    if 'edad' not in df.columns:
        df['edad'] = 30
    df['edad'] = df['edad'].astype(float).fillna(30.0)
    enc = None
    if 'genero' in df.columns:
        enc = _make_onehot_encoder(handle_unknown='ignore')
        gen = enc.fit_transform(df[['genero']].fillna('X'))
        gen_names = [f"gen_{c}" for c in enc.categories_[0]]
        gen_df = pd.DataFrame(gen, columns=gen_names, index=df.index)
        df = pd.concat([df.reset_index(drop=True), gen_df.reset_index(drop=True)], axis=1)
    artifacts = {'enc_user_genero': enc}
    # Note: we save same artifact file; consider saving separate artifacts as needed
    joblib.dump(artifacts, MODEL_ARTIFACT)
    return df, artifacts

def load_csvs(path_data="data"):
    items = pd.read_csv(f"{path_data}/items.csv")
    users = pd.read_csv(f"{path_data}/users.csv")
    ratings = pd.read_csv(f"{path_data}/ratings.csv")
    pairs = pd.read_csv(f"{path_data}/pairs_feedback.csv")
    return items, users, ratings, pairs

if __name__ == "__main__":
    items, users, ratings, pairs = load_csvs()
    items_p, art1 = preprocess_items(items)
    users_p, art2 = preprocess_users(users)
    print("Preprocessed items:", items_p.shape)
    print("Preprocessed users:", users_p.shape)
