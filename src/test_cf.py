# src/test_cf.py
import sys
import pandas as pd
import numpy as np
from src.cognitive import load_user_sim
from src.cf import predict_cf_for_user_item, predict_cf_for_user

def main():
    print("=== SANITY CHECK: Filtrado Colaborativo (CF) ===")
    # Cargar datos
    ratings = pd.read_csv("data/ratings.csv")
    items = pd.read_csv("data/items.csv")
    users = pd.read_csv("data/users.csv")

    print(f"Loaded: users={users.shape}, items={items.shape}, ratings={ratings.shape}")

    # Cargar artefactos cognitivos
    try:
        users_list, sim = load_user_sim()
    except Exception as e:
        print("ERROR al cargar artefactos cognitivos:", e)
        sys.exit(1)

    print(f"Usuarios con patrón cognitivo: {len(users_list)}")
    # elegimos un usuario que esté en users_list si es posible
    # buscamos un user_id presente en ratings y users_list
    candidate_users = [u for u in ratings['user_id'].unique() if u in users_list]
    if not candidate_users:
        print("No hay usuarios en common entre ratings y models/users_list.npy. Revisa user_id types.")
        sys.exit(1)

    user_id = candidate_users[0]
    # elegimos un ítem cualquiera (puede ser uno que el usuario ya calificó)
    item_id = int(items['item_id'].iloc[0])

    print(f"\nSeleccionado user_id = {user_id}, item_id = {item_id}")

    # Predicción puntual
    pred_single = predict_cf_for_user_item(user_id, item_id, ratings, users_list, sim, k=20)
    print(f"\nPredicción CF (user {user_id}, item {item_id}) = {pred_single:.3f}")

    # Top-N CF para ese usuario
    topn = predict_cf_for_user(user_id, ratings, users_list, sim, items, k=20)[:10]
    print("\nTop-10 recomendaciones CF (item_id, score):")
    for iid, score in topn:
        # si existe columna 'title' o 'name' en items, mostrarla
        title = None
        if 'title' in items.columns:
            title = items.loc[items['item_id']==iid, 'title'].values[0]
        elif 'name' in items.columns:
            title = items.loc[items['item_id']==iid, 'name'].values[0]
        if title is not None:
            print(f" - {iid} ({title[:40]}) -> {score:.3f}")
        else:
            print(f" - {iid} -> {score:.3f}")

    # Mostrar vecinos más similares (top 10) del usuario en la matriz cognitiva
    idx = int(np.where(users_list == user_id)[0][0])
    sims_user = sim[idx]
    sorted_idx = np.argsort(sims_user)[::-1]
    top_neighbors = sorted_idx[1:11]  # excluir self
    print("\nTop-10 vecinos (user_id, similitud):")
    for ni in top_neighbors:
        print(f" - {users_list[ni]} -> {sims_user[ni]:.4f}")

    # Comprobación: cuántos usuarios calificaron el primer ítem
    cnt = ratings.loc[ratings['item_id']==item_id].shape[0]
    print(f"\nNúmero de ratings para item {item_id}: {cnt}")

if __name__ == "__main__":
    main()
