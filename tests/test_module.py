# tests/test_module.py
import pandas as pd
from src.preprocess import preprocess_items, preprocess_users
from src.cognitive import build_user_cognitive_pattern, compute_user_cogsim

def test_preprocess_and_cognitive_small():
    items = pd.DataFrame([
        {'item_id': 1, 'R':4.5, 'K':10, 'N':100, 'type':'hotel'},
        {'item_id': 2, 'R':4.0, 'K':20, 'N':50, 'type':'restaurant'},
        {'item_id': 3, 'R':3.5, 'K':30, 'N':10, 'type':'attraction'},
    ])
    users = pd.DataFrame([{'user_id': 10, 'edad':30, 'genero':'M'}])
    pairs = pd.DataFrame([{'user_id':10,'item_a':1,'item_b':2}])
    items_p, _ = preprocess_items(items)
    users_p, _ = preprocess_users(users)
    user_cog = build_user_cognitive_pattern(pairs, items_p)
    users_list, sim = compute_user_cogsim(user_cog)
    assert user_cog.shape[0] == 1
    assert sim.shape[0] == 1
