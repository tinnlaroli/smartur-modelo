# src/evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from fusion import recommend_top3
import json

def precision_at_k(true_items, recommended_items, k=3):
    return sum(1 for r in recommended_items[:k] if r in true_items) / k

def ndcg_at_k(true_items_with_relevance, recommended_items, k=3):
    # true_items_with_relevance: dict item->relevance (e.g., rating)
    def dcg(rs):
        return sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(rs))
    rels = [true_items_with_relevance.get(it, 0) for it in recommended_items[:k]]
    dcg_v = dcg(rels)
    ideal = sorted(true_items_with_relevance.values(), reverse=True)[:k]
    idcg = dcg(ideal) if ideal else 1.0
    return dcg_v / idcg

# Small evaluation harness example: compute Precision@3 across sample users
def eval_topk(users_sample, candidate_pool_fn, ratings_df, users_df, items_df, users_list, sim_matrix, rf_model, alpha=0.6):
    precisions = []
    ndcgs = []
    for u in users_sample:
        # true relevant items: items rated >=4 in test set (this is example; in practice split temporally)
        true_items = set(ratings_df.loc[(ratings_df['user_id']==u) & (ratings_df['rating']>=4),'item_id'].tolist())
        if not true_items:
            continue
        candidates = candidate_pool_fn(u)  # implement this externally
        res = recommend_top3(u, candidates, ratings_df, users_df, items_df, users_list, sim_matrix, rf_model, alpha=alpha)
        rec_items = [r['item_id'] for r in res]
        precisions.append(precision_at_k(true_items, rec_items, k=3))
        # build relevance dict
        rels = {it: ratings_df.loc[(ratings_df['user_id']==u)&(ratings_df['item_id']==it),'rating'].max() for it in true_items}
        ndcgs.append(ndcg_at_k(rels, rec_items, k=3))
    return {'precision@3': float(np.mean(precisions)) if precisions else 0.0, 'ndcg@3': float(np.mean(ndcgs)) if ndcgs else 0.0}
