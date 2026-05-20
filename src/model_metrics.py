"""
Comparativa formal de los 3 algoritmos ML y persistencia de métricas / configuración óptima.
"""
import json
import os
from math import sqrt

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from cf import predict_cf_pearson

_DIR = os.path.dirname(os.path.abspath(__file__))
_METRICS_PATH = os.path.join(_DIR, '..', 'models', 'algorithm_metrics.json')

DEFAULT_METRICS = {
    'best_algorithm': 'hybrid',
    'best_alpha': 0.2,
    'local_blend': {'rf': 0.55, 'gbm': 0.45},
    'algorithms': {},
    'ranking': {},
}


def _rmse_mae(actuals, preds):
    actuals = np.asarray(actuals, dtype=float)
    preds = np.asarray(preds, dtype=float)
    return {
        'rmse': float(sqrt(mean_squared_error(actuals, preds))),
        'mae': float(mean_absolute_error(actuals, preds)),
    }


def compare_algorithms(engine, rf_model, gbm_model, sample_size=800, hybrid_alpha=0.2):
    """
    Evalúa CF+KNN, Random Forest y Gradient Boosting; elige el mejor por RMSE.
    Retorna métricas y pesos recomendados para producción.
    """
    from evaluate import _infer_user_context

    n_eval = min(sample_size, len(engine.test_data))
    test_sample = engine.test_data.sample(n_eval, random_state=42)

    actuals, preds_cf, preds_rf, preds_gbm = [], [], [], []
    user_contexts = {
        uid: _infer_user_context(uid, engine)
        for uid in test_sample['user_id'].unique()
    }

    for _, row in test_sample.iterrows():
        try:
            p_cf = predict_cf_pearson(row['user_id'], row['business_id'], engine)
            ctx = user_contexts.get(row['user_id'])
            p_rf = float(
                rf_model.predict_with_context([row['business_id']], user_context=ctx)[0]
            )
            p_gbm = float(
                gbm_model.predict_with_context([row['business_id']], user_context=ctx)[0]
            )
            if np.isnan(p_cf) or np.isnan(p_rf) or np.isnan(p_gbm):
                continue
            actuals.append(row['stars'])
            preds_cf.append(p_cf)
            preds_rf.append(p_rf)
            preds_gbm.append(p_gbm)
        except Exception:
            continue

    if len(actuals) < 50:
        return DEFAULT_METRICS

    actuals = np.array(actuals)
    preds_cf = np.array(preds_cf)
    preds_rf = np.array(preds_rf)
    preds_gbm = np.array(preds_gbm)

    media = float(engine.train_data['stars'].mean())
    preds_baseline = np.full_like(actuals, media)

    metrics = {
        'baseline': _rmse_mae(actuals, preds_baseline),
        'cf_knn_pearson': _rmse_mae(actuals, preds_cf),
        'random_forest': _rmse_mae(actuals, preds_rf),
        'gradient_boosting': _rmse_mae(actuals, preds_gbm),
    }

    alphas = np.arange(0.0, 1.05, 0.1)
    best_alpha, best_hybrid_rmse = hybrid_alpha, float('inf')
    for alpha in alphas:
        hybrid = alpha * preds_cf + (1 - alpha) * preds_rf
        rmse = sqrt(mean_squared_error(actuals, hybrid))
        if rmse < best_hybrid_rmse:
            best_hybrid_rmse = rmse
            best_alpha = float(alpha)

    hybrid_preds = best_alpha * preds_cf + (1 - best_alpha) * preds_rf
    metrics['hybrid_cf_rf'] = {
        **_rmse_mae(actuals, hybrid_preds),
        'alpha': best_alpha,
    }

    triple = 0.15 * preds_cf + 0.50 * preds_rf + 0.35 * preds_gbm
    metrics['hybrid_triple'] = _rmse_mae(actuals, triple)

    candidates = {
        'cf_knn_pearson': metrics['cf_knn_pearson']['rmse'],
        'random_forest': metrics['random_forest']['rmse'],
        'gradient_boosting': metrics['gradient_boosting']['rmse'],
        'hybrid_cf_rf': metrics['hybrid_cf_rf']['rmse'],
        'hybrid_triple': metrics['hybrid_triple']['rmse'],
    }
    best_algorithm = min(candidates, key=candidates.get)

    rf_rmse = metrics['random_forest']['rmse']
    gbm_rmse = metrics['gradient_boosting']['rmse']
    total_inv = (1 / rf_rmse) + (1 / gbm_rmse)
    w_rf = (1 / rf_rmse) / total_inv
    w_gbm = (1 / gbm_rmse) / total_inv

    result = {
        'best_algorithm': best_algorithm,
        'best_alpha': best_alpha if best_algorithm == 'hybrid_cf_rf' else hybrid_alpha,
        'local_blend': {'rf': round(w_rf, 3), 'gbm': round(w_gbm, 3)},
        'algorithms': metrics,
        'sample_size': len(actuals),
    }
    return result


def save_metrics(metrics):
    os.makedirs(os.path.dirname(_METRICS_PATH), exist_ok=True)
    with open(_METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def load_metrics():
    if not os.path.exists(_METRICS_PATH):
        return dict(DEFAULT_METRICS)
    try:
        with open(_METRICS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        merged = dict(DEFAULT_METRICS)
        merged.update(data)
        if 'local_blend' not in merged or not merged['local_blend']:
            merged['local_blend'] = DEFAULT_METRICS['local_blend']
        return merged
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_METRICS)
