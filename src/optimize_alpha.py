"""
Grid-search para encontrar el alpha optimo del sistema hibrido SMARTUR.
Prueba valores de 0.0 a 1.0 y reporta RMSE/MAE de cada combinacion.
"""
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from engine import SmarturEngine
from rf_model import SmarturContextModel
from cf import predict_cf_pearson


def optimize(sample_size=1000):
    print("=== Optimizacion de alpha para SMARTUR v2 ===\n")

    engine = SmarturEngine()
    engine.prepare_pearson_matrix()

    context_model = SmarturContextModel()
    context_model.train(engine.train_data)

    n_eval = min(sample_size, len(engine.test_data))
    test_sample = engine.test_data.sample(n_eval, random_state=42)

    actuals, cf_preds, rf_preds = [], [], []
    errores = 0

    total = len(test_sample)
    print(f"Pre-computando {n_eval} predicciones CF y RF...")
    for idx, (_, row) in enumerate(test_sample.iterrows()):
        if idx % 100 == 0:
            sys.stdout.write(f"\r  Progreso: {idx}/{total} ({idx/total*100:.0f}%)")
            sys.stdout.flush()
        try:
            p_cf = predict_cf_pearson(row['user_id'], row['business_id'], engine)
            p_rf = float(context_model.predict_context([row['business_id']])[0])
            if np.isnan(p_cf) or np.isnan(p_rf):
                errores += 1
                continue
            actuals.append(row['stars'])
            cf_preds.append(p_cf)
            rf_preds.append(p_rf)
        except Exception:
            errores += 1

    print(f"\r  Progreso: {total}/{total} (100%)     ")

    actuals = np.array(actuals)
    cf_preds = np.array(cf_preds)
    rf_preds = np.array(rf_preds)

    print(f"\nPredicciones validas: {len(actuals)} (errores: {errores})\n")

    # Grid search
    alphas = np.arange(0.0, 1.05, 0.1)
    best_alpha, best_rmse = 0.0, float('inf')

    print("  alpha |  RMSE   |  MAE")
    print("  ------|---------|--------")

    for alpha in alphas:
        hybrid = alpha * cf_preds + (1 - alpha) * rf_preds
        rmse = sqrt(mean_squared_error(actuals, hybrid))
        mae = mean_absolute_error(actuals, hybrid)
        marker = ""
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            marker = " <-- best"
        print(f"  {alpha:.1f}   | {rmse:.4f}  | {mae:.4f}{marker}")

    print(f"\n======================================")
    print(f"  ALPHA OPTIMO: {best_alpha:.1f}  (RMSE = {best_rmse:.4f})")
    print(f"======================================")
    print(f"\nActualiza alpha={best_alpha} en fusion.py y api.py")

    return best_alpha


if __name__ == "__main__":
    optimize()
