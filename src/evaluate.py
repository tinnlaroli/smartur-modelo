import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from engine import SmarturEngine
from rf_model import SmarturContextModel
from cf import predict_cf_pearson


def evaluar_modelo(sample_size=1000):
    print("=== Evaluacion SMARTUR v2 ===\n")

    engine = SmarturEngine()
    engine.prepare_pearson_matrix()

    context_model = SmarturContextModel()
    context_model.train(engine.train_data)

    n_test = len(engine.test_data)
    n_eval = min(sample_size, n_test)
    test_sample = engine.test_data.sample(n_eval, random_state=42)

    actuals, preds_cf, preds_rf, preds_hybrid = [], [], [], []
    errores = 0

    print(f"Evaluando {n_eval} de {n_test} interacciones del test set...")

    total = len(test_sample)
    for idx, (_, row) in enumerate(test_sample.iterrows()):
        if idx % 100 == 0:
            pct = idx / total * 100
            sys.stdout.write(f"\r  Progreso: {idx}/{total} ({pct:.0f}%)")
            sys.stdout.flush()

        try:
            p_cf = predict_cf_pearson(row['user_id'], row['business_id'], engine)
            p_rf = float(context_model.predict_context([row['business_id']])[0])
            if np.isnan(p_cf) or np.isnan(p_rf):
                errores += 1
                continue
            p_hybrid = (0.1 * p_cf) + (0.9 * p_rf)

            actuals.append(row['stars'])
            preds_cf.append(p_cf)
            preds_rf.append(p_rf)
            preds_hybrid.append(p_hybrid)
        except Exception as e:
            errores += 1
            if errores <= 3:
                print(f"\n  [warn] Error en u={row['user_id']}, b={row['business_id']}: {e}")

    print(f"\r  Progreso: {total}/{total} (100%)     ")

    actuals = np.array(actuals)
    preds_cf = np.array(preds_cf)
    preds_rf = np.array(preds_rf)
    preds_hybrid = np.array(preds_hybrid)

    media_global = engine.train_data['stars'].mean()
    preds_baseline = np.full_like(actuals, media_global)

    print(f"\nPredicciones exitosas: {len(actuals)}")
    print(f"Predicciones fallidas: {errores}")

    print("\n+--------------------------------+---------+--------+")
    print("| Componente                     |  RMSE   |  MAE   |")
    print("+--------------------------------+---------+--------+")

    for nombre, preds in [
        ("Baseline (media global)", preds_baseline),
        ("CF (Pearson + KNN)", preds_cf),
        ("RF (Random Forest)", preds_rf),
        ("Hibrido (a=0.1)", preds_hybrid),
    ]:
        rmse = sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)
        print(f"| {nombre:<30s} | {rmse:.4f}  | {mae:.4f} |")

    print("+--------------------------------+---------+--------+")

    errors_abs = np.abs(actuals - preds_hybrid)
    print("\nDistribucion de error absoluto (Hibrido):")
    for threshold in [0.5, 1.0, 1.5, 2.0]:
        pct = (errors_abs <= threshold).mean() * 100
        print(f"  |error| <= {threshold}: {pct:.1f}%")


if __name__ == "__main__":
    evaluar_modelo()
