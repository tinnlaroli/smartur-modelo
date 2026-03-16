from engine import SmarturEngine
from rf_model import SmarturContextModel
from fusion import recommend_hybrid


def ejecutar_smartur():
    print("=== Iniciando SMARTUR v2 ===")

    engine = SmarturEngine()
    engine.prepare_pearson_matrix()

    context_model = SmarturContextModel()
    print("Entrenando Random Forest con el 80% de los datos...")
    context_model.train(engine.train_data)

    test_user = engine.test_data['user_id'].iloc[0]
    print(f"\nRecomendación Top 5 para usuario: {test_user}")

    recomendaciones = recommend_hybrid(
        test_user, engine, context_model, alpha=0.1, top_n=5
    )

    print("\n--- RESULTADOS SMARTUR ---")
    for i, rec in enumerate(recomendaciones, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Score: {rec['score']}  (CF: {rec['pred_cf']}, RF: {rec['pred_rf']})")
        print("-" * 40)


if __name__ == "__main__":
    ejecutar_smartur()
