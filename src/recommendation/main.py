from engine import SmarturEngine
from rf_model import SmarturContextModel
from fusion import recommend_hybrid
import pandas as pd

def ejecutar_smartur():
    print("=== Iniciando SMARTUR v2 (Yelp Engine) ===")
    
    # 1. Inicializar el motor y preparar la matriz de Pearson
    # Esto consumirá RAM, así que cierra pestañas de Chrome si puedes
    engine = SmarturEngine()
    engine.prepare_pearson_matrix()
    
    # 2. Inicializar y entrenar el Random Forest con el contexto de Yelp
    context_model = SmarturContextModel()
    print("Entrenando Random Forest con el 80% de los datos...")
    context_model.train(engine.train_data)
    
    # 3. Probar una recomendación Híbrida
    # Elegimos un user_id que exista en el dataset de prueba
    test_user = engine.test_data['user_id'].iloc[0]
    print(f"\nGenerando recomendación Top 3 para el usuario: {test_user}")
    
    recomendaciones = recommend_hybrid(test_user, engine, context_model, alpha=0.7)
    
    print("\n--- RESULTADOS SMARTUR ---")
    for i, rec in enumerate(recomendaciones, 1):
        print(f"{i}. Negocio ID: {rec['business_id']}")
        print(f"   Score Final: {rec['final_score']} (CF: {rec['cf_part']}, RF: {rec['rf_part']})")
        print("-" * 30)

if __name__ == "__main__":
    ejecutar_smartur()