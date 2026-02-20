import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from engine import SmarturEngine
from rf_model import SmarturContextModel
from cf import predict_cf_pearson

def evaluar_modelo():
    print("=== Iniciando Evaluación de SMARTUR v2 ===")
    
    # 1. Preparar Motor y Cargar Datos
    engine = SmarturEngine()
    engine.prepare_pearson_matrix()
    
    context_model = SmarturContextModel()
    context_model.train(engine.train_data)
    
    # 2. Tomar una muestra del set de prueba (20%) para evaluar
    # Evaluamos 500 interacciones aleatorias para que no tarde demasiado
    test_sample = engine.test_data.sample(500, random_state=42)
    
    actuals = []
    predictions = []
    
    print(f"Evaluando {len(test_sample)} predicciones...")
    
    for _, row in test_sample.iterrows():
        u_id = row['user_id']
        b_id = row['business_id']
        real_rating = row['stars']
        
        # Predicción Híbrida (Alpha 0.7)
        try:
            p_cf = predict_cf_pearson(u_id, b_id, engine)
            # Para el RF, como es un solo negocio, pedimos la predicción
            p_rf = context_model.predict_context([b_id])[0]
            
            p_hybrid = (0.7 * p_cf) + (0.3 * p_rf)
            
            actuals.append(real_rating)
            predictions.append(p_hybrid)
        except:
            continue
            
    # 3. Calcular Métricas
    rmse = sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    print("\n--- MÉTRICAS DE PRECISIÓN ---")
    print(f"RMSE (Error Cuadrático Medio): {round(rmse, 4)}")
    print(f"MAE (Error Absoluto Medio):   {round(mae, 4)}")
    print("------------------------------")
    print("Nota: Un RMSE cercano a 1.0 es excelente en datasets reales de Yelp.")

if __name__ == "__main__":
    evaluar_modelo()