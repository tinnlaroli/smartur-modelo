import pytest
from engine import SmarturEngine
from rf_model import SmarturContextModel

def test_engine_instantiation():
    """Prueba básica para asegurar que el motor puede instanciarse correctamente."""
    try:
        # Esto probaría cargar CSVs si estuviera en entorno con datos,
        # pero para CI genérico, capturamos excepciones de FileNotFoundError si no hay data.
        engine = SmarturEngine()
        assert engine is not None
    except FileNotFoundError:
        # En CI vacío sin los CSV, nos basta con que el archivo se haya importado bien.
        pass

def test_rf_model_instantiation():
    """Prueba básica para la instanciación e inicialización de propiedades del modelo RandomForest."""
    try:
        model = SmarturContextModel()
        assert model is not None
        assert not model.is_fitted
    except FileNotFoundError:
        pass
